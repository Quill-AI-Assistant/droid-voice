"""reward — the JUDGE: a pure-numpy reward model over the operator's keep/drop votes.

The Phase-0 spike of the droid-voice rebuild. The judge maps a (candidate patch, context) to a
scalar "ear utility" learned from the votes. It is NOT a generator — it only RANKS
candidates the FROZEN synth already produced (generate_arrangement + jitter). That
is the whole architectural fix: the model can no longer flatten phrases, because it
never picks knobs.

Closed-form GP regression on +1/-1 (keep/drop) labels — the simplest judge that
answers "does ranking-by-a-learned-model beat the plain synth by ear?".

MLX-NATIVE learning core: the kernel / solve / score run on Apple-Silicon MLX
(float32, unified memory / GPU) when available, with a transparent numpy fallback
(set DROID_MLX=0 to force numpy). Persisted models stay plain numpy .npz, so the
play path loads with no MLX dependency. The FROZEN synth stays numpy (its cue WAVs
are sha256-byte-identity-gated — never change its float math). No torch / sklearn.
"""
import copy
import json
import math
import os

import numpy as np

try:
    import mlx.core as mx
    _MLX_OK = True
except Exception:                        # non-Apple / MLX absent -> numpy fallback
    mx = None
    _MLX_OK = False


def mlx_enabled():
    """MLX powers the learning core unless it's absent or DROID_MLX=0."""
    return _MLX_OK and os.environ.get("DROID_MLX", "1") != "0"


def backend():
    return "mlx" if mlx_enabled() else "numpy"

from dvoice import emotion as de
from dvoice import features as F

# Fixed reg in standardized feature space. keep/drop is a noisy human signal, so a
# real noise floor (regularize, don't interpolate) — the small-data discipline the
# trainer already uses. Tuned by the Phase-0 held-out ranking accuracy + the ear.
DEFAULT_LS = 1.0
DEFAULT_NOISE = 0.5

# Single source of truth for the train gate + minimum dataset size (used by both
# droid-train AND the studio so they can't disagree). GATE_AUC is applied to the
# leak-free WITHIN-CASE held-out AUC (fit_and_eval), where 0.50 is a true coin-flip,
# so 0.60 means the judge ranks within a case rather than via the (valence, arousal)
# leak. MIN_ROWS is the floor below which a 20% holdout has too few keep/drop pairs
# PER CASE for the AUC to mean anything.
GATE_AUC = 0.60
MIN_ROWS = 60

# Bump when featurize() changes shape/meaning — a persisted judge with a different
# version is rejected on load (stale features -> silently-wrong ranking otherwise).
FEATURE_VERSION = 2

# Gesture vocabulary in a FROZEN order so the structural histogram dim is stable
# across train (votes) and play (candidates). The knob codec (patch_to_knobs) drops
# `g`/`tex`; the judge was blind to the most perceptually salient axis. We add it back.
GESTURE_ORDER = ("bend", "boing", "chirp", "dip", "fall", "flat",
                 "rise", "sag", "stutter", "trill", "waver")


def struct_feats(patch):
    """Structural / perceptual features the knob codec throws away: note count, the
    gesture-contour histogram, and texture presence (warble / ring / sample-&-hold).
    These are what make two phrases sound DIFFERENT, so the judge must see them."""
    notes = patch.get("notes") or []
    n = len(notes)
    hist = np.zeros(len(GESTURE_ORDER), dtype=np.float64)
    warble = ring = sh = 0.0
    for note in notes:
        g = note.get("g")
        if g in GESTURE_ORDER:
            hist[GESTURE_ORDER.index(g)] += 1.0
        tex = note.get("tex") or []
        if "warble" in tex or float(note.get("fm_depth", 0.0)) > 0.0:
            warble += 1.0
        if float(note.get("ring_depth", 0.0)) > 0.03:
            ring += 1.0
        if float(note.get("sh_rate", 0.0)) > 0.5:
            sh += 1.0
    if n:
        hist /= n
        warble, ring, sh = warble / n, ring / n, sh / n
    return np.concatenate([[n / 6.0], hist, [warble, ring, sh]]).astype(np.float64)


def featurize(patch, valence, arousal):
    """A candidate -> feature vector: squashed 57-knob code + structural feats
    (gesture/texture/length) + (valence, arousal). Reuses the FROZEN codec for the
    knobs and adds back the perceptual structure the codec drops."""
    z = F.squash(F.patch_to_knobs(patch))                       # logit-unit, N_KNOBS
    s = struct_feats(patch)                                     # 15-d structure
    return np.concatenate([z, s, [float(valence), float(arousal)]]).astype(np.float64)


def load_votes(path):
    """preferences.jsonl -> (X, y, meta). keep=+1, drop=-1. Rows without a usable
    patch, or 'ab'/unknown votes, are skipped. Tolerant of blank/garbage lines."""
    X, y, meta = [], [], []
    if not os.path.exists(path):
        return np.zeros((0, 0)), np.zeros((0,)), []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("tombstoned"):                             # Phase-1 soft-delete
                continue
            vote, patch = r.get("vote"), r.get("winner_patch")
            if patch is None or vote not in ("keep", "drop"):
                continue
            rv, ra = r.get("valence"), r.get("arousal")
            if rv is None or ra is None:            # no circumplex coords -> not placeable
                continue
            v, a = float(rv), float(ra)
            try:
                X.append(featurize(patch, v, a))
            except Exception:
                continue
            y.append(1.0 if vote == "keep" else -1.0)
            meta.append({"target": r.get("target"), "vote": vote, "v": v, "a": a})
    if not X:
        return np.zeros((0, 0)), np.zeros((0,)), []
    return np.asarray(X), np.asarray(y), meta


def load_dataset(profile):
    """The NEW source of truth: profiles/<p>/dataset.jsonl via the store (tombstones
    already applied). Returns (X, y, meta). label y/k -> +1 (good/anchor), n -> -1."""
    from dvoice import store
    X, y, meta = [], [], []
    for r in store.load(profile):
        patch, label = r.get("patch"), r.get("label")
        ctx = r.get("ctx") or [0.0, 0.0]
        if patch is None or label not in store.VOTE_LABELS:
            continue
        v, a = float(ctx[0]), float(ctx[1])
        try:
            X.append(featurize(patch, v, a))
        except Exception:
            continue
        y.append(-1.0 if label == "n" else 1.0)
        meta.append({"case": r.get("case"), "label": label, "v": v, "a": a})
    if not X:
        return np.zeros((0, 0)), np.zeros((0,)), []
    return np.asarray(X), np.asarray(y), meta


def _median_lengthscale(Xs, seed=0, cap=400):
    """Median-heuristic length-scale: the median pairwise distance in standardized
    space. A scalar ls=1.0 collapses the Matern kernel to a near-neighbour lookup in
    high-d (d~59 here); the median heuristic sets the kernel's reach to the data's
    actual spread so it generalizes instead of memorizing."""
    n = Xs.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:min(n, cap)]                     # subsample for the O(n^2)
    S = Xs[idx]
    d2 = (S * S).sum(1)[:, None] + (S * S).sum(1)[None, :] - 2.0 * (S @ S.T)
    iu = np.triu_indices(len(S), k=1)
    dist = np.sqrt(np.maximum(d2[iu], 0.0))
    med = float(np.median(dist)) if dist.size else 1.0
    return med if med > 1e-6 else 1.0


_S5 = math.sqrt(5.0)


def _matern52_mlx(Xa, Xb, ls, sv):
    """Matérn-5/2 ARD covariance on MLX (float32) — same formula as features.matern52,
    returns an mx array. Used by fit/score when MLX is enabled."""
    A = mx.array(np.asarray(Xa), dtype=mx.float32) / mx.array(np.asarray(ls), dtype=mx.float32)
    B = mx.array(np.asarray(Xb), dtype=mx.float32) / mx.array(np.asarray(ls), dtype=mx.float32)
    aa = mx.expand_dims(mx.sum(A * A, axis=1), 1)
    bb = mx.expand_dims(mx.sum(B * B, axis=1), 0)
    d2 = mx.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
    r = mx.sqrt(d2)
    return float(sv) * (1.0 + _S5 * r + (5.0 / 3.0) * d2) * mx.exp(-_S5 * r)


def _solve_mlx(K, y):
    """Solve K alpha = y on MLX, falling back to numpy if the MLX solve is unavailable."""
    b = mx.array(np.asarray(y), dtype=mx.float32)
    try:
        return np.asarray(mx.linalg.solve(K, b, stream=mx.cpu), dtype=np.float64)
    except TypeError:
        return np.asarray(mx.linalg.solve(K, b), dtype=np.float64)
    except Exception:
        return np.linalg.solve(np.asarray(K, dtype=np.float64), np.asarray(y, dtype=np.float64))


def fit(X, y, length_scale="median", noise=DEFAULT_NOISE):
    """Closed-form GP regression -> a judge dict (numpy-stored, MLX-computed when on).
    Standardizes inputs; guards zero-variance dims; median-heuristic length-scale."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)                           # zero-variance guard
    Xs = (X - mu) / sd
    ls_scalar = _median_lengthscale(Xs) if length_scale == "median" else float(length_scale)
    ls = np.full(d, ls_scalar)
    if mlx_enabled():
        K = _matern52_mlx(Xs, Xs, ls, 1.0) + mx.eye(n, dtype=mx.float32) * float(noise + 1e-8)
        alpha = _solve_mlx(K, y)
    else:
        K = F.matern52(Xs, Xs, ls, 1.0)
        K[np.diag_indices(n)] += float(noise) + 1e-8
        alpha = np.linalg.solve(K, y)
    return {"in_mu": mu, "in_sd": sd, "Xs": Xs, "length_scales": ls,
            "signal_var": 1.0, "alpha": np.asarray(alpha, dtype=np.float64),
            "n": int(n), "d": int(d), "ls": float(ls_scalar), "backend": backend()}


def score(judge, phi):
    """Utility of a feature vector (1-d) or batch ((m,d)): k_star @ alpha (MLX or numpy)."""
    single = np.asarray(phi).ndim == 1
    q = np.atleast_2d(np.asarray(phi, dtype=np.float64))
    qs = (q - judge["in_mu"]) / judge["in_sd"]
    if mlx_enabled():
        k = _matern52_mlx(qs, judge["Xs"], judge["length_scales"], judge["signal_var"])
        alpha = mx.array(np.asarray(judge["alpha"]), dtype=mx.float32).reshape((-1, 1))
        out = np.asarray(k @ alpha, dtype=np.float64)[:, 0]
    else:
        k = F.matern52(qs, judge["Xs"], judge["length_scales"], judge["signal_var"])
        out = k @ judge["alpha"]
    return float(out[0]) if single else np.asarray(out)


def _scale_tempo(patch, factor):
    """Scale a phrase's overall LENGTH: multiply every note dur/decay + the gap by
    `factor` (clamped to bounds). Lets candidates differ in duration, not just contour."""
    p = copy.deepcopy(patch)
    for n in p.get("notes", []):
        if "dur" in n:
            n["dur"] = n["dur"] * factor
        if "decay_tau" in n:
            n["decay_tau"] = n["decay_tau"] * factor
    if "gap" in p:
        p["gap"] = p["gap"] * factor
    return de._clamp_patch(p)


def _vary_timbre(patch, rng):
    """Roll a TIMBRE profile onto a candidate so the pool spans bright<->dark,
    clean<->warbly, dry<->metallic, smooth<->gritty, tight<->wide — not just contour
    (the thing that made candidates 'sound almost the same'). Only sets knobs; the
    FROZEN synth renders any patch; _clamp_patch keeps it in bounds."""
    p = copy.deepcopy(patch)
    lo, hi, _ = de.GLOBAL_BOUNDS["lp_cutoff"]
    p["lp_cutoff"] = p.get("lp_cutoff", 2600.0) * (2.0 ** rng.uniform(-0.55, 0.55))
    dlo, dhi, _ = de.GLOBAL_BOUNDS["detune_cents"]
    p["detune_cents"] = float(rng.uniform(dlo, dhi))
    warble = float(rng.uniform(0.0, 12.0))                  # clean .. heavy vibrato
    fm_rate = float(rng.uniform(5.0, 12.0))
    ring = float(rng.choice([0.0, 0.0, rng.uniform(0.12, 0.5)]))     # 1/3 metallic
    ring_hz = float(rng.uniform(90.0, 380.0))
    sh = float(rng.choice([0.0, 0.0, 0.0, rng.uniform(5.0, 16.0)]))  # 1/4 gritty
    for n in p.get("notes", []):
        n["fm_depth"] = warble
        n["fm_rate"] = fm_rate
        if ring > 0:
            n["ring_depth"] = ring
            n["ring_hz"] = ring_hz
        if sh > 0:
            n["sh_rate"] = sh
    return de._clamp_patch(p)


def diverse_candidates(profile, valence, arousal, text="", *, k=12, sigma=0.18,
                       va_jolt=0.13, seed=0, kind=None, name=None, tags=()):
    """A STRUCTURALLY diverse candidate pool — candidates differ in NOTE COUNT, LENGTH,
    contour and texture, not just pitch (the old pool was all 4-note ~0.7s phrases).

    VOCALIZATION cases synthesize their own burst archetype (laugh/sigh/hmm/...) and vary
    it by re-seeding the builder + a small (v,a) nudge + a tempo stretch + jitter.
    EMOTION / EXPRESSION cases use the phrase generator with a note-count sweep CENTERED
    on the emotion's own count (so the pool reflects the feeling's energy and spans the
    1-8 range), a contour-excursion sweep (range_scale), texture (ring), a tempo stretch
    (~0.66x..1.6x length), then jitter. candidate[0] is always the exact default."""
    rng = np.random.default_rng(seed)
    if kind == "vocalization":
        cands = [de.generate_vocalization(profile, name, valence, arousal, tags=tags, seed=0)]
        for _ in range(max(0, k - 1)):
            v = float(np.clip(valence + rng.normal(0.0, va_jolt * 0.6), -1.0, 1.0))
            a = float(np.clip(arousal + rng.normal(0.0, va_jolt * 0.6), -1.0, 1.0))
            p = de.generate_vocalization(profile, name, v, a, tags=tags,
                                         seed=int(rng.integers(1, 1_000_000)))
            p = _scale_tempo(p, float(2.0 ** rng.uniform(-0.4, 0.5)))      # length variety
            cands.append(_vary_timbre(de.jitter(p, sigma, rng), rng))      # + timbre variance
        return cands
    cands = [de.generate_arrangement(profile, valence, arousal, text)]     # exact default
    has_text = bool((text or "").strip())                                  # a phrase pins the length
    base_n = len(cands[0].get("notes", []))                                # the emotion's own count
    # sweep length AROUND the emotion's energy (not a fixed 2-5 band) so the pool both
    # REFLECTS the feeling and varies in length — clamped to the full 1-8 range.
    syl_choices = [max(1, min(8, base_n + d)) for d in (-2, -1, 0, 0, 1, 2)]
    for _ in range(max(0, k - 1)):
        v = float(np.clip(valence + rng.normal(0.0, va_jolt), -1.0, 1.0))
        a = float(np.clip(arousal + rng.normal(0.0, va_jolt), -1.0, 1.0))
        rs = float(rng.uniform(0.5, 1.4))
        ring = float(rng.choice([0.0, 0.0, 0.0, rng.uniform(0.1, 0.4)]))
        # With a phrase, let the TEXT drive note count (don't override it) so the
        # phrase's length survives candidate generation; the pool still varies via
        # (v,a) jitter, range, ring, tempo, timbre. No phrase -> sweep the count.
        syl = None if has_text else int(rng.choice(syl_choices))
        # roll a cadence: mostly the emotion's own (None), sometimes a different rhythm,
        # so repeats feel like "the same feeling told a little differently"
        rhythm = rng.choice([None, None, None, "accelerando", "even", "syncopation"])
        p = de.generate_arrangement(profile, v, a, text, ring=ring, range_scale=rs,
                                    force_syl=syl, rhythm=rhythm)
        p = _scale_tempo(p, float(2.0 ** rng.uniform(-0.6, 0.7)))          # ~0.66x..1.6x length
        cands.append(_vary_timbre(de.jitter(p, sigma, rng), rng))          # + timbre variance
    return cands


def _select_index(scores, phis, *, temperature=0.0, novelty=0.0, top=0, recent=None,
                  in_mu=None, in_sd=None, rng=None):
    """Choose a candidate index from the judge scores.

    temperature<=0 -> greedy argmax: deterministic, the byte-stable path cues/tests rely on.
    temperature>0  -> softmax-sample over the scores so repeated calls don't return the same
    utterance (this is what kills the Best-of-N determinism). `top` (>0) first restricts to
    the top-N candidates by score, so a high temperature can't wander into bad-sounding
    low-GP candidates (a quality floor). `novelty` (>0) subtracts a penalty for cosine
    similarity to `recent` plays, so a session doesn't repeat itself. When the judge's
    feature scaling (`in_mu`/`in_sd`) is supplied the similarity is computed in that
    standardized space — otherwise raw-feature cosine is dominated by a few big-scale dims."""
    s = np.asarray(scores, dtype=np.float64).copy()
    if novelty > 0.0 and recent is not None and len(recent) > 0:
        P = np.asarray(phis, dtype=np.float64)
        Rm = np.atleast_2d(np.asarray(recent, dtype=np.float64))
        if Rm.shape[1] != P.shape[1]:                         # stale ring after a feature-version bump → ignore
            Rm = None
    if novelty > 0.0 and recent is not None and len(recent) > 0 and Rm is not None:
        if in_mu is not None and in_sd is not None:           # judge's own scaling → meaningful sim
            in_mu = np.asarray(in_mu, dtype=np.float64)
            in_sd = np.asarray(in_sd, dtype=np.float64) + 1e-9
            P = (P - in_mu) / in_sd
            Rm = (Rm - in_mu) / in_sd
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        Rn = Rm / (np.linalg.norm(Rm, axis=1, keepdims=True) + 1e-9)
        s = s - float(novelty) * (Pn @ Rn.T).max(axis=1)      # nearest recent play per candidate
    if temperature is None or temperature <= 0.0:
        return int(np.argmax(s))
    if top and 0 < top < len(s):                              # quality floor: keep only the best `top`
        keep = np.argpartition(s, -top)[-top:]
        masked = np.full_like(s, -np.inf)
        masked[keep] = s[keep]
        s = masked
    finite = np.isfinite(s)
    if rng is None:
        rng = np.random.default_rng()
    if not np.any(finite):                                    # all scores NaN/-inf -> uniform (no crash)
        return int(rng.integers(len(s)))
    p = np.where(finite, np.exp((s - s[finite].max()) / float(temperature)), 0.0)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def best_of_n(judge, profile, valence, arousal, text="", *, k=12, sigma=0.18, seed=0,
              seed_patch=None, structural=True, kind=None, name=None, tags=(),
              temperature=0.0, novelty=0.0, top=0, recent=None):
    """GENERATE k candidates, JUDGE them at the TARGET (v,a), return the best.
    candidate[0] is the exact default, so Best-of-N can only match-or-beat it under the
    judge. structural=True uses diverse_candidates (varied contour/texture/length, or
    burst variants for vocalizations); False is the knob-jitter-only pool. seed_patch
    (when set) forces jitter-around-a-fixed-base (e.g. to test in the votes' distribution)."""
    def _base():
        if kind == "vocalization":
            return de.generate_vocalization(profile, name, valence, arousal, tags=tags)
        return de.generate_arrangement(profile, valence, arousal, text)

    if seed_patch is not None:
        rng = np.random.default_rng(seed)
        base = copy.deepcopy(seed_patch)
        cands = [base] + [de.jitter(base, sigma, rng) for _ in range(max(0, k - 1))]
    elif structural:
        cands = diverse_candidates(profile, valence, arousal, text, k=k, sigma=sigma,
                                   seed=seed, kind=kind, name=name, tags=tags)
    else:
        rng = np.random.default_rng(seed)
        base = _base()
        cands = [base] + [de.jitter(base, sigma, rng) for _ in range(max(0, k - 1))]
    # Judge each candidate AS a <target-(v,a)> utterance (ctx fixed at the target).
    phis = np.asarray([featurize(c, valence, arousal) for c in cands])
    scores = score(judge, phis)
    sel_rng = np.random.default_rng(seed)
    best_i = _select_index(scores, phis, temperature=temperature, novelty=novelty,
                           top=top, recent=recent, rng=sel_rng,
                           in_mu=judge.get("in_mu"), in_sd=judge.get("in_sd"))
    return {
        "patch": cands[best_i], "best_i": best_i, "scores": scores,
        "candidates": cands, "argmax_i": int(np.argmax(scores)),
        "base_score": float(scores[0]), "best_score": float(scores[best_i]),
    }


def diverse_set(judge, profile, valence, arousal, text="", *, n=5, pool_k=24, top_frac=0.5,
                seed=None, kind=None, name=None, tags=()):
    """Pick N candidates that are mutually as DISTINCT as possible while staying judge-approved.

    Best-of-N optimizes QUALITY, so repeated picks cluster (audibly same-y). For a "play N
    different" set we want SPREAD: generate an oversampled pool, keep the top `top_frac` by
    judge score (the quality floor), then farthest-point select N in the judge's standardized
    feature space (greedy max-min distance, seeded from the highest-scored). The result is N
    judge-approved takes that are as far apart as the case allows — not N near-clones."""
    pool = diverse_candidates(profile, valence, arousal, text, k=max(pool_k, n), seed=seed,
                              kind=kind, name=name, tags=tags)
    phis = np.asarray([featurize(c, valence, arousal) for c in pool])
    if judge is not None:
        scores = np.asarray(score(judge, phis), dtype=np.float64)
        mu, sd = judge.get("in_mu"), judge.get("in_sd")
    else:
        scores, mu, sd = np.zeros(len(pool)), None, None
    keep = min(len(pool), max(n, int(round(len(pool) * float(top_frac)))))  # never promise more than the pool
    idx = list(np.argsort(scores)[::-1][:keep])                 # quality floor: judge's best `keep`
    Z = phis[idx]
    if mu is not None and sd is not None:
        Z = (Z - np.asarray(mu, float)) / (np.asarray(sd, float) + 1e-9)
    chosen = [0]                                                # start from the highest-scored
    while len(chosen) < min(n, len(idx)):
        d = np.min(np.stack([np.linalg.norm(Z - Z[c], axis=1) for c in chosen]), axis=0)
        for c in chosen:
            d[c] = -1.0
        chosen.append(int(np.argmax(d)))                        # the one farthest from all chosen
    return [pool[idx[c]] for c in chosen]


def arrange_by_judge(judge, profile, valence, arousal, text="", *, k=12, sigma=0.18,
                     seed=0, kind=None, name=None, tags=(), oversample=2):
    """Curate a collection batch: generate an OVERSAMPLED diverse pool, score it with the
    active judge, and return the top-k arranged BEST-FIRST — so the operator votes on the
    most promising candidates instead of a random flood of duds (active selection). With
    no judge it falls back to a plain diverse batch (still varied, just unordered)."""
    pool = diverse_candidates(profile, valence, arousal, text,
                              k=max(k, oversample * k), sigma=sigma, seed=seed,
                              kind=kind, name=name, tags=tags)
    if judge is None or len(pool) <= k:
        return pool[:k]
    phis = np.asarray([featurize(c, valence, arousal) for c in pool])
    s = np.asarray(score(judge, phis)).ravel()
    order = np.argsort(s)[::-1]                       # best predicted first
    return [pool[int(i)] for i in order[:k]]


def heldout_ranking_accuracy(X, y, *, groups=None, frac=0.2, seed=0, **hp):
    """Honest de-risk metric: fit on a train split, then over held-out (keep, drop)
    pairs report the fraction where the judge scores keep > drop (pairwise AUC).
    0.5 = no signal learned; >0.5 = the judge generalizes.

    groups (one label per row — e.g. the case name) restricts pairing to WITHIN a
    group. This is the ONLY ranking that matters at play time: best_of_n scores every
    candidate for a case at that case's single (valence, arousal), so the (v,a) dims
    are constant across the ranked set and cannot help. Without grouping, keeps and
    drops from DIFFERENT cases get paired, and the judge can win pairs by reading the
    leaked (v,a) coordinate — inflating the AUC above how the voice actually ranks
    what you'll hear. Pass groups for the honest gate; omit for the raw global metric
    (kept for the structural-discrimination fixtures that hold (v,a) fixed)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(2, int(round(n * frac)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    judge = fit(X[train_idx], y[train_idx], **hp)
    s = score(judge, X[test_idx])
    yt = y[test_idx]
    n_pos, n_neg = int((yt > 0).sum()), int((yt < 0).sum())
    base = {"n_train": int(len(train_idx)), "n_test_pos": n_pos, "n_test_neg": n_neg}
    if groups is not None:                              # within-case pairing (honest gate)
        g = np.asarray(groups, dtype=object)[test_idx]
        pairs = []
        for grp in {x for x in g.tolist()}:
            m = np.array([x == grp for x in g.tolist()])
            sp, sn = s[m][yt[m] > 0], s[m][yt[m] < 0]
            if len(sp) and len(sn):
                pairs.append((sp[:, None] > sn[None, :]).ravel())
        if not pairs:
            return {"auc": float("nan"), **base}
        return {"auc": float(np.concatenate(pairs).mean()), **base}
    pos, neg = s[yt > 0], s[yt < 0]                     # raw global pairing (back-compat)
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": float("nan"), **base}
    return {"auc": float((pos[:, None] > neg[None, :]).mean()), **base}


def fit_and_eval(profile, seeds=(0, 1, 2, 3, 4)):
    """Read the active dataset, fit the judge, and score it by mean held-out WITHIN-CASE
    pairwise AUC over several splits (the leak-free gate). Returns a summary dict
    (judge=None when no votes). Shared by `droid train` and the studio so the gate
    metric is computed one way."""
    import collections
    X, y, meta = load_dataset(profile)
    n = len(y)
    groups = [m["case"] for m in meta]
    aucs = [heldout_ranking_accuracy(X, y, groups=groups, seed=s).get("auc") for s in seeds] if n >= 4 else []
    aucs = [a for a in aucs if a == a]
    auc = float(np.mean(aucs)) if aucs else float("nan")
    return {
        "judge": fit(X, y) if n else None,
        "auc": auc, "n": int(n),
        "n_pos": int((y > 0).sum()) if n else 0,
        "n_neg": int((y < 0).sum()) if n else 0,
        "by_case": dict(collections.Counter(m["case"] for m in meta)),
    }


def sweep_fit(profile, noises=(0.3, 0.5, 0.8), ls_muls=(0.7, 1.0, 1.5),
              seeds=(0, 1, 2, 3, 4)):
    """Train the BEST judge: grid-search (noise × length-scale) by mean held-out
    WITHIN-CASE AUC on a SELECTION split of the seeds, then report the winner's AUC on
    a DISJOINT holdout split (so the reported number is not the max-over-grid it was
    selected on — that selection bias overstates generalization), and fit on all votes
    with the winner. Same shape as fit_and_eval plus the chosen hp + the grid.
    Length-scales are multiples of the median heuristic."""
    import collections
    X, y, meta = load_dataset(profile)
    n = len(y)
    if n < 4:
        return {"judge": None, "auc": float("nan"), "n": int(n), "n_pos": 0, "n_neg": 0,
                "by_case": {}, "grid": [], "ls": None, "noise": None}
    groups = [m["case"] for m in meta]
    sel_seeds = tuple(seeds[:max(1, len(seeds) - 2)])         # choose hp on these
    rep_seeds = tuple(seeds[len(sel_seeds):]) or sel_seeds    # report on these (disjoint when possible)
    base_ls = fit(X, y)["ls"]                                 # median heuristic
    grid, best = [], None
    for noise in noises:
        for mul in ls_muls:
            ls = base_ls * mul
            aucs = [heldout_ranking_accuracy(X, y, groups=groups, seed=s, length_scale=ls, noise=noise).get("auc")
                    for s in sel_seeds]
            aucs = [a for a in aucs if a == a]
            auc = float(np.mean(aucs)) if aucs else float("nan")
            grid.append({"ls": ls, "noise": noise, "auc": auc})
            if auc == auc and (best is None or auc > best["auc"]):
                best = {"ls": ls, "noise": noise, "auc": auc}
    if best is None:                                          # every cell NaN (single-label / tiny split):
        judge = fit(X, y)                                     # fall back like fit_and_eval, don't crash
        return {"judge": judge, "auc": float("nan"), "ls": judge["ls"], "noise": DEFAULT_NOISE,
                "n": int(n), "n_pos": int((y > 0).sum()), "n_neg": int((y < 0).sum()),
                "by_case": dict(collections.Counter(m["case"] for m in meta)), "grid": grid}
    rep = [heldout_ranking_accuracy(X, y, groups=groups, seed=s,                  # winner on held-out seeds
                                    length_scale=best["ls"], noise=best["noise"]).get("auc")
           for s in rep_seeds]
    rep = [a for a in rep if a == a]
    rep_auc = float(np.mean(rep)) if rep else best["auc"]
    judge = fit(X, y, length_scale=best["ls"], noise=best["noise"])
    return {"judge": judge, "auc": rep_auc, "ls": best["ls"], "noise": best["noise"],
            "n": int(n), "n_pos": int((y > 0).sum()), "n_neg": int((y < 0).sum()),
            "by_case": dict(collections.Counter(m["case"] for m in meta)), "grid": grid}


# ── persistence (numpy .npz; pure-numpy load on the play path) ────────────────
def save_judge(judge, path):
    """Persist a fitted judge as .npz (atomic temp + os.replace). Stamps the feature
    version + dim so a stale-feature judge is rejected on load, never silently used."""
    import os as _os
    tmp = path + ".tmp"
    np.savez(tmp,
             in_mu=judge["in_mu"], in_sd=judge["in_sd"], Xs=judge["Xs"],
             length_scales=judge["length_scales"],
             signal_var=np.float64(judge["signal_var"]),
             alpha=judge["alpha"],
             feature_version=np.int64(FEATURE_VERSION),
             d=np.int64(judge.get("d", judge["Xs"].shape[1])),
             ls=np.float64(judge.get("ls", float(judge["length_scales"][0]))))
    real = tmp + ".npz" if not _os.path.exists(tmp) and _os.path.exists(tmp + ".npz") else tmp
    _os.replace(real, path)
    return path


def load_judge(path):
    """Load a judge .npz, or None if absent / feature-version mismatch (-> caller
    falls back to the analytic floor — safe by design)."""
    import os as _os
    if not _os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if int(z["feature_version"]) != FEATURE_VERSION:
            return None
        return {"in_mu": z["in_mu"], "in_sd": z["in_sd"], "Xs": z["Xs"],
                "length_scales": z["length_scales"],
                "signal_var": float(z["signal_var"]), "alpha": z["alpha"],
                "d": int(z["d"]), "ls": float(z["ls"]),
                "n": int(z["Xs"].shape[0])}
    except Exception:
        return None
