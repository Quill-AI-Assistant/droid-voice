"""features — the feature/target codec for the tiny droid control model (Task M).

This is the deterministic bridge between (valence, arousal, text) and the synth's
patch knobs. It owns:

  * the COND VECTOR X (26-dim): [valence, arousal, prosody(8), text_emb_proj(16)],
  * the FROZEN KNOB_LAYOUT and the N_KNOBS-dim TARGET Y (N_SLOTS fixed slots),
  * squash()/unsquash(): map raw synth values <-> a bounded [0,1] code (so the
    model regresses in a normalised space and can never emit out-of-bounds audio),
  * patch_to_knobs()/knobs_to_patch(): patch dict <-> the N_KNOBS-knob vector,
  * the LOGIT-RESIDUAL framing: Y is a residual over the emotion_to_patch analytic
    prior, so 2-8 calibrated anchors are enough to learn from.

Everything imports the param space from droid_emotion (the keystone) — there is
ONE source of truth for NOTE_BOUNDS / GLOBAL_BOUNDS / the analytic prior. No audio
is made here; the synth (droid_synth.render_patch) is the deterministic decoder.

stdlib + numpy only on the structural path. The text embedding (Qwen3) is OPTIONAL
and zeroed when sentence_transformers is unavailable — prosody still drives the
structure.
"""

import hashlib
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dvoice import emotion as de  # noqa: E402  (the keystone — param space + analytic prior)

# ── FROZEN target layout ──────────────────────────────────────────────────────
# The model output is a FIXED-WIDTH N_SLOTS-slot phrase. Each slot has 6 per-note
# knobs; a per-slot gate decides whether the slot is an audible note (gate>=0.5) or
# empty. Then 3 global knobs. Order is FROZEN — model.npz weights depend on it.
# N_SLOTS=6 covers the longest phrase generate_arrangement emits (the warbly voice, long text);
# a 4-slot layout silently truncated 5-6 note phrases (the model's headline case).
N_SLOTS = 6
# ring_depth + sh_rate are regressed so the model carries the ARP-2600 droid
# colour (metallic ring + sample-&-hold stutter); ring_hz is a fixed carrier
# constant (RING_HZ), a timbre choice, not regressed.
NOTE_KNOBS = ["f0", "f1", "dur", "decay_tau", "fm_rate", "fm_depth", "ring_depth", "sh_rate"]  # 8 per slot
GLOBAL_KNOBS = ["gap", "detune_cents", "lp_cutoff"]                    # 3
# KNOB_LAYOUT is an explicit, ordered list of (kind, slot, name): N_SLOTS*6 note
# knobs + N_SLOTS gates + 3 globals. kind in {"note","gate","global"}.
KNOB_LAYOUT = (
    [("note", s, k) for s in range(N_SLOTS) for k in NOTE_KNOBS]       # N_SLOTS*6
    + [("gate", s, None) for s in range(N_SLOTS)]                      # N_SLOTS
    + [("global", None, k) for k in GLOBAL_KNOBS]                      # 3
)
N_KNOBS = len(KNOB_LAYOUT)
assert N_KNOBS == N_SLOTS * (len(NOTE_KNOBS) + 1) + len(GLOBAL_KNOBS), \
    f"KNOB_LAYOUT inconsistent: got {N_KNOBS}"

# fm_rate is squashed into a tight musical band (the synth ignores fm_rate when
# fm_depth==0, and a runaway rate sounds bad) — FROZEN per contract.
FM_RATE_LO, FM_RATE_HI = 4.0, 12.0
RING_HZ = 200.0   # fixed ring-mod carrier; only ring_depth + sh_rate are regressed

# Condition vector geometry (FROZEN).
PROSODY_DIM = 8
TEXT_EMB_PROJ_DIM = 16
COND_DIM = 2 + PROSODY_DIM + TEXT_EMB_PROJ_DIM                         # 26
assert COND_DIM == 26

# Frozen seed for the Gaussian random projection that compresses the 1024-d Qwen
# embedding to 16-d. Fixed so the projection is identical across train/inference.
_PROJ_SEED = 1729
_QWEN_DIM = 1024


# ── bounds helpers ────────────────────────────────────────────────────────────
def _note_bound(name):
    """(lo, hi, kind) for a per-note knob. fm_rate uses the tight FM band; the
    rest come from droid_emotion.NOTE_BOUNDS."""
    if name == "fm_rate":
        return (FM_RATE_LO, FM_RATE_HI, "mul")
    return de.NOTE_BOUNDS[name]


def _global_bound(name):
    return de.GLOBAL_BOUNDS[name]


def _logit(p, eps=1e-4):
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    # numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _to_unit(val, lo, hi, kind):
    """Map a raw synth value into [0,1] given its bound + scale kind."""
    val = min(hi, max(lo, float(val)))
    if kind == "mul":
        llo, lhi, lv = math.log(lo), math.log(hi), math.log(max(val, 1e-9))
        return (lv - llo) / (lhi - llo) if lhi > llo else 0.0
    return (val - lo) / (hi - lo) if hi > lo else 0.0


def _from_unit(u, lo, hi, kind):
    """Inverse of _to_unit — map [0,1] back to the raw synth value (clamped)."""
    u = min(1.0, max(0.0, float(u)))
    if kind == "mul":
        llo, lhi = math.log(lo), math.log(hi)
        return float(math.exp(llo + u * (lhi - llo)))
    return float(lo + u * (hi - lo))


# ── prosody (stdlib, deterministic) ───────────────────────────────────────────
_VOWELS = set("aeiouy")


def _count_syllables(text):
    groups = 0
    prev = False
    for ch in text.lower():
        is_v = ch in _VOWELS
        if is_v and not prev:
            groups += 1
        prev = is_v
    return groups


def prosody_feats(text):
    """8-d deterministic prosody vector from raw text (stdlib only):
    [n_syllables_norm, n_words_norm, has_q, has_excl, has_period, has_comma,
     mean_word_len_norm, ellipsis_flag]. All in roughly [0,1]."""
    t = (text or "").strip()
    words = t.split()
    n_words = len(words)
    n_syl = _count_syllables(t)
    mean_wl = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    feats = [
        min(n_syl / 12.0, 1.0),                 # n_syllables_norm
        min(n_words / 12.0, 1.0),               # n_words_norm
        1.0 if "?" in t else 0.0,               # has_q
        1.0 if "!" in t else 0.0,               # has_excl
        1.0 if "." in t else 0.0,               # has_period
        1.0 if "," in t else 0.0,               # has_comma
        min(mean_wl / 10.0, 1.0),               # mean_word_len_norm
        1.0 if (t.endswith("...") or t.endswith("…")) else 0.0,  # ellipsis_flag
    ]
    return np.asarray(feats, dtype=np.float64)


# ── text embedding (OPTIONAL; zeroed when unavailable) ────────────────────────
def make_projection(seed=_PROJ_SEED, in_dim=_QWEN_DIM, out_dim=TEXT_EMB_PROJ_DIM):
    """Frozen seeded Gaussian projection (1024 -> 16). Same matrix train+infer."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / math.sqrt(in_dim), size=(in_dim, out_dim))


_EMBED_MODEL = None
_EMBED_TRIED = False


def _get_embed_model():
    """Lazily load Qwen3-Embedding-0.6B via sentence_transformers; None if absent.
    Loaded once; failure is cached so the play path never repeatedly stalls.

    Opt-IN: this optional text-conditioned feature stays OFF unless DROID_USE_EMBED
    is set, so a fresh clone never triggers a multi-hundred-MB model download. The
    shipped `say`/demo paths are prosody-only and never reach this. DROID_NO_EMBED
    is still honoured as an explicit hard-off (overrides DROID_USE_EMBED)."""
    global _EMBED_MODEL, _EMBED_TRIED
    if _EMBED_TRIED:
        return _EMBED_MODEL
    _EMBED_TRIED = True
    if os.environ.get("DROID_NO_EMBED") or not os.environ.get("DROID_USE_EMBED"):
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    except Exception:
        _EMBED_MODEL = None
    return _EMBED_MODEL


def text_embedding(text, proj, *, use_model=True):
    """Project Qwen3-Embedding-0.6B(text) -> 16-d. ZEROS when the model or the
    library is unavailable, or use_model=False, or text empty. Deterministic given
    a fixed `proj`. prosody still carries the structure when this is zero."""
    t = (text or "").strip()
    if not t or not use_model:
        return np.zeros(TEXT_EMB_PROJ_DIM, dtype=np.float64)
    model = _get_embed_model()
    if model is None:
        return np.zeros(TEXT_EMB_PROJ_DIM, dtype=np.float64)
    try:
        emb = np.asarray(model.encode([t], normalize_embeddings=True)[0], dtype=np.float64)
        if emb.shape[0] != proj.shape[0]:                 # dim guard
            return np.zeros(TEXT_EMB_PROJ_DIM, dtype=np.float64)
        return emb @ proj
    except Exception:
        return np.zeros(TEXT_EMB_PROJ_DIM, dtype=np.float64)


def cond_vector(valence, arousal, text, proj, *, use_model=True):
    """The 26-d condition vector X: [valence, arousal, prosody(8), emb_proj(16)]."""
    v = float(min(1.0, max(-1.0, valence)))
    a = float(min(1.0, max(-1.0, arousal)))
    pro = prosody_feats(text)
    emb = text_embedding(text, proj, use_model=use_model)
    return np.concatenate([[v, a], pro, emb]).astype(np.float64)


# ── patch <-> knob vector ─────────────────────────────────────────────────────
def patch_to_knobs(patch):
    """Encode a patch dict into the RAW knob vector (NOT residual). Each note
    fills a slot 0..3 (extra notes beyond 4 are dropped; per-note `mid` glides are
    not regressed). Empty slots get gate 0 and prior-ish defaults so the vector is
    well-formed. Globals from the patch or GLOBAL_DEFAULT."""
    notes = patch.get("notes") or []
    knobs = np.zeros(N_KNOBS, dtype=np.float64)
    for i, (kind, slot, name) in enumerate(KNOB_LAYOUT):
        if kind == "note":
            if slot < len(notes):
                n = notes[slot]
                if name == "f1":
                    knobs[i] = float(n.get("f1", n.get("f0", 480.0)))
                elif name == "decay_tau":
                    knobs[i] = float(n.get("decay_tau", float(n.get("dur", 0.16)) * 0.45))
                elif name == "fm_rate":
                    knobs[i] = float(n.get("fm_rate", FM_RATE_LO))
                elif name == "fm_depth":
                    knobs[i] = float(n.get("fm_depth", 0.0))
                else:                                       # f0, dur
                    knobs[i] = float(n.get(name, _note_bound(name)[0]))
            else:                                           # empty slot default
                lo, hi, _ = _note_bound(name)
                knobs[i] = lo
        elif kind == "gate":
            knobs[i] = 1.0 if slot < len(notes) else 0.0
        else:                                               # global
            knobs[i] = float(patch.get(name, de.GLOBAL_DEFAULT[name]))
    return knobs


def knobs_to_patch(knobs, prosody=None):
    """Decode a RAW knob vector into a renderable patch dict. A slot becomes a
    note iff its gate >= 0.5 (slot 0 is always kept so there is always >=1 note).
    `prosody` (the 8-d feats) parametrically adds `mid` glides — a question/excl
    bends the terminal note up — these are derived, NOT regressed (per contract)."""
    knobs = np.asarray(knobs, dtype=np.float64)
    slot_vals = {s: {} for s in range(N_SLOTS)}
    gates = {}
    globals_ = {}
    for i, (kind, slot, name) in enumerate(KNOB_LAYOUT):
        if kind == "note":
            slot_vals[slot][name] = float(knobs[i])
        elif kind == "gate":
            gates[slot] = float(knobs[i])
        else:
            globals_[name] = float(knobs[i])

    notes = []
    for s in range(N_SLOTS):
        if s == 0 or gates.get(s, 0.0) >= 0.5:              # always keep slot 0
            v = slot_vals[s]
            note = {
                "f0": v["f0"], "f1": v["f1"],
                "dur": v["dur"], "decay_tau": v["decay_tau"],
                "fm_rate": v["fm_rate"], "fm_depth": v["fm_depth"],
            }
            rd = v.get("ring_depth", 0.0)
            if rd > 0.03:                          # carry the ARP-2600 ring/metal
                note["ring_depth"] = rd
                note["ring_hz"] = RING_HZ          # fixed carrier (timbre, not regressed)
            shr = v.get("sh_rate", 0.0)
            if shr > 0.5:                          # carry sample-&-hold stutter
                note["sh_rate"] = shr
            notes.append(note)

    patch = {
        "notes": notes,
        "gap": globals_["gap"],
        "detune_cents": globals_["detune_cents"],
        "lp_cutoff": globals_["lp_cutoff"],
    }

    # Parametric glide on the terminal note from prosody (?,! => up-bend). Derived,
    # NOT regressed (per contract). A question LIFTS the terminal pitch (a clean
    # rising terminal — the canonical interrogative contour); an exclamation
    # over-shoots then settles (an emphatic arch). Both stay inside the f1 bound.
    if prosody is not None and notes:
        has_q = float(prosody[2]) > 0.5
        has_excl = float(prosody[3]) > 0.5
        f1_hi = de.NOTE_BOUNDS["f1"][1]
        term = notes[-1]
        if has_q:                                   # rise: lift f1 above f0, glide up
            base = max(term["f0"], term["f1"])
            term["f1"] = float(min(f1_hi, base * 1.18))
            mid_pt = float(min(term["f1"] * 0.92, term["f1"]))
            term["mid"] = [[0.5, mid_pt]]           # monotone-up -> reads as rise
        elif has_excl:                              # emphatic over-shoot then settle
            top = float(min(f1_hi, max(term["f0"], term["f1"]) * 1.22))
            term["mid"] = [[0.45, top]]

    return de._clamp_patch(patch)


# ── squash / unsquash (residual-over-analytic) ────────────────────────────────
def _knob_unit(knobs):
    """Map the RAW knobs into [0,1] per-dim (using each knob's bound). Gates are
    already in [0,1] (clamped)."""
    out = np.zeros(N_KNOBS, dtype=np.float64)
    for i, (kind, slot, name) in enumerate(KNOB_LAYOUT):
        if kind == "gate":
            out[i] = min(1.0, max(0.0, float(knobs[i])))
        elif kind == "note":
            lo, hi, k = _note_bound(name)
            out[i] = _to_unit(knobs[i], lo, hi, k)
        else:
            lo, hi, k = _global_bound(name)
            out[i] = _to_unit(knobs[i], lo, hi, k)
    return out


def _unit_knob(unit):
    """Inverse of _knob_unit: [0,1] per-dim -> raw knobs."""
    out = np.zeros(N_KNOBS, dtype=np.float64)
    for i, (kind, slot, name) in enumerate(KNOB_LAYOUT):
        if kind == "gate":
            out[i] = min(1.0, max(0.0, float(unit[i])))
        elif kind == "note":
            lo, hi, k = _note_bound(name)
            out[i] = _from_unit(unit[i], lo, hi, k)
        else:
            lo, hi, k = _global_bound(name)
            out[i] = _from_unit(unit[i], lo, hi, k)
    return out


def squash(raw, layout=KNOB_LAYOUT, bounds=None):
    """Encode RAW knobs -> the model's LOGIT-UNIT space. Each knob is first mapped
    to [0,1] by its bound, then logit-transformed so an MLP with a linear head can
    represent it on the whole real line. `layout`/`bounds` accepted for signature
    compatibility; the canonical layout/bounds come from this module + droid_emotion."""
    unit = _knob_unit(np.asarray(raw, dtype=np.float64))
    return np.array([_logit(u) for u in unit], dtype=np.float64)


def unsquash(knob, layout=KNOB_LAYOUT, bounds=None):
    """Inverse of squash: LOGIT-UNIT space -> RAW knobs (clamped in-bounds)."""
    unit = np.array([_sigmoid(z) for z in np.asarray(knob, dtype=np.float64)],
                    dtype=np.float64)
    return _unit_knob(unit)


# ── residual framing over the analytic prior ──────────────────────────────────
def prior_knobs(valence, arousal, profile):
    """The analytic prior (emotion_to_patch) encoded as RAW knobs — the zero of the
    residual space. The model learns Y = squash(target) - squash(prior)."""
    return patch_to_knobs(de.emotion_to_patch(valence, arousal, profile))


def residual_target(patch, valence, arousal, profile):
    """LOGIT-RESIDUAL target Y for a calibrated patch: squash(patch) - squash(prior)."""
    return squash(patch_to_knobs(patch)) - squash(prior_knobs(valence, arousal, profile))


def patch_from_residual(residual, valence, arousal, profile, prosody=None):
    """Decode a predicted LOGIT-RESIDUAL back into a renderable patch:
    unsquash(prior_logit + residual) -> knobs -> patch."""
    z = squash(prior_knobs(valence, arousal, profile)) + np.asarray(residual, dtype=np.float64)
    return knobs_to_patch(unsquash(z), prosody=prosody)


# ── Gaussian-process control model (SHARED kernel: train fits, play predicts) ──
# This is the single source of truth for the kernel so the train-time fit and the
# play-time forward can NEVER drift apart (a silent mismatch would degrade quietly
# but stay in-bounds via the clamp — caught by the kernel-fixture test). Pure numpy:
# the `droid say` play path imports this and stays sklearn/torch-free.
def matern52(Xa, Xb, length_scales, signal_var):
    """Matérn-5/2 ARD covariance. Xa:(na,d), Xb:(nb,d), length_scales:(d,) -> (na,nb).
    k(r) = signal_var * (1 + sqrt5 r + 5/3 r^2) * exp(-sqrt5 r), r = ||(Xa-Xb)/ls||."""
    Xa = np.atleast_2d(np.asarray(Xa, dtype=np.float64))
    Xb = np.atleast_2d(np.asarray(Xb, dtype=np.float64))
    ls = np.asarray(length_scales, dtype=np.float64)
    A = Xa / ls
    B = Xb / ls
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    d2 = np.maximum(d2, 0.0)
    r = np.sqrt(d2)
    s5 = math.sqrt(5.0)
    return float(signal_var) * (1.0 + s5 * r + (5.0 / 3.0) * d2) * np.exp(-s5 * r)


def gp_predict(model, x):
    """Pure-numpy GP/KRR posterior MEAN over the stored support points:
        k_star(query, Xs) @ alpha
    `model` carries in_mu/in_sd (the input scaler), Xs (n,26) standardized support
    points, length_scales (26,), signal_var, alpha (n, N_KNOBS). x is a RAW 26-d
    cond vector (or (m,26)); standardized here. Returns the (N_KNOBS,) logit-residual
    (or (m, N_KNOBS)). No sklearn/scipy — only the kernel above + a matmul."""
    single = np.asarray(x).ndim == 1
    xq = np.atleast_2d(np.asarray(x, dtype=np.float64))
    xs = (xq - model["in_mu"]) / model["in_sd"]
    k = matern52(xs, model["Xs"], model["length_scales"], model["signal_var"])  # (m,n)
    out = k @ model["alpha"]                                                    # (m, N_KNOBS)
    return out[0] if single else out


# ── bounds hash guard ─────────────────────────────────────────────────────────
def bounds_hash():
    """sha1 over (NOTE_BOUNDS, GLOBAL_BOUNDS, the fm_rate band, the knob layout).
    The `droid say` play path loads a model ONLY if this matches the model's stored
    hash — else it falls back to the analytic prior (bounds drift => stale weights)."""
    h = hashlib.sha1()
    for k in sorted(de.NOTE_BOUNDS):
        h.update(f"{k}:{de.NOTE_BOUNDS[k]}".encode())
    for k in sorted(de.GLOBAL_BOUNDS):
        h.update(f"{k}:{de.GLOBAL_BOUNDS[k]}".encode())
    h.update(f"fm_rate:{(FM_RATE_LO, FM_RATE_HI)}".encode())
    h.update(repr(KNOB_LAYOUT).encode())
    return h.hexdigest()
