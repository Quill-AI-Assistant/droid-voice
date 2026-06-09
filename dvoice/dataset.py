"""build_dataset — assemble the (X, Y, W) training table for the tiny control
model (Task M).

The model learns (valence, arousal[, text]) -> LOGIT-RESIDUAL over the analytic
prior. Calibrated anchors are scarce (2-8 per profile), so the table mixes three
sources, each weighted:

  1. GOLD anchors (weight 1.0) — every entry in profiles/<p>/emotions.json. Y is
     the residual of the calibrated patch over emotion_to_patch(v,a). These are
     your ear, the only ground truth.
  2. JITTER augmentation (weight ~1.0) — each gold anchor jittered K times (import
     jitter() from droid_emotion) with perturbed prosody, so the model sees a
     neighbourhood around each anchor and learns a smooth manifold, not 8 points.
  3. Juslin&Laukka RULE-GRID (weight 0.3) — a 9x9 grid over (valence, arousal) with
     a ZERO residual target (i.e. "trust the analytic prior here"). This regularises
     the empty regions of V-A space toward the prior so an untrained corner still
     sounds sane. (Juslin & Laukka 2003: emotion -> acoustic cue mapping; here the
     analytic prior already encodes those cues, so the grid just anchors to it.)

Everything routes through features.py (the codec) and droid_emotion (the param
space + prior). stdlib + numpy only. Writes dataset-<profile>.npz for inspection.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dvoice import emotion as de   # noqa: E402
from dvoice import features as F         # noqa: E402
from dvoice import ROOT             # noqa: E402

PROFILES = os.path.join(ROOT, "profiles")


def _emotions_path(profile):
    return os.path.join(PROFILES, profile, "emotions.json")


def _load_anchors(profile):
    """Read profiles/<p>/emotions.json -> list of (name, valence, arousal, patch).
    Back-compat: entries without transcript/grammar are fine; entries must carry
    valence/arousal/patch (the frozen core)."""
    path = _emotions_path(profile)
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    out = []
    for name, e in data.items():
        if not isinstance(e, dict) or "patch" not in e:
            continue
        v = float(e.get("valence", 0.0))
        a = float(e.get("arousal", 0.0))
        out.append((name, v, a, e["patch"]))
    return out


def _perturb_prosody(pro, rng, sigma=0.05):
    """Small noise on the prosody feats for the jitter rows (keep flags binary)."""
    out = pro.copy()
    # continuous dims: 0 n_syl,1 n_words,6 mean_wl
    for i in (0, 1, 6):
        out[i] = float(min(1.0, max(0.0, out[i] + rng.normal(0.0, sigma))))
    return out


def build(profile, *, jitter_k=0, jitter_sigma=0.06, grid=5, seed=0, exclude_name=None):
    """Build (X, Y, W, meta) for `profile`.

    X: (N, 26) condition vectors. Y: (N, N_KNOBS=57) logit-residual targets. W: (N,)
    per-row weight (the GP trainer maps W -> per-row observation NOISE; higher weight
    = lower noise = trust the row more). meta: dict with bounds_hash, counts, proj.

    GP REGIME (default): jitter_k=0 — the kernel's smoothness replaces the old 40x
    patch-jitter (which only injected label noise around each anchor). Sources:
      * GOLD anchors (w=1.0): every emotions.json entry. Y = residual of the
        calibrated patch over the analytic prior. Your ear — ground truth.
      * far-field GRID (w=0.3, zero residual = "trust the prior here"): a small grid
        over V-A so untuned corners revert to the analytic prior. For a decaying
        kernel this is light Tikhonov anchoring; grid=0 disables it entirely.
    `exclude_name` holds ONE anchor out (for leave-one-anchor-out CV in droid-eval).
    Text embedding is OFF here (anchors have no text) — prosody neutral; text only
    steers at inference. jitter_k>0 still works (legacy MLP augmentation) but is OFF
    by default — the GP does not want manufactured label noise."""
    rng = np.random.default_rng(seed)
    proj = F.make_projection()
    anchors = [a for a in _load_anchors(profile) if a[0] != exclude_name]

    X, Y, W = [], [], []

    # Neutral prosody (no text at calibration time): zeros except a tiny default.
    neutral_pro = F.prosody_feats("")

    # 1. gold anchors (+ optional legacy jitter neighbourhood when jitter_k>0)
    for (_name, v, a, patch) in anchors:
        x = np.concatenate([[v, a], neutral_pro, np.zeros(F.TEXT_EMB_PROJ_DIM)])
        y = F.residual_target(patch, v, a, profile)
        X.append(x); Y.append(y); W.append(1.0)
        for _ in range(jitter_k):                       # OFF by default in the GP regime
            jp = de.jitter(patch, jitter_sigma, rng)
            jv = float(min(1.0, max(-1.0, v + rng.normal(0.0, 0.04))))
            ja = float(min(1.0, max(-1.0, a + rng.normal(0.0, 0.04))))
            pro = _perturb_prosody(neutral_pro, rng)
            xj = np.concatenate([[jv, ja], pro, np.zeros(F.TEXT_EMB_PROJ_DIM)])
            yj = F.residual_target(jp, jv, ja, profile)
            X.append(xj); Y.append(yj); W.append(0.9)

    # 2. far-field grid: zero residual (trust the prior) over V-A. grid<=0 disables.
    if grid and grid > 0:
        vs = np.linspace(-0.9, 0.9, grid)
        as_ = np.linspace(-0.9, 0.9, grid)
        for v in vs:
            for a in as_:
                xg = np.concatenate([[float(v), float(a)], neutral_pro,
                                     np.zeros(F.TEXT_EMB_PROJ_DIM)])
                X.append(xg); Y.append(np.zeros(F.N_KNOBS)); W.append(0.3)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    meta = {
        "profile": profile,
        "bounds_hash": F.bounds_hash(),
        "n_anchors": len(anchors),
        "n_rows": int(X.shape[0]),
        "jitter_k": jitter_k,
        "jitter_sigma": jitter_sigma,
        "grid": grid,
        "seed": seed,
        "exclude_name": exclude_name,
        "proj": proj,
        "cond_dim": F.COND_DIM,
        "knob_dim": F.N_KNOBS,
    }
    return X, Y, W, meta


def write_dataset(profile, **kw):
    """Build + persist dataset-<profile>.npz (for inspection / debug)."""
    X, Y, W, meta = build(profile, **kw)
    out = os.path.join(ROOT, f"dataset-{profile}.npz")
    np.savez(out, X=X, Y=Y, W=W, proj=meta["proj"],
             bounds_hash=meta["bounds_hash"], n_anchors=meta["n_anchors"],
             n_rows=meta["n_rows"])
    return out, meta


if __name__ == "__main__":
    prof = sys.argv[1] if len(sys.argv) > 1 else "qd"
    path, meta = write_dataset(prof)
    print(f"wrote {path}")
    print(f"  profile={meta['profile']}  anchors={meta['n_anchors']}  "
          f"rows={meta['n_rows']}  bounds_hash={meta['bounds_hash'][:12]}")
