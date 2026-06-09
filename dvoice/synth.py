"""droid_synth — the shared DSP engine for the droid sound layer.

One soft sci-fi synthesis engine (sine + a little 2nd/3rd harmonic, two slightly
detuned oscillators, raised-cosine attack, exponential "pluck" decay, gentle
one-pole low-pass). Both the fixed-cue renderer (render-cues.py) and the
interactive `droid` studio import from here, so there is one
source of truth for how a droid cue is made.

A *patch* is a JSON-serialisable description of one cue that `render_patch`
turns into audio — this is what the studio searches over:

    {
      "notes": [ {"f0":760, "f1":600, "dur":0.085, "decay_tau":0.030,
                   "fm_rate":0, "fm_depth":0}, ... ],
      "gap": 0.012,            # seconds between notes (seq)
      "detune_cents": 7.0,     # second-oscillator detune
      "lp_cutoff": 2600.0      # one-pole low-pass corner
    }
"""

import hashlib
import json
import math

import numpy as np

SR = 44100

# Default timbre (a patch may override detune/lp per render).
HARMONICS = (1.0, 0.22, 0.10)   # fundamental + a little 2nd/3rd → soft, not buzzy
DETUNE_CENTS = 7.0
LP_CUTOFF_HZ = 2600.0


def _glide(points, total_dur):
    """Per-sample frequency array. `points` = [(time_frac, hz), ...]; exponential
    (musical) interpolation between control points."""
    n = max(1, int(total_dur * SR))
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    fracs = np.array([p[0] for p in points])
    freqs = np.log(np.array([p[1] for p in points]))
    return np.exp(np.interp(t, fracs, freqs))


def _onepole_lp(x, cutoff):
    a = math.exp(-2.0 * math.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(x.size):           # short buffers (<1s) — a python loop is fine
        acc = (1.0 - a) * x[i] + a * acc
        y[i] = acc
    return y


def _osc(freq_hz, fm_rate=0.0, fm_depth=0.0):
    """Sum-of-harmonics oscillator with optional vibrato; frequency is a
    per-sample array so glides/bends are free."""
    n = freq_hz.size
    t = np.arange(n) / SR
    vib = fm_depth * np.sin(2 * math.pi * fm_rate * t) if fm_rate else 0.0
    inst = freq_hz + vib
    phase = 2 * math.pi * np.cumsum(inst) / SR
    out = np.zeros(n)
    for h, amp in enumerate(HARMONICS, start=1):
        out += amp * np.sin(h * phase)
    return out


def blip(points, dur, *, attack=0.006, decay_tau=None, fm_rate=0.0, fm_depth=0.0,
         detune_cents=DETUNE_CENTS, lp_cutoff=LP_CUTOFF_HZ,
         ring_hz=0.0, ring_depth=0.0, sh_rate=0.0, sh_random=0.0, sh_seed=0):
    """One glided tone: two detuned oscillators, soft attack + exp decay, LP.

    Optional ARP-2600 droid colour (off by default, so plain cues are unchanged):
    - sh_rate: sample-&-hold pitch stepping (steps/sec) — the stuttery R2 feel.
    - sh_random: 0 = the classic DETERMINISTIC staircase (latches the glide); >0 =
      STOCHASTIC sample-&-hold — each held step is jittered by ±sh_random OCTAVES
      around the glide (the real R2 "computer thinking" burble was a random source,
      not a staircase; ~0.42 ≈ ±a fourth). Seeded from patch contents (sh_seed) so
      a given patch always renders identically — determinism preserved.
    - ring_hz / ring_depth: ring modulation (0..1 = subtle AM .. full ring) for a
      metallic timbre."""
    freq = _glide(points, dur)
    n = freq.size
    if sh_rate > 0:                         # stepped, held pitch
        step = max(1, int(SR / sh_rate))
        if sh_random > 0:                   # stochastic S&H ("thinking"): seeded jitter
            rng = np.random.default_rng(sh_seed)
            nsteps = (n + step - 1) // step
            starts = np.minimum(np.arange(nsteps) * step, n - 1)
            factor = 2.0 ** rng.uniform(-sh_random, sh_random, size=nsteps)
            freq = np.repeat(freq[starts] * factor, step)[:n]
        else:                              # deterministic staircase (UNCHANGED)
            idx = np.minimum((np.arange(n) // step) * step, n - 1)
            freq = freq[idx]
    det = freq * (2.0 ** (detune_cents / 1200.0))
    sig = 0.5 * _osc(freq, fm_rate, fm_depth) + 0.5 * _osc(det, fm_rate, fm_depth)
    if ring_hz > 0 and ring_depth > 0:      # ring/AM modulation → metallic colour
        t = np.arange(n) / SR
        carrier = np.sin(2 * math.pi * ring_hz * t)
        sig = sig * ((1.0 - ring_depth) + ring_depth * carrier)

    env = np.ones(n)
    a = max(1, min(int(attack * SR), n))   # clamp: very short cues are all-attack
    env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, a))   # raised-cosine in
    tau = decay_tau if decay_tau is not None else dur * 0.45
    t = np.arange(n) / SR
    env *= np.exp(-t / tau)
    sig *= env
    return _onepole_lp(sig, lp_cutoff)


def seq(parts, gap=0.012):
    """Concatenate blips with a short gap → multi-note motifs."""
    out = []
    g = np.zeros(int(gap * SR))
    for i, p in enumerate(parts):
        if i:
            out.append(g)
        out.append(p)
    return np.concatenate(out)


def _seed_from_note(nspec):
    """A stable seed derived from the note's own contents (NOT time/global state),
    so the stochastic S&H renders identically for a given patch — the calibrator
    re-renders the same patch repeatedly and must get the same audio every time."""
    blob = json.dumps(nspec, sort_keys=True, default=float).encode()
    return int.from_bytes(hashlib.sha1(blob).digest()[:8], "big")


def render_patch(patch, gain=0.18):
    """Render a patch dict to a mono float array, peak-normalised to `gain`
    (every cue tops out at the same soft level; final loudness is trimmed by the
    player's volume). A note may carry optional `mid` control points
    [[frac, hz], ...] for multi-point glides (e.g. the 3-point error/compact)."""
    notes = patch.get("notes") or []
    if not notes:
        return np.zeros(1)
    detune = float(patch.get("detune_cents", DETUNE_CENTS))
    lp = float(patch.get("lp_cutoff", LP_CUTOFF_HZ))
    parts = []
    for nspec in notes:
        f0 = float(nspec["f0"])
        f1 = float(nspec.get("f1", f0))
        points = [(0.0, f0)]
        for frac, hz in nspec.get("mid", []):
            points.append((float(frac), float(hz)))
        points.append((1.0, f1))
        parts.append(blip(
            points, float(nspec["dur"]),
            decay_tau=float(nspec.get("decay_tau", nspec["dur"] * 0.45)),
            fm_rate=float(nspec.get("fm_rate", 0.0)),
            fm_depth=float(nspec.get("fm_depth", 0.0)),
            ring_hz=float(nspec.get("ring_hz", 0.0)),
            ring_depth=float(nspec.get("ring_depth", 0.0)),
            sh_rate=float(nspec.get("sh_rate", 0.0)),
            sh_random=min(1.0, max(0.0, float(nspec.get("sh_random", 0.0)))),
            sh_seed=_seed_from_note(nspec),
            detune_cents=detune, lp_cutoff=lp,
        ))
    sig = parts[0] if len(parts) == 1 else seq(parts, float(patch.get("gap", 0.012)))
    peak = np.max(np.abs(sig)) or 1.0
    return sig / peak * gain


def write_wav(path, x):
    from scipy.io import wavfile
    pcm = np.int16(np.clip(x, -1.0, 1.0) * 32767)
    wavfile.write(str(path), SR, pcm)
