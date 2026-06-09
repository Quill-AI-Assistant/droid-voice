"""droid_emotion — the shared emotional-voice core for the droid sound layer.

This is the KEYSTONE module of the emotional droid-voice system. It is the ONE
source of truth for:

  * the search/param space (NOTE_BOUNDS, GLOBAL_BOUNDS, GLOBAL_DEFAULT),
  * the per-character voice bias (CHARACTER) and emotion anchors (EMOTIONS),
  * the ANALYTIC PRIOR  emotion_to_patch()  (the zero-residual baseline the tiny
    control model regresses against, and the calibrator's snap-back variant),
  * the ARRANGEMENT GENERATOR  generate_arrangement()  (P2 phrase grammar — the
    emotion SEED used by the studio collect path and `droid say`),
  * the search primitives  jitter() / mean_patch() / _clamp_patch(),
  * the atomic store writer  finalize(),
  * the symbolic  transcript()  used by the studio, --show and `droid say`.

Everything renders through the FROZEN engine `droid_synth.render_patch` — there
is no new audio path. A *patch* is exactly the engine's patch dict (notes[] + gap
+ detune_cents + lp_cutoff); the new fields (ring/sh on notes, g/tex advisory
tags) are all optional, so an old patch and the qd clean cues render unchanged.

The `droid` studio (`droid collect`) is a thin CLI over this module; `droid-train`
and `droid-say` import the same constants and primitives so the param space, clamp
bounds and analytic prior never diverge.
"""

import copy
import hashlib
import json
import math
import os

import numpy as np

from dvoice.synth import render_patch, write_wav
from dvoice import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(ROOT, "profiles")        # data lives at the SERVICE root, not in the package
CATALOG = os.path.join(ROOT, "catalog.json")


# ── search bounds (FROZEN) ────────────────────────────────────────────────────
# Per-note searched params and the global ones. ring/sh/fm_depth are jittered
# ONLY when the seed note already carries the key (so clean cues stay clean —
# see jitter()).  ring_hz/ring_depth/sh_rate are the additive ARP-2600 colour.
NOTE_BOUNDS = {  # name: (lo, hi, kind)
    "f0": (120.0, 1300.0, "mul"),
    "f1": (120.0, 1300.0, "mul"),
    "dur": (0.03, 0.55, "mul"),
    "decay_tau": (0.012, 0.40, "mul"),
    "fm_depth": (0.0, 16.0, "add"),
    "ring_hz": (40.0, 900.0, "mul"),
    "ring_depth": (0.0, 0.6, "add"),
    "sh_rate": (0.0, 40.0, "add"),
}
GLOBAL_BOUNDS = {
    "detune_cents": (0.0, 20.0, "add"),
    "lp_cutoff": (1200.0, 4200.0, "mul"),
    "gap": (0.004, 0.09, "mul"),
}
GLOBAL_DEFAULT = {"detune_cents": 7.0, "lp_cutoff": 2600.0, "gap": 0.012}

# Lock groups (calibrator can freeze a group so the search holds it fixed).
LOCK_GROUPS = {
    "pitch": {"f0", "f1", "mid"},
    "time": {"dur", "decay_tau", "gap"},
    "warble": {"fm_rate", "fm_depth"},
    "timbre": {"lp_cutoff", "detune_cents"},
    "metal": {"ring_hz", "ring_depth", "sh_rate"},
}


# ── seed cue patches (mirror render-cues.py's build()) ───────────────────────
# Calibration of a CUE starts from the known-good first-pass voice; the warbly voice starts
# here too (character-biased), then the operator steers it by ear.
SEED = {
    "click": {"notes": [{"f0": 760, "f1": 600, "dur": 0.085, "decay_tau": 0.030}]},
    "light-on": {"notes": [{"f0": 523, "f1": 540, "dur": 0.10, "decay_tau": 0.09},
                            {"f0": 740, "f1": 784, "dur": 0.16, "decay_tau": 0.12}]},
    "light-off": {"notes": [{"f0": 700, "f1": 680, "dur": 0.10, "decay_tau": 0.09},
                            {"f0": 470, "f1": 440, "dur": 0.18, "decay_tau": 0.13}]},
    "error": {"notes": [{"f0": 320, "f1": 270, "mid": [[0.6, 300]], "dur": 0.42,
                         "decay_tau": 0.30, "fm_rate": 13.0, "fm_depth": 10.0}]},
    "session-start": {"notes": [{"f0": 392, "f1": 410, "dur": 0.09, "decay_tau": 0.08},
                                {"f0": 523, "f1": 540, "dur": 0.09, "decay_tau": 0.08},
                                {"f0": 700, "f1": 740, "dur": 0.16, "decay_tau": 0.13}]},
    "session-end": {"notes": [{"f0": 700, "f1": 680, "dur": 0.09, "decay_tau": 0.08},
                              {"f0": 523, "f1": 510, "dur": 0.09, "decay_tau": 0.08},
                              {"f0": 392, "f1": 370, "dur": 0.18, "decay_tau": 0.14}]},
    "subagent-spawn": {"notes": [{"f0": 560, "f1": 880, "dur": 0.11, "decay_tau": 0.07}]},
    "subagent-complete": {"notes": [{"f0": 740, "f1": 720, "dur": 0.09, "decay_tau": 0.08},
                                    {"f0": 587, "f1": 560, "dur": 0.13, "decay_tau": 0.11}]},
    "memory-save": {"notes": [{"f0": 660, "f1": 650, "dur": 0.05, "decay_tau": 0.025},
                              {"f0": 660, "f1": 650, "dur": 0.05, "decay_tau": 0.025}], "gap": 0.05},
    "compact": {"notes": [{"f0": 620, "f1": 320, "mid": [[0.5, 460]], "dur": 0.34, "decay_tau": 0.26}]},
}

# ── per-profile character ("the voice") ──────────────────────────────────────
# qd (the one droid voice: warm, warbly, longer/spacious,
# arched and exploratory.
# FROZEN bias values (warm/warbly/arched).
CHARACTER = {
    "qd": {"f0_mul": 1.12, "dur_mul": 1.30, "detune": 13.0, "lp": 2200.0, "warble": 4.0, "rise": 0.06},
}

# ── emotion anchors — the (valence, arousal) reference set the tiny control
# model trains on. Coordinates span the voice's documented emotional range.
# The FIRST 8 are the FROZEN regression seed: coords UNCHANGED (emotions.json
# + the tiny control model regress against them). The next 10 tile the empty
# quadrants — Q4 (positive valence, negative arousal) was entirely unvoiceable
# before the signed-arousal fix. The ONE justified coord choice: playful sits at
# (0.55, 0.6) NOT (0.7, 0.6) so it does not stack on happy in Q1.
EMOTIONS = {
    # ── 8 frozen anchors (control-model regression seed; coords UNCHANGED) ──
    "neutral":     (0.0, 0.2),
    "curious":     (0.4, 0.5),    # the DOMINANT default state
    "happy":       (0.7, 0.6),
    "recognition": (0.5, 0.3),
    "wistful":     (-0.3, -0.4),
    "sad":         (-0.5, -0.3),
    "worried":     (-0.4, 0.4),
    "alarmed":     (-0.6, 0.8),
    # ── 10 added anchors (pure data; tile empty regions) ──
    "excited":     (0.6, 0.9),    # Q1 top-arousal: widest/brightest/fastest
    "elated":      (0.85, 0.7),   # Q1 peak-valence: very wide, warm vibrato
    "playful":     (0.55, 0.6),   # Q1 boing signature; nudged off happy(0.7,0.6)
    "proud":       (0.65, 0.4),   # Q1/Q4 boundary: broad confident rise-and-hold
    "content":     (0.6, -0.15),  # Q4 FIRST positive-low anchor (was impossible)
    "serene":      (0.5, -0.55),  # Q4 deep calm: slow, low, dim-but-pleasant
    "confident":   (0.5, 0.2),    # steady gentle rise, minimal vibrato
    "tired":       (-0.15, -0.75),# bottom-arousal: very slow, dark, drooping
    "bored":       (-0.35, -0.5), # range-floor: near-monotone flat, dull
    "frustrated":  (-0.55, 0.55), # Q2 harsh: clipped down-stabs + light ring
}


# ── search / analytic primitives (verbatim moves from the legacy calibrator) ──
def _clamp_patch(patch):
    p = copy.deepcopy(patch)
    for n in p["notes"]:
        for k, (lo, hi, _kind) in NOTE_BOUNDS.items():
            if k in n:
                n[k] = float(min(hi, max(lo, n[k])))
    for k, (lo, hi, _kind) in GLOBAL_BOUNDS.items():
        if k in p:
            p[k] = float(min(hi, max(lo, p[k])))
    return p


def apply_character(patch, profile):
    """Bias a neutral cue patch toward the profile's voice (qd: warm/warbly/
    spacious)."""
    c = CHARACTER.get(profile, CHARACTER["qd"])
    p = copy.deepcopy(patch)
    for n in p["notes"]:
        n["f0"] = float(n["f0"]) * c["f0_mul"]
        n["f1"] = float(n.get("f1", n["f0"])) * c["f0_mul"] * (1.0 + c["rise"])
        n["dur"] = float(n["dur"]) * c["dur_mul"]
        if "decay_tau" in n:
            n["decay_tau"] = float(n["decay_tau"]) * c["dur_mul"]
        if c["warble"] and "fm_depth" not in n:
            n["fm_depth"] = c["warble"]
            n.setdefault("fm_rate", 9.0)
    p["detune_cents"] = c["detune"]
    p["lp_cutoff"] = c["lp"]
    return _clamp_patch(p)


def emotion_to_patch(valence, arousal, profile):
    """Map (valence, arousal) -> synth params on the profile's character base —
    the ANALYTIC PRIOR. arousal: pitch height + speed + warble; valence: contour
    direction + brightness. A two-note gesture (a small vocalisation, not a tick).
    Kept as the model's zero-residual target space and the calibrator's snap-back
    variant — generate_arrangement() is the richer phrase SEED."""
    c = CHARACTER.get(profile, CHARACTER["qd"])
    base_f0 = 480.0 * c["f0_mul"] * (1.0 + 0.5 * arousal)
    f1 = base_f0 * (1.0 + 0.35 * valence + c["rise"])
    midf = math.sqrt(base_f0 * f1)
    dur = 0.22 * c["dur_mul"] * (1.0 - 0.4 * max(0.0, arousal))
    decay = dur * 0.7
    warble = c["warble"] + 4.0 * max(0.0, arousal)
    lp = c["lp"] + 500.0 * valence + 400.0 * arousal
    patch = {
        "notes": [
            {"f0": base_f0, "f1": midf, "dur": dur * 0.5, "decay_tau": decay * 0.5,
             "fm_rate": 9.0, "fm_depth": warble},
            {"f0": midf, "f1": f1, "dur": dur, "decay_tau": decay,
             "fm_rate": 9.0, "fm_depth": warble},
        ],
        "gap": 0.02, "detune_cents": c["detune"], "lp_cutoff": lp,
    }
    return _clamp_patch(patch)


def _jit(val, lo, hi, kind, sigma, rng):
    if kind == "mul":
        v = val * math.exp(rng.normal(0.0, sigma))
    else:  # additive noise scaled by the parameter's own range
        v = val + rng.normal(0.0, sigma) * (hi - lo) * 0.5
    return float(min(hi, max(lo, v)))


def _locked_keys(locks):
    """Expand lock-group names -> the set of param keys they freeze."""
    keys = set()
    for grp in locks:
        keys |= LOCK_GROUPS.get(grp, set())
    return keys


def jitter(patch, sigma, rng, locks=frozenset()):
    """Perturb a patch around itself for the preference search.

    ring/sh/fm_depth are gated: only jittered if the seed note already carries
    the key (preserves clean cues). `locks` is a set of lock-GROUP names
    ('pitch','time','warble','timbre','metal'); any param in a locked group is
    held fixed. locks=empty => identical to the legacy calibrator path."""
    locked = _locked_keys(locks)
    p = copy.deepcopy(patch)
    for n in p["notes"]:
        for k, (lo, hi, kind) in NOTE_BOUNDS.items():
            if k in locked:
                continue
            if k in ("fm_depth", "ring_hz", "ring_depth", "sh_rate") and k not in n:
                continue  # don't add warble/metal to a clean cue
            n[k] = _jit(float(n.get(k, lo)), lo, hi, kind, sigma, rng)
    for k, (lo, hi, kind) in GLOBAL_BOUNDS.items():
        if k in locked:
            continue
        if k == "gap" and len(p["notes"]) < 2:
            continue
        p[k] = _jit(float(p.get(k, GLOBAL_DEFAULT[k])), lo, hi, kind, sigma, rng)
    return p


def mean_patch(patches):
    """Element-wise mean of a list of patches (retained convenience; NOT the
    default search anchor — the anchor is a real kept variant)."""
    base = copy.deepcopy(patches[0])
    for ni, n in enumerate(base["notes"]):
        for k in ("f0", "f1", "dur", "decay_tau"):
            n[k] = float(np.mean([p["notes"][ni][k] for p in patches]))
        if "fm_depth" in n:
            n["fm_depth"] = float(np.mean([p["notes"][ni].get("fm_depth", 0.0) for p in patches]))
    for k in ("detune_cents", "lp_cutoff", "gap"):
        if any(k in p for p in patches):
            base[k] = float(np.mean([p.get(k, GLOBAL_DEFAULT[k]) for p in patches]))
    return base


# ── ARRANGEMENT (P2 phrase grammar) ──────────────────────────────────────────
# A cue is a PHRASE with grammar, not a beep. The character supplies defaults
# (register, brightness, syllable budget, onset, terminal bias); (valence,arousal)
# + a few prosody cues from the text choose the gesture contour. The qd voice
# speaks in arched, open, warbly phrases.
GRAMMAR = {
    "qd": {
        "reg0": 560.0, "lp0": 2200.0, "syl_base": 4, "dur0": 0.16, "gap0": 0.026,
        "fm_rate": 9.0, "fm_min": 3.5, "detune": 13.0, "onset": True,
        "terminal_bias": "rise",
        # quadrant textures:
        "fm_rate_nervous": 11.5,   # Q2 flutter rate (jittery, not warm)
        "ring_default": 0.0,       # non-alarm states stay ring-free
    },
}

GESTURES = {"rise", "fall", "bend", "dip", "flat", "trill", "chirp", "stutter",
            "boing", "waver", "sag"}


def gesture_recipe(name, f0, valence, arousal, *, fm_rate=9.0, fm_depth=0.0,
                   ring=0.0, dur=0.16, decay_mul=0.62, range_scale=1.0):
    """One gesture -> a note-field dict, given a base pitch. Returns a renderable
    note (f0/f1[/mid]/dur/decay_tau) plus advisory 'g'/'tex' tags. Pure.

    range_scale (default 1.0 => BYTE-IDENTICAL for every legacy caller/test) is an
    optional openness multiplier on the contour EXCURSION (the part above/below
    f0): open (~1.4) for high +v/+a, compressed (~0.35) for low arousal. It is
    folded with a low-arousal taper (1 + 0.4*min(0,arousal)) so negative-arousal
    states read near-monotone-but-alive instead of dead-flat. At range_scale=1.0
    AND arousal>=0 the expressions reduce EXACTLY to the original ones."""
    note = {"f0": float(f0), "dur": float(dur), "decay_tau": float(dur * decay_mul),
            "fm_rate": float(fm_rate), "fm_depth": float(fm_depth), "g": name}
    tex = []
    # widened valence-driven openness; arousal opens the top
    span = 1.0 + 0.45 * abs(valence) + 0.30 * max(0.0, arousal)
    # low-energy taper: arousal<0 compresses the excursion (tired/bored ~alive)
    exc = range_scale * (1.0 + 0.4 * min(0.0, arousal))
    if name == "rise":
        note["f1"] = f0 * (1.0 + (span - 1.0) * exc)
    elif name == "fall":
        note["f1"] = f0 / (1.0 + (span - 1.0) * exc)
    elif name == "chirp":                       # quick wide upward onset
        note["f1"] = f0 * (1.0 + (0.25 + 0.5 * max(0.0, arousal)) * exc)
        note["dur"] = float(dur * 0.7)
        note["decay_tau"] = float(note["dur"] * 0.55)
    elif name == "bend":                        # up then settle (arched syllable)
        top = f0 * (1.0 + 0.22 * span * exc)
        note["mid"] = [[0.45, float(top)]]
        note["f1"] = f0 * (1.0 + 0.06 * valence)
    elif name == "dip":                         # down then recover
        bot = f0 / (1.0 + 0.22 * span * exc)
        note["mid"] = [[0.45, float(bot)]]
        note["f1"] = f0 * (1.0 - 0.04)
    elif name == "trill":                       # warbly, sustained
        note["f1"] = f0 * (1.0 + 0.04 * valence)
        note["fm_depth"] = max(fm_depth, 6.0)
        note["fm_rate"] = max(fm_rate, 11.0)
        tex.append("warble")
    elif name == "stutter":                     # sample&hold processing texture
        # S&H steps the pitch GLIDE, so a stutter must move in pitch to be heard
        # (a flat tone would step to itself = silent texture). A small rising drift
        # gives the R2 "processing / thinking" stepped warble.
        note["f1"] = f0 * (1.0 + 0.12 + 0.10 * max(0.0, arousal))
        note["sh_rate"] = 18.0 + 14.0 * max(0.0, arousal)
        # STOCHASTIC sample-&-hold (the R2 "data burble") — was a deterministic
        # staircase (sh_random 0); the engine seeds the randomness from patch content
        # so re-renders stay byte-identical. Bounded [0,1] octaves.
        note["sh_random"] = float(min(1.0, 0.30 + 0.10 * max(0.0, arousal)))
        tex.append("stutter")
    elif name == "boing":                       # playful multi-point bounce
        note["mid"] = [[0.25, float(f0 * 1.18 * span)],
                       [0.55, float(f0 * 0.94)],
                       [0.80, float(f0 * 1.10)]]
        note["f1"] = f0 * 1.04
    elif name == "waver":                       # Q2 nervous narrow up-down shimmer
        note["mid"] = [[0.30, float(f0 * 1.05)],
                       [0.60, float(f0 * 0.97)],
                       [0.85, float(f0 * 1.03)]]
        note["f1"] = f0 * 1.0
    elif name == "sag":                         # downward droop / pitch drift (tired)
        note["mid"] = [[0.50, float(f0 * 0.95)]]
        note["f1"] = f0 * 0.88
    else:                                       # flat
        note["f1"] = f0
    if ring > 0:
        # ring tied to NOTE PITCH (k*f0, clamped to the [40,900] bound) so the metallic
        # ratio stays consistent note-to-note instead of drifting against a fixed absolute.
        note["ring_hz"] = float(min(900.0, max(40.0, 0.7 * f0)))
        note["ring_depth"] = float(ring)
        tex.append("ring")
    if fm_depth > 0 and "warble" not in tex:
        tex.append("warble")
    if tex:
        note["tex"] = tex
    return note


def _syllable_count(text, base):
    """Coarse syllable budget from the text (vowel groups), clamped near `base`."""
    if not text:
        return base
    groups = 0
    prev_vowel = False
    for ch in text.lower():
        is_v = ch in "aeiouy"
        if is_v and not prev_vowel:
            groups += 1
        prev_vowel = is_v
    if groups <= 0:
        return base
    return int(max(2, min(base + 2, round((base + groups) / 2))))


def _quadrant(valence, arousal):
    """Pure (v,a) -> emotional quadrant. The split arousal pivot at 0.0 lets the
    Q3/Q4 (low-arousal) families read as genuinely calm/heavy now that arousal is
    signed end-to-end.
      Q1 = positive valence, high arousal  (excited/elated/happy/playful/curious)
      Q4 = positive valence, low arousal   (content/serene/confident/proud)
      Q2 = negative valence, high arousal  (worried/alarmed/frustrated)  ROUGH
      Q3 = negative valence, low arousal   (sad/wistful/tired/bored)
    'high'/'low' split on arousal >= 0 so the new Q4 anchors (a<0) tile cleanly."""
    pos = valence >= 0.0
    hi = arousal >= 0.0
    if pos and hi:
        return "Q1"
    if pos and not hi:
        return "Q4"
    if (not pos) and hi:
        return "Q2"
    return "Q3"


def _rhythm_pattern(valence, arousal, n_mid, rng, override=None):
    """Per-emotion RHYTHM (a `timing_mul` array over the middle notes) — the expressive
    axis that makes a phrase feel alive and further separates same-valence emotions.
    accelerando = mounting energy, ritardando = winding down/heavy, syncopation = bouncy/
    agitated off-beat, even = steady/composed. Multiplies dur0 via the existing mechanism
    (no new knob/bound); the caller layers the seeded jitter + plateau-break on top."""
    if n_mid <= 0:
        return np.ones(0)
    pat = override or _rhythm_name(valence, arousal)
    if pat == "syncopation":                             # playful/curious; frustrated/alarmed
        return np.array([0.7 if j % 2 == 0 else 1.3 for j in range(n_mid)])
    if pat == "accelerando":                             # excited/elated/happy
        return np.linspace(1.15, 0.7, n_mid)
    if pat == "ritardando":                              # sad/wistful/tired droop
        return np.linspace(0.9, 1.4, n_mid)
    base = np.full(n_mid, 0.85)                          # even/composed: neutral/proud/content/serene/bored
    base[rng.integers(0, n_mid)] = 1.25                  # one emphasis (the speech-like cadence)
    return base


def _rhythm_name(valence, arousal):
    """The cadence (v,a) selects. accelerando=mounting, ritardando=winding-down,
    syncopation=bouncy/agitated, even=steady. bored is the dull monotone -> even."""
    bored = (-0.45 <= valence <= -0.30 and -0.65 <= arousal <= -0.45)
    if (0.30 <= valence <= 0.65 and 0.45 <= arousal <= 0.70) \
            or (valence <= -0.40 and arousal >= 0.45):
        return "syncopation"
    if arousal >= 0.60:
        return "accelerando"
    if valence < 0.0 and arousal <= -0.25 and not bored:
        return "ritardando"
    return "even"


def _signature(valence, arousal, quad, n_mid, ring):
    """Per-emotion contour SIGNATURE → (onset, mids). Distinct gesture *rhythms* so
    same-quadrant emotions are qualitatively different shapes, not just magnitude-graded
    (the within-quadrant collapse the metric flagged: playful≈proud≈curious). Reuses only
    the existing 11 gestures, re-sequenced; keyed to (v,a) neighbourhoods (open boxes) and
    layered over the quadrant default so any uncovered (v,a) still gets a sensible plan."""
    def cyc(seq):                                        # tile a pattern across n_mid slots
        return [seq[j % len(seq)] for j in range(n_mid)]
    if quad == "Q1":                                     # positive + aroused
        if valence >= 0.55 and arousal >= 0.65:          # excited / elated: staccato darts
            return "chirp", cyc(["rise", "chirp"])
        if valence >= 0.65 and 0.45 <= arousal < 0.65:   # happy: lyrical rising arches (excl. low-arousal proud)
            return "chirp", cyc(["bend", "rise"])
        if 0.45 <= valence < 0.65 and arousal >= 0.5:    # playful: bounce
            return "chirp", cyc(["boing", "bend"])
        if 0.30 <= valence < 0.5 and arousal < 0.6:      # curious: probe + one data-burble
            return "bend", cyc(["rise", "stutter", "bend"])
        if valence >= 0.45 and arousal < 0.45:           # proud / confident: rise-and-hold
            return "bend", (["rise"] + ["flat"] * (n_mid - 1)) if n_mid else []
        m = ["rise" if j % 2 == 0 else "bend" for j in range(n_mid)]   # Q1 default / neutral
        return ("chirp" if arousal >= 0.2 else "bend"), m
    if quad == "Q4":                                     # positive + calm
        if arousal <= -0.45:                             # serene: still / level (rhythm lengthens)
            return "bend", ["flat"] * n_mid
        return "bend", cyc(["flat", "dip"])              # content: gentle settle
    if quad == "Q3":                                     # negative + calm
        if arousal <= -0.55:                             # tired: heavy slow droop
            return "sag", cyc(["sag", "flat"])
        return "bend", cyc(["sag", "dip"])               # sad / wistful: droop
    # Q2 (negative + aroused): textured key-presence — unchanged (certified distinct)
    if ring >= 0.4:
        return "stutter", ["waver"] * n_mid              # alarmed
    if ring > 0:
        return "fall", ["fall"] * n_mid                  # frustrated
    return "bend", ["waver"] * n_mid                     # worried


def _contour_plan(character, valence, arousal, terminal, n_syl, *, ring=0.0):
    """An ordered list of gesture NAMES of length n_syl — the per-emotion contour
    family. The terminal (already chosen by punctuation/valence) is honoured as the
    last element so the question/`!`/`...`/negative-statement overrides still win.

    Any non-qd character falls back to a terse plan (onset-less flat/dip + terminal)
    so the question-terminal discriminator is untouched; qd is the only shipped voice.
    All quadrant texture (boing/waver/sag/stutter, family middles) is qd-ONLY."""
    n_syl = max(1, int(n_syl))
    if n_syl == 1:
        return [terminal]

    if character != "qd":
        # non-qd fallback: onset-less; flat (or dip on negative valence) middles + terminal.
        mid_g = "flat" if valence >= 0 else "dip"
        return [mid_g] * (n_syl - 1) + [terminal]

    quad = _quadrant(valence, arousal)
    n_mid = n_syl - 2                                    # between onset and terminal
    # ── recognition "aha" signature: a soft rise-then-settle (NOT a bright chirp
    # ascent). Separates recognition(0.5,0.3) from confident/proud. Own terminal.
    if 0.35 <= valence <= 0.6 and 0.25 <= arousal <= 0.4:
        return ["bend"] + ["dip"] * n_mid + ["dip"]
    # ── bored: monotone floor (near-flat, dull), distinct from the Q3 droopers. Own terminal.
    if quad == "Q3" and -0.45 <= valence <= -0.30 and -0.65 <= arousal <= -0.45:
        return ["flat"] + ["flat"] * n_mid + ["dip"]
    onset, mids = _signature(valence, arousal, quad, n_mid, ring)
    return [onset] + mids + [terminal]


# ── melodic MODE: the non-verbal VALENCE cue (Juslin&Laukka 2003; Frontiers 2013) ──
# Sound encodes arousal reliably but valence weakly; the cue that DOES carry valence in
# tonal material is consonance/interval. Successive middle pitches step by consonant
# intervals for +v (M3/P5/M6 — "bright/major") and minor-2nd/tritone clusters for -v
# ("dark/dissonant" — distress cries are minor-2nd-heavy). Ladders OSCILLATE around the
# register (it's the step QUALITY that reads, not absolute height) to avoid f0 clipping.
# Per-note f0 is arbitrary so this never touches the FROZEN synth.
_MODE_MAJOR = (0.0, 4.0, 7.0, 4.0, 9.0, 5.0)        # M3 / P5 / M6 / P4 — consonant, bright
_MODE_MINOR = (0.0, -1.0, -2.0, -1.0, -6.0, -1.0)   # minor-2nd cluster + a tritone — dark


def _mode_degree(valence, slot):
    """Semitone offset for middle `slot`, |valence|-scaled (full intervals by |v|>=0.5,
    ~flat near neutral). +v selects the consonant ladder, -v the dissonant one."""
    vscale = min(1.0, abs(valence) / 0.5)
    ladder = _MODE_MAJOR if valence >= 0.0 else _MODE_MINOR
    return ladder[slot % len(ladder)] * vscale


def generate_arrangement(character, valence, arousal, text="", *, ring=0.0,
                         range_scale=None, force_syl=None, rhythm=None):
    """Build a full PHRASE patch for (character, valence, arousal[, text]).

    force_syl (optional) overrides the derived syllable/note count (clamped 1..N_SLOTS) —
    used by the candidate generator to vary phrase LENGTH/structure across a batch.
    Default (None) derives the count from (v,a): the qd energy band (1-N_SLOTS
    notes, count tracks arousal; a typed phrase fills toward the top but never past
    N_SLOTS, the frozen feature-interface slot count; non-qd fallback: 1-2).

    qd: a 1-6 note phrase (note COUNT tracks arousal — calm is sparse, excited is
    busy) whose CONTOUR FAMILY is chosen by (valence,arousal)
    quadrant (Q1 ascending+bright, Q4 gentle/slow, Q3 drooping/dark, Q2 textured),
    every content syllable warbled at >= fm_min so the studio can search warble.
    Any non-qd character falls back to a 1-2 note clipped phrase (no onset,
    flat/falling conclusive terminal, fm_depth 0 except when ring>0, e.g. the alarm
    special). The result is a plain patch dict — renderable by render_patch
    unchanged, clamped to bounds.

    Signed-arousal end-to-end: register/tempo/brightness/gap all use FULL signed
    arousal so negative-arousal states are slower/lower/darker/breathier than
    neutral (Q4/Q3 were unvoiceable before). range_scale (auto-derived from v,a if
    None) opens the contour for +v/+a and floors it for low arousal."""
    g = GRAMMAR.get(character, GRAMMAR["qd"])
    reg = g["reg0"] * (1.0 + 0.45 * arousal)             # FULL signed arousal
    warbly = (character == "qd")    # the warbly voice

    # ── ENERGY-scaled note count: the count IS the emotion's energy signal ──
    # arousal LEADS (calm => 1-2 sparse notes, excited => 5-6 busy ones); valence
    # nudges. Driven DIRECTLY off (v,a). The old code averaged this target back
    # toward syl_base (4) and floored it at 3, so calm and excited both rendered as
    # ~4 beeps — the "every emotion sounds like 4 notes" bug. A typed line then adds
    # beeps ON TOP (a longer line audibly = more notes) but never shrinks the count
    # below the feeling's own. Byte-gated cues never call this path (render-cues.py).
    has_text = bool((text or "").strip())
    n_syl = round(g["syl_base"] + 3.5 * arousal + 0.6 * valence)   # raw target ~1..8; clamped to N_SLOTS below
    if has_text:                                         # words extend, never shorten
        n_syl = max(n_syl, _syllable_count(text, g["syl_base"]))
    from dvoice.features import N_SLOTS                   # the frozen feature interface models
    #   exactly N_SLOTS notes; the generator must never emit more — notes past N_SLOTS are
    #   dropped by features.patch_to_knobs and the count is /N_SLOTS-normalized in
    #   reward.struct_feats, so a 7-8 note phrase would be ranked on truncated, >1.0-scaled feats.
    if warbly:
        n_syl = int(max(1, min(N_SLOTS, n_syl)))         # 1..N_SLOTS note range (calm→sparse, excited→busy)
    else:
        n_syl = int(max(1, min(3 if has_text else 2, n_syl)))   # 1-2 anchor / 1-3 phrased
    if force_syl is not None:                            # explicit override for diversity
        n_syl = int(max(1, min(N_SLOTS, force_syl)))

    # prosody flags from the text choose the terminal gesture (overrides on top)
    t = (text or "").strip()
    q = t.endswith("?")
    excl = t.endswith("!")
    ellip = t.endswith("...") or t.endswith("…")
    terminal = g["terminal_bias"]
    if q:
        terminal = "rise"
    elif excl:
        terminal = "chirp" if warbly else "fall"
    elif ellip:
        terminal = "trill" if warbly else "flat"
    elif valence < -0.15:
        terminal = "fall" if not warbly else "dip"        # Q2/Q3 warbly -> dip (frozen)
    elif warbly:
        # per-quadrant contour-family terminal (BELOW the punctuation/valence block)
        quad = _quadrant(valence, arousal)
        if quad == "Q1":
            terminal = "rise"                            # open/ascending close
        elif quad == "Q4":
            terminal = "rise" if arousal >= -0.3 else "flat"  # soft settle vs flat
        elif quad == "Q3":
            terminal = "dip"                             # negative-low droop (e.g. tired v=-0.15)
        # Q2 already handled by the valence<-0.15 branch above (a>0 keeps dip)

    # ── range openness as a real (v,a) axis ──
    if range_scale is None:
        if arousal < -0.2:                               # bored/tired/sad: compress
            range_scale = max(0.35, 0.85 + 0.5 * arousal)
        else:                                            # open for +v/+a
            range_scale = 1.0 + 0.25 * max(0.0, valence) + 0.20 * max(0.0, arousal)
        range_scale = max(0.35, min(1.45, range_scale))

    # ── warble keyed to v AND distress (fm_min floor preserved) ──
    fm_depth = 0.0
    fm_rate = g["fm_rate"]
    if warbly:
        distress = max(0.0, arousal * (-valence))        # a*(-v) nervous-flutter term
        fm_depth = max(g["fm_min"],
                       g["fm_min"] + 3.0 * max(0.0, arousal) + 2.5 * max(0.0, valence)
                       + 2.5 * distress)
        if valence < -0.2 and arousal > 0.45:            # Q2 distress: jittery rate
            fm_rate = g.get("fm_rate_nervous", 11.5) + 1.5 * max(0.0, arousal)
        elif arousal < -0.2:                             # calm Q4/Q3: warm slow rate
            fm_rate = 6.0
    elif ring > 0:                                        # non-qd alarm special
        fm_depth = 2.0

    # ── ring/sh activation on the NORMAL warbly path (was alarm-only/dormant) ──
    ring_amt = ring
    if warbly and ring <= 0.0:                             # gate to Q2 distress only
        if valence < -0.2 and arousal > 0.45:
            ring_amt = max(0.0, min(0.55, 0.25 + 0.25 * arousal))
            if arousal < 0.7:                            # frustrated: grit, capped
                ring_amt = min(ring_amt, 0.25)

    # ── dark "low-battery" grit: a faint sample-hold burble on NEGATIVE-low-arousal
    # middles (sad/wistful/tired/bored). Roughness reads as negative valence (Frontiers
    # 2013) — this separates the dark cluster from the CLEAN positive-low states
    # (content/serene), the wistful≈serene collapse. Well below the Q2 alarm sh (18+),
    # and gated to arousal<0 so it never touches the Q2 distress contract.
    grit = 0.0
    if warbly and valence < -0.15 and arousal < -0.1:
        grit = 6.0 + 3.0 * min(1.0, -valence)            # ~6-9 sh_rate

    # ── time + gap with signed arousal (negative LENGTHENS / breathes) ──
    dur0 = g["dur0"] * (1.0 - 0.25 * max(0.0, arousal) + 0.45 * max(0.0, -arousal))
    gap = g["gap0"] * (1.0 - 0.30 * max(0.0, arousal) + 0.60 * max(0.0, -arousal))
    gap = float(min(0.09, max(0.004, gap)))

    # ── build the phrase from the per-quadrant contour plan ──
    plan = _contour_plan(character, valence, arousal, terminal, n_syl, ring=ring_amt)
    # ── qd micro-timing + plateau-break (pace like speech, not a metronome) ──
    # Every middle shared dur0 and the SYMMETRIC mid-arch put two middles at the SAME
    # pitch, so a multi-syllable cue read as a held tone. Give most middles a short
    # 'passing' duration with ~1 'emphasized' longer one, and nudge a middle that lands
    # within ~0.6 semitone of the previous note off the plateau. CONTENT-SEEDED (sha1 of
    # character|v|a|text|len) so a re-render is byte-identical; qd-ONLY so the non-qd
    # cues stay frozen; no NOTE/GLOBAL bound or knob-layout change (bounds_hash unaffected).
    n_mid = max(0, len(plan) - 2)
    if warbly and n_mid:
        _seed = int(hashlib.sha1(
            f"{character}|{valence:.4f}|{arousal:.4f}|{text}|{len(plan)}".encode()).hexdigest()[:8], 16)
        _rng = np.random.default_rng(_seed)
        timing_mul = _rhythm_pattern(valence, arousal, n_mid, _rng, override=rhythm)  # cadence
        timing_mul *= _rng.uniform(0.95, 1.05, n_mid)    # tiny per-syllable jitter
        plateau_break = _rng.uniform(0.8, 1.6, n_mid) * _rng.choice((-1.0, 1.0), n_mid)  # semitones
    notes = []
    prev_f0 = None
    for i, gname in enumerate(plan):
        is_term = (i == len(plan) - 1)
        is_onset = (i == 0 and warbly and not is_term)
        if is_onset:
            f0 = reg * 0.92
            dur = dur0 * 0.7
        elif is_term:
            f0 = reg * (1.0 + 0.12 * valence)
            dur = dur0 * (1.5 if warbly else 1.3)
        else:
            frac = (i) / max(1, len(plan) - 1)
            # mid-arch lift: SIGN follows valence (negative valence SAGS)
            sign = 1.0 if valence >= 0 else -1.0
            lift = 1.0 + 0.18 * math.sin(math.pi * frac) * sign + 0.10 * valence
            f0 = reg * lift
            if warbly:                                    # melodic MODE = the valence cue
                f0 *= 2.0 ** (_mode_degree(valence, i - 1) / 12.0)
            dur = dur0                                   # middle content syllable
            if warbly and n_mid:                          # micro-timing + plateau-break
                mid_idx = i - 1                          # slot 1 -> middle 0 (onset is slot 0)
                dur = dur0 * float(timing_mul[mid_idx])
                if prev_f0 and abs(12.0 * math.log2(f0 / prev_f0)) < 0.6:
                    f0 *= 2.0 ** (float(plateau_break[mid_idx]) / 12.0)
        # ring is applied to every note of a distress phrase (qd Q2) or the
        # non-qd alarm special; stutter notes still self-add their sh_rate.
        note_ring = ring_amt if ring_amt > 0 else 0.0
        notes.append(gesture_recipe(
            gname, f0, valence, arousal,
            fm_rate=fm_rate, fm_depth=fm_depth, ring=note_ring, dur=dur,
            range_scale=range_scale))
        if grit > 0.0 and not is_onset and not is_term \
                and float(notes[-1].get("sh_rate", 0.0)) <= 0.0:   # dark burble on middles
            notes[-1]["sh_rate"] = grit
        prev_f0 = f0

    # Audible question: lift the terminal a semitone so a '?' actually RISES (a plain
    # quadrant 'rise' terminal was inaudible vs a statement). warbly-only, and only when
    # the text is a question, so anchor renders (text='') stay byte-identical.
    if warbly and q and notes:
        notes[-1]["f1"] = float(notes[-1].get("f1", notes[-1]["f0"])) * 2.0 ** (1.0 / 12.0)

    patch = {
        "notes": notes,
        "gap": gap,
        "detune_cents": g["detune"],
        "lp_cutoff": g["lp0"] + 700.0 * valence + 400.0 * arousal,   # +v brighter/cleaner
    }
    return _clamp_patch(patch)


# ── symbolic transcript (pure, one glyph-cluster per note) ───────────────────
# Vision-disability operator: glyphs must NOT rely on colour; an ASCII fallback
# is provided. One cluster per note: register · contour · warble · ring · stutter.
_REG_GLYPH = [("▁", "_"), ("▃", "."), ("▅", "-"), ("▇", "=")]  # low..high
_REG_ASCII = ["_", ".", "-", "="]


def _contour_glyph(note, ascii):
    f0 = float(note.get("f0", 0.0))
    f1 = float(note.get("f1", f0))
    mid = note.get("mid") or []
    g = note.get("g")
    # boing/waver read as an up-arch; sag reads as a fall (HONEST glyphs — the
    # vision-disability operator reads the glyph, not the colour/texture).
    if g in ("bend", "boing", "waver") or (mid and float(mid[0][1]) > max(f0, f1)):
        return ("^" if ascii else "∧")        # arch up
    if g == "sag":
        return ("\\" if ascii else "↘")       # downward droop / drift
    if g == "dip" or (mid and float(mid[0][1]) < min(f0, f1)):
        return ("v" if ascii else "∨")        # dip down
    if g == "trill":
        return ("~" if ascii else "〜")        # warble
    ratio = f1 / f0 if f0 else 1.0
    if ratio > 1.06:
        return ("/" if ascii else "↗")        # rise
    if ratio < 0.94:
        return ("\\" if ascii else "↘")       # fall
    return ("-" if ascii else "→")            # flat


def _note_cluster(note, ascii):
    f0 = float(note.get("f0", 0.0))
    # register bucket by pitch
    bucket = 0 if f0 < 360 else 1 if f0 < 560 else 2 if f0 < 820 else 3
    reg = _REG_ASCII[bucket] if ascii else _REG_GLYPH[bucket][0]
    cluster = reg + _contour_glyph(note, ascii)
    if float(note.get("fm_depth", 0.0)) > 0.5:
        cluster += ("*" if ascii else "✻")    # warble star
    if float(note.get("ring_depth", 0.0)) > 0.0 and float(note.get("ring_hz", 0.0)) > 0:
        cluster += ("#" if ascii else "⦿")    # ring/metal
    if float(note.get("sh_rate", 0.0)) > 0.0:
        cluster += (":" if ascii else "⋮")    # stutter / sample-hold
    return cluster


def transcript(patch, ascii=False):
    """A pure symbolic transcript of a patch: one glyph-cluster per note
    (register · contour · warble · ring · stutter), gap-joined. ascii=True uses a
    plain-ASCII fallback set for terminals/fonts that cannot render the glyphs."""
    notes = patch.get("notes") or []
    if not notes:
        return ""
    sep = " " if ascii else " "
    return sep.join(_note_cluster(n, ascii) for n in notes)


# ── grammar metadata (optional, written by the calibrator) ───────────────────
def grammar_meta(patch):
    """Derive the optional emotions.json `grammar` block from a patch:
    {syllable_count, contour: asc|arch|desc|flat, terminal: rise|fall|flat|trill}."""
    notes = patch.get("notes") or []
    n = len(notes)
    if n == 0:
        return {"syllable_count": 0, "contour": "flat", "terminal": "flat"}
    f0s = [float(x.get("f0", 0.0)) for x in notes]
    # overall contour
    if n == 1:
        contour = "flat"
    else:
        peak_i = max(range(n), key=lambda i: f0s[i])
        if f0s[-1] > f0s[0] * 1.05:
            contour = "asc"
        elif f0s[-1] < f0s[0] * 0.95:
            contour = "desc"
        elif 0 < peak_i < n - 1:
            contour = "arch"
        else:
            contour = "flat"
    last = notes[-1]
    g = last.get("g")
    if g in ("rise", "chirp", "boing"):
        terminal = "rise"
    elif g in ("fall", "dip", "sag"):
        terminal = "fall"
    elif g == "trill":
        terminal = "trill"
    else:
        f0 = float(last.get("f0", 0.0)); f1 = float(last.get("f1", f0))
        terminal = "rise" if f1 > f0 * 1.06 else "fall" if f1 < f0 * 0.94 else "flat"
    return {"syllable_count": n, "contour": contour, "terminal": terminal}


# ── store helpers + atomic writer ─────────────────────────────────────────────
def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _store_path(profile, *parts):
    return os.path.join(PROFILES, profile, *parts)


def resolve_target(profile, arg):
    """Return (target, seed_patch). target['kind'] is 'emotion' or 'cue'.

    Emotion anchors (curious, happy, ...) calibrate the (valence,arousal)
    reference set -> emotions.json + emotions/<name>.wav, SEEDED FROM THE
    ARRANGEMENT GENERATOR (a full phrase). Cues (catalog event or slug) calibrate
    a deployed sound -> patches.json + <slug>.wav, seeded from the
    character-biased first-pass voice. A saved store entry always wins as the
    resume seed."""
    if arg in EMOTIONS:
        v, a = EMOTIONS[arg]
        store = _store_path(profile, "emotions.json")
        saved = _load_json(store).get(arg)
        if saved and "patch" in saved:
            seed = copy.deepcopy(saved["patch"])
        else:
            seed = generate_arrangement(profile, v, a)
        return {"kind": "emotion", "key": arg, "valence": v, "arousal": a,
                "store": store, "wav": _store_path(profile, "emotions", f"{arg}.wav")}, seed
    events = (_load_json(CATALOG).get("events") or {})
    slug = events[arg]["slug"] if arg in events else arg
    store = _store_path(profile, "patches.json")
    saved = _load_json(store).get(slug)
    if saved:
        seed = copy.deepcopy(saved)
    elif slug in SEED:
        seed = apply_character(SEED[slug], profile)   # bias toward the profile's voice
    else:
        return None, None
    return {"kind": "cue", "key": slug, "store": store,
            "wav": _store_path(profile, f"{slug}.wav")}, seed


def finalize(target, patch, gain):
    """Atomically render+write the WAV and update the JSON store (temp file +
    os.replace + fsync, so a crash mid-write can never corrupt a profile). Cues
    store slug->patch in patches.json; emotions store
    name->{valence,arousal,patch[,transcript,grammar]} in emotions.json (the
    labeled reference set the tiny control model trains on)."""
    os.makedirs(os.path.dirname(target["wav"]), exist_ok=True)
    write_wav(target["wav"] + ".tmp", render_patch(patch, gain))
    os.replace(target["wav"] + ".tmp", target["wav"])
    store = _load_json(target["store"])
    if target["kind"] == "emotion":
        entry = {"valence": target["valence"], "arousal": target["arousal"], "patch": patch}
        entry["transcript"] = transcript(patch)
        entry["grammar"] = grammar_meta(patch)
        store[target["key"]] = entry
    else:
        store[target["key"]] = patch
    with open(target["store"] + ".tmp", "w") as f:
        json.dump(store, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(target["store"] + ".tmp", target["store"])
    return target["wav"]


# ── VOCALIZATIONS — non-verbal affect bursts ─────────────────────────────────
# A vocalization is NOT a phrase: it is a short non-lexical burst (laugh, sigh,
# hmm, chirp, gasp, growl, beep) with its OWN structure, so a laugh actually sounds
# like a laugh. (valence, arousal) STEER each archetype (pitch / speed / brightness /
# count) but do not define it. Each builder returns a patch dict; rng gives the
# structural variation `collect` needs; _clamp_patch keeps it in bounds.
VOC_SYNONYMS = {
    "laugh": {"laugh", "giggle", "chuckle", "haha", "cackle", "delight", "amused"},
    "sigh":  {"sigh", "exhale", "weary", "resigned", "deflate", "tired"},
    "hmm":   {"hmm", "hum", "ponder", "mmm", "think", "thinking", "consider", "doubt"},
    "chirp": {"chirp", "greet", "hello", "hi", "greeting", "trill", "cheer"},
    "gasp":  {"gasp", "surprise", "startle", "oh", "alarm", "shock"},
    "growl": {"growl", "grr", "annoyed", "frustrated", "warn", "displeased", "grumble"},
    "beep":  {"beep", "ack", "acknowledge", "ok", "okay", "confirm", "ready", "online",
              "yes", "affirm", "affirmative", "yep", "agree"},
    "buzz":  {"buzz", "no", "nope", "deny", "reject", "negative", "wrong", "decline"},
}


def voc_archetype(name, tags=()):
    """Map a case name + tags to a burst archetype, or None (-> (v,a) fallback)."""
    toks = set(str(name).lower().replace("-", " ").replace("_", " ").split())
    toks |= {str(t).lower() for t in (tags or ())}
    for arch, syn in VOC_SYNONYMS.items():
        if toks & syn:
            return arch
    return None


def _voc_fallback(valence, arousal):
    if arousal >= 0.4:
        return "laugh" if valence >= 0.3 else "gasp"
    if arousal <= -0.2:
        return "sigh" if valence < 0.2 else "hmm"
    return "chirp" if valence >= 0.2 else "hmm"


def _jf(rng, oct_):                       # multiplicative jitter, ± oct_ octaves
    return float(2.0 ** rng.uniform(-oct_, oct_))


def _voc_laugh(c, v, a, rng):
    """ha-ha-ha: 3-5 short bright pulses, each a quick bounce, stepping down."""
    n = int(min(5, max(3, round(3 + 2 * max(0.0, a) + int(rng.integers(0, 2))))))
    base = 520.0 * c["f0_mul"] * (1.0 + 0.35 * max(0.0, a)) * (1.0 + 0.08 * v) * _jf(rng, 0.12)
    notes = []
    for i in range(n):
        peak = base * (1.0 - 0.06 * i) * _jf(rng, 0.04)
        dur = 0.075 * c["dur_mul"] * _jf(rng, 0.10)
        notes.append({"f0": peak * 1.04, "f1": peak * 0.80, "mid": [[0.35, peak * 1.14]],
                      "dur": dur, "decay_tau": dur * 0.5, "fm_rate": 9.0, "fm_depth": 2.0})
    return {"notes": notes, "gap": 0.034 * _jf(rng, 0.15),
            "detune_cents": c["detune"], "lp_cutoff": 3200.0}


def _voc_sigh(c, v, a, rng):
    """one long descending exhale, breathy (dark), optional settle."""
    f0 = 470.0 * c["f0_mul"] * (1.0 + 0.2 * a) * (1.0 + 0.1 * v) * _jf(rng, 0.10)
    f1 = f0 * (0.45 + 0.05 * rng.uniform(-1, 1))
    dur = (0.50 + 0.12 * (-min(0.0, a))) * c["dur_mul"] * _jf(rng, 0.08)
    notes = [{"f0": f0, "f1": f1, "mid": [[0.30, f0 * 0.88]], "dur": dur,
              "decay_tau": dur * 0.7, "fm_rate": 5.0, "fm_depth": 3.0}]
    if rng.uniform() < 0.5:
        notes.append({"f0": f1 * 0.98, "f1": f1 * 0.9, "dur": dur * 0.4, "decay_tau": dur * 0.3})
    return {"notes": notes, "gap": 0.02, "detune_cents": c["detune"], "lp_cutoff": 1800.0}


def _voc_hmm(c, v, a, rng):
    """a low, almost-flat hum with a small dip-and-return — pondering."""
    f0 = 235.0 * c["f0_mul"] * (1.0 + 0.12 * a) * _jf(rng, 0.10)
    dur = 0.42 * c["dur_mul"] * _jf(rng, 0.10)
    return {"notes": [{"f0": f0, "f1": f0 * 0.99, "mid": [[0.5, f0 * 0.93]], "dur": dur,
                       "decay_tau": dur * 0.7, "fm_rate": 5.0, "fm_depth": 2.5}],
            "gap": 0.02, "detune_cents": c["detune"], "lp_cutoff": 1600.0}


def _voc_chirp(c, v, a, rng):
    """1-2 short bright rising chirps — a friendly greeting."""
    n = 1 + int(rng.integers(0, 2))
    base = 620.0 * c["f0_mul"] * (1.0 + 0.30 * a) * (1.0 + 0.08 * v) * _jf(rng, 0.10)
    notes = []
    for i in range(n):
        f0 = base * (1.0 + 0.12 * i)
        notes.append({"f0": f0, "f1": f0 * (1.45 + 0.10 * rng.uniform(-1, 1)),
                      "dur": 0.085 * c["dur_mul"], "decay_tau": 0.05,
                      "fm_rate": 9.0, "fm_depth": 1.5})
    return {"notes": notes, "gap": 0.03, "detune_cents": c["detune"], "lp_cutoff": 3400.0}


def _voc_gasp(c, v, a, rng):
    """a very short, sharp upward intake — surprise."""
    f0 = 480.0 * c["f0_mul"] * (1.0 + 0.2 * a) * _jf(rng, 0.08)
    return {"notes": [{"f0": f0, "f1": f0 * (1.6 + 0.15 * rng.uniform(-1, 1)),
                       "dur": 0.06 * c["dur_mul"], "decay_tau": 0.035,
                       "fm_rate": 9.0, "fm_depth": 1.0}],
            "gap": 0.02, "detune_cents": c["detune"], "lp_cutoff": 3600.0}


def _voc_growl(c, v, a, rng):
    """a low, gritty, sustained burble — displeasure (sample-&-hold + ring)."""
    f0 = 175.0 * c["f0_mul"] * _jf(rng, 0.10)
    dur = 0.42 * c["dur_mul"] * _jf(rng, 0.08)
    return {"notes": [{"f0": f0, "f1": f0 * 0.92, "dur": dur, "decay_tau": dur * 0.7,
                       "fm_rate": 7.0, "fm_depth": 4.0, "sh_rate": 11.0,
                       "ring_hz": 90.0, "ring_depth": 0.2}],
            "gap": 0.02, "detune_cents": c["detune"], "lp_cutoff": 1500.0}


def _voc_beep(c, v, a, rng):
    """two short bright blips, slightly RISING — a positive acknowledgement (yes)."""
    f0 = 660.0 * c["f0_mul"] * (1.0 + 0.1 * a) * _jf(rng, 0.06)
    n1 = {"f0": f0, "f1": f0 * 1.04, "dur": 0.05, "decay_tau": 0.025}
    n2 = {"f0": f0 * 1.12, "f1": f0 * 1.16, "dur": 0.05, "decay_tau": 0.025}
    return {"notes": [n1, n2], "gap": 0.05,
            "detune_cents": c["detune"], "lp_cutoff": 3000.0}


def _voc_buzz(c, v, a, rng):
    """a short, low, descending double 'bzzt' — a negative acknowledgement (no)."""
    f0 = 300.0 * c["f0_mul"] * _jf(rng, 0.06)
    n1 = {"f0": f0, "f1": f0 * 0.72, "dur": 0.07, "decay_tau": 0.04,
          "sh_rate": 7.0, "fm_rate": 6.0, "fm_depth": 3.0}
    n2 = {"f0": f0 * 0.8, "f1": f0 * 0.58, "dur": 0.08, "decay_tau": 0.05,
          "sh_rate": 7.0, "fm_rate": 6.0, "fm_depth": 3.0}
    return {"notes": [n1, n2], "gap": 0.04,
            "detune_cents": c["detune"], "lp_cutoff": 1700.0}


VOC_BUILDERS = {
    "laugh": _voc_laugh, "sigh": _voc_sigh, "hmm": _voc_hmm, "chirp": _voc_chirp,
    "gasp": _voc_gasp, "growl": _voc_growl, "beep": _voc_beep, "buzz": _voc_buzz,
}


def generate_vocalization(character, name, valence, arousal, *, tags=(), seed=0):
    """Build a non-verbal affect-burst patch for a vocalization case. The archetype
    comes from the name/tags (laugh/sigh/hmm/chirp/gasp/growl/beep), falling back to
    an (valence, arousal) heuristic. seed varies the structural choices (for collect)."""
    c = CHARACTER.get(character, CHARACTER["qd"])
    arch = voc_archetype(name, tags) or _voc_fallback(valence, arousal)
    rng = np.random.default_rng(int(seed))
    return _clamp_patch(VOC_BUILDERS[arch](c, float(valence), float(arousal), rng))
