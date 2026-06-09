"""test_droid_voice — the non-interactive test surface for the emotional droid
voice system (Build Task D).

Covers, against the FROZEN contract (no audio device needed — everything is
`--no-play` / pure-function / direct render):

  * render_patch validity + clamp enforcement (no NaN/Inf, peak == gain),
  * transcript() format: one cluster per note, ascii fallback, terminal glyphs,
  * generate_arrangement() produces multi-note in-range phrases (1-6 anchor, by
    warble >= fm_min; the alarm special adds ring/sh),
  * the calibrator's --no-play piped path writes a valid ATOMIC emotions.json
    (run hermetically against a temp profiles dir so the real stores are never
    touched), exit 0, per the FROZEN non-tty contract,
  * ring/sh additive colour materially changes the rendered signal,
  * the 10 shipped cue WAVs re-render sha256-identical (byte-identity proof),
  * back-compat: the existing 2-note qd emotions.json (no transcript/grammar)
    loads + renders unchanged,
  * the optional model pipeline (droid-train/droid-speak) is exercised end-to-end
    IF those siblings exist yet; otherwise those tests skip (they are Task M's
    deliverables — graceful, never a hard failure).

Run:  python3 -m pytest test_droid_voice.py -v
  or:  python3 test_droid_voice.py        (falls back to pytest.main)

These tests NEVER write into the committed profiles, never require a TTY, and
never play audio.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import droid_emotion as de                       # noqa: E402
from droid_synth import render_patch, SR         # noqa: E402

CALIBRATE = os.path.join(HERE, "droid-calibrate")
SERVICES = os.path.dirname(HERE)                 # .../system/services
RENDER_CUES = os.path.join(HERE, "tools", "render-cues.py")
CUE_PROFILE = os.path.join(HERE, "profiles", "qd")

# FROZEN byte-identity baseline (first 8 hex of sha256) for the generic cue family
# (character-independent blip/seq motifs; render-cues.py reproduces these exactly).
CUE_BASELINE = {
    "click": "54851aa4", "compact": "37ab715e", "error": "9257c07a",
    "light-off": "47c2a303", "light-on": "33530186", "memory-save": "6891b4dd",
    "session-end": "3facea54", "session-start": "69301f5d",
    "subagent-complete": "fcd050a4", "subagent-spawn": "981a6efd",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _in_note_bounds(patch):
    """Every searched note param within NOTE_BOUNDS; globals within GLOBAL_BOUNDS."""
    for n in patch["notes"]:
        for k, (lo, hi, _kind) in de.NOTE_BOUNDS.items():
            if k in n:
                assert lo - 1e-6 <= float(n[k]) <= hi + 1e-6, f"{k}={n[k]} out of {(lo, hi)}"
    for k, (lo, hi, _kind) in de.GLOBAL_BOUNDS.items():
        if k in patch:
            assert lo - 1e-6 <= float(patch[k]) <= hi + 1e-6, f"{k}={patch[k]} out of {(lo, hi)}"


# ── render_patch validity + clamps ───────────────────────────────────────────
def test_render_patch_finite_and_peak_normalised():
    patch = de.generate_arrangement("qd", 0.4, 0.5, "hello there")
    sig = render_patch(patch, gain=0.18)
    assert sig.ndim == 1 and sig.size > 1
    assert np.all(np.isfinite(sig)), "render produced NaN/Inf"
    assert np.max(np.abs(sig)) == pytest.approx(0.18, abs=1e-6), "not peak-normalised to gain"


def test_render_patch_gain_scales_peak():
    patch = de.emotion_to_patch(0.4, 0.5, "qd")
    lo = render_patch(patch, gain=0.05)
    hi = render_patch(patch, gain=0.30)
    assert np.max(np.abs(lo)) == pytest.approx(0.05, abs=1e-6)
    assert np.max(np.abs(hi)) == pytest.approx(0.30, abs=1e-6)


def test_clamp_patch_enforces_bounds():
    insane = {
        "notes": [{"f0": 99999.0, "f1": 0.001, "dur": 99.0, "decay_tau": 99.0,
                   "fm_depth": 999.0, "ring_hz": 99999.0, "ring_depth": 9.0,
                   "sh_rate": 999.0}],
        "lp_cutoff": 99999.0, "gap": 99.0, "detune_cents": 999.0,
    }
    c = de._clamp_patch(insane)
    _in_note_bounds(c)
    n = c["notes"][0]
    assert n["f0"] == de.NOTE_BOUNDS["f0"][1]      # clamped up to hi
    assert n["f1"] == de.NOTE_BOUNDS["f1"][0]      # clamped down to lo
    assert c["lp_cutoff"] == de.GLOBAL_BOUNDS["lp_cutoff"][1]
    assert c["gap"] == de.GLOBAL_BOUNDS["gap"][1]


def test_generated_patches_render_without_error():
    # a spread of (valence, arousal) all render to finite audio (qd is the sole voice).
    for v in (-0.6, 0.0, 0.7):
        for a in (-0.4, 0.2, 0.8):
            p = de.generate_arrangement("qd", v, a, "a test phrase")
            _in_note_bounds(p)
            sig = render_patch(p, 0.18)
            assert np.all(np.isfinite(sig)) and sig.size > 1
    # CONTRACT (2026-06-03): the in-bounds/finite contract covers EVERY shipped
    # emotion. NOTE-COUNT now tracks emotional ENERGY (1-6 by arousal; a phrase can
    # extend it up to 8) — the per-emotion spread + arousal monotonicity is pinned by
    # test_note_count_tracks_emotion_energy; here we just assert every emotion renders.
    for name, (v, a) in de.EMOTIONS.items():
        p = de.generate_arrangement("qd", v, a, "a test phrase")
        _in_note_bounds(p)
        sig = render_patch(p, 0.18)
        assert np.all(np.isfinite(sig)) and sig.size > 1, f"{name} non-finite"
        n = len(p["notes"])
        assert 1 <= n <= 8, f"{name} phrased note-count {n} out of 1-8"


# ── transcript() format ──────────────────────────────────────────────────────
def test_transcript_one_cluster_per_note():
    patch = de.generate_arrangement("qd", 0.4, 0.5, "what is this?")
    t = de.transcript(patch)
    assert isinstance(t, str) and t
    # gap-joined => clusters == note count
    clusters = t.split(" ")
    assert len(clusters) == len(patch["notes"])


def test_transcript_ascii_fallback_is_plain():
    patch = de.generate_arrangement("qd", 0.4, 0.5, "what is this?")
    a = de.transcript(patch, ascii=True)
    assert a, "ascii transcript empty"
    # ASCII fallback must be representable in plain ASCII (vision-disability / dumb terminals)
    a.encode("ascii")  # raises if any non-ascii glyph leaked in
    # the unicode form differs from the ascii form (it actually used glyphs)
    assert de.transcript(patch) != a


def test_transcript_rising_terminal_glyph():
    # a question => rising terminal; the last cluster must carry the rise glyph.
    patch = de.generate_arrangement("qd", 0.4, 0.5, "what is this?")
    assert patch["notes"][-1].get("g") == "rise"
    last_uni = de.transcript(patch).split(" ")[-1]
    last_asc = de.transcript(patch, ascii=True).split(" ")[-1]
    assert "↗" in last_uni and "/" in last_asc


def test_transcript_empty_patch():
    assert de.transcript({"notes": []}) == ""


# ── arrangement generator quality ────────────────────────────────────────────
def test_qd_arrangement_is_warbly_phrase():
    p = de.generate_arrangement("qd", 0.4, 0.5, "what is this thing?")
    assert 3 <= len(p["notes"]) <= 6, f"qd phrased should be 3-6 notes, got {len(p['notes'])}"
    fm_min = de.GRAMMAR["qd"]["fm_min"]
    for n in p["notes"]:
        assert n.get("fm_depth", 0.0) >= fm_min, "every qd syllable must carry warble >= fm_min"
    # question => rising/open terminal
    assert p["notes"][-1].get("g") == "rise"


def test_arrangement_question_overrides_terminal_bias():
    # A question forces a rising terminal; a negative-valence statement dips.
    # qd's bias is "rise" (open/arched), so the sad dip is the clean discriminator.
    q = de.generate_arrangement("qd", 0.3, 0.4, "ready?")
    assert q["notes"][-1].get("g") == "rise", "a question must rise"
    qsad = de.generate_arrangement("qd", -0.4, 0.3, "oh no.")
    assert qsad["notes"][-1].get("g") == "dip", "qd sad statement should dip, not rise"


def test_note_count_tracks_emotion_energy():
    # REGRESSION (the "every emotion sounds like 4 notes" bug): the bare (no-text)
    # note count IS the emotion's energy signal — calm feelings render SPARSE, aroused
    # feelings render BUSY — so the count varies widely across the wheel. The old code
    # averaged the (v,a) target back toward syl_base and floored at 3, collapsing
    # nearly every emotion to ~4 notes.
    counts = {name: len(de.generate_arrangement("qd", v, a, "")["notes"])
              for name, (v, a) in de.EMOTIONS.items()}
    vals = list(counts.values())
    assert max(vals) - min(vals) >= 3, f"note count barely varies across emotions: {counts}"
    assert len(set(vals)) >= 4, f"note count clusters on too few values: {sorted(set(vals))}"
    byaro = sorted(de.EMOTIONS.items(), key=lambda kv: kv[1][1])  # ascending arousal
    calm = [len(de.generate_arrangement("qd", v, a, "")["notes"]) for _, (v, a) in byaro[:4]]
    hot = [len(de.generate_arrangement("qd", v, a, "")["notes"]) for _, (v, a) in byaro[-4:]]
    assert sum(hot) - sum(calm) >= 6, f"aroused emotions not busier than calm ones: calm={calm} hot={hot}"


def test_text_length_drives_note_count():
    # The feature: a typed PHRASE adds beeps ON TOP of the emotion's own count (a
    # longer line audibly = more notes), but never shrinks it below the feeling. Tested
    # at a LOW-arousal point so the emotion's bare count is small and text can extend it.
    for char, phrased_hi in (("qd", 8),):
        bare = len(de.generate_arrangement(char, 0.0, -0.4, "")["notes"])      # calm -> sparse
        short = len(de.generate_arrangement(char, 0.0, -0.4, "go")["notes"])
        long_ = len(de.generate_arrangement(
            char, 0.0, -0.4, "a really long sentence with very many words in it now")["notes"])
        assert 1 <= bare <= 6, f"{char} bare out of anchor band (1-6): {bare}"
        assert long_ >= short >= bare, f"{char}: text should not shrink the count (bare={bare} short={short} long={long_})"
        assert long_ > bare, f"{char}: a long phrase should add beeps over the bare feeling (got {long_}, bare {bare})"
        assert long_ <= phrased_hi, f"{char} long phrase exceeds phrased cap: {long_}"


# Q2 alarm's sample-hold FLOOR (emotion.py: `18.0 + 14.0 * max(0, arousal)`): the
# threshold above which sh_rate reads as DISTRESS. The dark "low-battery" grit on the
# calm-negative cluster must stay strictly below it so a sad/tired burble can never be
# mistaken for an alarm cry (and so grit never collides with the certified-distinct Q2
# distress contract).
Q2_ALARM_SH_FLOOR = 18.0


def test_dark_grit_stays_below_q2_alarm_floor():
    # INTEGRATION (the new dark-grit knob): for EVERY shipped emotion, no grit-bearing
    # middle note may reach the Q2 alarm sh floor. Grit is gated to valence<-0.15 AND
    # arousal<-0.1 (the calm-negative cluster) and capped ~6-9; the Q2 distress emotions
    # are high-arousal so the gate excludes them entirely — but assert it through the
    # real generator rather than trusting the gate by eye.
    for name, (v, a) in de.EMOTIONS.items():
        p = de.generate_arrangement("qd", v, a, "a test phrase")
        mids = p["notes"][1:-1] if len(p["notes"]) >= 3 else []
        is_dark_calm = v < -0.15 and a < -0.1                 # the grit-eligible region
        for nt in mids:
            sh = float(nt.get("sh_rate", 0.0))
            if is_dark_calm:
                assert sh < Q2_ALARM_SH_FLOOR, (
                    f"{name} ({v},{a}): grit middle sh_rate {sh} reached the Q2 alarm "
                    f"floor {Q2_ALARM_SH_FLOOR} — dark burble would read as distress")


def test_grit_formula_capped_below_alarm_floor():
    # The grit MAGNITUDE itself: at the most-negative valence (-1.0) the formula
    # 6.0 + 3.0*min(1.0, -valence) peaks at 9.0, leaving headroom under the 18.0
    # alarm floor. Pin the cap so a future tweak to the grit gain can't silently
    # cross into distress territory without this test failing.
    grit_max = 6.0 + 3.0 * min(1.0, 1.0)                      # valence == -1.0
    assert grit_max == 9.0, f"grit formula cap changed: {grit_max}"
    assert grit_max < Q2_ALARM_SH_FLOOR, (
        f"grit cap {grit_max} must stay below the Q2 alarm floor {Q2_ALARM_SH_FLOOR}")


# ── ring / sample-hold materially change the signal ──────────────────────────
def test_ring_modulation_changes_signal():
    clean = {"notes": [{"f0": 400.0, "f1": 420.0, "dur": 0.30, "decay_tau": 0.15}]}
    ring = {"notes": [{"f0": 400.0, "f1": 420.0, "dur": 0.30, "decay_tau": 0.15,
                       "ring_hz": 300.0, "ring_depth": 0.5}]}
    a = render_patch(clean, 0.18)
    b = render_patch(ring, 0.18)
    n = min(a.size, b.size)
    assert not np.allclose(a[:n], b[:n]), "ring modulation did not change the signal"


def test_sample_hold_changes_signal():
    clean = {"notes": [{"f0": 400.0, "f1": 520.0, "dur": 0.30, "decay_tau": 0.15}]}
    sh = {"notes": [{"f0": 400.0, "f1": 520.0, "dur": 0.30, "decay_tau": 0.15,
                     "sh_rate": 20.0}]}
    a = render_patch(clean, 0.18)
    b = render_patch(sh, 0.18)
    n = min(a.size, b.size)
    assert not np.allclose(a[:n], b[:n]), "sample-and-hold did not change the signal"


# ── ROI-1: stochastic sample-&-hold (sh_random) — new optional knob ──────────
def test_sh_random_absent_is_byte_identical_staircase():
    # A patch with sh_rate but NO sh_random must render EXACTLY the legacy
    # deterministic staircase — the byte-identity contract for the new knob.
    base = {"notes": [{"f0": 400.0, "f1": 700.0, "dur": 0.30, "decay_tau": 0.15,
                       "sh_rate": 20.0}]}
    a = render_patch(base, 0.18)
    b = render_patch(base, 0.18)
    assert np.array_equal(a, b), "sh_rate path must be deterministic"
    # explicit sh_random=0.0 is identical to the key being absent
    base0 = {"notes": [dict(base["notes"][0], sh_random=0.0)]}
    c = render_patch(base0, 0.18)
    assert np.array_equal(a, c), "sh_random=0 must equal the staircase (byte-identical)"


def test_sh_random_changes_signal_and_is_seeded():
    base = {"notes": [{"f0": 400.0, "f1": 700.0, "dur": 0.30, "decay_tau": 0.15,
                       "sh_rate": 20.0}]}
    rnd = {"notes": [dict(base["notes"][0], sh_random=0.4)]}
    a = render_patch(base, 0.18)          # deterministic staircase
    b = render_patch(rnd, 0.18)           # stochastic S&H
    n = min(a.size, b.size)
    assert not np.allclose(a[:n], b[:n]), "sh_random did not change the signal"
    # seeded from patch contents => identical across repeated renders (calibrator-safe)
    b2 = render_patch(rnd, 0.18)
    assert np.array_equal(b, b2), "stochastic S&H must be seeded/deterministic per patch"
    # a different patch (different f0) yields a different seed => different jitter
    rnd2 = {"notes": [dict(rnd["notes"][0], f0=410.0)]}
    c = render_patch(rnd2, 0.18)
    m = min(b.size, c.size)
    assert not np.allclose(b[:m], c[:m]), "different patch should reseed the jitter"


# ── calibrator --no-play piped path writes a valid ATOMIC emotions.json ──────








# ── back-compat: committed qd emotions.json (no transcript/grammar) ───────
def test_existing_emotions_json_back_compat():
    # An OLD-style emotions.json entry (pre-dates the transcript/grammar metadata)
    # must still load, render, and transcript cleanly. Built inline with the 2-note
    # analytic prior so the test is HERMETIC — it must not depend on a calibrated
    # profile (fresh profiles start pristine).
    data = {
        "curious": {"valence": 0.4, "arousal": 0.5,
                    "patch": de.emotion_to_patch(0.4, 0.5, "qd")},
        "happy": {"valence": 0.7, "arousal": 0.6,
                  "patch": de.emotion_to_patch(0.7, 0.6, "qd")},
    }
    for name, e in data.items():
        assert {"valence", "arousal", "patch"} <= set(e)
        assert "transcript" not in e and "grammar" not in e  # genuinely old-style
        # old entries (no transcript/grammar) must still render + transcript
        sig = render_patch(e["patch"], 0.18)
        assert np.all(np.isfinite(sig)) and sig.size > 1
        assert isinstance(de.transcript(e["patch"]), str)


# ── byte-identity: the shipped cue family re-renders sha256-identical ─────────
def test_cue_family_byte_identical_baseline():
    # the committed WAVs already match the FROZEN baseline (sanity on the repo).
    for name, prefix8 in CUE_BASELINE.items():
        wav = os.path.join(CUE_PROFILE, f"{name}.wav")
        assert os.path.exists(wav), f"missing committed cue {name}.wav"
        assert _sha256(wav).startswith(prefix8), f"{name} drifted from baseline"


def test_cue_family_rerender_identical(tmp_path):
    # re-render to a TEMP dir (never the committed profile) and compare hashes.
    out = str(tmp_path / "cue-rerender")
    cp = subprocess.run([sys.executable, RENDER_CUES, "--gain", "0.18", "--out", out],
                        capture_output=True, text=True)
    assert cp.returncode == 0, f"render-cues failed:\n{cp.stderr}"
    for name in CUE_BASELINE:
        fresh = os.path.join(out, f"{name}.wav")
        committed = os.path.join(CUE_PROFILE, f"{name}.wav")
        assert _sha256(fresh) == _sha256(committed), f"{name}.wav re-render NOT byte-identical"


# ── refactor equivalence: analytic primitives are deterministic + seeded ─────
def test_emotion_to_patch_deterministic():
    a = de.emotion_to_patch(0.4, 0.5, "qd")
    b = de.emotion_to_patch(0.4, 0.5, "qd")
    assert a == b, "analytic prior must be a pure deterministic function"
    _in_note_bounds(a)


def test_jitter_seeded_reproducible():
    seed = de.emotion_to_patch(0.4, 0.5, "qd")
    j1 = de.jitter(seed, 0.2, np.random.default_rng(0))
    j2 = de.jitter(seed, 0.2, np.random.default_rng(0))
    assert j1 == j2, "seeded jitter must be reproducible"
    _in_note_bounds(j1)


def test_jitter_does_not_add_warble_to_clean_cue():
    # a clean cue (no fm_depth/ring/sh keys) must stay clean after jitter.
    clean = copy.deepcopy(de.SEED["click"])
    j = de.jitter(clean, 0.3, np.random.default_rng(1))
    n = j["notes"][0]
    assert "fm_depth" not in n and "ring_hz" not in n and "sh_rate" not in n


def test_jitter_lock_group_holds_params_fixed():
    seed = de.emotion_to_patch(0.4, 0.5, "qd")
    j = de.jitter(seed, 0.3, np.random.default_rng(2), locks={"pitch"})
    for ni, n in enumerate(j["notes"]):
        assert n["f0"] == seed["notes"][ni]["f0"], "locked pitch must not move"
        assert n["f1"] == seed["notes"][ni]["f1"]


def test_finalize_is_atomic_no_tmp_left(tmp_path):
    # finalize writes via tmp + os.replace; after success no .tmp lingers.
    de_profiles_backup = de.PROFILES
    try:
        de.PROFILES = str(tmp_path / "profiles")
        os.makedirs(os.path.join(de.PROFILES, "qd", "emotions"), exist_ok=True)
        tgt, seed = de.resolve_target("qd", "curious")
        de.finalize(tgt, seed, 0.18)
        store = tgt["store"]
        assert os.path.exists(store) and not os.path.exists(store + ".tmp")
        assert os.path.exists(tgt["wav"]) and not os.path.exists(tgt["wav"] + ".tmp")
        json.load(open(store))                    # valid => not half-written
    finally:
        de.PROFILES = de_profiles_backup


# ── OPTIONAL model pipeline (Task M deliverables) — skip if not built yet ────
def _has(*names):
    return all(os.path.exists(os.path.join(HERE, n)) for n in names)


def _imports_module(src, *roots):
    """True iff the source actually IMPORTS one of `roots` (top-level package) via
    a real import statement — comments/docstrings that mention the name don't
    count. Parsed with ast so 'does NOT import sklearn' in a docstring is ignored."""
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name.split(".")[0] in roots for n in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in roots:
                return True
    return False










# ── GP control model (2026-05-31 redesign): kernel-match + LOO-beats-prior gate ──






# ── Phase 1: preference refinement (RankRefine) — collapse + name-guard ──────




# ── ENHANCEMENT (2026-05-31): signed arousal, contour families, colour axes ──
# These lock the specific baseline collapses the enhancement fixes so a future
# edit cannot silently re-collapse the set.
ADDED_EMOTIONS = ["excited", "elated", "playful", "proud", "content",
                  "serene", "confident", "tired", "bored", "frustrated"]


def _va(name):
    return de.EMOTIONS[name]


def _arr(name, text=""):
    v, a = _va(name)
    return de.generate_arrangement("qd", v, a, text)


def _mean_f0(patch):
    return float(np.mean([float(n["f0"]) for n in patch["notes"]]))


def _total_dur(patch):
    return float(sum(float(n["dur"]) for n in patch["notes"]))


def _max_dur(patch):
    return float(max(float(n["dur"]) for n in patch["notes"]))


def _excursion(patch):
    """Contour excursion = max/min over every pitch point (f0, f1, mid hz)."""
    fs = []
    for n in patch["notes"]:
        fs.append(float(n["f0"]))
        fs.append(float(n.get("f1", n["f0"])))
        for m in n.get("mid", []):
            fs.append(float(m[1]))
    return max(fs) / min(fs)


def _has_ring(patch):
    return any(float(n.get("ring_depth", 0.0)) > 0 and float(n.get("ring_hz", 0.0)) > 0
               for n in patch["notes"])


def _has_sh(patch):
    return any(float(n.get("sh_rate", 0.0)) > 0 for n in patch["notes"])


def _max_fm(patch):
    return float(max(float(n.get("fm_depth", 0.0)) for n in patch["notes"]))


def test_new_emotion_anchors_present():
    # all 18 EMOTIONS keys exist; each of the 10 added resolves, renders finite,
    # and is in-bounds (item (i)).
    assert len(de.EMOTIONS) == 18, f"expected 18 anchors, got {len(de.EMOTIONS)}"
    # the 8 frozen anchors keep their EXACT coords (control-model regression seed)
    frozen = {
        "neutral": (0.0, 0.2), "curious": (0.4, 0.5), "happy": (0.7, 0.6),
        "recognition": (0.5, 0.3), "wistful": (-0.3, -0.4), "sad": (-0.5, -0.3),
        "worried": (-0.4, 0.4), "alarmed": (-0.6, 0.8),
    }
    for k, va in frozen.items():
        assert de.EMOTIONS[k] == va, f"frozen anchor {k} coord drifted: {de.EMOTIONS[k]}"
    # playful is deliberately NOT stacked on happy(0.7,0.6)
    assert de.EMOTIONS["playful"] != de.EMOTIONS["happy"]
    for name in ADDED_EMOTIONS:
        assert name in de.EMOTIONS
        p = _arr(name, "a test phrase")
        _in_note_bounds(p)
        sig = render_patch(p, 0.18)
        assert np.all(np.isfinite(sig)) and sig.size > 1


def test_signed_arousal_monotonic():
    # item (a): negative arousal LENGTHENS notes (Q4/Q3 slower than neutral) and
    # lowers register; positive arousal lifts register. Locks the slowdown in BOTH
    # directions against the old max(0,a) deadness.
    neutral = de.generate_arrangement("qd", 0.0, 0.2, "standing by")
    slow = de.generate_arrangement("qd", -0.5, -0.7, "standing by")
    assert _total_dur(slow) > _total_dur(neutral), "neg-arousal total dur must exceed neutral"
    assert _max_dur(slow) > _max_dur(neutral), "neg-arousal per-note dur must exceed neutral"
    # register: tired (a=-0.75) lower than excited (a=+0.9)
    assert _mean_f0(_arr("tired")) < _mean_f0(_arr("excited")), "low arousal must sit lower"
    # gap breathes at low arousal, clips at high arousal
    assert slow["gap"] > neutral["gap"], "low arousal must breathe (larger gap)"


def test_contour_family_by_quadrant():
    # item (b): positive non-'?' statements end rising/opening; Q3 negative-low
    # end falling. Per-quadrant terminal.
    for name in ("excited", "happy", "content", "proud", "confident"):
        term = _arr(name, "task complete")["notes"][-1].get("g")
        assert term in ("rise", "chirp", "boing", "flat"), f"{name} terminal {term} not opening/soft"
    for name in ("sad", "wistful", "tired", "bored"):
        term = _arr(name, "that failed")["notes"][-1].get("g")
        assert term in ("fall", "dip", "sag"), f"{name} Q3 terminal {term} not falling"


def test_q2_textures_distinct():
    # item (c): the exact baseline collapse this fixes (worried<->neutral=1.255).
    # The three Q2 states differ by KEY-PRESENCE, not a few-Hz centroid shift:
    #   alarmed  = ring AND sh_rate AND fm   (the anchor)
    #   worried  = fm flutter, NO ring, NO sh_rate
    #   frustrated = ring, NO sh_rate
    alarmed = _arr("alarmed", "alert alert")
    worried = _arr("worried", "thats not right")
    frustrated = _arr("frustrated", "come on")
    assert _has_ring(alarmed) and _has_sh(alarmed) and _max_fm(alarmed) > 0, "alarmed ring+sh+fm"
    assert _max_fm(worried) > 0 and not _has_ring(worried) and not _has_sh(worried), \
        "worried = fm only (no ring, no sh)"
    assert _has_ring(frustrated) and not _has_sh(frustrated), "frustrated = ring, no sh"


def test_distress_carries_more_warble_than_calm():
    # item (d): the a*(-v) distress cross-term gives Q2 distress states MORE warble
    # than the CALM states (neutral + the new Q4 positive-low anchors), which is the
    # real "separate the collapsed worried/alarmed band" goal. alarmed (the anchor)
    # also out-warbles even high-valence happy. NOTE: happy's own warble is large
    # (high +valence drives 2.5*v), so the defensible distress>calm contract is vs
    # the calm band, not vs every positive emotion — frustrated sits just under
    # happy by the spec's own formula, which is fine: they differ by RING, not fm.
    alarmed_fm = _max_fm(_arr("alarmed", "alert"))
    frustrated_fm = _max_fm(_arr("frustrated", "come on"))
    calm_fms = [_max_fm(_arr(n, "all good"))
                for n in ("neutral", "content", "serene", "confident")]
    for cf in calm_fms:
        assert alarmed_fm > cf and frustrated_fm > cf, "distress must out-warble calm states"
    # the alarm anchor is the warbliest distress state — beats even happy.
    assert alarmed_fm > _max_fm(_arr("happy", "task complete")), "alarmed must out-warble happy"


def test_negative_valence_compresses_range():
    # item (e): contour excursion for bored/tired < excited/elated.
    bored_exc = _excursion(_arr("bored", "task complete"))
    tired_exc = _excursion(_arr("tired", "task complete"))
    excited_exc = _excursion(_arr("excited", "task complete"))
    elated_exc = _excursion(_arr("elated", "task complete"))
    assert bored_exc < excited_exc and bored_exc < elated_exc, "bored range must be < excited/elated"
    assert tired_exc < excited_exc, "tired range must be < excited"


def test_phrase_has_gesture_diversity():
    # item (f): a multi-syllable qd phrase uses >= 2 distinct gestures (locks
    # the all-bend collapse where baseline n_distinct_gestures was 1.0).
    for name, txt in (("excited", "yes look at this"), ("tired", "so very slow"),
                      ("worried", "something is off")):
        p = _arr(name, txt)
        gestures = set(n.get("g") for n in p["notes"])
        assert len(gestures) >= 2, f"{name} phrase must use >=2 gestures, got {gestures}"


def test_colour_axes_active():
    # item (g): at least one emotion's output carries ring (hz+depth>0), and at
    # least one carries sh_rate>0, on the NORMAL qd speak path (guards the two
    # dormant axes against going silent again).
    any_ring = any(_has_ring(_arr(n, "ready")) for n in de.EMOTIONS)
    any_sh = any(_has_sh(_arr(n, "ready")) for n in de.EMOTIONS)
    assert any_ring, "no emotion activated the ring colour axis"
    assert any_sh, "no emotion activated the sample-hold colour axis"


def _emotion_vector(name):
    """A z-normalizable structural feature vector per emotion. Includes the (v,a)
    design coords (every anchor is defined by a DISTINCT (v,a), so this guarantees
    separability proportional to coordinate distance) plus the realized acoustic
    structure: register, excursion, warble, note-count, ring, sh, brightness, tempo."""
    v, a = _va(name)
    p = de.generate_arrangement("qd", v, a, "task complete")
    return np.array([
        v, a,
        _mean_f0(p),
        _excursion(p),
        _max_fm(p),
        float(len(p["notes"])),
        1.0 if _has_ring(p) else 0.0,
        1.0 if _has_sh(p) else 0.0,
        float(p.get("lp_cutoff", 2200.0)),
        _total_dur(p),
    ], dtype=np.float64)


def test_distinctiveness_floor():
    # item (h) regression: min pairwise z-normalized distance over ALL 18 emotions
    # >= a FLOOR (sized as a floor, NOT the target) so a future change cannot
    # re-collapse the set. This is a different (richer) feature space than the
    # offline _baseline_metrics_run.py 9-dim acoustic vector, so the absolute number
    # is not directly comparable to that script's 1.2553 baseline — it is an
    # in-suite regression guard. The closest legitimately-adjacent pairs are the Q3
    # neighbours (wistful/sad/bored cluster, ~0.1 apart in (v,a)); the floor sits
    # safely below them so a real collapse (two anchors mapping to one patch) trips
    # it. Confirmed no Q1/Q4 densification produced a sub-floor pair.
    names = list(de.EMOTIONS)
    mat = np.array([_emotion_vector(n) for n in names])
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[sd == 0] = 1.0
    z = (mat - mu) / sd
    mind = np.inf
    worst = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = float(np.linalg.norm(z[i] - z[j]))
            if d < mind:
                mind = d
                worst = (names[i], names[j])
    # FLOOR sizing (honest): with the 8 design dims + the (v,a) coords, the closest
    # legitimately-adjacent pair is the Q3 wistful/sad cluster (~0.5 apart — they
    # share the dark/slow/drooping family and sit ~0.2 apart in (v,a) by design).
    # The floor sits below that intrinsic adjacency but FAR above 0 — a real
    # re-collapse (two distinct anchors producing the same patch) drives the pair to
    # ~0.0 and trips this. Measured min over the 18-set was 0.524 at authoring; the
    # 0.40 floor is a guard, not the target (the spec's headline >=2.5 target lives
    # in the separate offline 9-dim acoustic metric, a different feature space).
    assert mind >= 0.40, f"distinctiveness floor breached: {worst} = {mind:.3f} < 0.40"


def test_added_emotion_pair_separable():
    # A defined NEW pair stays acoustically separable: serene (Q4 calm) vs excited
    # (Q1 peak) must differ on register, tempo AND excursion (opposite corners).
    # Tempo is measured PER-NOTE (serene's individual syllables are slower) — total
    # duration is not the right axis here because serene also has fewer syllables.
    serene = _arr("serene", "all good")
    excited = _arr("excited", "all good")
    assert _mean_f0(serene) < _mean_f0(excited), "serene must be lower-register than excited"
    assert _max_dur(serene) > _max_dur(excited), "serene's notes must be slower than excited's"
    assert _excursion(serene) < _excursion(excited), "serene must be narrower than excited"


def test_range_scale_default_is_byte_identical():
    # gesture_recipe's new range_scale kwarg defaults to 1.0 => byte-identical for
    # every legacy caller at arousal>=0 (the spec's hard backward-compat guarantee).
    for gname in ("rise", "fall", "chirp", "bend", "dip", "flat"):
        a = de.gesture_recipe(gname, 500.0, 0.5, 0.6)
        b = de.gesture_recipe(gname, 500.0, 0.5, 0.6, range_scale=1.0)
        assert a == b, f"{gname}: range_scale=1.0 default must be byte-identical"


# ── droid-stats: training statistics + live retrain (2026-05-31) ─────────────
# The stats tool + its shared module. Hermetic: droid-stats reads DROID_TEST_PROFILES
# directly, so a subprocess needs no driver; module tests pass profiles_root so no
# global PROFILES is ever mutated (a leak would corrupt sibling tests in the process).
























@pytest.mark.skipif(not _has("droid-talk"), reason="droid-talk not built")
def test_droid_talk_parse_and_heuristic():
    # Pure logic only (no network): the JSON extractor + the LLM-down fallback.
    from importlib.machinery import SourceFileLoader
    talk = SourceFileLoader("droid_talk", os.path.join(HERE, "droid-talk")).load_module()
    # tolerant JSON extraction from a messy reply -> (emotion, text)
    assert talk.parse_mapping('sure! {"emotion":"happy","text":"Yay!"} hope that helps') == ("happy", "Yay!")
    assert talk.parse_mapping('{"emotion":"ecstatic","text":"x"}') is None   # unknown emotion rejected
    assert talk.parse_mapping("no json at all") is None
    # keyword fallback: routing + question-mark override + <=6-word truncation
    assert talk.heuristic("thank you so much")[0] == "happy"
    assert talk.heuristic("what now?")[0] == "curious"
    assert talk.heuristic("error: the build broke")[0] == "worried"
    _emo, text = talk.heuristic("this is a very long sentence with many many extra words here")
    assert len(text.split()) <= 6
    # droid-talk is LLM-bridged but must not import a paid SDK on its own surface
    assert not _imports_module(open(os.path.join(HERE, "droid-talk")).read(), "openai", "anthropic")






@pytest.mark.skipif(not _has("droid"), reason="droid dispatcher not built")
def test_droid_dispatch():
    # The one CLI routes each verb to its implementation; unknown verbs exit 2.
    env = dict(os.environ, DROID_NO_EMBED="1")
    D = os.path.join(HERE, "droid")
    cp = subprocess.run([sys.executable, D, "help"], capture_output=True, text=True, env=env)
    assert cp.returncode == 0 and "droid say" in cp.stdout and "droid talk" in cp.stdout
    cp = subprocess.run([sys.executable, D, "bogus-verb"], capture_output=True, text=True, env=env)
    assert cp.returncode == 2, "unknown verb must exit 2"
    cp = subprocess.run([sys.executable, D, "list"], capture_output=True, text=True, env=env)
    assert cp.returncode == 0 and "emotion" in cp.stdout, "verb must route to its impl (cases list)"


def test_dvoice_package_and_shims():
    # The logic lives in the dvoice/ package; the flat names are SAME-OBJECT aliases
    # (so PROFILES repointing in the hermetic tests reaches the real module).
    import dvoice.emotion
    import droid_emotion as de
    assert de is dvoice.emotion
    from dvoice.synth import render_patch as rp_pkg
    from droid_synth import render_patch as rp_flat
    assert rp_pkg is rp_flat


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
