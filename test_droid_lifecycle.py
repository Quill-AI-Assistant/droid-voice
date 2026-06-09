"""test_droid_lifecycle — the NEW droid-voice lifecycle (cases / store / reward /
collect / review). Hermetic: every test repoints de.PROFILES at a tmp dir, so the
shipped profiles/qd data is never touched.

    DROID_NO_EMBED=1 python3 -m pytest test_droid_lifecycle.py -q
"""
import os
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("DROID_NO_EMBED", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dvoice import cases as C        # noqa: E402
from dvoice import emotion as de     # noqa: E402
from dvoice import reward as R       # noqa: E402
from dvoice import store             # noqa: E402


def test_no_dvoice_module_shadows_stdlib():
    """dvoice/ is on sys.path (flat-shim back-compat), so any module here named like a
    stdlib module silently shadows it. `tty.py` did exactly that and crashed every
    interactive verb (tty.setcbreak vanished). Guard against the whole class."""
    import tty as stdlib_tty
    assert hasattr(stdlib_tty, "setcbreak"), "stdlib tty is shadowed — a dvoice module collides"
    dvoice_dir = os.path.join(HERE, "dvoice")
    stems = {f[:-3] for f in os.listdir(dvoice_dir) if f.endswith(".py")}
    clashes = stems & set(getattr(sys, "stdlib_module_names", set()))
    assert not clashes, f"dvoice modules shadow stdlib: {clashes}"


@pytest.fixture
def profile(tmp_path, monkeypatch):
    """A hermetic profile dir; cases/store read de.PROFILES at call time."""
    monkeypatch.setattr(de, "PROFILES", str(tmp_path))
    os.makedirs(tmp_path / "qd", exist_ok=True)
    return "qd"


def _patch(v=0.4, a=0.5):
    return de.generate_arrangement("qd", v, a, "")


# ── featurize / structural features ──────────────────────────────────────────
def test_featurize_dim_and_version():
    p = _patch()
    f = R.featurize(p, 0.4, 0.5)
    assert f.shape[0] == 74 and np.isfinite(f).all()
    assert R.FEATURE_VERSION == 2


def test_struct_feats_sees_gestures():
    p = _patch(0.6, 0.5)
    s = R.struct_feats(p)
    assert s.shape[0] == 15
    assert s[1:1 + len(R.GESTURE_ORDER)].sum() > 0     # at least one gesture counted


# ── judge: fit / score / generalize ──────────────────────────────────────────
def _separable_set(n=40, seed=0):
    """Positives = high-arousal rises, negatives = low/dippy — separable by design."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        v, a = rng.uniform(0.3, 0.9), rng.uniform(0.4, 0.9)
        X.append(R.featurize(de.generate_arrangement("qd", v, a, ""), v, a)); y.append(1.0)
        v2, a2 = rng.uniform(-0.9, -0.3), rng.uniform(-0.9, -0.3)
        X.append(R.featurize(de.generate_arrangement("qd", v2, a2, ""), v2, a2)); y.append(-1.0)
    return np.asarray(X), np.asarray(y)


def test_judge_ranks_positives_above_negatives():
    X, y = _separable_set()
    j = R.fit(X, y)
    s = R.score(j, X)
    assert s[y > 0].mean() > s[y < 0].mean()


def test_heldout_auc_beats_chance():
    X, y = _separable_set(n=60)
    auc = R.heldout_ranking_accuracy(X, y, seed=1)["auc"]
    assert auc > 0.6


# ── judge: STRUCTURAL discrimination, NOT the (v,a) leak ──────────────────────
# _separable_set above is separable on (valence, arousal) alone, and featurize
# (reward.py:93) appends raw (v,a) as the last two dims — so its AUC proves the
# fit/score PLUMBING works, not that the judge learned anything STRUCTURAL. The
# fixture below holds (v,a) non-discriminative and forces the signal into the 15-d
# struct_feats block (note-count + warble/ring/grit), so a judge that only read the
# leaked (v,a) scores ~chance. (The teeth test proves exactly that.)
def _set_texture(patch, *, fm, ring, sh):
    """Force every note's texture knobs so struct_feats' warble/ring/grit counts are
    deterministic (clean vs textured), independent of the synth's stochastic timbre."""
    for nn in patch.get("notes", []):
        nn["fm_depth"] = fm
        nn["ring_depth"] = ring
        nn["sh_rate"] = sh
    return patch


def _structural_set(n=40, seed=0):
    """Positives (terse + clean) and negatives (busy + textured) at an IDENTICALLY-
    distributed (v,a): the SAME (v,a) draw feeds both classes each row, so the two raw
    (v,a) dims carry NO class signal. The classes differ ONLY in structure — note count
    {2,3} vs {4,5} and clean vs warble+ring+grit — which lives in struct_feats."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        v = float(rng.uniform(-0.6, 0.6))                 # one draw, BOTH classes
        a = float(rng.uniform(-0.6, 0.6))
        pos = de.generate_arrangement("qd", v, a, "", force_syl=int(rng.choice([2, 3])))
        neg = de.generate_arrangement("qd", v, a, "", force_syl=int(rng.choice([4, 5])), ring=0.35)
        _set_texture(pos, fm=0.0, ring=0.0, sh=0.0)        # clean
        _set_texture(neg, fm=8.0, ring=0.35, sh=8.0)       # warble + ring + grit
        X.append(R.featurize(pos, v, a)); y.append(1.0)
        X.append(R.featurize(neg, v, a)); y.append(-1.0)
    return np.asarray(X), np.asarray(y)


def test_judge_discriminates_structure_at_fixed_va():
    """Held-out AUC must come from STRUCTURE, not the (v,a) featurize passes through.
    With (v,a) identically distributed across classes, only a judge that learned
    struct_feats (note-count / texture) can rank keep > drop."""
    X, y = _structural_set(n=60)
    auc = R.heldout_ranking_accuracy(X, y, seed=1)["auc"]
    assert auc > 0.6, f"judge did not learn structure at fixed (v,a): AUC={auc:.3f}"


def test_structural_set_has_teeth_va_alone_is_chance():
    """Teeth check: ablate every feature EXCEPT the trailing (v,a) and the same fixture
    is no longer separable — so the test above genuinely measures structural learning,
    not a (v,a) read-off. If this ever clears the bar, the structural test is a
    tautology like _separable_set."""
    X, y = _structural_set(n=60)
    X_va_only = X.copy()
    X_va_only[:, :-2] = 0.0                                # keep only (valence, arousal)
    auc = R.heldout_ranking_accuracy(X_va_only, y, seed=1)["auc"]
    assert auc < 0.6, f"(v,a) alone should be ~chance but AUC={auc:.3f} — fixture leaks (v,a)"


def test_median_lengthscale_not_unit():
    X, y = _separable_set()
    j = R.fit(X, y)                                    # default length_scale="median"
    assert j["ls"] > 1.5                               # high-d -> median heuristic, not 1.0


def test_verb_scripts_parse():
    """The extensionless verb scripts (droid, droid-studio, droid-collect, …) aren't
    imported by pytest, so a syntax error in them slips past every other test. Compile
    each one. (A nested-quote f-string in droid-studio once crashed the whole studio.)"""
    import ast
    for fn in ("droid", "droid-studio", "droid-collect", "droid-review", "droid-cases",
               "droid-train", "droid-say", "droid-demo", "droid-eval", "droid-models",
               "droid-doctor", "droid-profiles", "droid-bootstrap", "droid-abtest",
               "web/server.py"):
        path = os.path.join(HERE, fn)
        ast.parse(open(path).read(), filename=fn)        # raises SyntaxError on failure


# ── every verb the dispatcher advertises is wired and runs ───────────────────
def _dispatcher_verbs():
    """Load the `droid` dispatcher's VERBS map without running main()."""
    ns = {"__name__": "droid_dispatch_test", "__file__": os.path.join(HERE, "droid")}
    exec(compile(open(os.path.join(HERE, "droid")).read(), "droid", "exec"), ns)
    return ns["VERBS"]


def test_dispatcher_targets_all_exist():
    for verb, (script, _prefix) in _dispatcher_verbs().items():
        assert os.path.exists(os.path.join(HERE, script)), f"verb '{verb}' -> missing {script}"


@pytest.mark.parametrize("verb", ["cases", "collect", "review", "train", "say",
                                  "demo", "eval", "models", "profiles", "doctor",
                                  "bootstrap", "abtest", "web"])
def test_verb_help_runs(verb):
    """`droid <verb> --help` routes through the dispatcher and exits cleanly (argparse)."""
    cp = subprocess.run([sys.executable, os.path.join(HERE, "droid"), verb, "--help"],
                        capture_output=True, text=True,
                        env=dict(os.environ, DROID_NO_EMBED="1"))
    assert cp.returncode == 0, f"droid {verb} --help failed:\n{cp.stderr}"
    assert "usage" in (cp.stdout + cp.stderr).lower()


def test_demo_floor_path_renders():
    """droid demo --tour on a judge-less profile uses the analytic floor and emits each beat."""
    cp = subprocess.run([sys.executable, os.path.join(HERE, "droid"), "demo",
                         "--tour", "--no-play", "--profile", "untrained-johndoe"],
                        capture_output=True, text=True,
                        env=dict(os.environ, DROID_NO_EMBED="1"))
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.count("♪") >= 10            # one beat per case, floor path


def test_diverse_set_spreads_and_quality():
    """diverse_set returns N judge-approved candidates that are structurally spread (the
    'play N different' selector), and degrades gracefully with no judge."""
    X, y = _separable_set()
    j = R.fit(X, y)
    ds = R.diverse_set(j, "qd", 0.65, 0.4, "", n=5, pool_k=20, seed=0)
    assert len(ds) == 5
    assert len({de.transcript(p) for p in ds}) >= 3            # spread, not near-clones
    Z = np.asarray([R.featurize(p, 0.65, 0.4) for p in ds])
    dmin = min(np.linalg.norm(Z[i] - Z[k]) for i in range(5) for k in range(i + 1, 5))
    assert dmin > 0.0                                          # no two identical
    assert len(R.diverse_set(None, "qd", 0.65, 0.4, "", n=4, pool_k=16, seed=0)) == 4  # no-judge floor


def test_demo_interactive_reacts_to_input():
    """`droid demo` is an interactive REPL: piped lines (a case, a tagged phrase) each react,
    and the inline [tag] is parsed out of the spoken text. EOF/q exits cleanly."""
    cp = subprocess.run([sys.executable, os.path.join(HERE, "droid"), "demo", "--no-play"],
                        input="proud\nthat failed [error]\n?\nq\n",
                        capture_output=True, text=True,
                        env=dict(os.environ, DROID_NO_EMBED="1"))
    assert cp.returncode == 0, cp.stderr
    assert "[proud]" in cp.stdout                          # bare case reacted
    assert "[error]" in cp.stdout and "that failed" in cp.stdout  # tag parsed from the phrase


def test_versions_roundtrip(profile):
    """The engine behind `droid models`: save → list → activate → soft-delete."""
    from dvoice import versions
    X, y = _separable_set()
    j = R.fit(X, y)
    v1 = versions.save_version(profile, j, auc=0.70, n_votes=len(y), dataset="t", gate=True, ts="t1")
    v2 = versions.save_version(profile, j, auc=0.71, n_votes=len(y), dataset="t", gate=True, ts="t2")
    assert [e["version"] for e in versions.list_versions(profile)] == [v1, v2]
    versions.set_active_version(profile, v1)
    assert versions.active_version(profile) == v1
    with pytest.raises(ValueError):
        versions.set_active_version(profile, 999)            # no such version
    versions.delete_version(profile, v2)
    assert [e["version"] for e in versions.list_versions(profile)] == [v1]


def test_reward_backend_reports():
    assert R.backend() in ("mlx", "numpy")
    assert R.fit(*_separable_set(n=20)).get("backend") in ("mlx", "numpy")


def test_mlx_numpy_parity(monkeypatch):
    """MLX (float32) and numpy (float64) must produce the same RANKING (the judge only
    ranks) — guards against an MLX-port regression."""
    X, y = _separable_set(n=50)
    monkeypatch.setenv("DROID_MLX", "0")               # force numpy
    sn = R.score(R.fit(X, y), X)
    monkeypatch.delenv("DROID_MLX", raising=False)
    if not R.mlx_enabled():                            # no MLX on this host -> nothing to compare
        return
    sm = R.score(R.fit(X, y), X)
    assert np.corrcoef(sn, sm)[0, 1] > 0.99            # rankings agree despite float32


# ── persistence ──────────────────────────────────────────────────────────────
def test_judge_save_load_roundtrip(tmp_path):
    X, y = _separable_set()
    j = R.fit(X, y)
    p = str(tmp_path / "judge.npz")
    R.save_judge(j, p)
    j2 = R.load_judge(p)
    assert j2 is not None
    assert np.allclose(R.score(j, X[:5]), R.score(j2, X[:5]))


def test_judge_load_rejects_version_mismatch(tmp_path, monkeypatch):
    X, y = _separable_set()
    p = str(tmp_path / "judge.npz")
    R.save_judge(R.fit(X, y), p)
    monkeypatch.setattr(R, "FEATURE_VERSION", R.FEATURE_VERSION + 99)
    assert R.load_judge(p) is None                     # stale features -> rejected


# ── store: append / load / tombstone ─────────────────────────────────────────
def test_store_append_load_tombstone(profile):
    store.append(profile, {"id": "a", "case": "proud", "label": "y", "patch": _patch(), "ctx": [0.6, 0.4]})
    store.append(profile, {"id": "b", "case": "proud", "label": "n", "patch": _patch(), "ctx": [0.6, 0.4]})
    assert len(store.load(profile)) == 2
    store.tombstone(profile, "a")
    rows = store.load(profile)
    assert len(rows) == 1 and rows[0]["id"] == "b"
    assert store.counts(profile)["proud"] == (1, 0, 1)


# ── cases: add / resolve / va ─────────────────────────────────────────────────
def test_cases_add_and_resolve(profile):
    C.add(profile, {"name": "smug", "type": "emotion", "valence": 0.5, "arousal": 0.1})
    C.add(profile, {"name": "greet", "type": "expression", "base": "content"})
    C.add(profile, {"name": "giggle", "type": "vocalization", "valence": 0.7, "arousal": 0.5})
    assert set(C.names(profile, "vocalization")) >= {"giggle"}
    r = C.resolve(profile, "greet")                    # expression resolves base emotion
    assert r["type"] == "expression"
    assert (r["valence"], r["arousal"]) == tuple(de.EMOTIONS["content"])
    assert C.resolve(profile, "smug")["valence"] == 0.5


def test_remove_case_sticks(profile):
    C.add(profile, {"name": "smug", "type": "emotion", "valence": 0.5, "arousal": 0.1})
    assert "smug" in C.names(profile, "emotion")
    C.remove(profile, "smug")
    assert "smug" not in C.names(profile, "emotion")
    assert C.resolve(profile, "smug") is None
    # removal also sticks for a built-in emotion (past the EMOTIONS fallback)
    assert "curious" in C.names(profile, "emotion")
    C.remove(profile, "curious")
    assert "curious" not in C.names(profile, "emotion")
    assert C.resolve(profile, "curious") is None


def test_cases_add_rejects_bad_type(profile):
    with pytest.raises(ValueError):
        C.add(profile, {"name": "x", "type": "bogus"})


# ── load_dataset label mapping + tombstone exclusion ─────────────────────────
def test_load_dataset_labels_and_tombstones(profile):
    for i, lab in enumerate(["y", "k", "n"]):
        store.append(profile, {"id": f"r{i}", "case": "proud", "type": "emotion",
                               "label": lab, "ctx": [0.6, 0.4], "patch": _patch()})
    X, y, meta = R.load_dataset(profile)
    assert len(y) == 3
    assert list(y) == [1.0, 1.0, -1.0]                 # y,k -> +1 ; n -> -1
    store.tombstone(profile, "r2")
    _, y2, _ = R.load_dataset(profile)
    assert len(y2) == 2 and all(v > 0 for v in y2)


# ── vocalization synthesis (real bursts, not phrases) ────────────────────────
VOC_ARCHES = ["laugh", "sigh", "hmm", "chirp", "gasp", "growl", "beep", "buzz"]


def test_vocalization_archetypes_distinct():
    sigs = {}
    for name in VOC_ARCHES:
        p = de.generate_vocalization("qd", name, 0.5, 0.4, seed=0)
        first = p["notes"][0]
        rising = first.get("f1", first["f0"]) > first["f0"]
        sigs[name] = (len(p["notes"]), rising, round(first["dur"], 1))
    assert sigs["laugh"][0] >= 3          # ha-ha-ha: many pulses
    assert sigs["hmm"][0] == 1            # one hum
    assert sigs["chirp"][1] and sigs["gasp"][1]   # rising bursts
    assert len(set(sigs.values())) >= 4   # archetypes are genuinely different


def test_voc_archetype_resolution():
    assert de.voc_archetype("chirp-greet", ["greet"]) == "chirp"
    assert de.voc_archetype("giggle", []) == "laugh"
    assert de.voc_archetype("yes", []) == "beep"          # affirmative -> beep
    assert de.voc_archetype("no", []) == "buzz"           # negative -> buzz
    assert de.voc_archetype("mystery", []) is None        # -> (v,a) fallback


def test_rename_migrates_votes(profile):
    from dvoice import sessions
    C.add(profile, {"name": "old", "type": "vocalization", "valence": 0.0,
                    "arousal": 0.2, "tags": ["ack"]})
    for lab in ("y", "n", "y"):
        sessions.log_vote(profile, C.resolve(profile, "old"), _patch(), lab, rnd=0, session="t")
    assert store.counts(profile).get("old", (0,))[0] == 3
    migrated = C.rename(profile, "old", "new")
    assert migrated == 3
    assert "old" not in C.names(profile, "vocalization") and C.resolve(profile, "old") is None
    assert "new" in C.names(profile, "vocalization")
    assert store.counts(profile).get("new", (0,))[0] == 3


def test_generate_vocalization_in_bounds():
    for name in VOC_ARCHES:
        for seed in range(3):
            p = de.generate_vocalization("qd", name, 0.3, 0.2, seed=seed)
            for note in p["notes"]:
                for kk, (lo, hi, _k) in de.NOTE_BOUNDS.items():
                    if kk in note:
                        assert lo - 1e-6 <= note[kk] <= hi + 1e-6, f"{name}.{kk} OOB"


def test_diverse_candidates_vocalization_path():
    cands = R.diverse_candidates("qd", 0.6, 0.5, "", k=8,
                                 kind="vocalization", name="laugh", tags=["delight"])
    assert len(cands) == 8
    assert len(cands[0]["notes"]) >= 3            # canonical laugh = multi-pulse burst


# ── candidate generation: structural diversity + analytic baseline ───────────
def test_diverse_candidates_are_structurally_varied():
    cands = R.diverse_candidates("qd", 0.65, 0.4, "", k=16, seed=0)
    sigs = {tuple(n.get("g") for n in c["notes"]) for c in cands}
    assert len(sigs) >= 3                              # not 16 copies of one contour


def test_arrange_by_judge_orders_best_first():
    X, y = _separable_set(n=40)
    j = R.fit(X, y)
    cands = R.arrange_by_judge(j, "qd", 0.6, 0.5, "", k=6, seed=0)
    assert len(cands) == 6
    s = [R.score(j, R.featurize(c, 0.6, 0.5)) for c in cands]
    assert all(s[i] >= s[i + 1] - 1e-6 for i in range(len(s) - 1)), "must be best-first"
    assert len(R.arrange_by_judge(None, "qd", 0.6, 0.5, "", k=6, seed=0)) == 6   # None -> plain


def test_candidates_vary_timbre():
    cands = R.diverse_candidates("qd", 0.65, 0.4, "", k=12, seed=1)
    lps = {round(c["lp_cutoff"]) for c in cands}
    warbles = {round(c["notes"][0].get("fm_depth", 0.0), 1) for c in cands}
    rings = sum(any(n.get("ring_depth", 0) > 0 for n in c["notes"]) for c in cands)
    grits = sum(any(n.get("sh_rate", 0) > 0 for n in c["notes"]) for c in cands)
    assert len(lps) >= 6, "brightness should vary"
    assert len(warbles) >= 5, "warble should vary"
    assert rings >= 1 and grits >= 1, "some candidates metallic / gritty"


def test_best_of_n_baseline_is_analytic():
    X, y = _separable_set()
    j = R.fit(X, y)
    res = R.best_of_n(j, "qd", 0.65, 0.4, "", k=8, seed=0)
    base = de.generate_arrangement("qd", 0.65, 0.4, "")
    assert res["candidates"][0]["notes"][0]["f0"] == base["notes"][0]["f0"]
    assert 0 <= res["best_i"] < 8


# ── stochastic selection (variety without losing quality) ────────────────────
def test_select_index_temperature_zero_is_argmax():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    phis = np.eye(4)
    assert R._select_index(scores, phis, temperature=0.0) == 3


def test_select_index_samples_with_variety_but_prefers_best():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    phis = np.eye(4)
    rng = np.random.default_rng(0)
    draws = [R._select_index(scores, phis, temperature=1.0, rng=rng) for _ in range(200)]
    assert len(set(draws)) >= 2                          # not deterministic
    assert max(set(draws), key=draws.count) == 3         # still favors the best


def test_select_index_top_caps_to_best_n():
    scores = np.array([5.0, 0.1, 0.1, 0.1])              # only index 0 is good
    phis = np.eye(4)
    rng = np.random.default_rng(0)
    draws = {R._select_index(scores, phis, temperature=2.0, top=1, rng=rng) for _ in range(50)}
    assert draws == {0}                                  # top=1 forces the single best


def test_select_index_novelty_avoids_recent():
    scores = np.array([3.0, 2.9])                        # candidate 0 wins on score...
    phis = np.array([[1.0, 0.0], [0.0, 1.0]])
    recent = np.array([[1.0, 0.0]])                      # ...but it repeats a recent play
    assert R._select_index(scores, phis, temperature=0.0, novelty=1.0, recent=recent) == 1


def test_select_index_ignores_stale_recent():
    """A recent-ring of the wrong width (e.g. left over after a FEATURE_VERSION bump) must
    be ignored, not crash the novelty matmul."""
    scores = np.array([0.0, 1.0, 2.0])
    phis = np.eye(3)
    stale = np.zeros((2, 5))                             # width 5 != phis width 3
    assert R._select_index(scores, phis, temperature=0.0, novelty=1.0, recent=stale) == 2


def test_best_of_n_default_is_deterministic_argmax():
    X, y = _separable_set()
    j = R.fit(X, y)
    a = R.best_of_n(j, "qd", 0.65, 0.4, "", k=8, seed=0)        # temp=0 default
    b = R.best_of_n(j, "qd", 0.65, 0.4, "", k=8, seed=0)
    assert a["best_i"] == b["best_i"] == a["argmax_i"]            # reproducible (cues/tests)


def test_best_of_n_temperature_varies_selection():
    X, y = _separable_set()
    j = R.fit(X, y)
    picks = {R.best_of_n(j, "qd", 0.65, 0.4, "", k=10, seed=s, temperature=0.9, top=0)["best_i"]
             for s in range(12)}
    assert len(picks) >= 2                               # sampling → different picks across seeds


# ── sessions.log_vote writes a well-formed row ───────────────────────────────
def test_sessions_log_vote(profile):
    from dvoice import sessions
    case = {"name": "proud", "type": "emotion", "valence": 0.65, "arousal": 0.4}
    sessions.log_vote(profile, case, _patch(), "y", rnd=0, session="t")
    rows = store.load(profile)
    assert len(rows) == 1 and rows[0]["case"] == "proud" and rows[0]["label"] == "y"
    assert rows[0]["ctx"] == [0.65, 0.4] and rows[0]["id"]


# ── datasets: create / switch / isolation ─────────────────────────────────────
def test_dataset_create_switch_isolation(profile):
    from dvoice import sessions
    case = {"name": "proud", "type": "emotion", "valence": 0.6, "arousal": 0.4}
    sessions.log_vote(profile, case, _patch(), "y", rnd=0, session="t")     # -> main
    store.create_dataset(profile, "exp", copy_from=None)                    # empty, now active
    assert store.active_dataset(profile) == "exp"
    assert len(store.load(profile)) == 0                                    # exp is empty
    sessions.log_vote(profile, case, _patch(), "n", rnd=0, session="t")     # -> exp
    assert len(store.load(profile)) == 1
    store.set_active_dataset(profile, "main")
    assert len(store.load(profile)) == 1                                    # main untouched
    names = [n for n, _ in store.list_datasets(profile)]
    assert "main" in names and "exp" in names


# ── model versions: save / activate / rollback / delete ──────────────────────
def test_version_save_activate_rollback(profile):
    from dvoice import versions
    X, y = _separable_set()
    j = R.fit(X, y)
    v1 = versions.save_version(profile, j, auc=0.9, n_votes=len(y), dataset="main", gate=True, ts="t1")
    v2 = versions.save_version(profile, j, auc=0.8, n_votes=len(y), dataset="main", gate=True, ts="t2")
    assert (v1, v2) == (1, 2)
    assert versions.active_version(profile) == 2
    versions.set_active_version(profile, 1)
    assert versions.active_version(profile) == 1
    assert versions.load_active_judge(profile) is not None
    assert len(versions.list_versions(profile)) == 2
    versions.delete_version(profile, 1)
    assert versions.active_version(profile) == 2          # rolled forward to a survivor
    assert len(versions.list_versions(profile)) == 1
