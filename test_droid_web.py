"""test_droid_web — the LOCAL web "vignette UI" server over the droid-voice engine.

Exercises web/server.py WITHOUT a browser: it spins the real ThreadingHTTPServer on
an ephemeral port in a background thread and drives it over loopback HTTP, and it also
calls the server's pure request-handling helpers directly (no socket) for the
fine-grained assertions. Hermetic: a fixture repoints de.PROFILES at a tmp dir, so the
shipped/live profiles/ (qd) is NEVER touched — mirrors the `profile`
fixture in test_droid_lifecycle.py.

    DROID_NO_EMBED=1 python3 -m pytest test_droid_web.py -q

The server imports the engine relative to the service root and reads DROID_NO_EMBED at
import time, so both are set BEFORE the first `from dvoice import …`.
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from http.server import ThreadingHTTPServer

import pytest

os.environ.setdefault("DROID_NO_EMBED", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dvoice import cases as C        # noqa: E402
from dvoice import emotion as de     # noqa: E402
from dvoice import reward as R       # noqa: E402
from dvoice import sessions          # noqa: E402
from dvoice import store             # noqa: E402
from dvoice import versions          # noqa: E402

# The server is the unit under test. If it isn't built yet, skip the whole module
# cleanly rather than erroring at collection time.
server = pytest.importorskip("web.server", reason="web/server.py not present yet")

# ─────────────────────────────────────────────────────────────────────────────
# Hermetic profile sandbox — never touch the real profiles/ dir.
# store / versions / cases all read de.PROFILES at call time (via the `de` ref),
# so a single monkeypatch on de.PROFILES isolates every write.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def profiles_root(tmp_path, monkeypatch):
    """Repoint de.PROFILES at a tmp dir. The UI's default 'sandbox' profile needs
    zero setup (built-in emotion fallback), so we don't pre-create it."""
    monkeypatch.setattr(de, "PROFILES", str(tmp_path))
    # Clear any in-process caches so tokens/patch_ids don't leak across tests.
    for name in ("WAV_CACHE", "PATCH_CACHE"):
        cache = getattr(server, name, None)
        if isinstance(cache, dict):
            cache.clear()
    return str(tmp_path)

# ── helper-call adapter ──────────────────────────────────────────────────────
# The server's request handling lives in pure helpers (handle_demo(body), …): each
# returns a bare dict on success and raises server.ApiError(status, payload) on a
# clean error. We normalize both into a (status, dict) tuple so the assertions read
# the same way for success and error paths (and a bad request never becomes a 500).
def _call(name, payload):
    fn = getattr(server, name, None)
    if fn is None:
        pytest.skip(f"web.server has no helper {name!r}")
    try:
        out = fn(payload) if payload is not None else fn()
    except server.ApiError as e:
        return e.status, e.payload
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], int):
        return out
    return 200, out

def _qs(**kw):
    """parse_qs-style params ({key:[value]}) — the shape the GET handlers expect."""
    return {k: [v] for k, v in kw.items() if v is not None}

def _cases(profile, kind=None):
    return _call("handle_cases", _qs(profile=profile, kind=kind))

# ─────────────────────────────────────────────────────────────────────────────
# Live-socket fixture — proves the wire path (routing + MIME + WAV bytes) works,
# not just the helpers.
# ─────────────────────────────────────────────────────────────────────────────
def _free_port():
    import socket
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

@pytest.fixture
def live_server(profiles_root):
    """Start the real ThreadingHTTPServer on an ephemeral loopback port."""
    handler = getattr(server, "Handler", None) or getattr(server, "DroidHandler", None)
    if handler is None:
        pytest.skip("web.server exposes no request Handler class")
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)

def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()

def _post(base, path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

# ── seeding helper: enough hermetic votes to clear the train gate ─────────────
def _seed_votes(profile, n=40):
    """Log n separable votes so R.fit_and_eval can fit a real, AUC-positive judge.
    High-valence/high-arousal candidates -> keep (y); low/low -> drop (n). The judge
    only has to RANK these apart, so the held-out AUC clears the 0.55 gate."""
    case_hi = {"name": "proud", "type": "emotion", "valence": 0.7, "arousal": 0.6}
    case_lo = {"name": "bored", "type": "emotion", "valence": -0.6, "arousal": -0.5}
    half = n // 2
    for i in range(half):
        hi = R.diverse_candidates(profile, 0.7, 0.6, "", k=4, seed=i)[i % 4]
        lo = R.diverse_candidates(profile, -0.6, -0.5, "", k=4, seed=100 + i)[i % 4]
        sessions.log_vote(profile, case_hi, hi, "y", rnd=0, session="seed")
        sessions.log_vote(profile, case_lo, lo, "n", rnd=0, session="seed")

# ═════════════════════════════════════════════════════════════════════════════
# 1. /api/cases returns the taxonomy
# ═════════════════════════════════════════════════════════════════════════════
def test_cases_returns_builtin_emotions(profiles_root):
    status, body = _cases("untrained-johndoe")
    assert status == 200
    assert body["profile"] == "untrained-johndoe"
    names = {c["name"] for c in body["cases"]}
    # 18 built-in emotions are the fallback for a never-seen profile.
    assert len(names) >= 18
    assert {"proud", "curious", "content"} <= names
    proud = next(c for c in body["cases"] if c["name"] == "proud")
    assert proud["type"] == "emotion"
    assert "valence" in proud and "arousal" in proud
    # per-case vote tally present and zero on a fresh sandbox
    assert proud["votes"]["total"] == 0
    # sandbox is writable + not protected
    assert body["writable"] is True and body["protected"] is False

def test_cases_kind_filter(profiles_root):
    status, body = _cases("untrained-johndoe", kind="emotion")
    assert status == 200
    assert body["cases"]
    assert all(c["type"] == "emotion" for c in body["cases"])

def test_cases_marks_protected_profile(profiles_root):
    # demoing/inspecting qd is read-only and must report protected=true
    os.makedirs(os.path.join(profiles_root, "qd"), exist_ok=True)
    status, body = _cases("qd")
    assert status == 200
    assert body["protected"] is True

# ═════════════════════════════════════════════════════════════════════════════
# 2. /api/demo renders valid WAV + variety (never the same twice)
# ═════════════════════════════════════════════════════════════════════════════
def test_demo_returns_wav_token_and_floor_source(profiles_root):
    status, body = _call("handle_demo", {"profile": "untrained-johndoe", "case": "proud"})
    assert status == 200
    assert body["case"] == "proud" and body["type"] == "emotion"
    assert body.get("transcript")            # glyph transcript present
    assert body["wav_token"]                 # token to fetch the audio
    # no judge on a fresh sandbox -> analytic floor, no version
    assert body.get("version") in (None,)
    assert "floor" in body.get("source", "").lower() or body.get("version") is None
    # the token resolves to real 16-bit PCM WAV bytes
    wav = server.WAV_CACHE[body["wav_token"]]
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert len(wav) > 1000

def test_demo_token_is_fresh_each_call(profiles_root):
    """Every demo stashes a fresh WAV under a new uuid token (so /api/wav can't collide
    and 'Again' always re-fetches)."""
    s1, b1 = _call("handle_demo", {"profile": "untrained-johndoe", "case": "proud"})
    s2, b2 = _call("handle_demo", {"profile": "untrained-johndoe", "case": "proud"})
    assert s1 == s2 == 200
    assert b1["wav_token"] != b2["wav_token"]
    assert b1["wav_token"] in server.WAV_CACHE and b2["wav_token"] in server.WAV_CACHE

def test_demo_with_judge_never_repeats(profiles_root):
    """The demo's whole point: with a judge active, best_of_n(seed=None) => never the
    same twice. (The judge-less analytic floor is intentionally deterministic, so a
    trained judge is what unlocks variety.)"""
    import datetime
    _seed_votes("untrained-johndoe", n=40)
    # Train + activate a judge via the ENGINE (the web server is demo-only — training
    # lives in the CLI). fit_and_eval -> save_version is exactly what `droid train` does.
    res = R.fit_and_eval("untrained-johndoe")
    assert res["judge"] is not None
    ver = versions.save_version("untrained-johndoe", res["judge"], auc=res["auc"], n_votes=res["n"],
                                dataset="main", gate=True,
                                ts=datetime.datetime.now().isoformat(timespec="seconds"))
    bodies = [_call("handle_demo", {"profile": "untrained-johndoe", "case": "proud"})[1]
              for _ in range(4)]
    audios = [server.WAV_CACHE[b["wav_token"]] for b in bodies]
    # at least two of the four reactions differ -> stochastic selection, not a loop
    assert len({a[:512] for a in audios}) >= 2
    # the source label advertises the judge + version
    assert "judge" in bodies[0]["source"].lower()
    assert bodies[0]["version"] == ver

def test_demo_adhoc_valence_arousal(profiles_root):
    """An ad-hoc 'v,a' case (no name in the taxonomy) is parsed by the server."""
    status, body = _call("handle_demo", {"profile": "untrained-johndoe", "case": "0.6,0.3"})
    assert status == 200
    assert abs(body["valence"] - 0.6) < 1e-6 and abs(body["arousal"] - 0.3) < 1e-6
    assert body["wav_token"] in server.WAV_CACHE

def test_demo_is_read_only_on_protected_profile(profiles_root):
    """Demo must be safe (zero writes) even on qd — no 409, nothing logged."""
    os.makedirs(os.path.join(profiles_root, "qd"), exist_ok=True)
    status, body = _call("handle_demo", {"profile": "qd", "case": "proud"})
    assert status == 200
    assert body["wav_token"] in server.WAV_CACHE
    assert sum(t for t, _k, _d in store.counts("qd").values()) == 0

def test_bad_profile_name_rejected_no_write(profiles_root):
    """A crafted profile name (case-variant on a case-insensitive FS, '..' traversal,
    trailing slash) must 400 bad_profile BEFORE touching the filesystem — closing the
    bypass where e.g. 'QD' or 'sandbox/../qd' reached a live dir. Exercised via the
    read-only /api/demo (the server is demo-only; every handler runs _safe_profile first)."""
    os.makedirs(os.path.join(profiles_root, "qd"), exist_ok=True)
    for bad in ("QD", "sandbox/../qd", "../escaped", "qd/"):
        status, body = _call("handle_demo", {"profile": bad, "case": "proud"})
        assert status == 400 and body.get("error") == "bad_profile", (bad, status, body)
    assert len(store.load("qd")) == 0                                  # nothing written
    assert not os.path.exists(os.path.join(os.path.dirname(profiles_root), "escaped"))

# ═════════════════════════════════════════════════════════════════════════════
# 8. live wire path — static page, JSON routing, WAV MIME, clean errors
# ═════════════════════════════════════════════════════════════════════════════
def test_live_static_index(live_server):
    status, ctype, body = _get(live_server, "/")
    assert status == 200
    assert "text/html" in ctype
    assert b"<" in body and len(body) > 100      # an actual HTML page

def test_live_static_assets_mime(live_server):
    _, ctype_js, js = _get(live_server, "/app.js")
    assert "javascript" in ctype_js and js
    _, ctype_css, css = _get(live_server, "/style.css")
    assert "css" in ctype_css and css

def test_live_profiles_lists_sandbox_default(live_server):
    status, _ctype, body = _get(live_server, "/api/profiles")
    data = json.loads(body)
    assert status == 200
    assert "untrained-johndoe" in data["profiles"]
    assert data["default"] == "untrained-johndoe"
    assert set(data["protected"]) >= {"qd"}
    assert data["labels"]["untrained-johndoe"] == "Untrained-JohnDoe"

def test_live_demo_then_wav_bytes(live_server):
    status, body = _post(live_server, "/api/demo", {"profile": "untrained-johndoe", "case": "proud"})
    assert status == 200
    token = body["wav_token"]
    s, ctype, wav = _get(live_server, "/api/wav?" + urllib.parse.urlencode({"token": token}))
    assert s == 200
    assert "audio/wav" in ctype
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"

def test_live_wav_missing_token_404(live_server):
    s, _ctype, _ = _get_status(live_server, "/api/wav?token=nope")
    assert s == 404

def test_live_state_endpoint(live_server):
    status, _ctype, body = _get(live_server, "/api/state?profile=untrained-johndoe")
    data = json.loads(body)
    assert status == 200
    assert data["profile"] == "untrained-johndoe"
    assert data["has_judge"] is False           # fresh untrained floor -> no judge
    assert data["active_version"] in (None,)

def test_live_unknown_route_is_clean_404(live_server):
    s, _ctype, _ = _get_status(live_server, "/api/does-not-exist")
    assert s == 404

def _get_status(base, path):
    """GET that tolerates 4xx/5xx without raising (urllib raises on >=400)."""
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()

# ═════════════════════════════════════════════════════════════════════════════
# words -> affect: interjections map to a case, content words set (v,a)
# ═════════════════════════════════════════════════════════════════════════════
def test_lexicon_maps_interjections_and_affect():
    from dvoice import lexicon as lex
    assert lex.text_to_affect("ha ha ha")["case"] == "laugh"
    assert lex.text_to_affect("lol!")["case"] == "laugh"
    assert lex.text_to_affect("ugh")["case"] == "frustrated"
    pos = lex.text_to_affect("this is wonderful")
    assert pos["matched"] and pos["valence"] > 0.4
    neg = lex.text_to_affect("everything is terrible and broken")
    assert neg["valence"] < -0.2
    assert lex.text_to_affect("buy some milk")["matched"] is False   # no affect words -> graceful

def test_demo_map_affect_shifts_va_from_text(profiles_root):
    # With map_affect, negative words pull valence below the bare case's (+0.4 curious).
    neg = _call("handle_demo", {"profile": "untrained-johndoe", "case": "curious",
                                "text": "everything is terrible and broken", "map_affect": True})
    assert neg[0] == 200 and neg[1]["valence"] is not None and neg[1]["valence"] < 0
    # Without the flag, free text is inert -> curious keeps its own positive valence.
    plain = _call("handle_demo", {"profile": "untrained-johndoe", "case": "curious",
                                  "text": "everything is terrible and broken"})
    assert plain[1]["valence"] > 0
