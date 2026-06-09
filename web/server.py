#!/usr/bin/env python3
"""droid-voice web server — a stdlib-only local UI over the droid-voice engine.

Serves a single-page frontend (web/index.html + app.js + style.css) and a JSON API
that orchestrates the EXISTING engine (dvoice.{cases,emotion,reward,versions,store,
sessions} + droid_synth). No Flask/Django/new deps — pure python3 http.server, so it
stays IP-clean and dependency-free.

DEMO-ONLY: the UI lets you LISTEN — pick/type a case (and optionally a phrase whose
words set the affect), hear a freshly-generated, never-repeating reaction. READ-ONLY
and safe on the live profile (qd); it never writes. Teaching the voice
(collect/vote/train/activate) happens in the `droid` CLI, not here.

Run:
    python3 web/server.py [--host 127.0.0.1] [--port 8765] [--profile untrained-johndoe]

Audio never touches disk: each rendered patch is encoded to WAV bytes in-memory and
parked in a bounded cache keyed by a token; the browser fetches /api/wav?token=...
The (large/nested) patch dicts also stay server-side, keyed by patch_id, so a vote
references an id rather than round-tripping the whole patch through the browser.

Request handling is split into pure handle_*(body|params)->dict helpers so the
pytest can drive the API without opening a socket.
"""
import argparse
import io
import json
import os
import re
import sys
import uuid
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ── service root on sys.path + dependency-light engine BEFORE importing dvoice ──
HERE = os.path.dirname(os.path.abspath(__file__))          # .../droid-sounds/web
SERVICE_ROOT = os.path.dirname(HERE)                       # .../droid-sounds
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)
os.environ.setdefault("DROID_NO_EMBED", "1")               # disable the embedding path

import numpy as np                                          # noqa: E402
from dvoice import cases as C        # noqa: E402
from dvoice import emotion as de     # noqa: E402
from dvoice import reward as R       # noqa: E402
from dvoice import versions          # noqa: E402
from dvoice import store             # noqa: E402
from dvoice import sessions          # noqa: E402
from dvoice import phrasebook as pb  # noqa: E402
from dvoice import lexicon as lex    # noqa: E402
from droid_synth import render_patch, SR   # noqa: E402

# ── policy constants ──────────────────────────────────────────────────────────
PROTECTED = ("qd",)                       # live profile — read-only here; never written
DEFAULT_PROFILE = "untrained-johndoe"     # virtual demo default — no judge → analytic floor
# per-case-type default sampling temperature (copied from droid-say TYPE_TEMP)
TYPE_TEMP = {"emotion": 0.8, "expression": 0.55, "vocalization": 0.45, "adhoc": 0.7}
CACHE_CAP = 256                            # bounded in-process caches
GAIN = 0.18

# ── bounded in-process caches (insertion-order eviction) ──────────────────────
WAV_CACHE = OrderedDict()                  # token(hex) -> wav bytes
PATCH_CACHE = OrderedDict()                # patch_id(hex) -> patch dict


def _cache_put(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > CACHE_CAP:
        cache.popitem(last=False)          # evict oldest


def _cache_get(cache, key):
    v = cache.get(key)
    if v is not None:
        cache.move_to_end(key)
    return v


# ── helpers ───────────────────────────────────────────────────────────────────
class ApiError(Exception):
    """A clean, status-coded API error (never a 500)."""
    def __init__(self, status, payload):
        super().__init__(str(payload))
        self.status = status
        self.payload = payload if isinstance(payload, dict) else {"error": str(payload)}


def _jsafe(x):
    """A NaN/Inf-safe float (-> None) for JSON. The judge AUC can be NaN."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _wav_bytes(patch, gain=GAIN):
    """Render a patch to 16-bit PCM WAV bytes in-memory (no temp file).

    render_patch returns a float64 numpy array in [-1,1] at SR=44100 mono; we clip,
    scale to int16, and let scipy.io.wavfile.write encode into a BytesIO (the same 3
    lines as synth.write_wav, but to a buffer instead of disk)."""
    from scipy.io import wavfile
    audio = render_patch(patch, gain)
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    buf = io.BytesIO()
    wavfile.write(buf, SR, pcm)
    return buf.getvalue()


def _stash_wav(patch, gain=GAIN):
    """Render + cache a patch's WAV; return its token."""
    token = uuid.uuid4().hex
    _cache_put(WAV_CACHE, token, _wav_bytes(patch, gain))
    return token


def _stash_patch(patch):
    """Cache a raw patch dict server-side; return its id."""
    pid = uuid.uuid4().hex
    _cache_put(PATCH_CACHE, pid, patch)
    return pid


def _resolve_case(profile, name, valence=None, arousal=None, text=""):
    """Resolve a case to a dict carrying name/type/valence/arousal/text/tags.

    Order: an explicit (valence,arousal) override -> ad-hoc; else C.resolve; else
    a bare 'v,a' ad-hoc parse (mirrors droid-say). Raises ApiError(400) on failure."""
    if valence is not None and arousal is not None:
        return {"name": name or f"{valence},{arousal}", "type": "adhoc",
                "valence": float(valence), "arousal": float(arousal), "text": text or "",
                "tags": []}
    case = C.resolve(profile, name) if name else None
    if case is None and name and "," in name:
        try:
            v, a = (float(x) for x in name.split(",", 1))
        except ValueError:
            raise ApiError(400, {"error": "bad_adhoc", "detail": f"cannot parse '{name}' as 'v,a'"})
        return {"name": name, "type": "adhoc", "valence": v, "arousal": a,
                "text": text or "", "tags": []}
    if case is None:
        raise ApiError(400, {"error": "unknown_case", "case": name})
    out = dict(case)
    out.setdefault("text", "")
    out.setdefault("tags", out.get("tags", []))
    return out


_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _safe_profile(p):
    """Validate a profile name BEFORE it touches the filesystem. Rejects path traversal
    ('../x', 'a/b'), case-variants of protected names ('QD' on a case-insensitive FS),
    and any non-token — closing the protected-guard bypass. Lowercase-only by design, so
    the exact-match PROTECTED check below can't be evaded by case folding."""
    if not isinstance(p, str) or not _PROFILE_RE.match(p):
        raise ApiError(400, {"error": "bad_profile", "profile": str(p)[:40]})
    return p






# ── API handlers (pure: take dict/params, return dict; raise ApiError on bad) ──
def _profile_label(p):
    """Demo display name for a profile key. The trained voice shows its LIVE judge
    version (Q-D6); the virtual default is the untrained floor."""
    if p == DEFAULT_PROFILE:
        return "Untrained-JohnDoe"
    v = versions.active_version(p)
    return f"Q-D{v}" if v is not None else "Q-D (untrained)"


def handle_profiles():
    """List the demo-selectable profiles + their display labels. PROFILES today
    holds only the trained voice (qd); the untrained default is virtual."""
    profiles = set()
    try:
        for name in os.listdir(de.PROFILES):
            if os.path.isdir(os.path.join(de.PROFILES, name)) and not name.startswith("."):
                profiles.add(name)
    except OSError:
        pass
    profiles.add(DEFAULT_PROFILE)
    # protected (trained) first, then the untrained default, then the rest
    ordered = [p for p in PROTECTED if p in profiles]
    if DEFAULT_PROFILE in profiles and DEFAULT_PROFILE not in ordered:
        ordered.append(DEFAULT_PROFILE)
    ordered += sorted(profiles - set(ordered))
    labels = {p: _profile_label(p) for p in ordered}
    return {"profiles": ordered, "default": DEFAULT_PROFILE,
            "protected": list(PROTECTED), "labels": labels}


def handle_cases(params):
    profile = _safe_profile((params.get("profile") or [DEFAULT_PROFILE])[0])
    kind = (params.get("kind") or [None])[0] or None
    counts = store.counts(profile)
    out = []
    for name in C.names(profile, kind):
        case = C.resolve(profile, name)
        if case is None:
            continue
        n_tot, n_keep, n_drop = counts.get(name, (0, 0, 0))
        out.append({
            "name": name,
            "type": case.get("type", "emotion"),
            "valence": _jsafe(case.get("valence")),
            "arousal": _jsafe(case.get("arousal")),
            "tags": case.get("tags", []),
            "example": pb.example(name),     # a short phrase the tour can render + show
            "votes": {"total": int(n_tot), "keep": int(n_keep), "drop": int(n_drop)},
        })
    return {"profile": profile, "cases": out,
            "protected": profile in PROTECTED,
            "writable": profile not in PROTECTED}


def handle_state(params):
    profile = _safe_profile((params.get("profile") or [DEFAULT_PROFILE])[0])
    judge = versions.load_active_judge(profile)
    vote_total = sum(t[0] for t in store.counts(profile).values())
    return {
        "profile": profile,
        "active_dataset": store.active_dataset(profile),
        "active_version": versions.active_version(profile),
        "has_judge": judge is not None,
        "datasets": [list(d) for d in store.list_datasets(profile)],
        "vote_total": int(vote_total),
        "protected": profile in PROTECTED,
    }


def handle_demo(body):
    """READ-ONLY. Generate one fresh reaction (Best-of-N if a judge exists, else the
    analytic floor) and stash its WAV. seed=None => never the same twice."""
    profile = _safe_profile(body.get("profile") or DEFAULT_PROFILE)
    case = _resolve_case(profile, body.get("case"),
                         valence=body.get("valence"), arousal=body.get("arousal"),
                         text=body.get("text", ""))
    v, a = case["valence"], case["arousal"]
    kind, name, tags = case.get("type"), case.get("name"), tuple(case.get("tags", ()))
    text = body.get("text") or case.get("text", "")

    # Make typed WORDS shape the affect (the synth is (v,a)-driven). Only when the React
    # path asks (map_affect) for FREE text — never for an explicit [tag], the tour, or an
    # ad-hoc (v,a) override. An interjection maps to a case ("ha"->laugh) when that case
    # exists on the profile; otherwise content words set (v,a). No hit -> the case stands.
    if body.get("map_affect") and text and body.get("valence") is None:
        aff = lex.text_to_affect(text)
        mapped = C.resolve(profile, aff["case"]) if aff.get("case") else None
        if mapped is not None:
            case = mapped
            v, a = case["valence"], case["arousal"]
            kind, name, tags = case.get("type"), case.get("name"), tuple(case.get("tags", ()))
        elif aff["matched"]:
            v, a = aff["valence"], aff["arousal"]

    temp = body.get("temperature")
    if temp is None:
        temp = TYPE_TEMP.get(kind, 0.7)
    k = int(body.get("k") or 12)
    top = int(body.get("top") or 6)
    novelty = body.get("novelty")
    novelty = 0.5 if novelty is None else float(novelty)

    judge = versions.load_active_judge(profile)
    version = versions.active_version(profile)
    if judge is not None:
        res = R.best_of_n(judge, profile, v, a, text, k=k, seed=None,
                          kind=kind, name=name, tags=tags,
                          temperature=float(temp), top=top, novelty=novelty, recent=None)
        patch = res["patch"]
        source = f"best-of-{k} (judge v{version}, T={float(temp):.2g})"
    else:
        if kind == "vocalization":
            patch = de.generate_vocalization(profile, name, v, a, tags=tags)
        else:
            patch = de.generate_arrangement(profile, v, a, text)
        source = "analytic floor (no judge)"
        version = None

    token = _stash_wav(patch)
    return {
        "case": name,
        "type": kind,
        "valence": _jsafe(v),
        "arousal": _jsafe(a),
        "transcript": de.transcript(patch),
        "source": source,
        "version": version,
        "wav_token": token,
    }


def handle_wav(params):
    """Return (wav_bytes,) for a token; raise ApiError(404) if evicted/missing.
    Returned to the HTTP layer which sets audio/wav + Content-Length."""
    token = (params.get("token") or [None])[0]
    data = _cache_get(WAV_CACHE, token) if token else None
    if data is None:
        raise ApiError(404, {"error": "wav_expired", "detail": "react again"})
    return data












# ── HTTP layer ──────────────────────────────────────────────────────────────
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}

# GET endpoints take parsed query params (dict[str,list]); POST take a parsed body dict.
# DEMO-ONLY server: only read routes + the read-only /api/demo. Training/voting happen
# in the CLI (droid collect / train / models); the old write routes were stripped.
GET_API = {
    "/api/profiles": lambda p: handle_profiles(),
    "/api/cases": handle_cases,
    "/api/state": handle_state,
}
POST_API = {
    "/api/demo": handle_demo,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "droid-voice-web/1.0"

    def log_message(self, fmt, *args):       # quieter logs
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- response helpers --
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, fname, content_type):
        path = os.path.join(HERE, fname)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            self._send_json({"error": "not_found", "file": fname}, 404)
            return
        self._send_bytes(data, content_type)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        if length > 4_000_000:                                # cap body — no unbounded read
            raise ApiError(413, {"error": "too_large"})
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, {"error": "bad_json"})
        if not isinstance(obj, dict):
            raise ApiError(400, {"error": "bad_json", "detail": "expected an object"})
        return obj

    # -- routing --
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in STATIC:
                fname, ctype = STATIC[route]
                self._send_static(fname, ctype)
                return
            if route == "/api/wav":
                data = handle_wav(parse_qs(parsed.query))
                self._send_bytes(data, "audio/wav")
                return
            if route in GET_API:
                self._send_json(GET_API[route](parse_qs(parsed.query)))
                return
            self._send_json({"error": "not_found", "path": route}, 404)
        except ApiError as e:
            self._send_json(e.payload, e.status)
        except BrokenPipeError:
            pass
        except Exception as e:           # NEVER 500 — clean JSON error
            self._send_json({"error": "internal", "detail": str(e)}, 400)

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route not in POST_API:
                self._send_json({"error": "not_found", "path": route}, 404)
                return
            body = self._read_body()
            self._send_json(POST_API[route](body))
        except ApiError as e:
            self._send_json(e.payload, e.status)
        except BrokenPipeError:
            pass
        except Exception as e:           # NEVER 500 — clean JSON error
            self._send_json({"error": "internal", "detail": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description="droid-voice local web UI server")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (local-only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--profile", default=DEFAULT_PROFILE,
                    help="informational default profile for the UI (the UI selects per-request)")
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"droid-voice web UI  ->  {url}")
    print(f"  service root : {SERVICE_ROOT}")
    print(f"  profiles dir : {de.PROFILES}")
    print(f"  default      : {args.profile}  (protected: {', '.join(PROTECTED)})")
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
