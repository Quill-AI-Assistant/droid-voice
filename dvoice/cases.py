"""cases — the droid-voice TAXONOMY registry (profiles/<p>/cases.json).

A *case* is any utterance you can select, collect votes on, and render. Three
composable types (affective-computing clean split):

  emotion       how it FEELS   — a point on the valence/arousal circumplex
  expression    what it DOES   — a communicative status (done/thinking/error...),
                                 rendered with a base emotion's colour
  vocalization  a non-verbal AFFECT BURST (laugh/sigh/hmm/chirp), optional v,a tint

The registry is data, not code: adding a new vocalization is appending ONE row —
no source edit. cases.json is JSONL (append-only friendly; later rows override
earlier by name, so edits win). Built-in EMOTIONS are a fallback when a name has no
explicit row, so the tool works before any migration.
"""
import json
import os

from dvoice import emotion as de

TYPES = ("emotion", "expression", "vocalization")


def path(profile):
    return os.path.join(de.PROFILES, profile, "cases.json")


def load(profile):
    """name -> case dict. Later rows override earlier (edits win). Tolerant of junk."""
    out = {}
    p = path(profile)
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            if c.get("name"):
                out[c["name"]] = c
    return out


def va(case, cases=None):
    """(valence, arousal) for a case. An expression resolves its `base` emotion;
    explicit valence/arousal on the case win. Defaults to neutral-ish."""
    if case.get("valence") is not None and case.get("arousal") is not None:
        return float(case["valence"]), float(case["arousal"])
    base = case.get("base")
    if base and base in de.EMOTIONS:
        v, a = de.EMOTIONS[base]
        return float(v), float(a)
    if cases and base in cases and cases[base] is not case:
        return va(cases[base], cases)
    return 0.0, 0.2


def resolve(profile, name):
    """A fully-resolved case dict (with concrete valence/arousal/text), or None.
    Falls back to a built-in EMOTION so the tool works pre-migration. A case marked
    removed resolves to None (it's out of the workflow)."""
    cases = load(profile)
    if name in cases:
        c = cases[name]
        if c.get("removed"):
            return None
        v, a = va(c, cases)
        return {"type": "emotion", "text": "", **c, "valence": v, "arousal": a}
    if name in de.EMOTIONS:
        v, a = de.EMOTIONS[name]
        return {"name": name, "type": "emotion", "text": "",
                "valence": float(v), "arousal": float(a)}
    return None


def names(profile, kind=None):
    """Sorted case names, optionally filtered by type. Unions built-in emotions but
    EXCLUDES any case marked removed (so 'remove' sticks past the built-in fallback)."""
    cases = load(profile)
    removed = {n for n, c in cases.items() if c.get("removed")}
    out = {n for n in cases if not cases[n].get("removed")}
    if kind in (None, "emotion"):
        out |= (set(de.EMOTIONS) - removed)
    if kind:
        out = {n for n in out if (cases.get(n, {}).get("type",
               "emotion" if n in de.EMOTIONS else None)) == kind}
    return sorted(out)


def rename(profile, old, new, **overrides):
    """Rename a case: create `new` from old's attributes (+overrides), MIGRATE its
    votes (across all datasets), then soft-remove `old`. Returns votes migrated."""
    from dvoice import store
    cur = resolve(profile, old) or {}
    case = {k: cur[k] for k in ("type", "valence", "arousal", "base", "function", "tags")
            if k in cur}
    case.setdefault("type", "emotion")
    case["name"] = new
    case.update(overrides)
    add(profile, case)
    migrated = store.relabel_case(profile, old, new)
    remove(profile, old)
    return migrated


def remove(profile, name):
    """Soft-remove a case (append a tombstone row; last-wins so it survives + is
    reversible by re-adding). Removed cases drop out of names()/resolve()."""
    cur = load(profile).get(name, {})
    typ = cur.get("type") or ("emotion" if name in de.EMOTIONS else "emotion")
    return add(profile, {"name": name, "type": typ, "removed": True})


def add(profile, case):
    """Append a case row (JSONL). Validates type + name. Returns the written case."""
    name = (case.get("name") or "").strip()
    typ = case.get("type")
    if not name:
        raise ValueError("case needs a name")
    if typ not in TYPES:
        raise ValueError(f"type must be one of {TYPES}, got {typ!r}")
    p = path(profile)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(case) + "\n")
    return case
