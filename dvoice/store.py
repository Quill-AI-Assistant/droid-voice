"""store — the vote dataset + studio state (profiles/<p>/).

Append-only votes. A removal is a tombstone row (never destroy a line — `trash > rm`,
auditable, reversible).

Datasets are versionable: `main` is dataset.jsonl; named snapshots live in
datasets/<name>.jsonl. The ACTIVE dataset (what collect appends to and train reads)
is held in studio.json. This lets you run experiments on separate vote sets.

Row shapes:
  vote:       {"id","case","type","ctx":[v,a],"label":"y|n|k","patch":{...},
               "round","ts","session","gen"}
  tombstone:  {"tombstone_id": <vote id>, "ts": ...}

label: y=good, n=bad, k=keep/gold-anchor. (s=skip writes NO row.)
"""
import json
import os

from dvoice import emotion as de

VOTE_LABELS = ("y", "n", "k")


# ── studio state (active dataset + active judge version) ──────────────────────
def _profile_dir(profile):
    return os.path.join(de.PROFILES, profile)


def state_path(profile):
    return os.path.join(_profile_dir(profile), "studio.json")


def get_state(profile):
    p = state_path(profile)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {"active_dataset": "main", "active_judge": None}


def set_state(profile, **kw):
    st = get_state(profile)
    st.update(kw)
    p = state_path(profile)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(st, open(tmp, "w"), indent=2)
    os.replace(tmp, p)
    return st


# ── dataset files + versioning ────────────────────────────────────────────────
def active_dataset(profile):
    return get_state(profile).get("active_dataset", "main")


def dataset_file(profile, name=None):
    """Path of a dataset by name. 'main' -> dataset.jsonl (back-compat);
    any other name -> datasets/<name>.jsonl."""
    name = name or active_dataset(profile)
    if name == "main":
        return os.path.join(_profile_dir(profile), "dataset.jsonl")
    return os.path.join(_profile_dir(profile), "datasets", f"{name}.jsonl")


def path(profile):
    """The ACTIVE dataset file (what append/load operate on)."""
    return dataset_file(profile, active_dataset(profile))


def list_datasets(profile):
    """[(name, n_active_votes)] for main + every datasets/<name>.jsonl."""
    names = ["main"]
    ddir = os.path.join(_profile_dir(profile), "datasets")
    if os.path.isdir(ddir):
        names += sorted(f[:-6] for f in os.listdir(ddir) if f.endswith(".jsonl"))
    return [(name, len(load(profile, dataset=name))) for name in names]


def create_dataset(profile, name, copy_from=None, activate=True):
    """Create a named dataset (empty, or copying another's raw lines). Optionally make
    it active. Refuses to clobber an existing dataset."""
    if name == "main":
        raise ValueError("'main' is the default dataset; pick another name")
    dst = dataset_file(profile, name)
    if os.path.exists(dst):
        raise ValueError(f"dataset '{name}' already exists")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy_from is not None:
        src = dataset_file(profile, copy_from)
        lines = open(src).read() if os.path.exists(src) else ""
        open(dst, "w").write(lines)
    else:
        open(dst, "w").close()
    if activate:
        set_state(profile, active_dataset=name)
    return dst


def set_active_dataset(profile, name):
    if name != "main" and not os.path.exists(dataset_file(profile, name)):
        raise ValueError(f"no dataset '{name}'")
    return set_state(profile, active_dataset=name)


# ── append / load / tombstone (operate on a dataset by name; default = active) ─
def append(profile, row, dataset=None):
    p = dataset_file(profile, dataset)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _read_raw(profile, dataset=None):
    rows = []
    p = dataset_file(profile, dataset)
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def load(profile, case=None, dataset=None):
    """Active (non-tombstoned) vote rows of a dataset, optionally filtered to a case."""
    raw = _read_raw(profile, dataset)
    dead = {r["tombstone_id"] for r in raw if r.get("tombstone_id") is not None}
    out = [r for r in raw
           if r.get("tombstone_id") is None
           and r.get("id") not in dead
           and r.get("label") in VOTE_LABELS]
    if case is not None:
        out = [r for r in out if r.get("case") == case]
    return out


def tombstone(profile, vote_id, ts=None, dataset=None):
    return append(profile, {"tombstone_id": vote_id, "ts": ts}, dataset=dataset)


def relabel_case(profile, old, new):
    """Rename a case across ALL datasets: rewrite every row where case==old to
    case==new (preserves order + other fields). One-time data maintenance so a case
    rename keeps its votes. Returns the number of rows migrated."""
    migrated = 0
    for name, _ in list_datasets(profile):
        p = dataset_file(profile, name)
        if not os.path.exists(p):
            continue
        rows = _read_raw(profile, dataset=name)
        hit = False
        for r in rows:
            if r.get("case") == old:
                r["case"] = new
                hit = True
                migrated += 1
        if hit:
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, p)
    return migrated


def counts(profile, dataset=None):
    """Per-case (n_total, n_keep, n_drop) over the active set — for menus/status."""
    out = {}
    for r in load(profile, dataset=dataset):
        c = r.get("case")
        n_tot, n_keep, n_drop = out.get(c, (0, 0, 0))
        if r["label"] == "n":
            n_drop += 1
        else:
            n_keep += 1
        out[c] = (n_tot + 1, n_keep, n_drop)
    return out
