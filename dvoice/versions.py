"""versions — model (judge) version management for the droid-voice studio.

Each train saves a NEW judge version: profiles/<p>/judges/v{N}.npz, recorded in
judges/index.json with its metadata (AUC, vote count, dataset, gate pass, timestamp).
The ACTIVE version (used by `say`) is a pointer in studio.json, so you can roll back
to an earlier judge without retraining. Deleting moves the file to judges/.trash/
(reversible — `trash > rm`).
"""
import json
import os

from dvoice import reward as R
from dvoice import store


def judges_dir(profile):
    return os.path.join(de_profiles(profile), "judges")


def de_profiles(profile):
    from dvoice import emotion as de
    return os.path.join(de.PROFILES, profile)


def index_path(profile):
    return os.path.join(judges_dir(profile), "index.json")


def read_index(profile):
    p = index_path(profile)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return []


def write_index(profile, index):
    p = index_path(profile)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(index, open(tmp, "w"), indent=2)
    os.replace(tmp, p)


def version_path(profile, v):
    return os.path.join(judges_dir(profile), f"v{int(v)}.npz")


def next_version(profile):
    idx = read_index(profile)
    return (max((e["version"] for e in idx), default=0) + 1)


def save_version(profile, judge, *, auc, n_votes, dataset, gate, ts=None, activate=True):
    """Persist `judge` as the next version + record metadata. Returns the version int."""
    v = next_version(profile)
    os.makedirs(judges_dir(profile), exist_ok=True)
    R.save_judge(judge, version_path(profile, v))
    idx = read_index(profile)
    idx.append({"version": v, "ts": ts, "auc": round(float(auc), 4),
                "n_votes": int(n_votes), "dataset": dataset, "gate": bool(gate)})
    write_index(profile, idx)
    if activate:
        store.set_state(profile, active_judge=v)
    return v


def list_versions(profile):
    return sorted(read_index(profile), key=lambda e: e["version"])


def active_version(profile):
    return store.get_state(profile).get("active_judge")


def set_active_version(profile, v):
    if not os.path.exists(version_path(profile, v)):
        raise ValueError(f"no judge v{v}")
    return store.set_state(profile, active_judge=int(v))


def load_active_judge(profile):
    """The active version's judge (pure-numpy). Falls back to a legacy judge.npz, then
    None (-> caller uses the analytic floor)."""
    v = active_version(profile)
    if v is not None and os.path.exists(version_path(profile, v)):
        j = R.load_judge(version_path(profile, v))
        if j is not None:
            return j
    legacy = os.path.join(de_profiles(profile), "judge.npz")
    return R.load_judge(legacy)


def delete_version(profile, v):
    """Soft-delete: move the .npz to judges/.trash/ and drop it from the index."""
    v = int(v)
    src = version_path(profile, v)
    if os.path.exists(src):
        tdir = os.path.join(judges_dir(profile), ".trash")
        os.makedirs(tdir, exist_ok=True)
        os.replace(src, os.path.join(tdir, f"v{v}.npz"))
    write_index(profile, [e for e in read_index(profile) if e["version"] != v])
    if active_version(profile) == v:
        remaining = [e["version"] for e in read_index(profile)]
        store.set_state(profile, active_judge=(max(remaining) if remaining else None))
