"""sessions — the shared interactive loops (collect + review).

ONE implementation of each playback-and-vote loop, used by both the thin CLI verbs
(droid-collect / droid-review) and the integrated studio (droid). Each loop owns its
own cbreak Console + temp render dir. Audio is the FROZEN synth.
"""
import os
import sys
import tempfile
import uuid
from datetime import datetime

from dvoice import colors
from dvoice import emotion as de
from dvoice import reward as R
from dvoice import store
from dvoice import versions
from dvoice import synth
from dvoice import term

COLLECT_HELP = "  →/y good · ←/n bad · s skip · r replay · . next batch · q quit"
REVIEW_HELP = "  x remove · f flip label · enter keep · r replay · q quit"
FLIP = {"y": "n", "n": "y"}
# Arrow control on collection: → good, ← bad (alias the y/n votes). ↓ also skips.
ARROW_VOTE = {"RIGHT": "y", "LEFT": "n", "DOWN": "s"}


def _now():
    return datetime.now().isoformat(timespec="seconds")


def log_vote(profile, case, patch, label, *, rnd, session, gen="collect"):
    """Append one vote row to the ACTIVE dataset. Pure (no audio)."""
    return store.append(profile, {
        "id": uuid.uuid4().hex[:12],
        "case": case["name"], "type": case.get("type", "emotion"),
        "ctx": [float(case["valence"]), float(case["arousal"])],
        "label": label, "patch": patch, "round": rnd,
        "ts": _now(), "session": session, "gen": gen,
    })


def _render(td, name, patch, gain):
    wav = os.path.join(td, name)
    synth.write_wav(wav, synth.render_patch(patch, gain))
    return wav


def collect(profile, case, *, k=12, sigma=0.18, gain=0.18, no_play=False):
    """Interactive collection loop for one case. Returns the per-label tally dict.
    Opens its own Console; refreshes diverse batches until the user quits."""
    v, a, text = case["valence"], case["arousal"], case.get("text", "")
    session = uuid.uuid4().hex
    tally = {"y": 0, "n": 0, "s": 0}
    have = store.counts(profile).get(case["name"], (0,))[0]
    judge = versions.load_active_judge(profile)        # arrange batches best-first if trained
    arr = "  (arranged by judge)" if judge is not None else ""
    print(f"\ncollect '{case['name']}' [{case.get('type')}] (v={v:+.2f} a={a:+.2f})  "
          f"{have} votes so far{arr}")
    print(COLLECT_HELP)
    rnd = 0
    with term.Console() as con, tempfile.TemporaryDirectory() as td:
        while True:
            cands = R.arrange_by_judge(judge, profile, v, a, text, k=k, sigma=sigma, seed=rnd,
                                       kind=case.get("type"), name=case["name"],
                                       tags=case.get("tags", ()))
            for i, patch in enumerate(cands):
                wav = _render(td, f"c{rnd}_{i}.wav", patch, gain)
                con.flush()
                # transient line during playback (overwritten by the result line below)
                sys.stdout.write(f"\r\033[K  [{rnd}.{i+1}/{len(cands)}]  {de.transcript(patch)}")
                sys.stdout.flush()
                while True:
                    key = term.play_and_wait(con, wav, no_play)
                    if key == "r":
                        continue
                    if key == "?":
                        print("\r\033[K" + COLLECT_HELP)
                        continue
                    break
                key = ARROW_VOTE.get(key, key)               # →/← -> y/n ; ↓ -> skip
                if key == "q":
                    print(f"\r\033[K  collected: {tally['y']} good {tally['n']} bad "
                          f"({tally['s']} skipped)")
                    return tally
                if key == ".":
                    break
                # Result line: FIXED-width columns first, variable-width transcript LAST,
                # so the result columns always align (the transcript no longer floats them).
                if key in ("y", "n"):
                    log_vote(profile, case, patch, key, rnd=rnd, session=session)
                    tally[key] += 1
                    cum = store.counts(profile).get(case["name"], (0,))[0]
                    mark = (colors.c(f"{'✓ good':<6}", "bgreen", "bold") if key == "y"
                            else colors.c(f"{'✗ bad':<6}", "bred", "bold"))
                    sess = f"{tally['y']}↑/{tally['n']}↓"
                    print(f"\r\033[K  {mark}  {sess:<10} {case['name']}:{cum}"
                          f"   {de.transcript(patch)}")
                else:
                    tally["s"] += 1
                    print(f"\r\033[K  " + colors.c(f"{'– skip':<6}", "dim")
                          + f"   {de.transcript(patch)}")
            rnd += 1


def _relog(profile, row, new_label):
    return store.append(profile, {
        "id": uuid.uuid4().hex[:12], "case": row["case"], "type": row.get("type", "emotion"),
        "ctx": row.get("ctx", [0.0, 0.0]), "label": new_label, "patch": row["patch"],
        "round": row.get("round"), "ts": _now(), "session": row.get("session"), "gen": "review",
    })


def apply_review_key(profile, row, key):
    """Effect one review keystroke; returns a status string or None (keep)."""
    if key == "x":
        store.tombstone(profile, row["id"], ts=_now())
        return "removed"
    if key == "f":
        store.tombstone(profile, row["id"])
        _relog(profile, row, FLIP.get(row["label"], "n"))
        return f"flipped -> {FLIP.get(row['label'], 'n')}"
    return None


def review(profile, case_name, *, gain=0.18, no_play=False):
    """Interactive review/prune loop for one case. Returns the change count."""
    rows = store.load(profile, case=case_name)
    if not rows:
        print(f"  no votes for '{case_name}' yet.")
        return 0
    print(f"\nreview '{case_name}' — {len(rows)} votes "
          f"({sum(r['label'] != 'n' for r in rows)} good / {sum(r['label'] == 'n' for r in rows)} bad)")
    print(REVIEW_HELP)
    changed = 0
    with term.Console() as con, tempfile.TemporaryDirectory() as td:
        for i, row in enumerate(rows):
            wav = _render(td, f"r{i}.wav", row["patch"], gain)
            con.flush()
            sys.stdout.write(f"\r  [{i+1}/{len(rows)}] [{row['label']}] {de.transcript(row['patch'])}   ")
            sys.stdout.flush()
            while True:
                key = term.play_and_wait(con, wav, no_play)
                if key == "r":
                    continue
                if key == "?":
                    print("\n" + REVIEW_HELP)
                    continue
                break
            if key == "q":
                break
            st = apply_review_key(profile, row, key)
            if st:
                changed += 1
                print(f"\n  -> {st}")
    print(f"\n  {changed} change(s).")
    return changed
