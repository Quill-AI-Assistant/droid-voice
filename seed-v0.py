#!/usr/bin/env python3
"""seed-v0 — write a designed v0 emotional voice for a profile.

    seed-v0.py <profile>            # e.g. qd

Calibration is by-ear (the `droid` studio / `droid collect`) and yours to
perfect, but this seeds a coherent, character-grounded STARTING voice for all 8
emotion anchors so the profile speaks well out of the box (and the tiny model has
something to learn from). Each anchor = the rich multi-syllable arrangement from
droid_emotion plus a touch of emotion-appropriate ARP-2600 droid colour (metallic
ring on all, sample-&-hold stutter on the agitated ones).

Writes profiles/<profile>/emotions.json (+ emotions/<name>.wav renders). Then
`droid train <profile>` learns it and `droid say <profile> <emotion>` plays
it. Re-run after editing TEXTURE to retune the whole v0.

NOTE: a profile's throwaway runtime state (gen/, events/, .recent_plays.json, .bak)
is gitignored; the shipped qd profile's authored data (cases.json, cue WAVs, dataset
+ judge) is committed so the demo runs out of the box. This script regenerates the
v0 emotion renders locally.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import droid_emotion as de        # noqa: E402
import droid_synth as ds          # noqa: E402

# (ring_depth, sh_rate) per emotion — subtle metallic ring on every voice, with
# sample-&-hold stutter added to the agitated/urgent emotions. Within bounds
# (ring_depth<=0.6, sh_rate<=40). ring_hz carrier is fixed (200 Hz).
TEXTURE = {
    "neutral":     (0.12, 0),
    "curious":     (0.18, 0),
    "happy":       (0.15, 0),
    "recognition": (0.20, 0),
    "wistful":     (0.12, 0),
    "sad":         (0.10, 0),
    "worried":     (0.28, 14),
    "alarmed":     (0.42, 20),
}
RING_HZ = 200.0
GAIN = 0.18


def seed(profile):
    pdir = os.path.join(HERE, "profiles", profile)
    os.makedirs(os.path.join(pdir, "emotions"), exist_ok=True)
    store = {}
    for name, (v, a) in de.EMOTIONS.items():
        patch = de.generate_arrangement(profile, v, a)
        ring, shr = TEXTURE.get(name, (0.12, 0))
        for n in patch["notes"]:
            n["ring_hz"] = RING_HZ
            n["ring_depth"] = ring
            if shr:
                n["sh_rate"] = shr
        patch = de._clamp_patch(patch)
        store[name] = {
            "valence": v, "arousal": a, "patch": patch,
            "transcript": de.transcript(patch), "grammar": de.grammar_meta(patch),
        }
        ds.write_wav(os.path.join(pdir, "emotions", f"{name}.wav"),
                     ds.render_patch(patch, GAIN))
        print(f"  {name:11s} {de.transcript(patch)}")
    with open(os.path.join(pdir, "emotions.json"), "w") as f:
        json.dump(store, f, indent=2)
        f.write("\n")
    print(f"→ wrote {len(store)} emotion anchors to {profile}/emotions.json")

    # phrasebook: render the named (emotion + text) utterances as droid phrases
    # (analytic/designed renders — `droid say` plays the model version at runtime).
    import phrasebook
    pbdir = os.path.join(pdir, "phrasebook")
    os.makedirs(pbdir, exist_ok=True)
    for name in phrasebook.names():
        v, a, text = phrasebook.resolve(name)
        patch = de.generate_arrangement(profile, v, a, text)
        ds.write_wav(os.path.join(pbdir, f"{name}.wav"), ds.render_patch(patch, GAIN))
    print(f"→ rendered {len(phrasebook.names())} phrasebook utterances → {profile}/phrasebook/")
    print(f"  next: ./droid train {profile} && ./droid say {profile} done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: seed-v0.py <profile>", file=sys.stderr)
        sys.exit(2)
    seed(sys.argv[1])
