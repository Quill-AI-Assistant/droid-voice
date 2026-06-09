#!/usr/bin/env python3
"""render-cues.py — synthesize the droid-voice cue family (macropad + session).

Output goes into the droid sound layer's qd profile
(system/services/droid-sounds/profiles/qd/); the catalog maps events to these
slugs. These cues are character-independent blip/seq motifs (the generic
macropad/session beep set), not the trained qd emotional voice.

The DSP engine (soft sine + a little 2nd/3rd harmonic, two detuned oscillators,
raised-cosine attack, exponential pluck decay, gentle low-pass) lives in the
shared module `droid_synth` so this renderer and the calibrator make cues the
same way. Cues differ only in their pitch *motif* — a classic sci-fi cockpit move
is a quick pitch glide, so each cue is a short glided blip or two.

Every cue is peak-normalised to MASTER_GAIN — re-tune the whole set's volume
with `--gain` and re-run.

    render-cues.py [--gain 0.18] [--out DIR] [--format wav|mp3]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# Shared DSP engine lives next door in the droid-sounds service.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_synth import SR, blip, seq, write_wav  # noqa: E402

MASTER_GAIN = 0.18  # peak headroom; soft by default. Override with --gain.


# ── the cue family — same engine, different pitch motif ──────────────────────
def build():
    cues = {}

    # transport / keypress tick: one short blip with a quick downward bend.
    cues["click"] = blip([(0.0, 760), (1.0, 600)], 0.085, decay_tau=0.030)

    # lights ON: two ascending notes (C5 → G5), confident but gentle.
    cues["light-on"] = seq([
        blip([(0.0, 523), (1.0, 540)], 0.10, decay_tau=0.09),
        blip([(0.0, 740), (1.0, 784)], 0.16, decay_tau=0.12),
    ])

    # lights OFF: two descending notes — mirror of ON.
    cues["light-off"] = seq([
        blip([(0.0, 700), (1.0, 680)], 0.10, decay_tau=0.09),
        blip([(0.0, 470), (1.0, 440)], 0.18, decay_tau=0.13),
    ])

    # offline / failure: a low, soft warble — non-harsh, slight downward drift.
    cues["error"] = blip([(0.0, 320), (0.6, 300), (1.0, 270)], 0.42,
                         decay_tau=0.30, fm_rate=13.0, fm_depth=10.0)

    # ── session-lifecycle cues (same soft engine; generic cue voice) ────
    # session-start: a gentle three-note "power up".
    cues["session-start"] = seq([
        blip([(0.0, 392), (1.0, 410)], 0.09, decay_tau=0.08),
        blip([(0.0, 523), (1.0, 540)], 0.09, decay_tau=0.08),
        blip([(0.0, 700), (1.0, 740)], 0.16, decay_tau=0.13),
    ])
    # session-end: the mirror — a soft three-note "power down".
    cues["session-end"] = seq([
        blip([(0.0, 700), (1.0, 680)], 0.09, decay_tau=0.08),
        blip([(0.0, 523), (1.0, 510)], 0.09, decay_tau=0.08),
        blip([(0.0, 392), (1.0, 370)], 0.18, decay_tau=0.14),
    ])
    # subagent-spawn: a quick upward chirp — "something woke up".
    cues["subagent-spawn"] = blip([(0.0, 560), (1.0, 880)], 0.11, decay_tau=0.07)
    # subagent-complete: a soft two-note settle — "done".
    cues["subagent-complete"] = seq([
        blip([(0.0, 740), (1.0, 720)], 0.09, decay_tau=0.08),
        blip([(0.0, 587), (1.0, 560)], 0.13, decay_tau=0.11),
    ])
    # memory-save: a tiny double-tick at one pitch — unobtrusive.
    cues["memory-save"] = seq([
        blip([(0.0, 660), (1.0, 650)], 0.05, decay_tau=0.025),
        blip([(0.0, 660), (1.0, 650)], 0.05, decay_tau=0.025),
    ], gap=0.05)
    # compact: a soft descending sweep — "winding down / folding".
    cues["compact"] = blip([(0.0, 620), (0.5, 460), (1.0, 320)], 0.34, decay_tau=0.26)

    return cues


def main():
    ap = argparse.ArgumentParser(description="Render the droid-voice cue family")
    ap.add_argument("--gain", type=float, default=MASTER_GAIN,
                    help=f"master peak gain 0..1 (default {MASTER_GAIN}); re-tune volume here")
    ap.add_argument("--out",
                    default=str(Path(__file__).parent.parent / "profiles" / "qd"),
                    help="output directory (default: the qd droid-sound profile)")
    ap.add_argument("--format", choices=["wav", "mp3"], default="wav")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, sig in build().items():
        peak = np.max(np.abs(sig)) or 1.0
        sig = sig / peak * args.gain          # peak-normalise to the master gain
        wav = out / f"{name}.wav"
        write_wav(wav, sig)
        final = wav
        if args.format == "mp3":
            mp3 = out / f"{name}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                            "-codec:a", "libmp3lame", "-q:a", "4", str(mp3)], check=True)
            wav.unlink()
            final = mp3
        dur = sig.size / SR
        manifest.append(f"{final.name}\t{dur:.3f}s\tgain={args.gain}")
        print(f"  rendered {final.name}  ({dur:.3f}s)")

    (out / "manifest.txt").write_text(
        f"# qd droid cues — gain={args.gain}, homogenous DSP family\n" + "\n".join(manifest) + "\n")
    print(f"OK {len(manifest)} cues → {out}  (gain={args.gain})")


if __name__ == "__main__":
    main()
