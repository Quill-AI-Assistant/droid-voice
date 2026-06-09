# droid-voice

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![tests 113 passing](https://img.shields.io/badge/tests-113%20passing-brightgreen) ![deps numpy + scipy](https://img.shields.io/badge/deps-numpy%20%2B%20scipy-lightgrey)

*A generate‑then‑judge expressive voice: a frozen DSP synth proposes, a tiny preference model
trained on your votes ranks — so it learns your taste without ever averaging the life out of the
sound.*

A **synthesized expressive droid voice** — the chirps, warbles and sighs of a science‑fiction
robot — generated entirely from your own DSP (no samples, no movie audio), that **learns which
sounds you like** from thumbs‑up / thumbs‑down votes.

The twist: the learned model is a **judge, not a generator**. A frozen synthesizer proposes many
candidate sounds; a small preference model — trained on your votes — ranks them; selection samples
a fresh top pick. Because the judge only ever *chooses among* lively candidates the synth already
made, it can refine the voice toward your taste but can never flatten it.

```
                 ┌─ generate ─┐   ┌─ judge ─┐   ┌─ sample ─┐
   (valence,     │  K diverse │ → │  score  │ → │ top‑N +  │ → ♪ a fresh, in‑character utterance
    arousal) ───▶│ candidates │   │  each   │   │ novelty  │
                 └────────────┘   └─────────┘   └──────────┘
   candidate #0 is always the plain analytic phrase — the always‑on quality floor
```

## Quick start

Requires **Python 3.10+**. Two dependencies (`numpy`, `scipy`); an optional MLX backend
accelerates training on Apple Silicon and is auto‑detected.

```bash
pip install -r requirements.txt

# 1) hear the voice from the terminal
./droid say proud
./droid say curious --text "what is this thing?"

# 2) open the browser demo (no build step, stdlib server)
./droid web            # → http://127.0.0.1:8765
```

**Audio playback:** terminal playback uses macOS `afplay`. On Linux/Windows the terminal
commands write the WAV but produce no sound (no fallback player is wired) — use the browser demo
(`./droid web`), which plays in‑browser and is fully cross‑platform.

The demo lets you pick a feeling — or just type words (`ha` → laugh, `ugh` → frustrated) — and hear
a freshly generated, never‑repeating reaction, and contrasts a **trained** voice (`qd`, the shipped
trained voice) against an **untrained** one (the analytic floor). The repo ships the judge as
actually tuned by ear — **v8, fit on 1057 real keep/drop votes** — so the demo plays a voice with
real taste out of the box (run `droid eval` for the honest held‑out *within‑case* ranking score —
the leak‑free gate metric). Teach it your own with `droid collect <case>`.

## How it works

1. **Generate.** A frozen numpy synth (`dvoice/synth.py`) renders a *patch* — a sequence of pitched,
   gliding notes with optional ring‑modulation and sample‑&‑hold "grit". `generate_arrangement`
   turns a `(valence, arousal)` target and an optional phrase into a structurally‑varied batch of
   candidates (different contours, lengths, textures).
2. **Judge.** A preference model (`dvoice/reward.py`) — a closed‑form Gaussian‑process ranker over a
   compact acoustic feature codec — scores every candidate. It is trained purely on your keep/drop
   votes (Best‑of‑N / rejection‑sampling style), so it never picks the synth's knobs; it only
   *ranks* what the synth produced.
3. **Sample.** Selection restricts to the judge's top‑N, softmax‑samples by temperature, and
   subtracts a novelty penalty measured against recent plays — so repeated asks give different good
   takes, never the same one twice. `temperature 0` collapses to the deterministic argmax used for
   the byte‑stable cue sounds.

A **case** is anything the voice reacts to: an `emotion` (a valence/arousal point — *curious,
proud*), an `expression` (a status coloured by a feeling — *done, error*), or a `vocalization` (a
non‑verbal burst with its own synthesis — *laugh, sigh, chirp*). The demo page (`droid web`) is the walkthrough and the playground in one.

## How emotions stay distinct

Within `generate_arrangement`, each emotion is voiced non-verbally across three axes — all driving the **frozen** synth, which never changes:

- **Melodic mode = valence.** Middle pitches step by consonant intervals (major-3rd / perfect-5th / major-6th) for positive feelings and minor-2nd / tritone clusters for negative ones — the cue that actually carries valence in tonal, word-free sound (Juslin & Laukka, 2003).
- **Per-emotion contour signatures.** Same-quadrant feelings get distinct gesture *rhythms*, not just more or less of one shape — *playful* bounces, *proud* rises and holds, *curious* probes with a data-burble — so neighbours on the wheel read as qualitatively different shapes.
- **Rhythm as an expressive axis.** Inter-note timing bends per feeling (accelerando when aroused, ritardando when tired, syncopation when playful), and every replay rolls a fresh rhythm/signature variant so repeats stay alive.

## The CLI

`droid` is a single dispatcher; bare `droid` opens an interactive studio (dashboard + menus). The
scriptable verbs:

| verb | does |
|---|---|
| `droid say <case>` | generate → judge‑rank → sample a fresh take (the always‑on analytic floor if no judge) |
| `droid demo` | interactive playground: type a case, a `"phrase [tag]"`, or a `v,a` point |
| `droid cases [list\|add]` | the taxonomy: emotion / expression / vocalization |
| `droid collect <case>` | vote y/n/k on diverse candidates (k=keep, s=skip) → append to the dataset |
| `droid train` | fit the judge, held‑out gate (pairwise AUC), save a new model version |
| `droid bootstrap <p>` | synthesize a v0 judge from a profile's *character* (no votes needed yet) |
| `droid eval / models / doctor` | judge AUC + coverage · manage versions · health‑check the stack |
| `droid web` | launch the browser demo (read‑only) |
| `droid cue <event>` | play a generated UI/notification cue (the beep layer) |

Models are versioned (`profiles/<p>/judges/vN.npz` with AUC/votes/date); the dataset is
append‑only with reversible tombstones.

## Layout

```
droid              dispatcher (bare `droid` → the studio)
droid-*            one script per verb
dvoice/            the engine package
  synth.py         FROZEN numpy DSP (the only audio path)
  emotion.py       param space, emotion anchors, generate_arrangement / generate_vocalization
  reward.py        the preference‑GP judge + Best‑of‑N + fit/eval
  features.py      acoustic feature codec + shared kernel
  cases.py store.py versions.py sessions.py term.py colors.py lexicon.py
web/               stdlib HTTP server + vanilla‑JS single‑page demo (no build)
tools/             render-cues.py — generate the UI cue family from pure DSP
profiles/qd/       the shipped voice: case taxonomy, cue WAVs, the by‑ear‑trained judge (v8)
```

## Tests

```bash
pip install pytest                         # test‑only dep, not pulled in by the runtime deps
DROID_NO_EMBED=1 python3 -m pytest -q      # 113 passing — pure‑function / no audio device
```

The frozen synth's cue WAVs are **sha256 byte‑identity gated** (a test re‑renders them and asserts
they are bit‑for‑bit identical), so a refactor can never silently drift the sound.

## Design notes

- **Judge, not generator.** An earlier design trained a model to *generate* the voice directly and
  it flattened — it regressed the synth's controls toward a dull average. Best‑of‑N over a frozen,
  expressive synth fixes that: the model can only choose among lively candidates.
- **No heavy deps on the play path.** Judges persist as plain numpy `.npz`; rendering and playback
  need only numpy/scipy. The optional MLX backend (`DROID_MLX=0` to force numpy) only accelerates
  training on Apple Silicon. No PyTorch / scikit‑learn at runtime.
- **Accessibility.** Every transcript has a plain‑ASCII fallback and never relies on colour alone.

## License

MIT — see [`LICENSE`](LICENSE). All audio is synthesized from original DSP; the affect lexicon is
hand‑authored and IP‑clean.
