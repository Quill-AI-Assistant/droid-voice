"""droid — synthesized droid-voice package.

The shared engine (synth/emotion/features/dataset/refine/stats/phrasebook) and the
CLI verbs (droid/cli/<verb>.py) live here; the top-level `droid` dispatcher routes to
the verbs, and the root `droid_synth.py` / `droid_emotion.py` / `phrasebook.py` are thin
top-level aliases so scripts and tests can import the engine by short name.

ROOT is the SERVICE directory (the parent of this package) where profiles/, catalog.json,
config.json and the data live — modules resolve those paths off ROOT, NOT off their own
file location, so moving a module into the package never repoints its data paths.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
