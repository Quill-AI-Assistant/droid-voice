"""Top-level alias — the keystone lives in dvoice/emotion.py.

A sys.modules ALIAS so `import droid_emotion as de` is the SAME object as
`dvoice.emotion` — preserving the mutable module globals (PROFILES, EMOTIONS, ...)
that the calibrator and the hermetic tests repoint at runtime.
"""
import sys

from dvoice import emotion as _emotion

sys.modules[__name__] = _emotion
