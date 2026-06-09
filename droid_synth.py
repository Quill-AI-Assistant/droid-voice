"""Top-level alias — the FROZEN synth engine lives in dvoice/synth.py.

Kept as a sys.modules ALIAS (not a re-export) so `import droid_synth` and
`from droid_synth import render_patch, write_wav, SR` resolve to the SAME module
object as `dvoice.synth` — preserving byte-identity and any shared module state.
"""
import sys

from dvoice import synth as _synth

sys.modules[__name__] = _synth
