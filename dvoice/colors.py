"""colors — tiny ANSI colour helper for the interactive CLI.

Colour REINFORCES, never replaces: every coloured thing also carries a glyph/word
(✓ good / ✗ bad / ★), so the UI stays legible without colour (low-vision, NO_COLOR,
piped output). Auto-disabled when stdout isn't a TTY, NO_COLOR is set, or TERM=dumb.
"""
import os
import sys

_CODES = {
    "reset": "0", "bold": "1", "dim": "2", "inv": "7",
    "red": "31", "green": "32", "yellow": "33", "blue": "34",
    "magenta": "35", "cyan": "36",
    "bred": "91", "bgreen": "92", "byellow": "93", "bcyan": "96", "bmagenta": "95",
}

_ENABLED = (sys.stdout.isatty()
            and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") != "dumb")


def enabled():
    return _ENABLED


def c(text, *styles):
    """Wrap text in the given ANSI styles (no-op when colour is disabled)."""
    if not _ENABLED or not styles:
        return str(text)
    pre = "".join(f"\033[{_CODES[s]}m" for s in styles if s in _CODES)
    return f"{pre}{text}\033[0m" if pre else str(text)
