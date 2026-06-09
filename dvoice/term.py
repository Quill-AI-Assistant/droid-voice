"""tty — the shared single-key terminal input layer for the interactive verbs.

Extracted verbatim-in-spirit from the legacy calibrator so `droid collect`
and `droid review` share ONE keyboard reader (cbreak mode, arrow + single-char
tokens, press-to-decide playback). stdlib only; isatty()-gated by the caller.
"""
import atexit
import os
import string
import subprocess
import sys

_ARROWS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
# Any letter / digit / common punctuation a verb might bind. (The old hand-listed set
# silently dropped m, d, c, '.' — breaking the studio's models/datasets menus.)
NORMAL = set(string.ascii_lowercase + string.digits + " .,?*+-=_/")


class Console:
    """cbreak terminal: raw-ish mode set ONCE on enter, restored on exit (finally +
    atexit so a crash never leaves the tty broken). getkey(timeout) returns a
    normalized token ('' on timeout); flush() drops buffered input (anti key-spam)."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = None
        import termios  # noqa: F401  (validate availability early)

    def __enter__(self):
        import termios
        import tty as _tty
        self.old = termios.tcgetattr(self.fd)
        _tty.setcbreak(self.fd)
        atexit.register(self._restore)
        return self

    def __exit__(self, *exc):
        self._restore()
        return False

    def _restore(self):
        if self.old is not None:
            import termios
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass
            self.old = None

    def flush(self):
        """Discard pending input — call before each candidate so keys typed during a
        gap don't roll forward onto the next sound."""
        import termios
        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except Exception:
            pass

    def getkey(self, timeout):
        """Normalized token or '' on timeout (None blocks). Reads raw bytes via
        os.read so no Python buffering swallows a press. Accepts BOTH CSI (ESC [ x)
        and SS3 (ESC O x) arrow encodings — application-cursor mode is common and a
        CSI-only check silently drops every arrow."""
        import select
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return ""
        data = os.read(self.fd, 8)
        if not data:
            return ""
        if data[:1] == b"\x1b":                          # arrow/escape (maybe split)
            tries = 0
            while len(data) < 3 and tries < 8:
                rr, _, _ = select.select([self.fd], [], [], 0.06)
                if not rr:
                    break
                more = os.read(self.fd, 8)
                if not more:
                    break
                data += more
                tries += 1
            if len(data) >= 3 and data[1:2] in (b"[", b"O"):
                return _ARROWS.get(chr(data[2]), "")
            return ""
        if data[:1] in (b"\r", b"\n"):
            return "ENTER"
        ch = chr(data[0]).lower()
        return ch if ch in NORMAL else ""


def play_and_wait(console, wav_path, no_play=False):
    """Play `wav_path`; return the FIRST key pressed. A key during playback stops
    afplay instantly (press-to-decide). If audio finishes with no key, block for one.
    no_play => just block for a key (headless/test stepping)."""
    if no_play:
        return console.getkey(None)
    proc = subprocess.Popen(["afplay", wav_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        k = console.getkey(0.02)
        if k:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except Exception:
                proc.kill()
            return k
    return console.getkey(None)
