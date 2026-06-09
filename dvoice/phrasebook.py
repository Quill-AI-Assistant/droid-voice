"""phrasebook — named droid utterances for emotions and statuses.

Each entry is (emotion, text): the emotion sets the colour (its valence/arousal
from droid_emotion.EMOTIONS), the text shapes the phrase's LENGTH (syllable count ->
note count) and TERMINAL gesture (a question rises, '!' emphasises, '...' trails) — the
word CONTENT itself is never spoken. The effect is strongest on the warbly voice
(3-6 notes); the terse voices move mainly via length + punctuation.

Play one:    droid say <profile> <name>              e.g.  droid say qd done
Override text: droid say <profile> <name> "..."       (your text wins)
Render all:  seed-v0.py <profile>  (writes profiles/<p>/phrasebook/<name>.wav)

These are USAGE shortcuts, not training anchors — the model trains on V-A emotion
anchors (emotions.json); text steers at inference. Add/edit freely; re-render
with seed-v0.py.
"""

from dvoice import emotion as de

# name -> (emotion, text). Curated to the droid's voice (curious, dry, hyper-competent).
PHRASES = {
    # ── expressive (emotions) ─────────────────────────────────────────────────
    "wondering":    ("curious", "what is this?"),
    "intrigued":    ("curious", "hm, now that is interesting"),
    "delighted":    ("happy", "yes, there it is"),
    "got-it":       ("happy", "got it"),
    "i-see":        ("recognition", "ah, i see it now"),
    "of-course":    ("recognition", "of course"),
    "noted":        ("neutral", "noted"),
    "dry":          ("neutral", "well, that happened"),
    "mourning":     ("wistful", "that one is gone now"),
    "disappointed": ("sad", "that did not work"),
    "uneasy":       ("worried", "something here is off"),
    "alarm":        ("alarmed", "warning, stop"),

    # ── added emotion phrases (the 10 new anchors) ────────────────────────────
    "thrilled":     ("excited", "yes! look at this!"),
    "wonderful":    ("elated", "wonderful!"),
    "playing":      ("playful", "hehe, watch this!"),
    "accomplished": ("proud", "i did it"),
    "settled":      ("content", "all good"),
    "calm":         ("serene", "mmm, calm"),
    "assured":      ("confident", "got it"),
    "weary":        ("tired", "so... slow"),
    "unbothered":   ("bored", "...whatever"),
    "fed-up":       ("frustrated", "come on!"),

    # ── social / conversational (pleasantries) ────────────────────────────────
    "greetings":     ("curious", "oh, hello there"),
    "welcome-back":  ("happy", "ah, you're back"),
    "farewell":      ("wistful", "until next time"),
    "thank-you":     ("happy", "thank you"),
    "youre-welcome": ("content", "anytime"),
    "oops":          ("playful", "oops"),
    "my-bad":        ("worried", "ah, my mistake"),
    "apology":       ("sad", "sorry about that"),
    "congrats":      ("excited", "well done!"),
    "agreed":        ("recognition", "agreed"),

    # ── statuses (operational) ────────────────────────────────────────────────
    "thinking":     ("curious", "let me think this through"),
    "searching":    ("curious", "searching"),
    "found":        ("recognition", "found it"),
    "working":      ("neutral", "working on it"),
    "building":     ("neutral", "building"),
    "done":         ("happy", "task complete"),
    "saved":        ("neutral", "saved"),
    "committed":    ("neutral", "committed"),
    "online":       ("curious", "online, ready"),
    "ending":       ("wistful", "session ending"),
    "error":        ("worried", "error detected"),
    "failed":       ("sad", "that failed"),
    "success":      ("happy", "success"),
    "critical":     ("alarmed", "critical, intervene"),
    "standing-by":  ("neutral", "standing by"),
    "waiting":      ("neutral", "waiting on you"),
}


# ── per-CASE example phrases ──────────────────────────────────────────────────
# A short representative line for each case (emotion / expression / vocalization),
# keyed by the CASE name (not the utterance name above). The web tour renders each
# case both bare AND with this phrase so the text->sound effect is hearable; the
# string is also shown on the row. Emotions/expressions: the text shapes length +
# terminal (see emotion.generate_arrangement). Vocalizations ignore text (burst
# archetype), so their string is a label only. Unknown cases -> "" (bare only).
CASE_EXAMPLES = {
    # emotions (the 18 circumplex anchors)
    "alarmed": "warning, stop!", "bored": "...whatever", "confident": "got it",
    "content": "all good", "curious": "what is this?", "elated": "wonderful!",
    "excited": "yes! look at this!", "frustrated": "come on!", "happy": "yes, there it is",
    "neutral": "noted", "playful": "hehe, watch this!", "proud": "i did it",
    "recognition": "ah, i see it now", "sad": "that did not work", "serene": "mmm, calm",
    "tired": "so... slow", "wistful": "until next time", "worried": "something is off",
    # expressions (operational statuses)
    "done": "task complete", "error": "error detected", "thinking": "let me think this through",
    "found": "found it", "working": "working on it", "building": "building",
    "saved": "saved", "committed": "committed", "online": "online, ready",
    "searching": "searching", "success": "success", "failed": "that failed",
    "critical": "critical, intervene!", "waiting": "waiting on you", "ending": "session ending",
    "standing-by": "standing by", "greeting": "oh, hello there", "chirp-greet": "oh, hello there",
    # vocalizations (label only — these render a burst archetype, text is ignored)
    "laugh": "*ha ha*", "sigh": "*sigh*", "hmm": "hmm...", "chirp": "*chirp*",
    "gasp": "*gasp!*", "growl": "*grr*", "beep": "yes", "buzz": "no",
}


def example(name):
    """A short example phrase for a case name, or '' if none. The tour uses this as
    the spoken phrase (emotions/expressions) or a display label (vocalizations)."""
    return CASE_EXAMPLES.get(name, "")


def resolve(name):
    """name -> (valence, arousal, text), or None if not a phrase."""
    entry = PHRASES.get(name)
    if not entry:
        return None
    emotion, text = entry
    v, a = de.EMOTIONS.get(emotion, (0.0, 0.2))
    return v, a, text


def names():
    return list(PHRASES)
