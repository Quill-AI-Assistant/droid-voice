"""lexicon — map free TEXT to droid AFFECT so typed words actually shape the sound.

The synth is driven by (valence, arousal) — the circumplex point IS the control input
(emotion.generate_arrangement) — so "make words matter" is a drop-in: map text to a
(valence, arousal), and/or to a vocalization CASE for sound-symbolic interjections that
a (v,a) point can't express ("ha ha ha" is a laugh, not a coordinate).

This is NOT speech — words are never spoken; they set the FEELING. Two tables:

  * INTERJECTION  — a token -> (case, v, a). "ha"/"lol" -> the `laugh` case, etc. The
    case plays the real burst when it exists on the profile; the (v,a) is the fallback
    when it doesn't (case sets differ per profile).
  * AFFECT_VA     — a content word -> (valence, arousal) in [-1, 1].

A phrase's affect = the average (v,a) of every token that hit either table; the case is
the first interjection seen. Unknown/function words contribute nothing (graceful). This
is a COMPACT, hand-authored STARTER set (IP-clean, stdlib-only). For broad coverage drop
a full affective lexicon (Warriner 2013 / NRC-VAD) into AFFECT_VA — same interface (note: many large VAD lexicons are research/non-commercial; vaderSentiment is MIT, valence-only).
"""
import re

# token -> (case_name, valence, arousal). Repeated-letter variants are normalised below
# (hahaha/haha/ha; hmmm/hmm; yesss/yes), so list one canonical form.
INTERJECTION = {
    "ha": ("laugh", 0.8, 0.6), "haha": ("laugh", 0.8, 0.65), "heh": ("laugh", 0.6, 0.4),
    "hehe": ("laugh", 0.7, 0.5), "lol": ("laugh", 0.75, 0.6), "lmao": ("laugh", 0.8, 0.7),
    "rofl": ("laugh", 0.8, 0.75), "haa": ("laugh", 0.75, 0.6),
    "hmm": ("thinking", 0.0, -0.1), "hm": ("thinking", 0.0, -0.1),
    "huh": ("curious", 0.1, 0.3), "eh": ("curious", -0.1, 0.0),
    "ugh": ("frustrated", -0.6, 0.5), "argh": ("frustrated", -0.6, 0.6),
    "grr": ("frustrated", -0.6, 0.6), "gah": ("frustrated", -0.5, 0.5),
    "wow": ("excited", 0.6, 0.7), "whoa": ("excited", 0.4, 0.7), "woah": ("excited", 0.4, 0.7),
    "yay": ("excited", 0.8, 0.7), "yyy": ("excited", 0.7, 0.7), "yippee": ("excited", 0.85, 0.8),
    "yupee": ("excited", 0.85, 0.8), "woohoo": ("excited", 0.85, 0.85), "yeehaw": ("excited", 0.8, 0.8),
    "sigh": ("sigh", -0.3, -0.3), "phew": ("sigh", 0.2, -0.2),
    "meh": ("bored", -0.3, -0.4), "blah": ("bored", -0.3, -0.3),
    "aha": ("recognition", 0.5, 0.4), "ah": ("recognition", 0.3, 0.1),
    "oh": ("curious", 0.0, 0.2), "ooh": ("curious", 0.3, 0.4), "ohh": ("curious", 0.1, 0.2),
    "oops": ("worried", -0.3, 0.3), "uhoh": ("worried", -0.4, 0.4), "yikes": ("alarmed", -0.5, 0.7),
    "eek": ("alarmed", -0.4, 0.7), "aww": ("content", 0.5, 0.0), "yum": ("happy", 0.6, 0.3),
    "boo": ("sad", -0.5, 0.2), "no": ("frustrated", -0.4, 0.4), "yes": ("excited", 0.5, 0.5),
}

# content word -> (valence, arousal) in [-1, 1]. Compact starter set of common affect words.
AFFECT_VA = {
    # positive, high arousal
    "amazing": (0.85, 0.7), "awesome": (0.8, 0.7), "wonderful": (0.85, 0.6), "great": (0.7, 0.5),
    "fantastic": (0.85, 0.7), "excellent": (0.8, 0.55), "love": (0.8, 0.6), "loved": (0.8, 0.55),
    "win": (0.7, 0.7), "won": (0.7, 0.65), "winning": (0.7, 0.7), "success": (0.7, 0.5),
    "perfect": (0.8, 0.5), "yes": (0.6, 0.5), "exciting": (0.7, 0.8), "excited": (0.65, 0.8),
    "happy": (0.75, 0.5), "joy": (0.8, 0.6), "celebrate": (0.8, 0.7), "best": (0.7, 0.5),
    "brilliant": (0.8, 0.6), "cool": (0.5, 0.4), "nice": (0.5, 0.2), "fun": (0.7, 0.6),
    # positive, low arousal
    "calm": (0.5, -0.6), "peaceful": (0.6, -0.6), "relax": (0.5, -0.6), "relaxed": (0.5, -0.6),
    "good": (0.5, 0.2), "fine": (0.3, -0.1), "okay": (0.2, -0.1), "ok": (0.2, -0.1),
    "gentle": (0.4, -0.4), "warm": (0.5, -0.2), "safe": (0.5, -0.3), "content": (0.6, -0.2),
    "thanks": (0.5, 0.2), "thank": (0.5, 0.2), "please": (0.3, 0.1), "soft": (0.3, -0.3),
    # negative, high arousal
    "angry": (-0.7, 0.7), "mad": (-0.6, 0.7), "hate": (-0.8, 0.6), "terrible": (-0.8, 0.6),
    "awful": (-0.7, 0.5), "horrible": (-0.8, 0.6), "fail": (-0.6, 0.4), "failed": (-0.6, 0.4),
    "error": (-0.5, 0.4), "crash": (-0.6, 0.6), "crashed": (-0.6, 0.6), "broken": (-0.5, 0.3),
    "wrong": (-0.5, 0.3), "stop": (-0.3, 0.6), "panic": (-0.7, 0.8), "scared": (-0.6, 0.6),
    "afraid": (-0.6, 0.55), "danger": (-0.6, 0.7), "alarm": (-0.5, 0.7), "warning": (-0.4, 0.6),
    "no": (-0.4, 0.4), "ugh": (-0.6, 0.5), "stuck": (-0.5, 0.3), "worried": (-0.4, 0.4),
    "annoying": (-0.5, 0.5), "stupid": (-0.5, 0.5),
    # negative, low arousal
    "sad": (-0.6, -0.3), "tired": (-0.2, -0.7), "bored": (-0.4, -0.5), "boring": (-0.4, -0.4),
    "slow": (-0.2, -0.3), "lost": (-0.4, -0.2), "down": (-0.5, -0.3), "lonely": (-0.6, -0.3),
    "sorry": (-0.3, -0.1), "sigh": (-0.3, -0.3), "meh": (-0.3, -0.4), "dull": (-0.4, -0.4),
    "sleepy": (-0.1, -0.7), "low": (-0.3, -0.3), "empty": (-0.5, -0.3),
    # curious / recognition
    "what": (0.0, 0.3), "why": (0.0, 0.3), "how": (0.0, 0.2), "wonder": (0.3, 0.3),
    "curious": (0.4, 0.5), "interesting": (0.4, 0.4), "found": (0.5, 0.3), "got": (0.4, 0.3),
    "done": (0.4, 0.2), "thinking": (0.0, 0.1), "maybe": (0.0, 0.0),
}

_RUN = re.compile(r"(.)\1{2,}")          # 3+ repeats of a char


def _norm_forms(tok):
    """Variants of a token to look up: itself, runs collapsed to 2, runs collapsed to 1.
    Lets hahaha/haha/ha and hmmm/hmm and yesss/yes all resolve."""
    forms = [tok]
    two = _RUN.sub(r"\1\1", tok)
    one = _RUN.sub(r"\1", tok)
    for f in (two, one):
        if f not in forms:
            forms.append(f)
    return forms


def text_to_affect(text):
    """Map free text -> {case, valence, arousal, matched}.

    case: the first interjection's case (or None). valence/arousal: the mean (v,a) of
    every token that hit INTERJECTION or AFFECT_VA, in [-1, 1] (0,0 when nothing hit).
    matched: whether any affect-bearing token was found. The caller decides whether the
    case is usable on the active profile, else falls back to (v,a)."""
    toks = re.findall(r"[a-z']+", (text or "").lower())
    case = None
    vs, as_ = [], []
    for tok in toks:
        hit = None
        for f in _norm_forms(tok):
            if f in INTERJECTION:
                c, v, a = INTERJECTION[f]
                if case is None:
                    case = c
                hit = (v, a)
                break
            if f in AFFECT_VA:
                hit = AFFECT_VA[f]
                break
        if hit is not None:
            vs.append(hit[0])
            as_.append(hit[1])
    if not vs:
        return {"case": None, "valence": 0.0, "arousal": 0.0, "matched": False}
    v = max(-1.0, min(1.0, sum(vs) / len(vs)))
    a = max(-1.0, min(1.0, sum(as_) / len(as_)))
    return {"case": case, "valence": float(v), "arousal": float(a), "matched": True}
