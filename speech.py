"""Planning what Astrid says aloud. Pure Python: no GTK, no audio, no numpy.

Two things were making her feel slow after the model itself was fixed. She
synthesized the WHOLE reply before any sound came out (Kokoro runs on the CPU at
about 8x real time, so 3.7 s of silence for a 475-character answer), and
analysis replies are simply long when spoken (about 28 s). This module decides
how a reply is cut up and how much of it is spoken:

  * the reply is split into chunks, so gui.py can start playing the first one
    while the rest are still being synthesized;
  * optionally (SPOKEN_LIMIT_CHARS, off by default) a long reply is cut after a
    few whole sentences, and a closing line says the rest is on screen. The
    transcript always has the full text.

It also works out where an interruption fell (text_after), so "go on" can resume,
and recognises a spoken request to read something out (read_request).

Every number here is a named constant because every one is a judgement about how
it sounds, which only the listener can settle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# The time to first sound is the time to synthesize the FIRST chunk, so it should
# be short. Too short and the second chunk is not ready when the first finishes
# playing, which is an audible gap: at ~8x real time one ordinary sentence of lead
# is plenty. A 60-character chunk is about 4 s of speech and ~0.5 s to synthesize.
FIRST_CHUNK_MIN_CHARS = 60
# Later chunks are merged up to about this size. Larger means fewer, smoother
# joins; smaller keeps each synthesis short, so a barge-in does not wait on one.
CHUNK_TARGET_CHARS = 220

# Cutting long replies is OFF (0): she speaks every reply in full, however long, and
# the listener breaks in with the click on the orb or the Stop button. "Read the rest" still
# works if the cut is turned on, and so does "go on" after an interruption (text_after below).
#
# To turn the cut on, set SPOKEN_LIMIT_CHARS to the number of characters of a long
# reply to speak, in whole sentences (280 is about 17 s). A reply up to
# SPEAK_ALL_UP_TO_CHARS is spoken whole regardless; the gap between the two
# numbers is deliberate, since a 401-character reply is not worth cutting to save
# one sentence.
SPEAK_ALL_UP_TO_CHARS = 400
SPOKEN_LIMIT_CHARS = 0
# Playback runs this far behind the clock: the player takes a moment to start and
# keeps a little audio buffered. Used to work out where an interruption fell.
PLAYBACK_LAG_SECONDS = 0.3
# Says where the rest is AND how to get it: a cut with no way back looked like a
# refusal. "Read the rest" is handled in gui.py without asking the language model,
# which cannot be trusted to read its own earlier reply back (asked to, it wrote
# new text).
CLOSING_LINE = "The rest is on screen. Say read the rest, and I will read it out."

# A full stop after these does not end a sentence.
_ABBREVIATIONS = frozenset({
    "e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "ms.", "st.", "no.",
    "approx.", "inc.", "jr.", "sr.", "fig.", "cf.", "al.",
})
_TERMINATOR = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
_OPENERS = "\"'(["
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by space and something that starts a sentence.

    Wrong in a few rare cases (a sentence that really does start lower-case, an
    abbreviation not in the list), and a wrong split only costs a slightly odd
    pause. Nothing is dropped or reordered: " ".join(result) equals the input
    with its whitespace normalised.
    """
    text = " ".join(text.split())
    if not text:
        return []
    sentences, start = [], 0
    for match in _TERMINATOR.finditer(text):
        end = match.end()
        rest = text[end:].lstrip()
        if not rest:
            continue
        lead = next((c for c in rest if c not in _OPENERS), "")
        # Upper case or a digit starts a sentence; so does a letter with no case at
        # all (CJK), but a lower-case letter means the sentence is continuing.
        if not (lead.isupper() or lead.isdigit() or (lead.isalpha() and not lead.islower())):
            continue
        word = text[start:end].split()[-1].lstrip(_OPENERS)
        # "e.g." and friends, or a capital initial like "J." -- a lower-case single
        # letter ("... option b.") does end a sentence. The initial guard also
        # swallows "Plan B." and "vitamin C.", which merges two sentences into one,
        # the harmless way round: a missed pause, not a pause in the middle of a name.
        if word.lower() in _ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", word):
            continue
        sentences.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _units(sentence: str) -> list[str]:
    """A sentence, or pieces of it if it is too long to synthesize in one go.

    A 400-character run-on sentence would put 3 s of silence in front of the
    first sound, so oversize sentences are cut at commas, semicolons and colons,
    and failing that at a space.
    """
    if len(sentence) <= CHUNK_TARGET_CHARS:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for clause in _CLAUSE_BREAK.split(sentence):
        while len(clause) > CHUNK_TARGET_CHARS:      # no punctuation to cut at
            cut = clause.rfind(" ", 0, CHUNK_TARGET_CHARS)
            cut = cut if cut > 0 else CHUNK_TARGET_CHARS
            if current:
                pieces.append(current)
                current = ""
            pieces.append(clause[:cut].strip())
            clause = clause[cut:].strip()
        if current and len(current) + 1 + len(clause) > CHUNK_TARGET_CHARS:
            pieces.append(current)
            current = clause
        else:
            current = (current + " " + clause).strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_sentences(sentences: list[str]) -> list[str]:
    """Merge sentences into chunks: a short first one, then up to the target.

    One rule: the chunk being built is closed when the next unit arrives and the
    chunk is already "full" for its position. The first chunk is full once it
    reaches FIRST_CHUNK_MIN_CHARS, so it is as short as can safely be (the time
    to first sound is its synthesis time). Later chunks are full when the next
    unit would not fit under CHUNK_TARGET_CHARS.
    """
    chunks: list[str] = []
    current = ""
    for unit in (u for s in sentences for u in _units(s)):
        if not current:
            current = unit
            continue
        if not chunks:
            full = len(current) >= FIRST_CHUNK_MIN_CHARS
        else:
            full = len(current) + 1 + len(unit) > CHUNK_TARGET_CHARS
        if full:
            chunks.append(current)
            current = unit
        else:
            current += " " + unit
    if current:
        chunks.append(current)
    return chunks


@dataclass
class SpeechPlan:
    chunks: list[str] = field(default_factory=list)
    closing: str = ""              # spoken after the chunks when the reply was cut
    omitted_chars: int = 0         # how much of the reply is left for the screen
    rest: str = ""                 # exactly that text, so it can be read out on request

    @property
    def spoken(self) -> list[str]:
        """Everything to synthesize, in order."""
        return self.chunks + ([self.closing] if self.closing else [])


def plan_speech(text: str, full: bool = False) -> SpeechPlan:
    """Decide what is spoken, and in which pieces, for an already-cleaned reply.

    `full` is the listener asking for all of it ("read it out"): chunked for
    streaming, never cut.
    """
    sentences = split_sentences(text)
    if not sentences:
        return SpeechPlan()
    if full or SPOKEN_LIMIT_CHARS <= 0 or len(" ".join(sentences)) <= SPEAK_ALL_UP_TO_CHARS:
        return SpeechPlan(chunk_sentences(sentences))
    kept, used = [], 0
    for sentence in sentences:
        added = len(sentence) + (1 if kept else 0)
        if kept and used + added > SPOKEN_LIMIT_CHARS:
            break                                      # always at least the first sentence
        kept.append(sentence)
        used += added
    if len(kept) == len(sentences):
        return SpeechPlan(chunk_sentences(sentences))  # one huge sentence: nothing to cut at
    rest = " ".join(sentences[len(kept):])
    return SpeechPlan(chunk_sentences(kept), CLOSING_LINE, len(rest), rest)


def text_after(pieces: list[str], seconds: list[float], elapsed: float) -> str:
    """What is still to be said after `elapsed` seconds of speech.

    `pieces` are the chunks in order and `seconds` the duration of each one that has
    been synthesized so far (a chunk with no duration yet has not started). The
    position inside the chunk being spoken is estimated from how much of its time
    has passed, and then backed up to the START OF THE SENTENCE she was in the
    middle of, so that resuming repeats a few words instead of starting mid-phrase.
    The result is always a suffix of the pieces joined, never a rewording.
    """
    t = elapsed - PLAYBACK_LAG_SECONDS
    start = 0.0
    for i, piece in enumerate(pieces):
        if i >= len(seconds):                     # not synthesized, so not started
            return " ".join(pieces[i:])
        end = start + seconds[i]
        if t < end:
            fraction = 0.0 if seconds[i] <= 0 else max(0.0, (t - start) / seconds[i])
            position = int(fraction * len(piece))
            sentences = split_sentences(piece)
            consumed, kept = 0, sentences
            for k, sentence in enumerate(sentences):
                if position < consumed + len(sentence) + 1:
                    kept = sentences[k:]
                    break
                consumed += len(sentence) + 1
            return " ".join(kept + pieces[i + 1:])
        start = end
    return ""                                     # it had all been said


# ---- asking her to read something out --------------------------------------------
# A request that is only "read it out" is answered from what she already said,
# not by the language model. Measured with the real model: asked to "read the rest
# of that out loud" it invented new content, and asked to "read it all out" it
# announced that it would and did nothing. Strict on purpose: the WHOLE utterance
# must be the request, so "read the rest of the story" and "read the log file" go
# to the model as before.
_FILLER = (r"(?:(?:ok|okay|hey|astrid|please|just|now|then|can you|could you|would you|will you|"
           r"go ahead and|i want you to|i would like you to|i'd like you to|i need you to)\s+)*")
_VERB = r"(?:read|say|speak|tell|play)\s+(?:me\s+)?"
_TAIL = r"(?:\s+(?:out|aloud|out loud|to me|for me|again|please|now))*"
_REST = (r"(?:the\s+rest(?:\s+of\s+(?:it|that|this|the\s+(?:reply|answer|response|text)))?|"
         r"the\s+remainder|the\s+remaining(?:\s+(?:part|bit|text))?|what(?:'s|\s+is)\s+left)")
_ALL = (r"(?:it(?:\s+all)?|that(?:\s+all)?|this(?:\s+all)?|all\s+of\s+(?:it|that|this)|all|everything|"
        r"(?:the\s+)?(?:whole|entire|full)(?:\s+(?:thing|reply|answer|response|text))?|"
        r"the\s+(?:reply|answer|response|text))")
_CONTINUE = r"(?:keep\s+(?:reading|going)|continue(?:\s+reading)?|go\s+on|read\s+on|carry\s+on)"
_REST_RE = re.compile("^" + _FILLER + _VERB + _REST + _TAIL + "$")
_ALL_RE = re.compile("^" + _FILLER + _VERB + _ALL + _TAIL + "$")
_CONTINUE_RE = re.compile("^" + _FILLER + _CONTINUE + _TAIL + "$")
_ALOUD_RE = re.compile(
    r"\b(?:out\s+loud|aloud|read\s+(?:it|that|this|them|these|those|the\s+rest|all|everything|"
    r"the\s+(?:whole|entire|full))|(?:say|speak)\s+(?:it|that|the\s+rest|all|everything)\b)")


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def read_request(text: str) -> str | None:
    """What `text` asks for, if it is nothing but a request to read out what she already
    said: "rest" (explicitly the remainder), "more" (go on, keep reading: the rest if
    there is any, otherwise just conversation) or "all"; else None."""
    t = _normalise(text)
    if _REST_RE.match(t):
        return "rest"
    if _CONTINUE_RE.match(t):
        return "more"
    if _ALL_RE.match(t):
        return "all"
    return None


def mentions_reading_aloud(text: str) -> bool:
    """The listener asked for something to be read OUT, inside a larger request
    ("read me the log out loud"). The reply is then spoken in full, not cut."""
    return bool(_ALOUD_RE.search(_normalise(text)))
