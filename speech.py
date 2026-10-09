"""Planning what Astrid says aloud. Pure Python: no GTK, no audio, no numpy.

Two things were making her feel slow after the model itself was fixed. She
synthesized the WHOLE reply before any sound came out (Kokoro runs on the CPU at
about 8x real time, so 3.7 s of silence for a 475-character answer), and
analysis replies are simply long when spoken (about 28 s). This module decides
how a reply is cut up and how much of it is spoken:

  * the reply is split into chunks, so gui.py can start playing the first one
    while the rest are still being synthesized;
  * a long reply is cut after a few whole sentences, and a closing line says the
    rest is on screen. The transcript always has the full text.

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

# A reply up to this long is spoken whole (about 24 s). Past it, only the first
# few sentences are spoken. The gap between the two numbers is deliberate: a
# 401-character reply is not worth cutting to save one sentence.
SPEAK_ALL_UP_TO_CHARS = 400
# How much of a long reply is spoken, in whole sentences (about 17 s).
# 0 turns the cap off and every reply is spoken in full.
SPOKEN_LIMIT_CHARS = 280
CLOSING_LINE = "The rest is on screen."

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

    @property
    def spoken(self) -> list[str]:
        """Everything to synthesize, in order."""
        return self.chunks + ([self.closing] if self.closing else [])


def plan_speech(text: str) -> SpeechPlan:
    """Decide what is spoken, and in which pieces, for an already-cleaned reply."""
    sentences = split_sentences(text)
    if not sentences:
        return SpeechPlan()
    if SPOKEN_LIMIT_CHARS <= 0 or len(" ".join(sentences)) <= SPEAK_ALL_UP_TO_CHARS:
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
    omitted = len(" ".join(sentences[len(kept):]))
    return SpeechPlan(chunk_sentences(kept), CLOSING_LINE, omitted)
