"""Token budgeting for Astrid's conversation. Pure Python: no GTK, no network.

Why this exists: the model runs with a fixed context window (persona.LLM_OPTIONS
["num_ctx"]). The system prompt and the tool schemas are always sent, and the
history used to be limited only by MESSAGE COUNT. One large upload, or a 50 KB
file read, could therefore push the system prompt out of the window, which is
exactly how a persona quietly stops working. Everything here exists to keep the
total under the window.

The estimator is content-aware because a single characters-per-token ratio is
badly wrong. Measured against the real model (30 samples, 2026-10-09):

    prose, code, JSON ........... 3.6 to 4.5 chars per token
    log lines, markdown tables .. 1.8 to 2.0
    UUIDs, hashes, IPs, base64 .. 1.1 to 1.4
    numeric CSV ................. 1.0

so a flat ratio either wastes most of the budget or overflows on data files.
The weights below are fitted by tools/calibrate_tokens.py (non-negative least
squares, validated leave-one-out) and pinned by tests/test_context_budget.py
against the recorded measurements in tests/fixtures/token_samples.json.

Known limit: non-ASCII text is charged a flat 1 token per character, which is
safe but over-counts Cyrillic and CJK by 1.5x to 3x.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

# --- fitted weights: tokens per unit of each feature -----------------------------
# Refit with `tools/calibrate_tokens.py --refit` (or re-measure) whenever the model
# or the feature definition changes. W_OTHER is fixed, not fitted: too few non-ASCII
# samples exist to fit it, and 1.0 is the safe direction.
W_WORD = 1.330          # per run of letters (a "word")
W_DIGIT = 1.291         # per digit
W_PUNCT = 0.109         # per punctuation character
W_ID = 0.816            # per char of a letters+digits run of 8+ chars (ids, hashes)
W_NEWLINE = 2.475       # per newline
W_OTHER = 1.0           # per non-ASCII character (fixed, conservative)
SAFETY_MARGIN = 1.32    # covers content unlike the calibration samples
FLOOR_CHARS_PER_TOKEN = 6.0   # never estimate fewer tokens than chars / 6
ID_RUN_MIN_LENGTH = 8
MESSAGE_OVERHEAD_TOKENS = 6   # role markers and template per message
REPLY_RESERVE_TOKENS = 1500   # room left for the model's own answer
MIN_USEFUL_TOKENS = 400       # below this a turn should stop calling tools

_ALNUM = re.compile(rb"[A-Za-z0-9]+")
_SPACES = b" \t\r"


def char_features(text: str) -> tuple[int, int, int, int, int, int]:
    """Return (words, digits, punctuation, id_chars, newlines, non_ascii).

    * words: runs of letters. A run that mixes letters and digits and is at least
      ID_RUN_MIN_LENGTH long (an id, hash, key) is counted in id_chars instead.
    * digits: every digit in pure-digit runs and in short mixed runs.
    """
    ascii_bytes = text.encode("ascii", "ignore")
    non_ascii = len(text) - len(ascii_bytes)
    words = digits = id_chars = consumed = 0
    for match in _ALNUM.finditer(ascii_bytes):
        run = match.group()
        consumed += len(run)
        n_digits = sum(48 <= b <= 57 for b in run)
        n_letters = len(run) - n_digits
        if n_digits and n_letters and len(run) >= ID_RUN_MIN_LENGTH:
            id_chars += len(run)
        elif n_letters:
            words += 1
            digits += n_digits
        else:
            digits += n_digits
    newlines = ascii_bytes.count(b"\n")
    spaces = len(ascii_bytes) - len(ascii_bytes.translate(None, _SPACES))
    punct = len(ascii_bytes) - consumed - spaces - newlines
    return words, digits, punct, id_chars, newlines, non_ascii


def raw_estimate(features: Sequence[int]) -> float:
    """Weighted token count for a feature tuple, before margin and floor."""
    words, digits, punct, id_chars, newlines, non_ascii = features
    return (W_WORD * words + W_DIGIT * digits + W_PUNCT * punct + W_ID * id_chars
            + W_NEWLINE * newlines + W_OTHER * non_ascii)


def estimate_tokens(text: str) -> int:
    """Conservative estimate of the tokens `text` costs. Never negative.

    The floor stops degenerate input (a megabyte of spaces, or one endless run of
    letters) from estimating near zero.
    """
    if not text:
        return 0
    weighted = raw_estimate(char_features(text))
    floor = len(text) / FLOOR_CHARS_PER_TOKEN
    return max(1, math.ceil(max(weighted, floor) * SAFETY_MARGIN))


def message_tokens(message: dict) -> int:
    """Estimate one chat message, including any tool-call arguments."""
    total = MESSAGE_OVERHEAD_TOKENS + estimate_tokens(str(message.get("content") or ""))
    calls = message.get("tool_calls")
    if calls:
        total += estimate_tokens(json.dumps(calls, ensure_ascii=False))
    return total


def messages_tokens(messages: Sequence[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def history_budget(num_ctx: int, system_text: str, tool_schemas: object,
                   reply_reserve: int = REPLY_RESERVE_TOKENS) -> int:
    """Tokens left for conversation history plus the current turn.

    The system prompt and tool schemas are sent on every request, so they come
    off the top, along with room for the reply.
    """
    fixed = (MESSAGE_OVERHEAD_TOKENS + estimate_tokens(system_text)
             + estimate_tokens(json.dumps(tool_schemas, ensure_ascii=False)))
    return max(0, num_ctx - fixed - reply_reserve)


# --- trimming ----------------------------------------------------------------
def split_turns(messages: Sequence[dict]) -> list[list[dict]]:
    """Group messages into turns: each starts at a "user" message and includes
    the assistant and tool messages that follow, so a tool_calls message is never
    separated from its tool result. Leading non-user messages are dropped.
    """
    turns: list[list[dict]] = []
    for message in messages:
        if message.get("role") == "user":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    return turns


def trim_to_budget(messages: Sequence[dict], *, max_messages: int,
                   max_tokens: int) -> list[dict]:
    """Drop the oldest whole turns until the history fits both limits.

    * messages[0] (the system prompt) is always kept.
    * Only whole turns are dropped, never part of one.
    * The newest turn is always kept, even if it alone is over budget: dropping
      the question being answered would be worse. Callers size the current turn
      (attachments, tool output) so that does not happen; see over_budget().
    Returns a new list; the input is not modified.
    """
    if not messages:
        return []
    system, rest = messages[0], messages[1:]
    turns = split_turns(rest)
    while len(turns) > 1 and sum(len(t) for t in turns) > max_messages:
        turns.pop(0)
    while len(turns) > 1 and sum(messages_tokens(t) for t in turns) > max_tokens:
        turns.pop(0)
    return [system] + [m for turn in turns for m in turn]


def over_budget(messages: Sequence[dict], *, max_tokens: int) -> bool:
    """True if the history after the system prompt exceeds `max_tokens`."""
    return messages_tokens(messages[1:]) > max_tokens


# --- clipping ----------------------------------------------------------------
def clip_to_tokens(text: str, max_tokens: int) -> str:
    """Longest prefix of `text` whose estimate fits in `max_tokens`.

    Because the estimate is never below len/FLOOR_CHARS_PER_TOKEN, no prefix longer
    than max_tokens * FLOOR_CHARS_PER_TOKEN can fit, so the search never has to
    scan more than that, however large `text` is.
    """
    if max_tokens <= 0:
        return ""
    text = text[: int(max_tokens * FLOOR_CHARS_PER_TOKEN) + 1]
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[:low]


CLIP_NOTE = "[output clipped to fit the conversation budget; ask for a narrower slice]"
PAGED_CLIP_NOTE = ("Only the first part of this window fit. It ends at "
                   "next_offset: continue from there if you still need more.")


class TurnBudget:
    """Tokens a single turn may still add, spent by tool results.

    Each tool result is clipped to what is left. When the budget runs low,
    `exhausted` tells the caller to stop offering tools and take its final
    answer pass instead of piling up more output.
    """

    def __init__(self, tokens_available: int) -> None:
        self._left = max(0, int(tokens_available))

    @property
    def remaining(self) -> int:
        return self._left

    @property
    def exhausted(self) -> bool:
        return self._left < MIN_USEFUL_TOKENS

    def charge(self, tokens: int) -> None:
        """Spend tokens on something other than a clipped result: the tool-call
        message the model wrote, or a message's fixed overhead."""
        self._left = max(0, self._left - int(tokens))

    def clip(self, text: str, note: str = CLIP_NOTE) -> str:
        """Return `text`, clipped to the remaining budget, and charge for it."""
        # A huge string cannot fit: clip before estimating so the cost of
        # deciding is bounded by the budget, not by the size of the input.
        if len(text) <= self._left * FLOOR_CHARS_PER_TOKEN:
            cost = estimate_tokens(text)
            if cost <= self._left:
                self._left -= cost
                return text
        keep = max(0, self._left - estimate_tokens(note) - MESSAGE_OVERHEAD_TOKENS)
        clipped = clip_to_tokens(text, keep)
        self._left = 0
        return clipped + "\n" + note


def clip_tool_result(result: dict, budget: TurnBudget) -> str:
    """The string the model will see for a tool result, clipped to `budget`.

    Clipping the serialised JSON would cut it mid-string and leave something that
    does not parse, and would lose the fields that matter most (the path, the
    next_offset, the exit code). So the single largest text field is shortened
    instead and everything else arrives intact, with "clipped": true set so the
    model knows it did not get all of it.
    """
    text = json.dumps(result, ensure_ascii=False)
    if estimate_tokens(text) <= budget.remaining:
        return budget.clip(text)               # fits: charged for, unchanged
    fields = [k for k, v in result.items() if isinstance(v, str)]
    if not fields:
        return budget.clip(text)
    field = max(fields, key=lambda k: len(result[k]))
    # A paged read (read_file) carries the byte offset its text starts at. If the
    # text is clipped, next_offset must move back to where the clip fell, or the
    # model would resume past the part it never received and skip it silently.
    paged = field == "content" and isinstance(result.get("offset"), int)
    shell = dict(result, clipped=True)
    shell[field] = CLIP_NOTE
    if paged:
        shell.update(next_offset=10**9, truncated=True, note=PAGED_CLIP_NOTE)
    room = budget.remaining - estimate_tokens(json.dumps(shell, ensure_ascii=False))
    for _ in range(4):                          # escaping makes the estimate drift
        if room <= 0:
            break
        kept = clip_to_tokens(result[field], room)
        shell[field] = kept + "\n" + CLIP_NOTE
        if paged:
            shell["next_offset"] = result["offset"] + len(kept.encode("utf-8"))
        out = json.dumps(shell, ensure_ascii=False)
        over = estimate_tokens(out) - budget.remaining
        if over <= 0:
            return budget.clip(out)
        room -= over + 1
    return budget.clip(json.dumps(shell | {field: CLIP_NOTE}, ensure_ascii=False))
