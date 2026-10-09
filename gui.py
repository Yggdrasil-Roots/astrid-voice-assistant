#!/usr/bin/env python3
import os

# Model files are already cached locally — never phone home for updates.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Reduce CUDA allocator fragmentation — VRAM is tight when STT + SDXL briefly overlap.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Imported here, above numpy/torch/ctranslate2/onnxruntime, because
# cap_native_thread_pools() sets environment variables those libraries read
# once at load time. Two independent libgomp copies end up in this process
# (one bundled in ctranslate2, one system-wide for torch); each otherwise
# spawns nproc=24 threads and spin-waits. Measured: 8 threads synthesise an
# 11s utterance ~25% FASTER than 24. This is an efficiency fix, not a
# stability one -- main-loop tick lateness stayed under 2ms either way.
import stability  # noqa: E402
stability.cap_native_thread_pools(8)

import atexit
import collections
import functools
import subprocess
import signal
import sys
import json
import math
import queue
import re
import threading
import time

import webrtcvad
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gtk, GLib, Pango, PangoCairo, Gdk, Gio

import numpy as np
import requests
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

import pwd

import attachments as attach_mod
import auth
import context_budget
import image_gen
import inputbar
import memory
import persona
import speech
import tools

ASTRID_INSTALL_DIR = "/opt/astrid"

# Timestamped logging to the file start.sh already redirects into. Without
# timestamps there is no way to line a freeze up against dmesg or the
# journal, which is what made the libasound segfault take an afternoon.
os.makedirs(auth.ASTRID_HOME, exist_ok=True)
_LOG_PATH = os.path.join(auth.ASTRID_HOME, "astrid.log")
log = stability.setup_logging(_LOG_PATH)
# faulthandler on the fatal signals, SIGUSR1 for an on-demand dump of every
# thread, and a threading.excepthook so a dying worker is never silent:
#     kill -USR1 $(pgrep -f "python3 -u gui.py")
stability.install_diagnostics()

# How long the wake loop may make no progress before the watchdog replaces
# it. Generous: SDXL image generation legitimately blocks the loop for tens
# of seconds, and a false restart mid-generation would be worse than waiting.
WATCHDOG_INTERVAL_SECONDS = 5
WATCHDOG_STALE_SECONDS = 180
WAKE_LOOP_BACKOFF_SECONDS = 1.0


def _get_user_name():
    """Each user's own display name — from a per-user override
    file if set, else their account's real name (GECOS), else username."""
    override_file = os.path.join(auth.ASTRID_HOME, "name")
    if os.path.exists(override_file):
        with open(override_file) as f:
            name = f.read().strip()
            if name:
                return name
    try:
        gecos = pwd.getpwuid(os.getuid()).pw_gecos.split(",")[0].strip()
        if gecos:
            return gecos.split()[0].capitalize()
    except Exception:
        pass
    return pwd.getpwuid(os.getuid()).pw_name.capitalize()


SAMPLE_RATE = 16000
# qwen2.5:14b-instruct-q5_K_M was deleted from Ollama when Astrid was
# archived on 2026-08-28. qwen3:14b is what this machine runs now: 9.3 GB
# resident against the 11.5 GB the old model needed, so the VRAM arithmetic
# in CLAUDE.md is looser than it was, not tighter. It is a reasoning model,
# which is why _stream_chat sends "think": False.
OLLAMA_MODEL = "huihui_ai/qwen3-abliterated:14b"
KOKORO_MODEL = os.path.join(ASTRID_INSTALL_DIR, "kokoro-v1.0.fp16.onnx")
KOKORO_VOICES = os.path.join(ASTRID_INSTALL_DIR, "voices-v1.0.bin")
# Character, sampling and voice settings all live in persona.py so she can be
# tuned without touching the app. Original prompt preserved in gui.py.bak.
ASSISTANT_NAME = persona.ASSISTANT_NAME
# NOT persona.USER_NAME: this is the shared multi-user install, so the name
# comes from whichever account is running (see _get_user_name). persona.py
# bakes its own USER_NAME into SYSTEM_PROMPT at import time, so substitute it
# here rather than editing persona.py — that keeps persona.py byte-identical
# to the copy in ~/voice-assistant and avoids re-diverging the two trees.
USER_NAME = _get_user_name()
# Per-account opt-in, off unless the file exists. persona.py is shared by
# every account on this machine, so the flirtation section is gated here
# rather than written unconditionally into the prompt.
#   enable:  touch ~/.astrid/flirt        disable:  rm ~/.astrid/flirt
_FLIRT_FILE = os.path.join(auth.ASTRID_HOME, "flirt")
FLIRT = os.path.exists(_FLIRT_FILE)
# Announced because the toggle is otherwise undetectable: the behaviour is
# rare by design and never signalled, so "is it on?" cannot be answered by
# listening. Deliberately a log line and not UI -- see the patch script.
print(f"[astrid] flirtation {'ON' if FLIRT else 'OFF'}"
      + ("" if FLIRT else f" (touch {_FLIRT_FILE} and relaunch to enable)"))
SYSTEM_PROMPT = persona.build_system_prompt(flirt=FLIRT).replace(
    persona.USER_NAME, USER_NAME)

MAX_HISTORY_MESSAGES = 20
MAX_TOOL_ITERATIONS = 6
# How long a command-approval dialog may sit unanswered before it counts as a
# no. Long enough to read an unfamiliar command properly; short enough that
# walking away from the machine cannot park the wake-loop thread.
COMMAND_CONFIRM_TIMEOUT_SECONDS = 120
# Typed input. Spoken replies to typed messages are off unless this file exists
# (the same per-user flag idiom as ~/.astrid/flirt); the window's switch makes it.
SPEAK_TYPED_FILE = os.path.join(auth.ASTRID_HOME, "speak_typed")
# Largest share of the context budget one typed message (text plus attachments)
# may take. Older turns are dropped to leave room for the rest.
MAX_MESSAGE_BUDGET_SHARE = 0.6
# One conversational turn at a time, spoken or typed. Module level (there is one
# window per process) so the wake loop and the typed-turn thread contend on the
# same lock without either needing a new instance attribute.
TURN_LOCK = threading.Lock()
# Speech synthesis is not thread-safe (espeak-ng, behind Kokoro's phonemiser, keeps
# global state) and a reply is synthesized chunk by chunk on a second thread. Every
# tts.create() call goes through this lock, so two can never overlap -- including a
# straggler left over from a reply that was interrupted, and the next reply's first
# chunk. Module level for the same reason as TURN_LOCK.
TTS_LOCK = threading.Lock()
# How far past the audio produced so far playback may run before it is cut off.
SPEAK_GRACE_SECONDS = 30.0


def player_argv(sample_rate):
    """The pw-play command for raw 16-bit mono audio on stdin. One place, so a
    test can swap in a recorder. Raw means no header, which is what lets chunks
    be appended to one running stream without a process start in between."""
    return ["pw-play", "--raw", "--rate", str(sample_rate), "--channels", "1",
            "--format", "s16", "-"]


def _one_turn_at_a_time(method):
    """Run `method` only if no other turn is in progress; otherwise drop it."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not TURN_LOCK.acquire(blocking=False):
            log.info("%s dropped: another turn is in progress", method.__name__)
            return False
        try:
            return method(self, *args, **kwargs)
        finally:
            TURN_LOCK.release()
    return wrapper

RUNES = "ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ"

# Norse forge / aurora palette (0..1 floats for Cairo)
BG = (0.035, 0.043, 0.055)
PANEL = (0.055, 0.066, 0.082)
GOLD = (0.83, 0.68, 0.25)
GOLD_DIM = (0.42, 0.34, 0.14)
ICE = (0.42, 0.85, 0.80)
TEXT = (0.90, 0.87, 0.80)
TEXT_DIM = (0.55, 0.52, 0.48)

STATE_LOADING = "loading"
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"
STATE_IMAGINING = "imagining"

STATUS_TEXT = {
    STATE_LOADING: "AWAKENING...",
    STATE_IDLE: "SAY \"ASTRID\" OR PRESS TO SPEAK",
    STATE_LISTENING: "LISTENING...",
    STATE_THINKING: "THINKING...",
    STATE_SPEAKING: "SPEAKING...",
    STATE_IMAGINING: "CONJURING AN IMAGE...",
}

# States in which she is busy on a turn: the status line shows what she is doing
# and counts the seconds, instead of one unchanging word.
WORKING_STATES = (STATE_THINKING, STATE_IMAGINING)

GENERATED_DIR = os.path.join(auth.ASTRID_HOME, "generated")

# -- wake word / VAD ----------------------------------------------------
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480
VAD_AGGRESSIVENESS = 2
VAD_TRIGGER_RATIO = 0.6
VAD_PRE_ROLL_FRAMES = 10  # ~300ms
VAD_SILENCE_HANGOVER_MS = 900
MIN_UTTERANCE_SECONDS = 0.3
CONVERSATION_WINDOW_SECONDS = 8
# Spoken on launch. She then listens for the reply for CONVERSATION_WINDOW_SECONDS
# before falling back to requiring the wake phrase.
OPENING_QUESTION = f"What request may I facilitate for you, {USER_NAME}?"


def is_wake_phrase(text):
    return "astrid" in text.lower()


CSS = """
window {
    background-color: #090b0e;
}
.title-label {
    color: #d4ad40;
    font-size: 30px;
    font-weight: bold;
    letter-spacing: 8px;
}
.subtitle-label {
    color: #6bd9cc;
    font-size: 11px;
    letter-spacing: 4px;
}
.status-label {
    color: #8c857a;
    font-size: 12px;
    letter-spacing: 3px;
}
.transcript {
    background-color: #0d1015;
    border: 1px solid #262b33;
    border-radius: 6px;
}
.transcript text {
    background-color: #0d1015;
    color: #e6ddcc;
    font-family: monospace;
    padding: 8px;
}
.image-frame {
    border: 1px solid #665221;
    border-radius: 6px;
    background-color: #0d1015;
}
.lock-prompt {
    color: #d4ad40;
    font-size: 14px;
    letter-spacing: 3px;
}
.lock-error {
    color: #e05a4f;
    font-size: 12px;
    letter-spacing: 2px;
}
.pin-entry {
    font-size: 22px;
    letter-spacing: 10px;
}
"""


def rms_level(chunk):
    return float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0


# Reasoning models wrap their scratchpad in <think>...</think>. _stream_chat
# asks Ollama to suppress it, but that key is ignored by older versions and by
# any model that emits the block regardless, so it is also stripped here --
# belt and braces, because the failure mode is her reading her own reasoning
# aloud for several minutes. DOTALL: the block spans lines.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)

# Markdown that survives VOICE_RULES, stripped before it reaches kokoro.
_MD_BULLET = re.compile(r"^[ \t]*[-*+•][ \t]+", re.M)
_MD_NUMBER = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.M)
_MD_HEADER = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.M)
_MD_EMPH = re.compile(r"(\*{1,3}|_{2,3})(.+?)\1", re.S)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")

# Number forms espeak-ng mispronounces. See patch-opt-speech-numbers.sh for the
# measurements -- plain integers are deliberately NOT touched, because they are
# already phoneme-identical to their spelled-out forms.
#
# Dotted quads first: four groups is an IP, and "dot" is how it is said. Two
# groups is a decimal or a version, both of which want "point" (Ubuntu 26.04 is
# spoken "twenty six point oh four", so the same rule serves both).
_NUM_DOTTED_QUAD = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
_NUM_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
# Only a comma sitting between digits with exactly three after it, so "1,024"
# is fixed while "for 3, 4 and 5" is left alone.
_NUM_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _speak_numbers(text):
    """Rewrite the number forms espeak keeps as punctuation rather than speech.

    Deliberately narrow: only the separators inside numbers. Plain integers,
    percentages, ordinals and negatives are already correct and are left
    untouched -- see the measurements in patch-opt-speech-numbers.sh.
    """
    text = _NUM_DOTTED_QUAD.sub(r"\1 dot \2 dot \3 dot \4", text)
    text = _NUM_THOUSANDS.sub("", text)
    text = _NUM_DECIMAL.sub(" point ", text)
    return text


def clean_for_speech(text):
    """Strip markup before TTS.

    persona.VOICE_RULES already forbids markdown, and she mostly complies --
    but relaying web_search results reliably produces bullet lists anyway, and
    kokoro reads "-" and "*" out as words. Prompt-level rules were measured as
    unreliable for this class of problem on 2026-08-17, so this is done in code
    where it is deterministic and costs no context.

    Lines are joined into flowing prose rather than dropped, because a list is
    still the answer -- it just has to be heard as a sentence. Hyphenated words
    and em-dash punctuation are deliberately left alone; only leading bullet
    markers are removed.
    """
    text = _THINK_BLOCK.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_EMPH.sub(r"\2", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_NUMBER.sub("", text)
    text = text.replace("`", "")
    text = _speak_numbers(text)

    out = []
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if out and not out[-1][-1] in ".!?:;,":
            out[-1] += "."
        out.append(line)
    spoken = re.sub(r"[ \t]{2,}", " ", " ".join(out)).strip()
    # Last step before synthesis: rewrite the handful of words espeak-ng reads
    # wrong in en-us. Speech only -- append_transcript() gets the real text.
    return stability.fix_pronunciation(spoken)


class Core(Gtk.DrawingArea):
    """Custom-drawn animated circular HUD — the assistant's visual 'presence'."""

    def __init__(self):
        super().__init__()
        self.set_content_width(300)
        self.set_content_height(300)
        self.state = STATE_LOADING
        self.level = 0.0
        self.phase = 0.0
        self.envelope = None
        self.envelope_rate = 0.0
        self.playback_start = 0.0
        self.set_draw_func(self.on_draw)
        GLib.timeout_add(33, self._tick)

    def _tick(self):
        self.phase += 0.033
        if self.state == STATE_SPEAKING and self.envelope is not None:
            elapsed = time.time() - self.playback_start
            idx = int(elapsed * self.envelope_rate)
            self.level = float(self.envelope[idx]) if 0 <= idx < len(self.envelope) else 0.0
        self.queue_draw()
        return True

    def set_state(self, state):
        self.state = state
        if state in (STATE_IDLE, STATE_THINKING, STATE_IMAGINING):
            self.level = 0.0

    def on_draw(self, area, cr, w, h):
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 12

        cr.set_source_rgb(*BG)
        cr.paint()

        state = self.state
        if state == STATE_LOADING:
            pulse = 0.5 + 0.5 * math.sin(self.phase * 3.0)
            core_r, ring_color, core_color, alpha, rotate = (
                radius * 0.22, GOLD_DIM, GOLD_DIM, 0.5 + 0.2 * pulse, self.phase * 2.2,
            )
        elif state == STATE_IDLE:
            pulse = 0.5 + 0.5 * math.sin(self.phase * 1.1)
            core_r, ring_color, core_color, alpha, rotate = (
                radius * 0.32 * (0.92 + 0.08 * pulse), GOLD_DIM, GOLD, 0.45 + 0.25 * pulse, self.phase * 0.05,
            )
        elif state == STATE_LISTENING:
            core_r, ring_color, core_color, alpha, rotate = (
                radius * 0.32 * (0.9 + 0.6 * self.level), ICE, ICE, 0.55 + 0.45 * self.level, self.phase * 0.2,
            )
        elif state == STATE_THINKING:
            core_r, ring_color, core_color, alpha, rotate = (
                radius * 0.30, GOLD, GOLD, 0.65, self.phase * 1.8,
            )
        elif state == STATE_IMAGINING:
            blend = ((GOLD[0] + ICE[0]) / 2, (GOLD[1] + ICE[1]) / 2, (GOLD[2] + ICE[2]) / 2)
            pulse = 0.5 + 0.5 * math.sin(self.phase * 2.4)
            core_r, ring_color, core_color, alpha, rotate = (
                radius * 0.30 * (0.9 + 0.1 * pulse), blend, blend, 0.6 + 0.25 * pulse, self.phase * 2.6,
            )
        else:  # speaking
            blend = ((GOLD[0] + ICE[0]) / 2, (GOLD[1] + ICE[1]) / 2, (GOLD[2] + ICE[2]) / 2)
            core_r, ring_color, core_color, alpha, rotate = (
                radius * 0.32 * (0.9 + 0.6 * self.level), GOLD, blend, 0.6 + 0.4 * self.level, self.phase * 0.7,
            )

        # decorative rune ring
        cr.save()
        cr.translate(cx, cy)
        n = 12
        for i in range(n):
            a = rotate * 0.15 + i * (2 * math.pi / n)
            rx, ry = math.cos(a) * radius * 0.90, math.sin(a) * radius * 0.90
            cr.save()
            cr.translate(rx, ry)
            cr.rotate(a + math.pi / 2)
            cr.set_source_rgba(*ring_color, 0.55)
            layout = PangoCairo.create_layout(cr)
            layout.set_font_description(Pango.FontDescription("Noto Sans Runic 15"))
            layout.set_text(RUNES[i % len(RUNES)], -1)
            tw, th = layout.get_pixel_size()
            cr.translate(-tw / 2, -th / 2)
            PangoCairo.show_layout(cr, layout)
            cr.restore()
        cr.restore()

        # soft glow (layered low-alpha circles, cheap bloom approximation)
        cr.save()
        cr.translate(cx, cy)
        for i, mult in enumerate((1.7, 1.4, 1.15)):
            cr.set_source_rgba(*ring_color, 0.05 * (i + 1) * (0.5 + 0.5 * alpha))
            cr.arc(0, 0, core_r * mult, 0, 2 * math.pi)
            cr.fill()
        cr.restore()

        # rotating segmented ring
        cr.save()
        cr.translate(cx, cy)
        cr.set_line_width(2.5)
        segs = 36
        for i in range(segs):
            a0 = rotate + i * (2 * math.pi / segs)
            a1 = a0 + (2 * math.pi / segs) * 0.55
            if state in (STATE_THINKING, STATE_IMAGINING):
                spin_speed = 1.8 if state == STATE_THINKING else 2.6
                trail = (i / segs * 2 * math.pi - (self.phase * spin_speed) % (2 * math.pi)) % (2 * math.pi)
                seg_alpha = max(0.05, 1.0 - trail / (math.pi * 0.6))
            else:
                seg_alpha = 0.30
            cr.new_sub_path()
            cr.set_source_rgba(*ring_color, min(seg_alpha, 0.9) * (0.6 + 0.4 * alpha))
            cr.arc(0, 0, radius * 0.72, a0, a1)
            cr.stroke()
        cr.restore()

        # core
        cr.save()
        cr.translate(cx, cy)
        for i, mult in enumerate((1.0, 0.72, 0.46, 0.22)):
            a = min(1.0, alpha * (0.28 + 0.22 * i))
            cr.set_source_rgba(*core_color, a)
            cr.arc(0, 0, max(core_r * mult, 1), 0, 2 * math.pi)
            cr.fill()
        cr.restore()


class _PwRecordStream:
    """Microphone capture as a subprocess, not as PortAudio.

    Every segfault in this machine's logs shares one signature: libasound
    snd_pcm_poll_descriptors_revents, called from libportaudio. Playback was
    moved to pw-play for that reason; this is the other half.

    Capture was believed exempt for a while, and that was wrong -- it cost a
    debugging cycle. `with sd.InputStream(...)` per utterance is the SAME
    open/close churn against the SAME pipewire ALSA shim. It is only slower
    to bite, because it is one open and one close per conversational turn
    rather than per sentence. So she would survive a handful of turns and then
    die with the identical stack. The symptom reads as "freezes after a short
    conversation" rather than "crashes at startup", which is what makes it so
    easy to misattribute to whatever changed most recently.

    One subprocess owns the device for the whole utterance. There is no
    callback to race and no shared C state in this process to corrupt.
    """

    def __init__(self, rate, frame_samples, latency_ms=VAD_FRAME_MS):
        self.rate = rate
        self.frame_bytes = frame_samples * 2   # s16 mono
        self.latency_ms = latency_ms
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["pw-record", "--raw", "--rate", str(self.rate), "--channels", "1",
             "--format", "s16", "--latency", "%dms" % self.latency_ms, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, bufsize=0)
        return self

    def read_frame(self):
        """One VAD frame, blocking. pw-record writes at real time, so this
        paces the loop by itself -- measured at 33 frames in 1.04s at
        16kHz/30ms.

        Loops until the frame is full. This is not optional and was got wrong
        once: bufsize=0 above makes proc.stdout a raw io.FileIO, and
        FileIO.read(n) is a SINGLE read(2) syscall that returns up to n bytes
        -- whatever happens to be in the pipe at that instant -- rather than
        filling the buffer the way a BufferedReader would. The first live run
        died instantly with "read 24 of 960 bytes": pw-record was perfectly
        healthy, 24 bytes was just the first fragment to arrive.

        Only a genuine EOF (an empty read) means the child died. Raise then,
        rather than returning silence: the wake loop's per-iteration handler
        turns a raise into one missed turn, whereas silently returning zeros
        makes her go quietly deaf, which is the failure nobody notices.
        """
        buf = bytearray()
        while len(buf) < self.frame_bytes:
            chunk = self.proc.stdout.read(self.frame_bytes - len(buf))
            if not chunk:   # b"" is EOF; None would mean non-blocking, which
                            # this pipe is not, but both mean "no more audio"
                raise RuntimeError("pw-record stopped producing audio "
                                   "(got %d of %d bytes before EOF)"
                                   % (len(buf), self.frame_bytes))
            buf += chunk
        return bytes(buf)

    def __exit__(self, *exc):
        if self.proc is None:
            return False
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        return False


def _system_prompt_with_memory():
    """Her prompt with the notes she has been asked to keep, and the file's mtime so a
    hand-edit can be noticed later. A module function, not a method, so the
    stand-in objects the older tests use are not required to grow new attributes."""
    stamp = memory.mtime()
    try:
        block = memory.render_block(memory.load(), persona.USER_NAME)
    except memory.MemoryFileError:
        log.warning("could not read the memory file; starting without it", exc_info=True)
        block = ""
    prompt = persona.build_system_prompt(flirt=FLIRT, memory=block).replace(persona.USER_NAME, USER_NAME)
    return prompt, stamp


def _refresh_memory_for(window, force=False):
    """Rebuild her prompt if the notes changed (a note was kept or forgotten, or the
    file was edited by hand). The notes are part of the system prompt, which comes out
    of the room left for the conversation, so the history budget is recomputed too.

    A no-op on an object that never loaded memory, which is what keeps the older
    stand-in windows in the tests working unchanged. Never raises: a failed refresh
    must not break a turn.
    """
    if getattr(window, "_memory_mtime", None) is None:
        return
    try:
        if not force and memory.mtime() == window._memory_mtime:
            return
        prompt, stamp = _system_prompt_with_memory()
        window._memory_mtime = stamp
        if window.messages and window.messages[0].get("role") == "system":
            window.messages[0]["content"] = prompt
        window._history_budget = context_budget.history_budget(
            persona.LLM_OPTIONS["num_ctx"], prompt, tools.TOOL_SCHEMAS)
    except Exception:
        log.exception("could not refresh her memory")


def _remember_on_request(window, text):
    """If `text` is "remember that ...", keep it (after his click) and return what to say.

    Done in code because the model does not: asked outright, it said "Noted. I will keep it in
    mind" and called no tool, 5 times in 5 (measured), which is a promise that is false the moment
    the window closes. The note is his own words, so standing preferences are allowed; the click,
    the secret check and the approval-bypass check still apply. Returns None if `text` is not that.
    """
    note = memory.remember_request(text)
    if not note:
        return None
    result = tools.remember(note, "you asked her to remember it", trusted=True)
    status = result.get("status")
    if status == "remembered":
        reply = "Kept."
    elif status == "already known":
        reply = "I already had that."
    elif status == "declined":
        reply = "Understood. I will not keep it."
    else:
        reply = result.get("error") or "I could not keep that."
    # The exchange is part of the conversation: "what did I just ask you to remember?" works.
    window.messages.append({"role": "user", "content": text})
    window.messages.append({"role": "assistant", "content": reply})
    return reply


def _honour_promise(window, user_text, reply):
    """If her reply promises to remember something and she did not actually call remember, make
    the promise true by asking (a click, as always), or correct it. See memory.promises_to_remember
    for why. Returns the reply, with the real outcome added when this fired.
    """
    if not reply or not memory.promises_to_remember(reply) or tools.remember_was_called():
        return reply
    note = memory.statement_about_him(user_text)
    if not note:
        return reply            # "Noted" about a command or a question: nothing to keep
    result = tools.remember(note, "She said she would keep this in mind.", trusted=True)
    status = result.get("status")
    if status == "remembered":
        said = "I have kept that."
    elif status == "already known":
        said = "I already had that."
    elif status == "declined":
        said = "I have not kept it."
    else:
        said = result.get("error") or "I could not keep that."
    if window.messages and window.messages[-1].get("role") == "assistant":
        window.messages[-1]["content"] = (window.messages[-1].get("content") or "") + " " + said
    return reply.rstrip() + " " + said


NOTHING_LEFT = "That was all of it."


def _earlier_reply_to_read_for(window, text):
    """What to read out if `text` is just a request to read out what she said.

    "the rest" is what was cut off by the length limit or left unsaid when she was
    stopped. "go on" is the same when there is some, and ordinary conversation when
    there is not ("go on" after a finished answer means elaborate, for the model).
    "read the rest" with nothing left gets a short spoken answer instead of the
    model, which would invent a rest.
    """
    wanted = speech.read_request(text)
    remaining = getattr(window, "_last_rest", "")
    if wanted == "rest":
        if remaining:
            return remaining
        return NOTHING_LEFT if getattr(window, "_last_full", "") else ""
    if wanted == "more":
        return remaining
    if wanted == "all":
        return getattr(window, "_last_full", "")
    return ""


class _PwPlayStream:
    """Speech playback as ONE pw-play subprocess for a whole reply.

    The reason playback is a subprocess at all: PortAudio segfaulted ten times in
    libasound, and a child process owning the device leaves no C state in this
    process to corrupt. Stop and barge-in are terminate(); there is no callback
    to race. That is kept exactly. What is new is that audio arrives in pieces
    while the reply is still being synthesized, so this is a stream you write to
    rather than a file handed over whole.

    write() blocks at the pipe's capacity (pw-play reads at playback rate), so it
    belongs on its own thread and never on the one that must notice a barge-in.
    """

    def __init__(self, sample_rate):
        self.proc = subprocess.Popen(
            player_argv(sample_rate), stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, data):
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def end_input(self):
        """EOF: pw-play finishes what it has been given, then exits."""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def running(self):
        return self.proc.poll() is None

    def stop(self):
        try:
            self.proc.terminate()
        except OSError:
            pass

    def close(self):
        """Make sure it is gone. Terminates first if it is still alive, so a
        writer blocked on a full pipe is released before stdin is closed."""
        if self.running():
            self.stop()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        self.end_input()


class _Envelope:
    """The loudness curve the orb follows, built up as chunks are synthesized.

    Chunks play back to back, so their curves join end to end. Any samples left
    over after the last whole block are carried into the next chunk, so the join
    never drifts: block i always means the same 1/30 s of the stream.
    """

    def __init__(self, sample_rate):
        self.block = max(1, sample_rate // 30)
        self.rate = sample_rate / self.block
        self._levels = np.zeros(0, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)

    def add(self, samples):
        data = np.concatenate([self._carry, np.asarray(samples, dtype=np.float32)])
        whole = len(data) // self.block
        if whole:
            blocks = data[:whole * self.block].reshape(whole, self.block)
            self._levels = np.concatenate(
                [self._levels, np.sqrt(np.mean(np.square(blocks), axis=1))])
        self._carry = data[whole * self.block:]

    def snapshot(self):
        """The curve so far, scaled so its loudest point is 1. A fresh array each
        time: the GTK thread reads whichever one it was last handed."""
        levels = self._levels
        if len(self._carry):
            levels = np.append(levels, rms_level(self._carry))
        peak = float(levels.max()) if len(levels) else 0.0
        return np.clip(levels / peak, 0.0, 1.0).astype(np.float32) if peak > 0 else levels.copy()


class AstridWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title=ASSISTANT_NAME)
        self.set_default_size(460, 800)

        self.stt = None
        self.tts = None
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.manual_trigger_event = threading.Event()
        self.cancel_event = threading.Event()
        # tools.py cannot import this module (gui imports tools), so the
        # approval prompt is handed down instead. Without this call, anything
        # off the read-only allowlist is refused rather than run.
        tools.set_confirm_handler(self._confirm_command)
        self._current_session = None
        # Barge-in / screen-lock signal to the playback thread. The dev tree
        # set this flag in both handlers but never created or read it, so
        # barge-in there raises AttributeError; that half-finished fix is why
        # it never reached /opt.
        self.stop_speaking = threading.Event()
        self.heartbeat = stability.Heartbeat("initialising")
        # Same inversion as set_confirm_handler above, and for a related
        # reason: a command runs on the wake-loop thread, and the watchdog
        # replaces that thread after WATCHDOG_STALE_SECONDS of no progress.
        # Without a beat, a slow `find` is indistinguishable from a hang --
        # the exact shape of the 2026-08-22 segfault, where the watchdog fired
        # during legitimate work and started a second wake loop.
        tools.set_heartbeat_handler(self.heartbeat.beat)
        self._wake_thread = None
        self._wake_generation = 0
        prompt, self._memory_mtime = _system_prompt_with_memory()
        self.messages = [{"role": "system", "content": prompt}]
        # Tokens left for history plus the current turn once the system prompt (now
        # including her notes), tool schemas and a reply reserve are subtracted.
        # History used to be limited by message COUNT only, so one big paste could
        # push the system prompt out of the model's context. See context_budget.py.
        self._history_budget = context_budget.history_budget(
            persona.LLM_OPTIONS["num_ctx"], prompt, tools.TOOL_SCHEMAS)
        tools.set_memory_changed_handler(lambda: _refresh_memory_for(self, force=True))
        self._speak_typed = os.path.exists(SPEAK_TYPED_FILE)
        os.makedirs(GENERATED_DIR, exist_ok=True)
        os.chmod(auth.ASTRID_HOME, 0o700)
        os.chmod(GENERATED_DIR, 0o700)

        # No PIN gate: she opens straight into the main view and starts
        # loading. `locked` is kept but repurposed -- it now means only "the
        # OS screen is locked". It still gates the mic and cuts speech
        # mid-sentence in that case, and clears again by itself when the
        # session unlocks, so a locked machine cannot be talked to but the
        # owner is never asked for anything.
        self.locked = False
        self._models_loading_started = True

        self.stack = Gtk.Stack()
        self.set_child(self.stack)
        self.stack.add_named(self._build_main_page(), "main")
        self.stack.set_visible_child_name("main")

        self._watch_screensaver()
        threading.Thread(target=self._load_models, daemon=True, name="model-load").start()
        GLib.timeout_add_seconds(WATCHDOG_INTERVAL_SECONDS, self._watchdog_tick)

    def _build_main_page(self):
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        root.set_margin_top(24)
        root.set_margin_bottom(20)
        root.set_margin_start(24)
        root.set_margin_end(24)

        title = Gtk.Label(label=ASSISTANT_NAME.upper())
        title.add_css_class("title-label")
        root.append(title)

        subtitle = Gtk.Label(label="LOCAL INTELLIGENCE · WARDED")
        subtitle.add_css_class("subtitle-label")
        subtitle.set_margin_bottom(18)
        root.append(subtitle)

        self.core = Core()
        core_wrap = Gtk.Box(halign=Gtk.Align.CENTER)
        core_wrap.append(self.core)
        root.append(core_wrap)

        click = Gtk.GestureClick()
        click.connect("released", self.on_core_click)
        self.core.add_controller(click)
        self.core.set_cursor(Gdk.Cursor.new_from_name("pointer"))

        self.status_label = Gtk.Label(label=STATUS_TEXT[STATE_LOADING])
        self.status_label.add_css_class("status-label")
        self.status_label.set_margin_top(14)
        self.status_label.set_margin_bottom(18)
        root.append(self.status_label)

        self.image_picture = Gtk.Picture()
        self.image_picture.add_css_class("image-frame")
        self.image_picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.image_picture.set_size_request(-1, 340)
        self.image_picture.set_margin_bottom(14)
        self.image_picture.set_visible(False)
        root.append(self.image_picture)

        scroller = Gtk.ScrolledWindow()
        scroller.set_vexpand(True)
        scroller.add_css_class("transcript")
        self.transcript_view = Gtk.TextView()
        self.transcript_view.set_editable(False)
        self.transcript_view.set_cursor_visible(False)
        self.transcript_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.transcript_buf = self.transcript_view.get_buffer()
        scroller.set_child(self.transcript_view)
        root.append(scroller)

        self.input_bar = inputbar.InputBar(
            on_submit=self._on_submit_text,
            on_stop=self._on_stop_turn,
            on_attach_paths=self._request_attach,
            on_speak_toggled=self._on_speak_toggled,
            speak_active=os.path.exists(SPEAK_TYPED_FILE),
        )
        root.append(self.input_bar)
        # A drop works anywhere on the window, not only over the bar.
        self.add_controller(self.input_bar.make_drop_target())

        return root

    # -- lock screen ------------------------------------------------------
    def _lock(self):
        """Screen locked: shut the mic and stop talking immediately.

        An in-progress reply is otherwise still audible to whoever is stood
        in front of a locked machine, which rather defeats locking it.
        """
        if self.locked:
            return False
        self.locked = True
        self._sync_input_state(self.core.state)
        self.cancel_event.set()
        # Flag only. This runs on the GTK main thread (screensaver D-Bus
        # signal -> GLib.idle_add), and calling into ALSA from here while the
        # wake thread is inside the playback stream segfaults libasound --
        # reproduced 5/5 by an earlier investigation, and almost certainly the
        # crash CLAUDE.md records for 2026-08-18. _speak() sees the flag and
        # tears the stream down on the thread that started it.
        self.stop_speaking.set()
        if self._current_session is not None:
            try:
                self._current_session.close()
            except Exception:
                pass
        return False

    def _unlock_screen(self):
        """Screen unlocked: resume. No PIN -- the OS session is the gate."""
        self.locked = False
        self.cancel_event.clear()
        self._sync_input_state(self.core.state)
        return False

    def _watch_screensaver(self):
        """Follow the OS screen lock: pause on lock, resume on unlock."""
        try:
            proxy = Gio.DBusProxy.new_for_bus_sync(
                Gio.BusType.SESSION, Gio.DBusProxyFlags.NONE, None,
                "org.gnome.ScreenSaver", "/org/gnome/ScreenSaver",
                "org.gnome.ScreenSaver", None,
            )
            proxy.connect("g-signal", self._on_screensaver_signal)
            self._screensaver_proxy = proxy  # keep alive — GC'd proxies drop their signal connection
        except GLib.Error as e:
            print(f"Screensaver watch unavailable, screen-lock auto-lock disabled: {e}")

    def _on_screensaver_signal(self, proxy, sender_name, signal_name, params):
        if signal_name == "ActiveChanged":
            (active,) = params.unpack()
            GLib.idle_add(self._lock if active else self._unlock_screen)

    # -- model loading --------------------------------------------------
    def _load_models(self):
        # cpu_threads: ctranslate2 defaults to one thread per core (24 here)
        # even on CUDA, in its own bundled libgomp that does not coordinate
        # with torch's. See stability.cap_native_thread_pools.
        self.stt = WhisperModel("distil-large-v3", device="cuda",
                                compute_type="int8_float16", cpu_threads=8)
        self.tts = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        GLib.idle_add(self._on_models_ready)

    def _on_models_ready(self):
        self._set_state(STATE_IDLE)
        threading.Thread(target=self._greet_then_listen, daemon=True).start()
        return False

    def _greet_then_listen(self):
        """Ask the opening question, then hand straight to the wake loop with
        conversation mode already on, so the answer needs no wake phrase.

        Deliberately sequential: _speak() returns only once playback has
        finished, so starting the loop after it is what stops her from
        capturing her own question. See patch-opt-opening-question.sh."""
        greeting = OPENING_QUESTION
        GLib.idle_add(self.append_transcript, ASSISTANT_NAME, greeting)
        self._speak(greeting)
        # Still sequential -- _speak() returns only once playback is done, so
        # she cannot capture her own question. The loop now gets its own named
        # thread so the watchdog can replace it.
        self._start_wake_loop(conversing=True)

    # -- UI helpers -------------------------------------------------------
    # What she is doing right now, shown on the status line while she works. Class
    # defaults so nothing depends on __init__ having run first.
    _activity = None
    _turn_started = 0.0
    _status_timer = None

    def _set_state(self, state):
        previous = self.core.state
        self.core.set_state(state)
        if state not in WORKING_STATES:
            self._activity = None
        elif previous not in WORKING_STATES:
            self._turn_started = time.monotonic()
        self._refresh_status()
        if state in WORKING_STATES and self._status_timer is None:
            self._status_timer = GLib.timeout_add(1000, self._tick_status)
        self._sync_input_state(state)
        return False

    def _note_activity(self, text):
        """Say what she is doing on the status line. Safe from any thread."""
        self._activity = text
        GLib.idle_add(self._refresh_status)

    def _status_text(self):
        state = self.core.state
        text = STATUS_TEXT[state]
        if state in WORKING_STATES:
            text = self._activity or text
            waited = int(time.monotonic() - self._turn_started)
            if waited >= 3:
                # A ticking count is what tells you she is still alive during a
                # slow step, which a static "THINKING..." never could.
                text = "%s  %ds" % (text, waited)
        return text

    def _refresh_status(self):
        self.status_label.set_text(self._status_text())
        return False

    def _tick_status(self):
        if self.core.state not in WORKING_STATES:
            self._status_timer = None
            return False
        self._refresh_status()
        return True

    def append_transcript(self, speaker, text):
        end = self.transcript_buf.get_end_iter()
        self.transcript_buf.insert(end, f"{speaker}: {text}\n\n")
        mark = self.transcript_buf.create_mark(None, self.transcript_buf.get_end_iter(), False)
        self.transcript_view.scroll_to_mark(mark, 0.0, False, 0, 0)
        return False

    # -- typed input ----------------------------------------------------------
    def _sync_input_state(self, state):
        """Keep the input bar's enabled and Send/Stop state in step with the
        window state. Main thread only."""
        bar = getattr(self, "input_bar", None)
        if bar is None:
            return False
        bar.set_busy(state in (STATE_THINKING, STATE_SPEAKING, STATE_IMAGINING))
        bar.set_enabled(state in (STATE_IDLE, STATE_LISTENING) and not self.locked)
        return False

    def _on_speak_toggled(self, active):
        self._speak_typed = bool(active)
        try:
            if active:
                with open(SPEAK_TYPED_FILE, "a"):
                    pass
                os.chmod(SPEAK_TYPED_FILE, 0o600)
            elif os.path.exists(SPEAK_TYPED_FILE):
                os.remove(SPEAK_TYPED_FILE)
        except OSError:
            log.exception("could not save the spoken-replies setting")

    def _on_stop_turn(self):
        """Stop pressed: cancel thinking and cut speech. These are the same flags
        a screen lock uses, so every thread unwinds the way it already does."""
        self.cancel_event.set()
        self.stop_speaking.set()

    def _request_attach(self, paths):
        self._attach_next(list(paths))

    def _attach_next(self, queue):
        """Validate and add files one at a time, pausing to ask about any that
        look private. Main thread only."""
        while queue:
            path = queue.pop(0)
            try:
                attach_mod.open_regular(path).close()
            except attach_mod.AttachmentError as exc:
                self.input_bar.show_status(
                    "%s: %s" % (attach_mod.display_name(path), exc), error=True)
                continue
            marker = attach_mod.sensitive_match(path)
            if marker is None:
                self.input_bar.add_attachment(path)
                continue
            self._confirm_sensitive_attach(path, marker, queue)
            return          # resumes from the dialog's callback

    def _confirm_sensitive_attach(self, path, marker, queue):
        dialog = Gtk.AlertDialog()
        dialog.set_modal(True)
        dialog.set_message("Attach a file that looks private?")
        dialog.set_detail(
            "%s\n\nIts path matches '%s', so it may hold keys, passwords or other "
            "credentials. Its text would be shown to the assistant and could appear "
            "in the conversation. Attach it anyway?"
            % (attach_mod.display_name(path), marker))
        dialog.set_buttons(["Cancel", "Attach"])
        dialog.set_cancel_button(0)
        dialog.set_default_button(0)         # Enter cancels; attaching is deliberate

        def answered(dlg, result):
            try:
                approved = dlg.choose_finish(result) == 1
            except Exception:
                approved = False
            if approved:
                self.input_bar.add_attachment(path)
            self._attach_next(queue)

        dialog.choose(self, None, answered)

    def _on_submit_text(self, text, paths):
        """Send pressed (main thread). The message appears at once; the work runs
        on a thread so reading a PDF or waiting on the model never freezes the
        window."""
        if self.locked:
            return
        self.cancel_event.clear()
        shown = text.strip()
        if paths:
            names = ", ".join(attach_mod.display_name(p) for p in paths)
            shown = (shown + "\n" if shown else "") + "[attached: %s]" % names
        self.append_transcript("You", shown)
        self.input_bar.clear()
        self._set_state(STATE_THINKING)      # also ends any microphone capture
        threading.Thread(target=self._run_text_turn, args=(text, list(paths)),
                         daemon=True, name="typed-turn").start()

    def _restore_submission(self, text, paths, status):
        """Put an unsent message back in the box so nothing the owner typed or
        pasted is lost. Main thread only."""
        self.input_bar.set_text(text)
        for path in paths:
            self.input_bar.add_attachment(path)
        if status:
            self.input_bar.show_status(status, error=True)
        return False

    def _run_text_turn(self, text, paths):
        if not TURN_LOCK.acquire(blocking=False):
            log.info("typed turn refused: another turn is in progress")
            GLib.idle_add(self._restore_submission, text, paths,
                          "She is busy. Try again in a moment.")
            return
        try:
            self._typed_turn(text, paths)
        except Exception:
            log.exception("typed turn failed")
            GLib.idle_add(self.append_transcript, ASSISTANT_NAME,
                          "Something went wrong with that message.")
            GLib.idle_add(self._set_state, STATE_IDLE)
        finally:
            TURN_LOCK.release()

    def _typed_turn(self, text, paths):
        self.heartbeat.beat("preparing a typed message")
        tools.begin_turn()
        again = None if paths else _earlier_reply_to_read_for(self, text)
        if again:
            # Asked for in words, so it is spoken even with the typed-replies switch off.
            self.cancel_event.clear()
            self._speak(again, full=True, remember=False)
            return
        if not paths and memory.remember_request(text):
            self.cancel_event.clear()
            said = _remember_on_request(self, text)
            GLib.idle_add(self.append_transcript, ASSISTANT_NAME, said)
            if self._speak_typed:
                self.heartbeat.beat("speaking")
                self._speak(said, full=True, remember=False)
            else:
                GLib.idle_add(self._set_state, STATE_IDLE)
            return
        self._full_next_reply = speech.mentions_reading_aloud(text)
        loaded, problems = [], []
        for path in paths:
            try:
                loaded.append(attach_mod.load_attachment(path))
            except attach_mod.AttachmentError as exc:
                problems.append("%s: %s" % (attach_mod.display_name(path), exc))
            self.heartbeat.beat("reading an attachment")
        for problem in problems:
            GLib.idle_add(self.append_transcript, "Note", "Could not attach " + problem)
        if not loaded and not text.strip():
            GLib.idle_add(self._restore_submission, text, paths, None)
            GLib.idle_add(self._set_state, STATE_IDLE)
            return
        try:
            built = attach_mod.build_fitted(
                text, loaded, int(self._history_budget * MAX_MESSAGE_BUDGET_SHARE))
        except attach_mod.AttachmentError as exc:
            GLib.idle_add(self.append_transcript, "Note", str(exc))
            GLib.idle_add(self._restore_submission, text, paths, None)
            GLib.idle_add(self._set_state, STATE_IDLE)
            return
        for note in built.notes:
            GLib.idle_add(self.append_transcript, "Note", note)
        if loaded:
            tools.note_content_ingested("an attached file")
        self.cancel_event.clear()
        self.heartbeat.beat("waiting on the model")
        reply = self._chat_with_tools(built.message)
        if reply is None:
            GLib.idle_add(self._set_state, STATE_IDLE)
            return
        if not paths:
            reply = _honour_promise(self, text, reply)
        GLib.idle_add(self.append_transcript, ASSISTANT_NAME, reply)
        if self._speak_typed or getattr(self, "_full_next_reply", False):
            self.heartbeat.beat("speaking")
            self._speak(reply)
        else:
            GLib.idle_add(self._set_state, STATE_IDLE)

    # -- interaction --------------------------------------------------------
    def on_core_click(self, gesture, n_press, x, y):
        state = self.core.state
        # Manual bypass: skip the wake-phrase check for whatever's captured next.
        if state == STATE_IDLE:
            self.manual_trigger_event.set()
            GLib.idle_add(self._set_state, STATE_LISTENING)
        elif state == STATE_SPEAKING:
            # Barge-in: set a flag and let the playback thread stop itself.
            #
            # This called sd.stop() from the GTK main thread while _speak()
            # was blocked in sd.wait() on the wake thread -- two threads in
            # ALSA on one stream, which was a hard segfault in libasound.
            # PortAudio is gone from this process entirely now (playback is
            # pw-play, capture is pw-record), so the invariant this protects
            # has nothing left to protect -- but the flag is still the right
            # shape and is kept.
            # (reproduced 5/5 in isolation by an earlier investigation whose
            # note survives in ~/voice-assistant/gui.py). The audio device is
            # now touched by exactly one thread. _speak()'s bounded wait
            # notices the flag, aborts the stream itself, and runs its normal
            # cleanup and state transition.
            self.stop_speaking.set()
        elif state in (STATE_THINKING, STATE_IMAGINING):
            # Abort mid-processing: closing the in-flight session interrupts
            # a blocking Ollama call, and cancel_event is checked between
            # SDXL denoising steps. _process_audio/_chat_with_tools notice
            # the flag and unwind quietly without speaking a reply.
            self.cancel_event.set()
            if self._current_session is not None:
                try:
                    self._current_session.close()
                except Exception:
                    pass
            GLib.idle_add(self._set_state, STATE_IDLE)

    def _capture_utterance(self, max_seconds=10, wait_timeout=None):
        """Block until one spoken utterance is captured via VAD, or None if
        interrupted or (when wait_timeout is set) nothing starts within that
        many seconds. With wait_timeout=None, waits indefinitely for speech
        to start — only capped once speech is actually in progress."""
        ring_buffer = collections.deque(maxlen=VAD_PRE_ROLL_FRAMES)
        triggered = False
        voiced_frames = []
        silence_run = 0
        max_silence_frames = int(VAD_SILENCE_HANGOVER_MS / VAD_FRAME_MS)
        max_frames_total = int(max_seconds * 1000 / VAD_FRAME_MS)
        wait_start = time.time()

        with _PwRecordStream(SAMPLE_RATE, VAD_FRAME_SAMPLES) as stream:
            while True:
                if self.locked or self.core.state not in (STATE_IDLE, STATE_LISTENING):
                    return None
                if not triggered and wait_timeout is not None and (time.time() - wait_start) > wait_timeout:
                    return None
                self.heartbeat.beat("capturing")
                pcm = stream.read_frame()
                data = np.frombuffer(pcm, dtype=np.int16)
                is_speech = self.vad.is_speech(pcm, SAMPLE_RATE)
                self.core.level = min(1.0, float(np.abs(data.astype(np.float32) / 32768.0).mean()) * 15)

                if not triggered:
                    ring_buffer.append((pcm, is_speech))
                    voiced = sum(1 for _, sp in ring_buffer if sp)
                    if voiced > VAD_TRIGGER_RATIO * ring_buffer.maxlen:
                        triggered = True
                        voiced_frames.extend(f for f, _ in ring_buffer)
                        ring_buffer.clear()
                else:
                    voiced_frames.append(pcm)
                    if is_speech:
                        silence_run = 0
                    else:
                        silence_run += 1
                        if silence_run > max_silence_frames:
                            break
                    if len(voiced_frames) > max_frames_total:
                        break

        if not voiced_frames:
            return None
        raw = b"".join(voiced_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def _start_wake_loop(self, conversing=False):
        """Start (or replace) the thread that owns the microphone.

        Each thread carries a generation number. The watchdog bumps it to
        abandon a wedged thread: a thread parked inside a native PortAudio
        read cannot be interrupted from Python, and calling more PortAudio at
        it can block too -- which is exactly how sd.stop() froze the UI. So
        the stuck one is left to rot and a fresh one opens a new stream.

        With capture on pw-record this is far less likely -- a blocking pipe
        read is interruptible in a way a native PortAudio read is not, and a
        dead child raises rather than parking forever -- but the generation
        counter is kept because abandoning a thread is still the only safe
        move if one ever does wedge.
        """
        self._wake_generation += 1
        generation = self._wake_generation
        self.heartbeat.beat("wake loop starting")
        self._wake_thread = threading.Thread(
            target=self._wake_loop, args=(conversing, generation),
            name="wake-loop-%d" % generation, daemon=True,
        )
        self._wake_thread.start()

    def _wake_loop(self, start_conversing=False, generation=None):
        """Runs for the app's lifetime. Listens for the wake phrase while
        idle; a manual click bypasses the phrase check for one utterance.
        After a reply, stays actively listening for a follow-up for
        CONVERSATION_WINDOW_SECONDS before falling back to requiring the
        wake phrase again.

        Every iteration is wrapped. This daemon thread is the ONLY thing that
        opens the microphone, and it used to be a bare `while True:` with no
        exception handling at all: a PortAudioError from the pipewire-alsa
        shim, a CUDA fault in whisper, or a malformed line from Ollama killed
        it outright. The GTK main loop was unaffected, so the window animated
        perfectly while she had stopped listening for good -- and the
        traceback went to a stdout nobody reads. This wrapper turns all of
        that into "she missed one turn".
        """
        conversing = start_conversing
        while True:
            if generation is not None and generation != self._wake_generation:
                log.info("wake loop generation %s superseded; exiting", generation)
                return
            try:
                conversing = self._wake_loop_iteration(conversing)
            except Exception:
                log.exception("wake loop iteration failed; recovering")
                conversing = False
                self.heartbeat.beat("recovering from an error")
                GLib.idle_add(self._set_state, STATE_IDLE)
                time.sleep(WAKE_LOOP_BACKOFF_SECONDS)

    def _wake_loop_iteration(self, conversing):
        """One listen/answer cycle. Returns whether to stay in conversation
        mode for the next one."""
        # Only a lock cancels conversation mode. This used to reset
        # `conversing` on any non-idle state, which silently killed the
        # follow-up window on EVERY turn: _speak() hands the STATE_IDLE
        # transition to the main loop via GLib.idle_add, so control gets
        # back here while the state is still STATE_SPEAKING, and the flag
        # _process_audio just set was cleared before it was ever read.
        #
        # STATE_LISTENING is accepted as well as STATE_IDLE. It used to
        # require exactly IDLE, so a click landing in the ~100ms window
        # between _speak() posting IDLE and this loop waking up set the state
        # to LISTENING and trapped the loop here forever -- nothing else ever
        # posts IDLE from that point. She would sit showing "listening" and
        # never hear anything again.
        while self.locked or self.core.state not in (STATE_IDLE, STATE_LISTENING):
            if self.locked:
                conversing = False
            self.heartbeat.beat("waiting to become idle")
            time.sleep(0.1)

        self.heartbeat.beat("listening")
        if conversing:
            GLib.idle_add(self._set_state, STATE_LISTENING)
            audio = self._capture_utterance(max_seconds=10, wait_timeout=CONVERSATION_WINDOW_SECONDS)
        else:
            audio = self._capture_utterance(max_seconds=6)

        manual = self.manual_trigger_event.is_set()
        self.manual_trigger_event.clear()
        was_conversing = conversing

        if audio is None or len(audio) < SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
            # A typed turn that preempted listening (it set THINKING, which ends the
            # capture) must not have its state reset to IDLE underneath it.
            if (manual or was_conversing) and not TURN_LOCK.locked():
                GLib.idle_add(self._set_state, STATE_IDLE)
            return False

        if manual or was_conversing:
            # Conversation mode is renewed only if this was genuinely a turn
            # -- see _process_audio's docstring for the loop this prevents.
            return bool(self._process_audio(audio))

        text = self._transcribe(audio)
        if is_wake_phrase(text):
            GLib.idle_add(self._set_state, STATE_LISTENING)
            command_audio = self._capture_utterance(max_seconds=10)
            if command_audio is not None and len(command_audio) >= SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
                return bool(self._process_audio(command_audio))
            GLib.idle_add(self._set_state, STATE_IDLE)
        return False

    def _watchdog_tick(self):
        """Replace the wake loop if it has stopped making progress.

        This is the backstop for the failure mode that started all of this:
        she is alive, the window animates, and she has silently stopped
        listening. Everything else here is prevention; this is the thing that
        gets her back without a restart.
        """
        if self.stt is None or self._wake_thread is None:
            return True  # still loading models
        age = self.heartbeat.age()
        if age < WATCHDOG_STALE_SECONDS:
            return True
        log.error(
            "watchdog: no progress for %.0fs (last activity: %s). "
            "Abandoning that thread and starting a fresh wake loop.",
            age, self.heartbeat.note(),
        )
        # Flag only -- same reason as the barge-in handler. If the abandoned
        # thread is genuinely wedged it will never act on this, so its stream
        # leaks; that is accepted. A leaked stream is survivable, a segfault
        # from touching ALSA off the wrong thread is not, and the replacement
        # loop opens its own input stream regardless.
        self.stop_speaking.set()
        GLib.idle_add(self._set_state, STATE_IDLE)
        self._start_wake_loop(conversing=False)
        return True

    def _transcribe(self, audio):
        """Transcribe, rejecting Whisper's inventions.

        Measured on this machine: distil-large-v3 called as
        `transcribe(audio, language="en")` returned "Thank you." for 5 of 5
        non-speech clips -- digital silence, hiss, room noise, mains hum, a
        thump. gui.py then joined every segment regardless of confidence, so
        a 0.02-confidence hallucination became a conversational turn she
        answered out loud. See stability.TRANSCRIBE_OPTIONS; the same clips
        now yield 0 of 5.
        """
        segments, _ = self.stt.transcribe(audio, **stability.TRANSCRIBE_OPTIONS)
        text = stability.transcript_from_segments(segments)
        if stability.is_probably_hallucination(text):
            log.debug("discarded probable hallucination: %r", text)
            return ""
        return text

    @_one_turn_at_a_time
    def _process_audio(self, audio):
        """Returns True if this was a real turn she answered.

        The caller uses that to decide whether to stay in conversation mode.
        It used to renew unconditionally, so a VAD trigger on room noise held
        the window open forever -- and since Whisper invents "Thank you." from
        silence, each false trigger became a turn she answered aloud, which
        renewed the window again. That is the loop where she keeps asking
        into an empty room.
        """
        GLib.idle_add(self._set_state, STATE_THINKING)
        self.heartbeat.beat("transcribing")
        tools.begin_turn()
        text = self._transcribe(audio)
        if not text:
            GLib.idle_add(self._set_state, STATE_IDLE)
            return False
        GLib.idle_add(self.append_transcript, "You", text)
        if text.strip().lower() in ("quit", "exit"):
            GLib.idle_add(self.get_application).quit()
            return False
        # "Read the rest" / "read it out": answered from what she already said, not
        # by the model, which writes new text instead of reading the old (measured).
        again = _earlier_reply_to_read_for(self, text)
        if again:
            self.cancel_event.clear()
            self.heartbeat.beat("speaking")
            self._speak(again, full=True, remember=False)
            return True
        if memory.remember_request(text):
            self.cancel_event.clear()
            self.heartbeat.beat("waiting for approval")
            said = _remember_on_request(self, text)
            GLib.idle_add(self.append_transcript, ASSISTANT_NAME, said)
            self.heartbeat.beat("speaking")
            self._speak(said, full=True, remember=False)
            return True
        self._full_next_reply = speech.mentions_reading_aloud(text)
        self.cancel_event.clear()
        self.heartbeat.beat("waiting on the model")
        reply = self._chat_with_tools(text)
        if reply is None:
            # Cancelled mid-processing -- unwind quietly, no spoken reply.
            GLib.idle_add(self._set_state, STATE_IDLE)
            return False
        reply = _honour_promise(self, text, reply)
        GLib.idle_add(self.append_transcript, ASSISTANT_NAME, reply)
        self.heartbeat.beat("speaking")
        self._speak(reply)
        return True

    def _stream_chat(self, use_tools=True):
        """POST /api/chat with stream=True and assemble the full message.
        Streaming (rather than one blocking call) gives us a checkpoint
        between each chunk to notice cancellation — closing a session from
        another thread does NOT reliably interrupt a blocked non-streaming
        read (tested: it doesn't). Tool-call decisions typically arrive
        whole in the first chunk regardless; plain text replies stream
        token-by-token, which is where cancellation actually matters most.
        use_tools=False omits the tools key entirely, which is how the
        caller forces a plain-text answer -- the model cannot emit a tool
        call for a tool it was never offered.

        Returns the assembled message dict, or None if cancelled."""
        self._current_session = requests.Session()
        content_parts = []
        tool_calls = None
        payload = {
            "model": OLLAMA_MODEL,
            "messages": self.messages,
            # num_ctx matters most here: Ollama's 4096 default was not
            # enough for tool schemas + persona + 20 turns of history,
            # and the system prompt is what got truncated first.
            "options": persona.LLM_OPTIONS,
            # qwen3 is a reasoning model and emits a <think> block before its
            # answer unless told not to. Astrid speaks every reply aloud, so
            # that block would be read out -- and it is long, which feeds
            # straight back into the watchdog defect fixed in stability.py.
            # Suppressed at the source; clean_for_speech strips any that slip
            # through, since this key is silently ignored by older Ollama.
            "think": False,
            "stream": True,
        }
        if use_tools:
            payload["tools"] = tools.TOOL_SCHEMAS
        try:
            with self._current_session.post(
                "http://localhost:11434/api/chat",
                json=payload,
                stream=True,
                timeout=60,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    self.heartbeat.beat("receiving the reply")
                    if self.cancel_event.is_set():
                        return None
                    if not line:
                        continue
                    obj = json.loads(line)
                    msg = obj.get("message", {})
                    if msg.get("tool_calls"):
                        tool_calls = msg["tool_calls"]
                    if msg.get("content"):
                        content_parts.append(msg["content"])
                    if obj.get("done"):
                        break
        finally:
            self._current_session = None

        message = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _chat_with_tools(self, user_text):
        """Returns the reply text, or None if cancelled partway through."""
        _refresh_memory_for(self)        # picks up a hand-edit of the notes, for one stat()
        self.messages.append({"role": "user", "content": user_text})
        # Trim BEFORE the call as well as after: this turn may be large.
        self._trim_history()
        # What this turn may still add, spent by tool results. Without it one big
        # read pushes the prompt past num_ctx, and then Ollama keeps only the first
        # few tokens and the tail ("truncating input prompt ... keep=4"): the
        # system prompt is what gets cut. Measured 2026-10-09 with a 30 KB result:
        # she lost her persona and every pass took seconds longer.
        budget = context_budget.TurnBudget(
            self._history_budget - context_budget.messages_tokens(self.messages[1:]))
        reply = None
        try:
            for step in range(MAX_TOOL_ITERATIONS):
                if self.cancel_event.is_set():
                    return None
                self._note_activity("THINKING..." if step == 0
                                    else "READING WHAT IT FOUND...")
                message = self._stream_chat()
                if message is None:
                    return None
                self.messages.append(message)
                calls = message.get("tool_calls")
                if not calls:
                    reply = message.get("content", "").strip() or "..."
                    break
                budget.charge(context_budget.message_tokens(message))
                for call in calls:
                    if self.cancel_event.is_set():
                        return None
                    name = call["function"]["name"]
                    args = call["function"].get("arguments", {})
                    self._note_activity(tools.describe_call(name, args))
                    if name == "generate_image":
                        try:
                            result = self._generate_image_tool(args.get("prompt", ""))
                        except image_gen.GenerationCancelled:
                            return None
                    else:
                        result = tools.call_tool(name, args)
                    budget.charge(context_budget.MESSAGE_OVERHEAD_TOKENS)
                    self.messages.append({
                        "role": "tool",
                        "content": context_budget.clip_tool_result(result, budget)})
                if self.cancel_event.is_set():
                    return None
                if budget.exhausted:
                    break
            if reply is None:
                # Either every pass came back as another tool call, or the turn's
                # token budget is spent. Instead of speaking a seed apology and
                # binning everything gathered so far, take one more pass with the
                # tools withheld so the model has to answer from the history it
                # already built.
                self._note_activity("PUTTING TOGETHER AN ANSWER...")
                final = self._stream_chat(use_tools=False)
                if final is None:
                    return None
                self.messages.append(final)
                reply = final.get("content", "").strip() or "Sorry, I couldn't finish that one."
        except requests.RequestException as e:
            reply = f"I couldn't reach the language model: {e}"

        self._trim_history()
        return reply

    def _trim_history(self):
        # Keep the system prompt and the most recent WHOLE turns, within both the
        # message cap and the token budget (context_budget.py). Only whole turns
        # are dropped, so a tool_calls message is never orphaned from its tool
        # result, and the newest turn is always kept. The token cap is what stops
        # one big upload or paste from pushing the system prompt out of context.
        self.messages = context_budget.trim_to_budget(
            self.messages, max_messages=MAX_HISTORY_MESSAGES,
            max_tokens=self._history_budget)

    def _confirm_command(self, command, reason="", title=None, approve="Run"):
        """Ask the user to approve a command. Returns True only on an explicit yes.

        `title` and `approve` let the same dialog serve an edit ("Apply") or an
        admin terminal ("Open terminal") without a second code path to get
        wrong; Cancel is always the default button.

        Called from the wake-loop thread, so the dialog is built on the main
        thread via idle_add and the result comes back through an Event.

        Two rules from stability.py apply directly here and are not optional:

          * NO UNBOUNDED WAIT. This is the wake-loop thread -- the one that
            owns the microphone. A dialog nobody answers must not park it
            forever, which is defect D1 in that module. Hence the deadline,
            and hence a timeout meaning "no".
          * BEAT THE HEARTBEAT WHILE WAITING. A fixed staleness threshold
            cannot supervise work whose duration is user-controlled, which is
            what cost a segfault on 2026-08-22 when playback of a long reply
            outran the watchdog. Somebody reading a command before clicking is
            exactly that shape, so the heartbeat is stamped every 100ms here.

        Fails closed everywhere: timeout, cancellation, a dialog that will not
        build, or an unreadable response all return False.
        """
        answered = threading.Event()
        box = {"approved": False}

        def present():
            try:
                dialog = Gtk.AlertDialog()
                dialog.set_modal(True)
                dialog.set_message(title or "Astrid wants to run a command")
                detail = command if not reason else "%s\n\n%s" % (command, reason)
                dialog.set_detail(detail)
                dialog.set_buttons(["Cancel", approve])
                dialog.set_cancel_button(0)
                dialog.set_default_button(0)   # Enter cancels; Run is deliberate

                def answered_cb(dlg, result):
                    try:
                        box["approved"] = dlg.choose_finish(result) == 1
                    except Exception:
                        box["approved"] = False
                    answered.set()

                dialog.choose(self, None, answered_cb)
            except Exception:
                log.exception("could not present the command confirmation")
                answered.set()
            return False

        GLib.idle_add(present)
        self._note_activity("WAITING FOR YOUR APPROVAL...")
        log.info("awaiting approval for: %s", command.splitlines()[0] if command else command)

        deadline = time.monotonic() + COMMAND_CONFIRM_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if answered.wait(0.1):
                break
            self.heartbeat.beat("waiting for you to approve a command")
            if self.cancel_event.is_set():
                log.info("command approval cancelled: %s", command)
                return False
        else:
            log.warning("command approval timed out after %ds: %s",
                        COMMAND_CONFIRM_TIMEOUT_SECONDS, command)
            return False

        log.info("command %s: %s",
                 "APPROVED" if box["approved"] else "declined", command)
        return box["approved"]

    def _generate_image_tool(self, prompt):
        """Tool call target for generate_image. Runs synchronously inside the
        _chat_with_tools loop (already off the main thread), so it's safe to
        block here. Returns a result dict for the LLM to see and reply to,
        rather than speaking directly — the model composes its own wrap-up
        from this."""
        if not prompt:
            return {"error": "no prompt given"}
        GLib.idle_add(self.image_picture.set_visible, False)
        GLib.idle_add(self._set_state, STATE_IMAGINING)
        # STT sits resident on GPU between turns; free its VRAM too so SDXL
        # has enough headroom, then restore it once generation is done.
        self.stt.model.unload_model(to_cpu=False)
        try:
            image = image_gen.generate(prompt, cancel_event=self.cancel_event)
        except image_gen.GenerationCancelled:
            self.stt.model.load_model()
            raise
        except Exception as e:
            # Guarded: on an OOM there may be no room to bring Whisper back
            # either, and a raise here would replace the real error and kill
            # the only mic-owning thread. Losing STT until the next restart is
            # bad; losing the wake loop is worse.
            try:
                self.stt.model.load_model()
            except Exception:
                log.exception("could not reload STT after a failed generation")
            GLib.idle_add(self._set_state, STATE_THINKING)
            return {"error": f"image generation failed: {e}"}
        self.stt.model.load_model()
        path = os.path.join(GENERATED_DIR, f"{int(time.time())}.png")
        image.save(path)
        GLib.idle_add(self.image_picture.set_filename, path)
        GLib.idle_add(self.image_picture.set_visible, True)
        GLib.idle_add(self._set_state, STATE_THINKING)
        return {"status": "image generated and now shown to the user", "prompt": prompt}

    def _synthesize(self, piece):
        """One chunk of speech -> (samples, sample rate), serialised by TTS_LOCK."""
        with TTS_LOCK:
            return self.tts.create(piece, voice=persona.TTS_VOICE,
                                   speed=persona.TTS_SPEED, lang=persona.TTS_LANG)

    @staticmethod
    def _pcm(samples):
        return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    def _speak(self, text, full=None, remember=True):
        """Say a reply aloud; returns only when playback is over.

        `full` speaks all of it instead of the first few sentences. None means
        "as the listener asked for this reply" (_full_next_reply, set when the
        request said "out loud" or "read it"), consumed here so it covers one
        reply only. `remember` records what was said so "read the rest" can find
        it; the rereading itself passes False so it does not overwrite its source.

        The reply is cut into chunks (speech.plan_speech), the FIRST is
        synthesized here and starts playing at once, and a second thread
        synthesizes the rest while it plays. Kokoro runs about 8x faster than
        real time, so after the first chunk it stays comfortably ahead. If the
        length limit is switched on (speech.SPOKEN_LIMIT_CHARS, off by default) a
        long reply is shortened to its first few sentences plus a closing line;
        the transcript already holds the full text.

        Thread layout: this thread synthesizes chunk 1, then only watches (stop,
        cancel, heartbeat, deadline) as before; `produce` synthesizes the rest;
        `feed` writes to the one pw-play process, because that write blocks.
        """
        self.heartbeat.beat("synthesising speech")
        self._note_activity("PREPARING TO SPEAK...")
        cleaned = clean_for_speech(text)
        if full is None:
            full = getattr(self, "_full_next_reply", False)
            self._full_next_reply = False
        plan = speech.plan_speech(cleaned, full=full)
        if remember:
            self._last_full = cleaned
        self._last_rest = plan.rest
        pieces = plan.spoken
        if not pieces:
            GLib.idle_add(self._set_state, STATE_IDLE)
            return
        first, sr = self._synthesize(pieces[0])
        try:
            player = _PwPlayStream(sr)
        except OSError:
            log.exception("pw-play unavailable; this reply is not spoken")
            GLib.idle_add(self._set_state, STATE_IDLE)
            return

        envelope = _Envelope(sr)
        envelope.add(first)
        self.core.envelope = envelope.snapshot()
        self.core.envelope_rate = envelope.rate
        self.core.playback_start = time.time()
        self.stop_speaking.clear()
        GLib.idle_add(self._set_state, STATE_SPEAKING)

        audio = queue.Queue()
        audio.put(self._pcm(first))
        abort = threading.Event()
        produced = {"seconds": len(first) / sr}   # audio synthesized so far
        durations = [len(first) / sr]             # per piece, so a stop can be placed in the text

        def produce():
            try:
                for piece in pieces[1:]:
                    if abort.is_set():
                        return
                    samples, rate = self._synthesize(piece)
                    if abort.is_set():
                        return
                    if rate != sr:
                        raise RuntimeError("sample rate changed mid-reply: %s -> %s" % (sr, rate))
                    envelope.add(samples)
                    self.core.envelope = envelope.snapshot()
                    produced["seconds"] += len(samples) / sr
                    durations.append(len(samples) / sr)
                    audio.put(self._pcm(samples))
            except Exception:
                log.exception("speech synthesis failed part-way; speaking what was ready")
            finally:
                audio.put(None)                    # end of stream, however it ended

        def feed():
            try:
                while not abort.is_set():
                    try:
                        data = audio.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if data is None:
                        break
                    player.write(data)
            except (BrokenPipeError, OSError, ValueError):
                pass                               # the player went away: nothing to feed
            finally:
                player.end_input()

        if len(pieces) > 1:
            threading.Thread(target=produce, daemon=True, name="speech-synth").start()
        else:
            audio.put(None)
        feeder = threading.Thread(target=feed, daemon=True, name="speech-feed")
        feeder.start()

        # The same bounded contract as before: never block past the audio that
        # exists plus a generous margin, and beat the heartbeat throughout so the
        # watchdog does not mistake long speech for a hang. The margin is measured
        # from audio actually produced, so a wedged synthesis ends playback 30 s
        # after the last sound instead of holding the wake loop forever.
        started = time.monotonic()
        ended = "done"
        while True:
            if self.stop_speaking.is_set() or self.cancel_event.is_set():
                ended = "stopped"
                player.stop()
                break
            if not player.running():
                break                              # drained and exited, or died
            if time.monotonic() > started + produced["seconds"] + SPEAK_GRACE_SECONDS:
                log.warning("speech overran %.1fs of audio; stopping", produced["seconds"])
                ended = "stopped"
                player.stop()
                break
            self.heartbeat.beat("speaking")
            time.sleep(0.05)

        # An in-flight synthesis is not waited for: it cannot be interrupted, and
        # waiting would delay listening by a second or two after a barge-in.
        # TTS_LOCK makes the next reply queue behind it instead of overlapping.
        abort.set()
        if ended == "stopped":
            # Where in the reply she was stopped, so "go on" can pick up from the
            # sentence she was in. Plus whatever the length limit had already held back.
            left = speech.text_after(plan.chunks, list(durations), time.time() - self.core.playback_start)
            self._last_rest = (left + " " + plan.rest).strip()
        player.close()
        feeder.join(1.0)
        self.core.envelope = None
        GLib.idle_add(self._set_state, STATE_IDLE)


def release_llm_vram():
    """Tell Ollama to evict the model as Astrid exits.

    Astrid's own ~1.3 GB (faster-whisper) is released by the driver when the
    process dies -- that part needs no help. Ollama is a separate long-running
    system service, so its ~11 GB stays resident for OLLAMA_KEEP_ALIVE (5
    minutes by default) after the last request, whether or not anything is
    still using it.

    keep_alive 0 asks for immediate eviction. Deliberately NOT done by setting
    OLLAMA_KEEP_ALIVE=0 globally -- the 5-minute default is what keeps her
    responsive mid-conversation, and CLAUDE.md documents that timer as
    load-bearing. This only fires on the way out. Best-effort: exiting matters
    more than reclaiming the memory a few minutes early.
    """
    try:
        loaded = requests.get(
            "http://localhost:11434/api/ps", timeout=5
        ).json().get("models", [])
    except Exception:
        return
    # Evict whatever is actually resident rather than naming one model. Naming
    # one is why ~10 GB sat on the card for the full 30-minute keep-alive after
    # she quit on 2026-08-29 -- and it is the same trap image_gen.unload_llm()
    # and comfyui-proxy both record. She can be pointed at qwen3:32b too.
    for m in loaded:
        name = m.get("model") or m.get("name")
        if not name:
            continue
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=5,
            )
        except Exception:
            pass


class AstridApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.local.astrid")

    def do_activate(self):
        provider = Gtk.CssProvider()
        provider.load_from_string(CSS + inputbar.CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        win = self.props.active_window
        if not win:
            win = AstridWindow(self)
        win.present()

    def do_shutdown(self):
        # Normal exit: window closed, or "quit"/"exit" spoken.
        release_llm_vram()
        Gtk.Application.do_shutdown(self)


if __name__ == "__main__":
    # Covers the paths do_shutdown does not: SIGTERM from a logout or a
    # `kill`, which by default terminates without running atexit handlers.
    atexit.register(release_llm_vram)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, lambda *_: sys.exit(0))

    AstridApp().run(None)
