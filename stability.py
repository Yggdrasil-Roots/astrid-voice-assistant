"""Bounded audio calls, transcription guards, a wake-loop watchdog, and the
diagnostics needed to prove where Astrid wedged -- next time, in seconds.

Nothing in here changes her character or what she can do. It exists because
three separate defects all present to the user as "she froze", and all three
are invisible from the app's own logs.

Import this FIRST, before numpy/torch/ctranslate2/onnxruntime, so
`cap_native_thread_pools()` can take effect. `sounddevice` is imported lazily
inside the functions for the same reason -- it pulls in numpy.

------------------------------------------------------------------------------
D1. `sd.wait()` blocks forever, and she goes silently deaf
------------------------------------------------------------------------------
`sd.wait()` is `threading.Event.wait()` with **no timeout**; the event is set
only by PortAudio's `finished_callback`. If the pipewire-alsa shim tears a
stream down without firing that callback, the wait never returns -- on
`_wake_loop`'s thread, the one thread that opens the microphone. The GTK main
loop is untouched, so the window keeps animating perfectly while she has
stopped listening forever. That is why this reads as "frozen, not crashed".

Observed live on 2026-08-19 (pid 17264): the wake-loop thread parked in
`futex_do_wait` with zero CPU, no audio streams open, and the process not
registered as a PipeWire client at all.

------------------------------------------------------------------------------
D2. `sd.stop()` is called from the GTK main thread, and libasound segfaults
------------------------------------------------------------------------------
Both `sd.stop()` call sites in /opt's gui.py run on the GTK main thread -- the
barge-in click handler, and `_lock()` off the screensaver D-Bus signal --
while `_speak()` is blocked in `sd.wait()` on the wake thread. Two threads
calling into ALSA on the same stream.

The dev tree at ~/voice-assistant already carries a comment recording this,
from an earlier investigation: *"That is a hard segfault inside libasound --
reproduced 5/5 in isolation, and it was killing her on every session."* That
is almost certainly the `libasound.so` segfault CLAUDE.md records for
2026-08-18. The fix was started there and never finished -- `stop_speaking`
is `.set()` in both handlers but was never initialised and never read, so
barge-in in that tree raises AttributeError -- and it never reached /opt at
all, which is what actually runs.

`Pa_StopStream` also *drains* before returning, so even without the segfault
this blocks the UI thread.

The invariant enforced here is the one that earlier investigation arrived at:
**exactly one thread ever touches the audio device.** The UI thread sets an
Event; the thread that called `sd.play()` is the only thread that stops it.
`abort_playback()` therefore must only be called from the playback thread.

------------------------------------------------------------------------------
D3. Whisper invents speech from silence, and she talks to herself
------------------------------------------------------------------------------
Measured on this machine, distil-large-v3 with gui.py's settings
(`transcribe(audio, language="en")` and nothing else) returned **"Thank you."
for 5 out of 5 non-speech clips** -- digital silence, faint hiss, room noise,
mains hum, and a single thump.

In conversation mode there is no wake-phrase gate: any VAD trigger over
MIN_UTTERANCE_SECONDS becomes a turn, and every processed turn sets
`conversing = True` again, renewing the window. So one false trigger is a
self-sustaining loop: noise -> "Thank you." -> she answers it -> she speaks ->
window renews -> noise. `condition_on_previous_text=True` (the library
default) makes it worse by feeding each hallucination back as the prompt for
the next one.

The guards below took that to 0 out of 5 on the same clips.

CLAUDE.md already records two prior failures in this same PortAudio ->
pipewire-alsa path (a native segfault in libasound, and an indefinite hang on
resample). D1 and D2 are the third and fourth shape of it. The rule this
module enforces: **no unbounded audio call, ever, and nothing that can block
on the GTK main thread.**
"""

import faulthandler
import logging
import os
import re
import signal
import sys
import threading
import time

log = logging.getLogger("astrid")


# ---------------------------------------------------------------- thread caps

def cap_native_thread_pools(threads=8):
    """Cap the OpenMP / BLAS pools and stop them spin-waiting.

    Must run before numpy, torch, ctranslate2 or onnxruntime are imported --
    each reads these once, at library load.

    This is an efficiency fix, NOT a fix for the freeze, and should not be
    sold as one. Measured on a desktop (24 logical cores, kokoro synthesising
    an 11s utterance): 24 threads took 1.242s, 8 threads took 0.927s --
    oversubscription was ~25% *slower*. GTK main-loop tick lateness stayed
    under 2ms at every thread count from 2 to 24, so the UI was never being
    starved by these pools; that hypothesis was tested and refuted.

    Two independent libgomp copies end up in this process (one bundled inside
    ctranslate2, one system-wide for torch). They do not coordinate, so each
    spawns its own pool of `nproc` threads and each spin-waits at a barrier.
    The passive wait policy is what stops the resulting churn: idle pool
    threads sleep on a futex instead of burning kernel time.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")


# --------------------------------------------------------------- audio safety

# How long past the audio's own duration to wait before declaring playback
# wedged. Generous on purpose: a slow drain is normal, a stuck stream is not.
PLAYBACK_GRACE_SECONDS = 5.0

# Poll interval while waiting for playback. 20ms is inaudible at the end of a
# sentence and keeps the loop cheap.
_PLAYBACK_POLL_SECONDS = 0.02


def abort_playback():
    """Stop playback immediately.

    **Call this only from the thread that called `sd.play()`.** Calling into
    ALSA from a second thread while the playback thread sits in the stream is
    the libasound segfault described at the top of this module. The UI thread
    must set an Event and let the playback thread act on it instead; that is
    what `wait_for_playback`'s `stop_event` is for.

    Uses Pa_AbortStream (discard buffers, return now) rather than
    Pa_StopStream (drain, then return), so a barge-in cuts her off rather
    than waiting out the buffered tail of the sentence.

    Never raises: a stuck sentence is better than an exception unwinding the
    one thread that owns the microphone.
    """
    import sounddevice as sd
    try:
        stream = sd.get_stream()
    except (RuntimeError, AttributeError):
        return  # play() was never called; nothing to abort
    except Exception:
        log.exception("stability: could not reach the playback stream")
        return
    try:
        stream.abort(ignore_errors=True)
    except Exception:
        log.exception("stability: aborting playback failed")


def wait_for_playback(expected_seconds, stop_event=None, cancel_event=None,
                      heartbeat=None):
    """Bounded replacement for `sd.wait()`. Call from the playback thread.

    Returns True if playback finished, was barged in on, or was cancelled;
    False if the deadline passed with the stream still active -- meaning the
    shim has wedged and the caller should carry on rather than block forever.

    `stop_event` is how barge-in and screen-lock reach this: the UI thread
    sets it, and the teardown happens *here*, on the thread that started
    playback, preserving the one-thread-owns-the-device invariant.

    `expected_seconds` is the true duration of the audio passed to sd.play().

    `heartbeat` must be passed by any caller that a watchdog is monitoring.
    Playback is the longest uninterrupted thing the wake-loop thread ever
    does, and its duration scales with the length of the reply while a
    watchdog's staleness threshold does not. Without a beat in here, a
    sufficiently long reply looks exactly like a hung thread: on 2026-08-22
    the watchdog fired at 181s of legitimate speech, abandoned the healthy
    wake loop and started a second one while this thread was still inside
    PortAudio -- breaking the one-thread-owns-the-device invariant at the top
    of this module and segfaulting in libasound. The beat costs nothing (this
    loop already spins every 20ms) and does not weaken the deadline below,
    so a genuinely wedged stream is still caught and abandoned on time.
    """
    import sounddevice as sd
    deadline = time.monotonic() + expected_seconds + PLAYBACK_GRACE_SECONDS
    while time.monotonic() < deadline:
        if heartbeat is not None:
            heartbeat.beat()
        if stop_event is not None and stop_event.is_set():
            log.debug("stability: playback stopped by request (barge-in or lock)")
            abort_playback()
            return True
        if cancel_event is not None and cancel_event.is_set():
            abort_playback()
            return True
        try:
            if not sd.get_stream().active:
                return True
        except (RuntimeError, AttributeError):
            return True  # stream already gone: finished
        except Exception:
            log.exception("stability: playback state unreadable; unwinding cleanly")
            return True
        time.sleep(_PLAYBACK_POLL_SECONDS)

    log.error(
        "stability: playback did not finish within %.1fs (audio was %.1fs). "
        "The PortAudio/pipewire-alsa stream has wedged; abandoning it.",
        expected_seconds + PLAYBACK_GRACE_SECONDS, expected_seconds,
    )
    abort_playback()
    return False


# -------------------------------------------------------- transcription guard

# Options that stop distil-large-v3 inventing speech from non-speech.
# `condition_on_previous_text=False` matters most for the self-talk loop: the
# default feeds each hallucination back as the prompt for the next one.
TRANSCRIBE_OPTIONS = dict(
    language="en",
    vad_filter=True,
    condition_on_previous_text=False,
    no_speech_threshold=0.5,
    log_prob_threshold=-0.8,
    temperature=0.0,
)

SEGMENT_NO_SPEECH_MAX = 0.5
SEGMENT_LOGPROB_MIN = -0.8

# What distil-large-v3 actually emits for silence on this machine, plus the
# usual YouTube-caption residue it was trained on. Matched after stripping
# punctuation and casing. Kept deliberately short -- a long blocklist starts
# swallowing real utterances, and the decoder options above do most of the work.
_HALLUCINATION_PHRASES = frozenset({
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "you", "bye", "goodbye", "okay", "ok", "oh",
    "please subscribe", "subtitles by the amara org community",
})

_PUNCT = str.maketrans("", "", ".,!?;:-\"'")


def is_probably_hallucination(text):
    """True if `text` looks like Whisper output for silence rather than speech.

    Also catches the repetition case ("Thank you. Thank you. Thank you.") by
    collapsing repeats before matching -- that is what silence longer than a
    couple of seconds produces.
    """
    if not text:
        return True
    normalised = " ".join(text.lower().translate(_PUNCT).split())
    if not normalised:
        return True
    if normalised in _HALLUCINATION_PHRASES:
        return True
    # "thank you thank you thank you" -> "thank you"
    words = normalised.split()
    for size in (1, 2, 3, 4):
        if len(words) >= size * 2 and len(words) % size == 0:
            first = words[:size]
            if all(words[i:i + size] == first for i in range(0, len(words), size)):
                if " ".join(first) in _HALLUCINATION_PHRASES:
                    return True
    return False


def transcript_from_segments(segments):
    """Join Whisper segments, dropping the ones that are not confident speech.

    faster-whisper hands back `no_speech_prob` and `avg_logprob` per segment
    and gui.py was joining every segment unconditionally, which is how a
    0.02-confidence "Thank you." became a conversational turn.
    """
    kept = []
    for seg in segments:
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        logprob = getattr(seg, "avg_logprob", 0.0)
        if no_speech > SEGMENT_NO_SPEECH_MAX or logprob < SEGMENT_LOGPROB_MIN:
            log.debug("stability: dropped low-confidence segment %r "
                      "(no_speech=%.2f avg_logprob=%.2f)", seg.text, no_speech, logprob)
            continue
        kept.append(seg.text)
    return " ".join(kept).strip()


# --------------------------------------------------------------- pronunciation

# Words espeak-ng gets wrong in en-us. Each was checked by phonemising it and
# its replacement -- see the table in CLAUDE.md.
#
# Substituted as SPELLINGS before synthesis, not as phonemes: kokoro's
# `is_phonemes=True` applies to the whole string, so mixing modes would mean
# hand-phonemising every reply.
#
# Deliberately short. Most of this machine's vocabulary is already correct in
# en-us -- Ollama, systemd, SDXL, Nvidia, Ubuntu, Kali, ALSA, GTK, NVMe,
# SQLite, Astrid and localhost all check out -- and every needless
# entry is a chance to mangle a word that was fine. Add to it from things
# actually heard to be wrong, not from things that look unusual.
#
# This affects SPEECH ONLY. The transcript pane still shows the real text.
PRONUNCIATION = {
    "SearXNG": "search X N G",     # was "seer eks-en-gee"
    "UFW": "U F W",                # was "uff-wuh"
    "qwen": "kwen",                # was "cue-wen"
    "CUDA": "koodah",              # was "cue-duh"
    "iptables": "I P tables",      # was "ip-tuh-bulz"
    "journalctl": "journal C T L",  # was "jur-nalk-tul"
    "kokoro": "koh koh roh",       # was "kuh-KOR-oh"
    "VRAM": "V RAM",               # was "vram" as one syllable
}

# Longest first, so a key that is a prefix of another cannot shadow it.
_PRONUNCIATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in
                      sorted(PRONUNCIATION, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_PRONUNCIATION_LOOKUP = {k.lower(): v for k, v in PRONUNCIATION.items()}


def fix_pronunciation(text):
    """Rewrite known-mispronounced words to spellings espeak reads correctly.

    Case-insensitive and word-bounded, so `sudo ufw status` and `UFW` both get
    fixed while a word that merely contains a key is left alone.
    """
    if not text:
        return text
    return _PRONUNCIATION_RE.sub(
        lambda m: _PRONUNCIATION_LOOKUP[m.group(0).lower()], text)


# ------------------------------------------------------------------ heartbeat

class Heartbeat:
    """A monotonic "this thread is still making progress" stamp.

    Cheap enough to call inside the VAD frame loop, which runs every 30ms.
    """

    def __init__(self, note="starting"):
        self._lock = threading.Lock()
        self._at = time.monotonic()
        self._note = note

    def beat(self, note=None):
        with self._lock:
            self._at = time.monotonic()
            if note is not None:
                self._note = note

    def age(self):
        with self._lock:
            return time.monotonic() - self._at

    def note(self):
        with self._lock:
            return self._note


# ---------------------------------------------------------------- diagnostics

def install_diagnostics():
    """Make the next wedge diagnosable in seconds, without root.

    Three things:

    * `faulthandler` on SIGSEGV/SIGABRT/SIGBUS/SIGFPE, so a native crash in
      libasound leaves a Python-side traceback in the log instead of one
      unresolved address in dmesg. CLAUDE.md records exactly that happening.
    * SIGUSR1 dumps every thread's stack to the log on demand:
          kill -USR1 $(pgrep -f 'python3 -u gui.py')
      This is the information that otherwise needs `sudo py-spy dump`, which
      is blocked here by yama ptrace_scope=1.
    * `threading.excepthook`, so a dying worker takes a logged traceback with
      it. `_wake_loop` is a bare `while True:` on a daemon thread -- when it
      dies the app keeps running and looks completely healthy, which is the
      worst possible thing to have to debug blind.

    Writes to stderr deliberately rather than opening its own handle:
    start.sh already redirects stderr into ~/.astrid/astrid.log for desktop
    launches, and a terminal launch wants it on the terminal. Opening the log
    file again here would interleave two independent writers into one file.
    """
    stream = sys.stderr
    faulthandler.enable(file=stream, all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True, chain=True)
    except (AttributeError, ValueError, OSError):
        log.warning("stability: SIGUSR1 stack dump unavailable here")

    def _thread_excepthook(args):
        if args.exc_type is SystemExit:
            return
        log.error("stability: thread %r died with an unhandled exception",
                  getattr(args.thread, "name", "?"),
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = _thread_excepthook
    return stream


def setup_logging(log_path=None, level=logging.INFO):
    """Timestamped logging to the same file start.sh already redirects into.

    Without timestamps there is no way to line a freeze up against `dmesg` or
    the journal, which is what made the libasound segfault take an afternoon.
    """
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("astrid")
    root.setLevel(level)
    if root.handlers:
        return root
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    root.addHandler(handler)
    # A file handler is added ONLY when stderr is a terminal. Under a desktop
    # launch start.sh has already redirected stderr into this same file, and
    # adding a second writer would duplicate every line.
    if log_path and sys.stderr.isatty():
        try:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            root.exception("stability: could not add file logging at %s", log_path)
    return root
