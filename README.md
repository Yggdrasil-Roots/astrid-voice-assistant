# astrid

> [!IMPORTANT]
> **Generic by design: you will need to modify this before use.** The persona, user name, location and time zone are placeholders, and the code reflects one machine's setup (audio, GPU, paths, model choice). All personal and sensitive details (names, locations, addresses, hostnames, device identifiers, credentials and personal notes) have been removed or replaced with placeholders. Expect to adapt names, paths, addresses, hardware assumptions and settings to your own environment, and review everything before you run or rely on it.

A local voice assistant for Linux with a GTK4 window. Everything runs on your
own machine: speech recognition, the language model, speech synthesis and
image generation. Nothing leaves the machine except the web searches you ask
for, which go through your own SearXNG instance (it queries search engines).

| Part | Used for |
|---|---|
| [Ollama](https://ollama.com) | the language model (`OLLAMA_MODEL` in `gui.py`) |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | speech to text (`distil-large-v3`, CUDA) |
| [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (ONNX) | text to speech |
| [diffusers](https://github.com/huggingface/diffusers) (SDXL) | image generation |
| [SearXNG](https://github.com/searxng/searxng) | private web search (JSON API on `localhost:8080`) |
| PipeWire (`pw-play`, `pw-record`) | audio in and out |
| `pdftotext` (poppler), `pandoc`, optionally `tesseract` | reading attached PDFs, Word/ODT/HTML/EPUB files, and text in images |

## What she can do

Tools exposed to the model: `get_current_datetime`, `get_system_info`,
`get_gpu_status`, `web_search`, `generate_image`, `list_files`, `read_file`,
`write_file`, `edit_file`, `run_command` and `open_terminal`. There is a PIN lock
screen (PBKDF2 hash in a `0600` file under `~/.astrid`).

You can also **type** to her (Enter sends, Shift+Enter adds a line) and **attach
files** with the button, by drag and drop, or by paste: text and code, PDF,
Word/ODT/HTML/EPUB (through `pandoc`), and images, whose text is read with
`tesseract`. She cannot see pictures, only read text in them. Spoken replies to
typed messages are off by default, with a switch.

While she works the status line says what she is doing ("READING app.log...",
"WAITING FOR YOUR APPROVAL...") with a seconds count, so a slow step looks
different from a hung one. Replies are spoken a sentence at a time, so the first
sound does not wait for the whole reply to be synthesized, and a long reply is
spoken only as far as its first few sentences, with a closing line saying the
rest is on screen. The thresholds are named constants at the top of `speech.py`;
setting `SPOKEN_LIMIT_CHARS = 0` speaks everything.

## Safety model

The command and file tools are triggered by voice and by typed text, and can be
fed by web content or by files you attach, so each is tiered. This is defence in
depth, not a sandbox.

1. `sudo`, `pkexec`, `doas`, `su` and `run0` are **refused outright** by
   `run_command`. `open_terminal` can open a *visible* terminal for a privileged
   command after a clearly worded approval (default button Cancel); you type the
   password into that terminal and the assistant never sees it. Set
   `TERMINAL_ADMIN_ENABLED = False` in `tools.py` to refuse it there too.
2. Read-only inspection (`df`, `ls`, `cat`, `grep`, `git log` and similar), and
   pipelines made only of those, run unattended. The allowlist matches flags the
   way the programs read them, because exact-string matching is not enough:
   `sort -oFILE`, `sort --out=FILE` (abbreviated long options are accepted by GNU
   and by uutils), `uniq IN OUT` and `git diff --output=FILE` all write files.
   Commands that print environment variables (`printenv`, `docker inspect`) ask.
3. Files she may **read** are anything under the home folder. Paths that look like
   credentials, browser profiles, token files, password stores or private keys
   (`SENSITIVE_PATH_MARKERS` in `tools.py`) are not refused but **put to you**,
   with the exact path, and open only if you click Allow. Recursive searches look
   inside the tree they are pointed at and ask if it holds such a name.
4. Files she may **write** are limited to `WRITE_DIRS`. `write_file` never
   overwrites. `edit_file` replaces one exact piece of text, shows you the change
   as a diff, keeps a `.bak` copy, writes atomically, and refuses private files.
5. Once a request has read file contents, `web_search` needs a click that shows
   the query (`TAINT_GATE_ENABLED`): a poisoned document should not be able to
   send what she just read out through a search.
6. Everything else needs an explicit click in the GTK dialog.

Text returned by web search or read from a file is treated as data, never as
instructions. Review `tools.py` yourself before running this on a machine you
care about: it executes commands and reads your files.

## Setting it up

This repository is the application code only. It does **not** include:

- a Python environment: there is no pinned requirements file yet. The imports
  tell you what is needed: PyGObject (`gi`), `faster-whisper`, `kokoro-onnx`,
  `torch` and `diffusers`, `webrtcvad`, `numpy`, `requests`
- the Kokoro files `kokoro-v1.0.fp16.onnx` and `voices-v1.0.bin`, expected next
  to `gui.py`
- Ollama models, or any SDXL weights
- a PIN: set it on first run; it lives in `~/.astrid`
- optional: `poppler-utils`, `pandoc` and `tesseract-ocr` for attachments. The
  extractors parse untrusted files, so install them from your distribution's
  security channel and keep them patched

Then run `./start.sh`. It starts Ollama and a local SearXNG if they are not
already answering. Copy nothing secret into this tree: replace the placeholder
`secret_key` in `searxng/config/settings.yml` with your own random value.

## Make it yours

`persona.py` is written for **one user**. Before using it, edit:

- `USER_NAME`, `LOCATION_NAME` and `TIME_ZONE_NAME` near the top
- the `WHO_YOU_ARE`, `HOW_YOU_SPEAK` and `EXAMPLES` blocks: the character, a
  dry understated British-leaning register, is deliberately specific, and the
  text refers to the user as "he"
- the `CANDOUR` block, which tells the model not to hedge on security topics
  because it was written for a security learner on their own machine

Note on the model: `gui.py` defaults to a community "abliterated" (refusal
removed) build of Qwen3 14B. The candour block assumes a model that will not
refuse; with a standard model it will likely soften or ignore it. Check the
licence and fitness of any model you choose.

Add your own private folders and file names to `SENSITIVE_PATH_MARKERS` in
`tools.py`; the list is generic and certainly misses things on your machine.

`./start.sh` and `gui.py` assume an installed copy; adjust paths such as
`ASTRID_INSTALL_DIR` in `gui.py` for yours.

## Notes from building it

- Audio goes through `pw-play` and `pw-record` subprocesses. PortAudio is not
  used: it segfaulted through the ALSA shim on the author's machine.
- Keep `OLLAMA_KV_CACHE_TYPE=f16`: on recent NVIDIA generations a quantised KV
  cache produced garbled tool calls.
- Never hardcode a terminal name; resolve it from `PATH`.
- Ollama silently truncates a prompt longer than the context window, keeping only
  the first few tokens and the tail ("truncating input prompt ... keep=4" in its
  log). A big tool result therefore removes the *system prompt*: the persona goes
  and every pass gets slower. Tool results are clipped to a per-turn token budget
  (`context_budget.py`), and `read_file` pages in small windows.
- Kokoro's phonemiser (espeak-ng) is not thread-safe. Speech is synthesized in
  chunks on a second thread, so every call goes through one lock.
- On a CPU-only ONNX runtime Kokoro runs at roughly 8x real time. Streaming the
  reply in chunks hides that: the first sound waits for one sentence, not all.

## Licence

MIT. See [LICENSE](LICENSE). Third-party components keep their own licences.
