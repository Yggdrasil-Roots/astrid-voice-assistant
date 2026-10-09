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

## What she can do

Tools exposed to the model: `get_current_datetime`, `get_system_info`,
`get_gpu_status`, `web_search`, `generate_image`, `list_files`, `read_file`,
`write_file`, `run_command` and `open_terminal`. There is a PIN lock screen
(PBKDF2 hash in a `0600` file under `~/.astrid`).

## Safety model for the command layer

`run_command` and `open_terminal` are triggered by voice and can be fed by web
content, so they are tiered:

1. `sudo`, `pkexec`, `doas`, `su` and `run0` are **refused outright** and never
   offered for approval.
2. Read-only inspection (`df`, `ls`, `cat`, `grep`, `git log` and similar) runs
   unattended, **except** paths that look like credentials (see
   `SENSITIVE_PATH_MARKERS` in `tools.py`), which need approval.
3. Everything else needs an explicit click in the GTK dialog.

Text returned by web search or read from a file is treated as data, never as
instructions. Review `tools.py` yourself before running this on a machine you
care about: it executes commands.

## Setting it up

This repository is the application code only. It does **not** include:

- a Python environment: there is no pinned requirements file yet. The imports
  tell you what is needed: PyGObject (`gi`), `faster-whisper`, `kokoro-onnx`,
  `torch` and `diffusers`, `webrtcvad`, `numpy`, `requests`
- the Kokoro files `kokoro-v1.0.fp16.onnx` and `voices-v1.0.bin`, expected next
  to `gui.py`
- Ollama models, or any SDXL weights
- a PIN: set it on first run; it lives in `~/.astrid`

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

`./start.sh` and `gui.py` assume an installed copy; adjust paths such as
`ASTRID_INSTALL_DIR` in `gui.py` for yours.

## Notes from building it

- Audio goes through `pw-play` and `pw-record` subprocesses. PortAudio is not
  used: it segfaulted through the ALSA shim on the author's machine.
- Keep `OLLAMA_KV_CACHE_TYPE=f16`: on recent NVIDIA generations a quantised KV
  cache produced garbled tool calls.
- Never hardcode a terminal name; resolve it from `PATH`.

## Licence

MIT. See [LICENSE](LICENSE). Third-party components keep their own licences.
