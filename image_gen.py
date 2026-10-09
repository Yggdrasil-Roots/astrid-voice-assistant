"""Lazy-loaded local image generation via SDXL + a realistic checkpoint
(RealVisXL). Kept resident in CPU RAM once loaded (not VRAM) via
enable_model_cpu_offload, so repeat generations after the first are fast.
The Ollama LLM must be unloaded by the caller before calling generate() —
SDXL needs most of the 16GB card to itself.
"""
import time

import requests
import torch
from diffusers import StableDiffusionXLPipeline

CHECKPOINT_PATH = "/opt/astrid/models/RealVisXL_V5.0_fp16.safetensors"
OLLAMA_URL = "http://localhost:11434"

_pipe = None


class GenerationCancelled(Exception):
    pass


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe
    pipe = StableDiffusionXLPipeline.from_single_file(
        CHECKPOINT_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    _pipe = pipe
    return _pipe


def unload_llm(timeout=30.0):
    """Free whatever Ollama is holding on the GPU, before SDXL needs the card.

    Asks /api/ps what is ACTUALLY loaded and evicts each model, rather than
    naming one. Naming one is the bug this replaced: the constant below still
    said qwen2.5:14b-instruct-q5_K_M months after that model was deleted from
    Ollama, so `ollama stop` matched nothing and exited 0, qwen3:14b stayed
    resident at ~10 GB of the 16 GB card, and SDXL OOM'd mid-UNet on
    2026-08-29. comfyui-proxy in the Open WebUI stack carries the same lesson
    in its docstring -- its earlier version hardcoded a name and "would
    silently miss qwen3:32b". Astrid can be pointed at qwen3:32b as well, so
    this was the same trap. Query what is loaded; never assume.

    Eviction is asynchronous. The old implementation slept one second and
    hoped; this polls until /api/ps reports nothing resident, which is the
    condition that actually matters, and gives up after `timeout` instead of
    blocking the wake-loop thread forever.

    Best-effort by design: if Ollama cannot be reached we return and let
    generation try anyway, rather than refusing to draw because bookkeeping
    failed. Mirrors comfyui-proxy's "FAILS SAFE" rule.
    """
    try:
        loaded = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
    except Exception:
        return  # unreachable: forward anyway, same as comfyui-proxy

    for m in loaded:
        name = m.get("model") or m.get("name")
        if not name:
            continue
        try:
            requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=30,
            )
        except Exception:
            pass

    # Wait for the driver to actually give the memory back. Ollama returns
    # from the eviction request before the VRAM is released.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if not requests.get(f"{OLLAMA_URL}/api/ps", timeout=5).json().get("models"):
                return
        except Exception:
            return
        time.sleep(0.25)


def generate(prompt: str, seed: int | None = None, cancel_event=None):
    """Generate a single image and return a PIL.Image. Blocking — run off the
    UI thread. If cancel_event is set partway through, raises
    GenerationCancelled (checked between denoising steps, so it aborts
    within roughly one step rather than running to completion)."""
    unload_llm()
    pipe = _load_pipeline()
    generator = torch.Generator("cpu")
    if seed is not None:
        generator = generator.manual_seed(seed)

    def check_cancel(pipe, step, timestep, callback_kwargs):
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled()
        return callback_kwargs

    image = pipe(
        prompt=prompt,
        negative_prompt="cartoon, illustration, anime, low quality, blurry, deformed",
        num_inference_steps=30,
        guidance_scale=6.5,
        width=1024,
        height=1024,
        generator=generator,
        callback_on_step_end=check_cancel if cancel_event is not None else None,
    ).images[0]
    return image
