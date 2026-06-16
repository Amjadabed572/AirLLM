"""Ollama / GGUF quantization runner (Tasks 5.3 / 5.4 — quantization sweep).

Why this exists on THIS hardware
--------------------------------
AirLLM's Q4/Q8 path goes through `bitsandbytes`, which needs a CUDA GPU with
compute capability >= 7.5 and several GB of VRAM. The lab machine's GPU is a
Maxwell GTX 950M (compute 5.0, 2 GB) — it cannot run bitsandbytes quantization.
So AirLLM here gives us the **FP16 layer-streaming** demonstration, and we get
the real **quantization comparison (FP16-ish vs Q8 vs Q4)** from **GGUF models
via Ollama**, which quantizes and runs entirely on the CPU (llama.cpp backend).

Both are legitimate, both produce real numbers. The assignment explicitly lists
Ollama and GGUF as in-scope tools.

Ollama returns *native* timing counters, which are more accurate than wall-clock
wrapping:
  * prompt_eval_count / prompt_eval_duration  -> prefill  -> TTFT
  * eval_count        / eval_duration         -> decode   -> TPOT, throughput
Durations are in nanoseconds.

Run standalone:  python -m experiments.run_ollama
Prereqs: install Ollama (https://ollama.com) and `uv pip install ollama`.
"""
from __future__ import annotations

import threading
import time

from src.metrics import RunMetrics
from src.prompts import Prompt

# Our quant label -> an Ollama GGUF tag for the same Qwen2.5-7B model family.
# q4_K_M and q8_0 are the standard llama.cpp quantization presets.
QUANT_TO_OLLAMA_TAG: dict[str, str] = {
    "q4": "qwen2.5:7b-instruct-q4_K_M",
    "q8": "qwen2.5:7b-instruct-q8_0",
    # "fp16" GGUF is large (~15 GB) and won't fit RAM here; AirLLM covers FP16.
    "fp16": "qwen2.5:7b-instruct-fp16",
}


def _sample_system_ram(stop: threading.Event, peak: list[float]) -> None:
    """Ollama runs the model in a *separate* server process, so per-process RSS
    of this script is meaningless. We sample whole-system used RAM instead and
    record the peak. Document this caveat in the report."""
    import psutil

    base = psutil.virtual_memory().used / (1024**3)
    while not stop.is_set():
        used = psutil.virtual_memory().used / (1024**3)
        peak[0] = max(peak[0], used - base)
        time.sleep(0.1)


def run_ollama(quant: str, prompt: Prompt, avg_power_w: float = 15.0) -> RunMetrics:
    if quant not in QUANT_TO_OLLAMA_TAG:
        raise ValueError(f"quant must be one of {list(QUANT_TO_OLLAMA_TAG)}")
    tag = QUANT_TO_OLLAMA_TAG[quant]
    m = RunMetrics(label=f"ollama-{quant}", model=tag, quantization=quant)

    try:
        import ollama
    except ImportError:
        m.failed = True
        m.failure_reason = "ImportError: `uv pip install ollama` and install the Ollama app"
        return m

    try:
        # Ensure the model is present (no-op if already pulled).
        ollama.pull(tag)

        stop = threading.Event()
        peak = [0.0]
        sampler = threading.Thread(target=_sample_system_ram, args=(stop, peak), daemon=True)
        sampler.start()

        wall0 = time.perf_counter()
        resp = ollama.generate(
            model=tag,
            prompt=prompt.text,
            options={"num_predict": prompt.max_new_tokens, "temperature": 0.0},
        )
        wall = time.perf_counter() - wall0

        stop.set()
        sampler.join()

        # Native Ollama counters (nanoseconds). Fall back to wall clock if absent.
        ns = 1e9
        m.prompt_tokens = int(resp.get("prompt_eval_count", 0))
        m.output_tokens = int(resp.get("eval_count", 0))
        load_s = resp.get("load_duration", 0) / ns
        prefill_s = resp.get("prompt_eval_duration", 0) / ns
        decode_s = resp.get("eval_duration", 0) / ns

        # TTFT = model load + prefill (first token appears after prefill finishes).
        m.ttft_s = load_s + prefill_s
        m.total_gen_s = m.ttft_s + decode_s if decode_s else wall
        # TPOT straight from decode counters (most accurate).
        if m.output_tokens > 1 and decode_s:
            m.tpot_s = decode_s / (m.output_tokens - 1)
        m.throughput_tok_s = m.output_tokens / decode_s if decode_s else 0.0
        m.peak_ram_gb = peak[0]
        m.peak_vram_gb = 0.0  # CPU/llama.cpp path
        m.est_energy_wh = avg_power_w * (m.total_gen_s / 3600.0)

    except Exception as e:  # noqa: BLE001 — record any runtime failure verbatim
        m.failed = True
        m.failure_reason = f"{type(e).__name__}: {e}"

    return m
