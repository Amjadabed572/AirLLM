"""Ollama / GGUF quantization runner (Tasks 5.3 / 5.4 — quantization sweep).

AirLLM's Q4/Q8 path needs bitsandbytes + a CUDA GPU of compute capability >= 7.5,
which the target machine's Maxwell GTX 950M (2 GB) cannot provide. So AirLLM gives
the FP16 layer-streaming demonstration, and the real quantization comparison
(Q8 vs Q4) comes from GGUF models via Ollama (llama.cpp, CPU). Both produce real
numbers; the assignment lists Ollama and GGUF as in-scope tools.

Ollama returns native ns-precision counters, more accurate than wall-clock:
  prompt_eval_count / prompt_eval_duration -> prefill -> TTFT
  eval_count        / eval_duration        -> decode  -> TPOT, throughput
"""
from __future__ import annotations

import threading
import time

from airllm_bench.constants import NS_PER_SECOND, QUANT_TO_OLLAMA_TAG
from airllm_bench.services.metrics import RunMetrics
from airllm_bench.services.prompts import Prompt


def _sample_system_ram(stop: threading.Event, peak: list[float]) -> None:
    """Sample whole-system used-RAM delta (Ollama runs in a separate process)."""
    import psutil

    base = psutil.virtual_memory().used / (1024**3)
    while not stop.is_set():
        used = psutil.virtual_memory().used / (1024**3)
        peak[0] = max(peak[0], used - base)
        time.sleep(0.1)


def run_ollama(quant: str, prompt: Prompt, avg_power_w: float = 15.0) -> RunMetrics:
    """Run one prompt through an Ollama GGUF model at the given quant level."""
    if quant not in QUANT_TO_OLLAMA_TAG:
        raise ValueError(f"quant must be one of {list(QUANT_TO_OLLAMA_TAG)}")
    tag = QUANT_TO_OLLAMA_TAG[quant]
    metrics = RunMetrics(label=f"ollama-{quant}", model=tag, quantization=quant)

    try:
        import ollama
    except ImportError:
        metrics.failed = True
        metrics.failure_reason = "ImportError: `uv pip install ollama` and install the Ollama app"
        return metrics

    try:
        ollama.pull(tag)  # no-op if already present

        stop = threading.Event()
        peak = [0.0]
        sampler = threading.Thread(target=_sample_system_ram, args=(stop, peak), daemon=True)
        sampler.start()

        wall0 = time.perf_counter()
        resp = ollama.generate(
            model=tag, prompt=prompt.text,
            options={"num_predict": prompt.max_new_tokens, "temperature": 0.0},
        )
        wall = time.perf_counter() - wall0

        stop.set()
        sampler.join()

        metrics.prompt_tokens = int(resp.get("prompt_eval_count", 0))
        metrics.output_tokens = int(resp.get("eval_count", 0))
        load_s = resp.get("load_duration", 0) / NS_PER_SECOND
        prefill_s = resp.get("prompt_eval_duration", 0) / NS_PER_SECOND
        decode_s = resp.get("eval_duration", 0) / NS_PER_SECOND

        metrics.ttft_s = load_s + prefill_s
        metrics.total_gen_s = metrics.ttft_s + decode_s if decode_s else wall
        if metrics.output_tokens > 1 and decode_s:
            metrics.tpot_s = decode_s / (metrics.output_tokens - 1)
        metrics.throughput_tok_s = metrics.output_tokens / decode_s if decode_s else 0.0
        metrics.peak_ram_gb = peak[0]
        metrics.est_energy_wh = avg_power_w * (metrics.total_gen_s / 3600.0)

    except Exception as exc:  # noqa: BLE001 — record any runtime failure verbatim
        metrics.failed = True
        metrics.failure_reason = f"{type(exc).__name__}: {exc}"

    return metrics
