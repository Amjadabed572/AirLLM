"""Baseline runner (Task 5.2).

Loads the model the *naive* way (full weights, no layer streaming). On a model
too big for RAM this cannot fit -- that failure IS the result (the memory-capacity
bottleneck). To get a deterministic, fast, honest "it does not fit" outcome we
give accelerate a realistic RAM budget (max_memory) and FORBID offloading (no
offload_folder): an oversized FP16 model then raises immediately rather than
silently disk-offloading and crawling, or being OOM-killed by the OS.
"""
from __future__ import annotations

import threading

from airllm_bench.services.metrics import (
    MemorySampler,
    RunMetrics,
    time_streaming_generation,
)
from airllm_bench.services.prompts import Prompt

# Realistic usable-RAM budget for the naive load on an 8 GB machine (OS + apps
# take ~2 GB). Offloading is disabled, so the model must fit this or fail.
CPU_RAM_BUDGET = "6GiB"


def run_baseline(model_id: str, prompt: Prompt, avg_power_w: float = 65.0) -> RunMetrics:
    """Attempt a naive full-RAM load; capture success metrics or the failure."""
    metrics = RunMetrics(label="baseline", model=model_id, quantization="fp16")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        inputs = tok(prompt.text, return_tensors="pt")
        metrics.prompt_tokens = int(inputs["input_ids"].shape[-1])

        with MemorySampler(avg_power_w=avg_power_w) as sampler:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={"cpu": CPU_RAM_BUDGET},
                low_cpu_mem_usage=True,
                # No offload_folder -> oversized model fails fast, no disk spill.
            )
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                **inputs.to(model.device), streamer=streamer,
                max_new_tokens=prompt.max_new_tokens, do_sample=False,
            )
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            time_streaming_generation(streamer, metrics)
            thread.join()

        metrics.peak_ram_gb = sampler.peak_ram_gb
        metrics.peak_vram_gb = sampler.peak_vram_gb
        metrics.est_energy_wh = sampler.energy_wh

    except (RuntimeError, MemoryError, OSError, ValueError) as exc:
        # ValueError: accelerate's "doesn't fit / needs offload". The rest are
        # real allocation failures. All are the bottleneck evidence.
        metrics.failed = True
        metrics.failure_reason = f"{type(exc).__name__}: {exc}"

    return metrics
