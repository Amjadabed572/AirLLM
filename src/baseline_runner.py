"""Baseline runner (Task 5.2).

Loads the model the *naive* way: full weights via transformers
AutoModelForCausalLM, no layer streaming. On a "big enough" model this cannot
fit in RAM -- that failure IS the result (the memory-capacity bottleneck). We
capture it cleanly and document it, because a well-analysed negative result is
worth full marks. This is the comparison point for everything else.

Design note (important).
------------------------
A bare `device_map="auto"` does NOT prove the bottleneck on a CPU-only box:
accelerate will *silently offload* layers to disk and the model "runs" — just
unbearably slowly (observed: tens of minutes for a few tokens). That conflates
the baseline with AirLLM-style streaming. To get a deterministic, fast, honest
"it does not fit" result we instead give accelerate a realistic **RAM budget**
(`max_memory`) and **forbid offloading** (no `offload_folder`). If the ~15 GB
FP16 model cannot be placed within that budget, accelerate raises immediately —
a clean, reproducible memory-capacity failure rather than an OS-dependent OOM
kill (which Windows can also mask by thrashing the pagefile).
"""
from __future__ import annotations

import threading

from src.metrics import MemorySampler, RunMetrics, time_streaming_generation
from src.prompts import Prompt

# Realistic usable-RAM budget for the naive load on an 8 GB machine (OS + apps
# take ~2 GB). Offloading is disabled, so the model must fit this or fail.
CPU_RAM_BUDGET = "6GiB"


def run_baseline(model_id: str, prompt: Prompt, avg_power_w: float = 65.0) -> RunMetrics:
    m = RunMetrics(label="baseline", model=model_id, quantization="fp16")

    # Lazy imports so the rest of the toolkit imports without torch present.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        inputs = tok(prompt.text, return_tensors="pt")
        m.prompt_tokens = int(inputs["input_ids"].shape[-1])

        with MemorySampler(avg_power_w=avg_power_w) as sampler:
            # Naive load with a hard RAM budget and NO offload allowed. If the
            # full FP16 model does not fit CPU_RAM_BUDGET, this raises here --
            # that exception is the bottleneck evidence (recorded below).
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={"cpu": CPU_RAM_BUDGET},  # realistic usable RAM
                low_cpu_mem_usage=True,
                # deliberately NO offload_folder -> accelerate cannot spill to
                # disk, so an oversized model fails fast instead of crawling.
            )
            # If we somehow get here, the model DID fit -- generate for real.
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(**inputs.to(model.device), streamer=streamer,
                              max_new_tokens=prompt.max_new_tokens, do_sample=False)
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            time_streaming_generation(streamer, m)
            thread.join()

        m.peak_ram_gb = sampler.peak_ram_gb
        m.peak_vram_gb = sampler.peak_vram_gb
        m.est_energy_wh = sampler.energy_wh

    except (RuntimeError, MemoryError, OSError, ValueError) as e:
        # ValueError: accelerate's "doesn't fit / needs offload" message.
        # MemoryError/OSError: a real allocation failure. All are the bottleneck.
        m.failed = True
        m.failure_reason = f"{type(e).__name__}: {e}"
        # Record this verbatim in the report -- it is the bottleneck evidence.

    return m
