"""Baseline runner (Task 5.2).

Loads the model the *naive* way: full weights via transformers
AutoModelForCausalLM, no layer streaming. On a "big enough" model this is
expected to OOM, thrash swap, or be unbearably slow on modest hardware --
that IS the result. We capture the failure cleanly and document it, because a
well-analysed negative result is worth full marks.

This is the comparison point for everything else.
"""
from __future__ import annotations

import threading

from src.metrics import MemorySampler, RunMetrics, time_streaming_generation
from src.prompts import Prompt


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
            # The line that typically fails on a massive model with small RAM/VRAM.
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
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

    except (RuntimeError, MemoryError, OSError) as e:  # OOM, swap death, etc.
        m.failed = True
        m.failure_reason = f"{type(e).__name__}: {e}"
        # Record this verbatim in the report -- it is the bottleneck evidence.

    return m
