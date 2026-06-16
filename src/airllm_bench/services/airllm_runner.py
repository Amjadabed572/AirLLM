"""AirLLM runner (Tasks 5.3 / 5.4).

Runs the SAME prompt through AirLLM, which streams one transformer layer into
memory at a time (layer sharding). This is what lets a model too big for RAM run
at all. Uses the general `AutoModel` class so the architecture is matched
automatically (avoids the Qwen class-mismatch error), and writes shards to a
configurable path so they do not flood the OS drive.
"""
from __future__ import annotations

import time

from airllm_bench.constants import QUANT_TO_COMPRESSION
from airllm_bench.services.metrics import MemorySampler, RunMetrics
from airllm_bench.services.prompts import Prompt


def run_airllm(
    model_id: str,
    prompt: Prompt,
    quant: str,
    layer_shards_saving_path: str,
    avg_power_w: float = 65.0,
) -> RunMetrics:
    """Run one prompt through AirLLM at the given quantization level."""
    if quant not in QUANT_TO_COMPRESSION:
        raise ValueError(f"quant must be one of {list(QUANT_TO_COMPRESSION)}")

    metrics = RunMetrics(label=f"airllm-{quant}", model=model_id, quantization=quant)

    import torch
    from airllm import AutoModel  # general class -> avoids class-mismatch error

    try:
        kwargs: dict = {"layer_shards_saving_path": layer_shards_saving_path}
        compression = QUANT_TO_COMPRESSION[quant]
        if compression is not None:
            kwargs["compression"] = compression  # needs bitsandbytes + CUDA GPU

        model = AutoModel.from_pretrained(model_id, **kwargs)

        enc = model.tokenizer(
            [prompt.text], return_tensors="pt", return_attention_mask=False,
            truncation=True, padding=False,
        )
        metrics.prompt_tokens = int(enc["input_ids"].shape[-1])

        with MemorySampler(avg_power_w=avg_power_w) as sampler:
            device = model.device if hasattr(model, "device") else "cpu"
            t0 = time.perf_counter()
            _ = model(enc["input_ids"])  # single forward ~ prefill cost (TTFT)
            metrics.ttft_s = time.perf_counter() - t0

            gen_t0 = time.perf_counter()
            out = model.generate(
                enc["input_ids"].to(device),
                max_new_tokens=prompt.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
            )
            metrics.total_gen_s = time.perf_counter() - gen_t0
            seq = out.sequences if hasattr(out, "sequences") else out
            metrics.output_tokens = int(seq.shape[-1]) - metrics.prompt_tokens

        metrics.peak_ram_gb = sampler.peak_ram_gb
        metrics.peak_vram_gb = sampler.peak_vram_gb
        metrics.est_energy_wh = sampler.energy_wh
        metrics.finalize()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except (RuntimeError, MemoryError, OSError, ValueError) as exc:
        metrics.failed = True
        metrics.failure_reason = f"{type(exc).__name__}: {exc}"

    return metrics
