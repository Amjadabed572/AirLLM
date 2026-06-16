"""AirLLM runner (Tasks 5.3 / 5.4).

Runs the SAME prompt through AirLLM, which streams one transformer layer into
memory at a time (layer sharding). This is what lets a massive model run on
hardware that could never hold it whole.

Key choices, exactly as the assignment's Do-list requires:
  * AutoModel (the GENERAL class) -> the toolkit matches the right architecture
    itself and avoids the Class-mismatch error.
  * layer_shards_saving_path -> shards are written to a FAST, DEDICATED drive,
    NOT the OS drive. On Windows point this at a fast NVMe path.
  * compression sweep: None (fp16) -> "8bit" (Q8) -> "4bit" (Q4).
"""
from __future__ import annotations

from src.metrics import MemorySampler, RunMetrics
from src.prompts import Prompt

# Map our quant label -> AirLLM `compression` argument.
QUANT_TO_COMPRESSION: dict[str, str | None] = {
    "fp16": None,
    "q8": "8bit",
    "q4": "4bit",
}


def run_airllm(
    model_id: str,
    prompt: Prompt,
    quant: str,
    layer_shards_saving_path: str,
    avg_power_w: float = 65.0,
) -> RunMetrics:
    if quant not in QUANT_TO_COMPRESSION:
        raise ValueError(f"quant must be one of {list(QUANT_TO_COMPRESSION)}")

    m = RunMetrics(label=f"airllm-{quant}", model=model_id, quantization=quant)

    import time

    import torch
    from airllm import AutoModel  # GENERAL class -> avoids class-mismatch error

    try:
        kwargs: dict = {"layer_shards_saving_path": layer_shards_saving_path}
        compression = QUANT_TO_COMPRESSION[quant]
        if compression is not None:
            kwargs["compression"] = compression  # needs bitsandbytes

        model = AutoModel.from_pretrained(model_id, **kwargs)

        enc = model.tokenizer(
            [prompt.text], return_tensors="pt", return_attention_mask=False,
            truncation=True, padding=False,
        )
        m.prompt_tokens = int(enc["input_ids"].shape[-1])

        with MemorySampler(avg_power_w=avg_power_w) as sampler:
            # AirLLM's generate is not a token streamer, so we time prefill
            # (first forward = TTFT proxy) separately from full generation.
            t0 = time.perf_counter()
            _ = model(enc["input_ids"])  # single forward ~ prefill cost
            m.ttft_s = time.perf_counter() - t0

            gen_t0 = time.perf_counter()
            out = model.generate(
                enc["input_ids"].to(model.device if hasattr(model, "device") else "cpu"),
                max_new_tokens=prompt.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
            )
            m.total_gen_s = time.perf_counter() - gen_t0
            seq = out.sequences if hasattr(out, "sequences") else out
            m.output_tokens = int(seq.shape[-1]) - m.prompt_tokens

        m.peak_ram_gb = sampler.peak_ram_gb
        m.peak_vram_gb = sampler.peak_vram_gb
        m.est_energy_wh = sampler.energy_wh
        m.finalize()

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except (RuntimeError, MemoryError, OSError, ValueError) as e:
        m.failed = True
        m.failure_reason = f"{type(e).__name__}: {e}"

    return m
