"""Run the AirLLM quantization sweep (fp16/q8/q4) and save results.

    python -m experiments.run_airllm
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import (  # noqa: E402
    AVG_POWER_W,
    LAYER_SHARDS_PATH,
    PROMPT_NAMES,
    QUANT_SWEEP,
)
from experiments.run_baseline import resolve_model  # noqa: E402
from src.airllm_runner import run_airllm  # noqa: E402
from src.prompts import PROMPTS  # noqa: E402


def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs(LAYER_SHARDS_PATH, exist_ok=True)
    model = resolve_model()
    by_name = {p.name: p for p in PROMPTS}
    print(f"[airllm] model={model}  shards={LAYER_SHARDS_PATH}")
    for quant in QUANT_SWEEP:
        for name in PROMPT_NAMES:
            prompt = by_name[name]
            print(f"  -> {quant} / prompt '{name}' ...", flush=True)
            m = run_airllm(model, prompt, quant, LAYER_SHARDS_PATH, avg_power_w=AVG_POWER_W)
            out = f"results/airllm_{quant}_{name}.json"
            m.save(out)
            status = "FAILED: " + m.failure_reason if m.failed else (
                f"TTFT={m.ttft_s:.2f}s  TPOT={m.tpot_s*1000:.1f}ms  "
                f"tok/s={m.throughput_tok_s:.2f}  "
                f"peakRAM={m.peak_ram_gb:.1f}GB  peakVRAM={m.peak_vram_gb:.1f}GB"
            )
            print(f"     {status}\n     saved -> {out}")


if __name__ == "__main__":
    main()
