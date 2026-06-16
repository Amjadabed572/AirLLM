"""Run the GGUF quantization sweep (Q4/Q8) via Ollama and save results.

This is the quantization comparison the assignment requires, runnable on a
CPU-only / weak-GPU machine where AirLLM's bitsandbytes path is unavailable.

    python -m experiments.run_ollama
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import AVG_POWER_W, OLLAMA_QUANT_SWEEP, PROMPT_NAMES  # noqa: E402
from src.ollama_runner import run_ollama  # noqa: E402
from src.prompts import PROMPTS  # noqa: E402


def main() -> None:
    os.makedirs("results", exist_ok=True)
    by_name = {p.name: p for p in PROMPTS}
    print(f"[ollama] quant sweep={OLLAMA_QUANT_SWEEP}")
    for quant in OLLAMA_QUANT_SWEEP:
        for name in PROMPT_NAMES:
            prompt = by_name[name]
            print(f"  -> {quant} / prompt '{name}' ...", flush=True)
            m = run_ollama(quant, prompt, avg_power_w=AVG_POWER_W)
            out = f"results/ollama_{quant}_{name}.json"
            m.save(out)
            status = "FAILED: " + m.failure_reason if m.failed else (
                f"TTFT={m.ttft_s:.2f}s  TPOT={m.tpot_s*1000:.1f}ms  "
                f"tok/s={m.throughput_tok_s:.2f}  peakRAM={m.peak_ram_gb:.1f}GB"
            )
            print(f"     {status}\n     saved -> {out}")


if __name__ == "__main__":
    main()
