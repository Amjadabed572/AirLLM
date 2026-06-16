"""Run the baseline (naive direct load) and save results.

    python -m experiments.run_baseline
"""
from __future__ import annotations

import json
import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import AVG_POWER_W, MODEL_ID, PROMPT_NAMES  # noqa: E402
from src.baseline_runner import run_baseline  # noqa: E402
from src.hardware import detect  # noqa: E402
from src.metrics import RunMetrics  # noqa: E402
from src.model_selector import recommend  # noqa: E402
from src.prompts import PROMPTS  # noqa: E402


def resolve_model() -> str:
    if MODEL_ID:
        return MODEL_ID
    hw = detect()
    return recommend(hw.ram_total_gb, hw.disk_free_gb).repo_id


def main() -> None:
    os.makedirs("results", exist_ok=True)
    model = resolve_model()
    by_name = {p.name: p for p in PROMPTS}
    print(f"[baseline] model={model}")
    for name in PROMPT_NAMES:
        prompt = by_name[name]
        out = f"results/baseline_{name}.json"
        # Provisional record: the naive load can hard-OOM and get KILLED by the
        # OS mid-load, leaving no chance to catch an exception. Write a failure
        # record up front so that outcome is still captured; we overwrite it
        # below only if the run actually returns.
        provisional = RunMetrics(label="baseline", model=model, quantization="fp16")
        provisional.failed = True
        provisional.failure_reason = (
            "Process killed by OS during naive in-RAM load (hard OOM) — model "
            "weights exceed physical RAM; the load did not complete. (If you see "
            "this, the run never returned to Python.)"
        )
        provisional.save(out)

        print(f"  -> prompt '{name}' ...", flush=True)
        m = run_baseline(model, prompt, avg_power_w=AVG_POWER_W)
        m.save(out)  # overwrite provisional with the actual outcome
        status = "FAILED: " + m.failure_reason if m.failed else (
            f"TTFT={m.ttft_s:.2f}s  TPOT={m.tpot_s*1000:.1f}ms  "
            f"tok/s={m.throughput_tok_s:.2f}  peakRAM={m.peak_ram_gb:.1f}GB"
        )
        print(f"     {status}\n     saved -> {out}")
    # Also stamp the hardware sheet.
    from dataclasses import asdict
    with open("results/hardware.json", "w") as f:
        json.dump(asdict(detect()), f, indent=2)


if __name__ == "__main__":
    main()
