"""Model selection helper.

The assignment wants a model that is "big enough to hurt, but not impossible"
to run on your hardware via AirLLM. AirLLM loads ONE transformer layer into
memory at a time, so the binding constraint is *disk* (to hold the shards) and
the size of a single layer, NOT the full model size in RAM. That is exactly why
AirLLM lets a 70B model run on a machine that could never hold it in RAM at once.

This table maps your total RAM to a sensible "hurts but works" choice. Override
freely in your config — justify your pick in the report (Task 5.1).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelChoice:
    repo_id: str
    params_b: float
    note: str


# Ordered from heaviest to lightest. We pick the heaviest model whose rough
# disk footprint (~2 bytes/param at FP16) fits comfortably in free disk and
# whose single-layer working set fits in the given RAM budget.
_CANDIDATES = [
    ModelChoice("Qwen/Qwen2.5-72B-Instruct", 72, "Flagship; needs ~145GB disk for FP16 shards"),
    ModelChoice("Qwen/Qwen2.5-32B-Instruct", 32, "Heavy but tractable on a good laptop+NVMe"),
    ModelChoice("Qwen/Qwen2.5-14B-Instruct", 14, "Sweet spot: clearly stresses 16GB RAM machines"),
    ModelChoice("Qwen/Qwen2.5-7B-Instruct", 7, "Light; use only for the smoke test"),
]


def recommend(ram_gb: float, disk_free_gb: float) -> ModelChoice:
    """Heuristic pick. Disk must hold ~2*params GB of FP16 shards (+ headroom).
    RAM only needs to hold a single layer + activations, so even 16GB RAM can
    run 14B-32B through AirLLM if the disk is large and fast enough."""
    for c in _CANDIDATES:
        disk_needed = c.params_b * 2.2  # FP16 shards + a little slack
        if disk_free_gb < disk_needed + 15:
            continue
        # Bigger models => bigger single layers; keep a soft RAM gate.
        if c.params_b >= 32 and ram_gb < 16:
            continue
        return c
    return _CANDIDATES[-1]  # fall back to the 7B smoke-test model


def explain(choice: ModelChoice, ram_gb: float, disk_free_gb: float) -> str:
    return (
        f"Recommended model: {choice.repo_id} (~{choice.params_b}B params)\n"
        f"  Reason: {choice.note}\n"
        f"  Your RAM: {ram_gb:.0f} GB | Free disk: {disk_free_gb:.0f} GB\n"
        f"  FP16 shards need ~{choice.params_b * 2.2:.0f} GB on disk.\n"
        f"  Justify this choice explicitly in the report (Task 5.1)."
    )


if __name__ == "__main__":
    from src.hardware import detect

    hw = detect()
    pick = recommend(hw.ram_total_gb, hw.disk_free_gb)
    print(explain(pick, hw.ram_total_gb, hw.disk_free_gb))
