"""Model selection helper (Task 5.1).

AirLLM loads ONE transformer layer into memory at a time, so the binding
constraint is *disk* (to hold the shards) and the size of a single layer, not the
full model size in RAM. This table maps detected RAM + free disk to a sensible
"hurts but works" choice; the final pick is justified in the report.
"""
from __future__ import annotations

from dataclasses import dataclass

from airllm_bench.constants import FP16_BYTES_PER_PARAM


@dataclass(frozen=True)
class ModelChoice:
    """A candidate model with the facts needed to justify picking it."""

    repo_id: str
    params_b: float
    note: str


# Heaviest to lightest. We pick the heaviest model whose FP16 shard footprint
# (~2 bytes/param) fits free disk and whose single layer fits the RAM budget.
_CANDIDATES: list[ModelChoice] = [
    ModelChoice("Qwen/Qwen2.5-72B-Instruct", 72, "Flagship; ~145GB disk for FP16 shards"),
    ModelChoice("Qwen/Qwen2.5-32B-Instruct", 32, "Heavy but tractable on a good laptop+NVMe"),
    ModelChoice("Qwen/Qwen2.5-14B-Instruct", 14, "Sweet spot for 16GB+ RAM machines with ample disk"),
    ModelChoice(
        "Qwen/Qwen2.5-7B-Instruct",
        7,
        "Right 'hurts but works' pick for an 8GB-RAM laptop: FP16 (~15GB) far "
        "exceeds RAM so the naive baseline fails, yet shards fit a ~40GB disk",
    ),
]


def recommend(ram_gb: float, disk_free_gb: float) -> ModelChoice:
    """Heuristic pick: disk must hold ~2*params GB of FP16 shards (+ headroom)."""
    for choice in _CANDIDATES:
        disk_needed = choice.params_b * FP16_BYTES_PER_PARAM * 1.1
        if disk_free_gb < disk_needed + 15:
            continue
        if choice.params_b >= 32 and ram_gb < 16:
            continue
        return choice
    return _CANDIDATES[-1]


def explain(choice: ModelChoice, ram_gb: float, disk_free_gb: float) -> str:
    """Human-readable justification for the report (Task 5.1)."""
    return (
        f"Recommended model: {choice.repo_id} (~{choice.params_b}B params)\n"
        f"  Reason: {choice.note}\n"
        f"  Your RAM: {ram_gb:.0f} GB | Free disk: {disk_free_gb:.0f} GB\n"
        f"  FP16 shards need ~{choice.params_b * FP16_BYTES_PER_PARAM:.0f} GB on disk."
    )
