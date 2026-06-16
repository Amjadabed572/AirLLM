"""Immutable project constants (guidelines §7.3).

Only true constants live here — physical/mathematical facts and fixed mappings.
Anything tunable (model id, prices, power, paths) belongs in config/*.json and is
read through shared.config.Config, never hardcoded.
"""
from __future__ import annotations

from enum import Enum

# Bytes per parameter when weights are stored at FP16 (used for size estimates).
FP16_BYTES_PER_PARAM: int = 2

# Nanoseconds per second — Ollama reports durations in ns.
NS_PER_SECOND: float = 1e9

# SATA-SSD sequential read ceiling (GB/s), for the roofline disk-tier note.
SATA_SSD_READ_GBPS: float = 0.5


class Quant(str, Enum):
    """Quantization levels compared across engines."""

    FP16 = "fp16"
    Q8 = "q8"
    Q4 = "q4"


# AirLLM `compression` argument per quantization label (None = full FP16).
QUANT_TO_COMPRESSION: dict[str, str | None] = {
    Quant.FP16.value: None,
    Quant.Q8.value: "8bit",
    Quant.Q4.value: "4bit",
}

# Ollama GGUF tags per quantization label, Qwen2.5-7B family (llama.cpp presets).
QUANT_TO_OLLAMA_TAG: dict[str, str] = {
    Quant.Q4.value: "qwen2.5:7b-instruct-q4_K_M",
    Quant.Q8.value: "qwen2.5:7b-instruct-q8_0",
    Quant.FP16.value: "qwen2.5:7b-instruct-fp16",
}
