"""Unit tests for constants and version metadata."""
from __future__ import annotations

from airllm_bench.constants import (
    QUANT_TO_COMPRESSION,
    QUANT_TO_OLLAMA_TAG,
    Quant,
)
from airllm_bench.shared.version import COMPATIBLE_CONFIG_VERSIONS, __version__


def test_version_is_baseline() -> None:
    assert __version__ == "1.00"
    assert __version__ in COMPATIBLE_CONFIG_VERSIONS


def test_quant_enum_values() -> None:
    assert {q.value for q in Quant} == {"fp16", "q8", "q4"}


def test_compression_map_matches_quant() -> None:
    assert QUANT_TO_COMPRESSION["fp16"] is None
    assert QUANT_TO_COMPRESSION["q4"] == "4bit"
    assert QUANT_TO_COMPRESSION["q8"] == "8bit"


def test_ollama_tags_cover_quant_levels() -> None:
    assert set(QUANT_TO_OLLAMA_TAG) == {"fp16", "q8", "q4"}
    assert "q4_K_M" in QUANT_TO_OLLAMA_TAG["q4"]


def test_package_exports_sdk_and_version() -> None:
    import airllm_bench

    assert airllm_bench.__version__ == "1.00"
    assert hasattr(airllm_bench, "AirLLMBenchSDK")
