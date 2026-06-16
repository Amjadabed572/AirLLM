"""Unit tests for the model selection heuristic."""
from __future__ import annotations

from airllm_bench.services.model_selector import explain, recommend


def test_small_disk_falls_back_to_7b() -> None:
    choice = recommend(ram_gb=8, disk_free_gb=40)
    assert "7B" in choice.repo_id


def test_large_machine_picks_heavier_model() -> None:
    choice = recommend(ram_gb=64, disk_free_gb=400)
    assert choice.params_b >= 32


def test_low_ram_gate_blocks_32b_plus() -> None:
    # Ample disk but only 8 GB RAM -> 32B/72B gated out, 14B fits disk.
    choice = recommend(ram_gb=8, disk_free_gb=400)
    assert choice.params_b <= 14


def test_tiny_machine_returns_fallback() -> None:
    choice = recommend(ram_gb=4, disk_free_gb=5)
    assert "7B" in choice.repo_id


def test_explain_contains_key_facts() -> None:
    choice = recommend(ram_gb=8, disk_free_gb=40)
    text = explain(choice, 8, 40)
    assert choice.repo_id in text
    assert "disk" in text.lower()
