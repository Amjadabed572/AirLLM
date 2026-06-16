"""Unit tests for analysis orchestration."""
from __future__ import annotations

import json
from pathlib import Path

from airllm_bench.services.analyze import (
    analyze,
    load_results,
    make_figures,
    markdown_table,
)
from airllm_bench.services.economics import Scenario


def _write_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    ok = {"label": "airllm-fp16", "quantization": "fp16", "ttft_s": 1.0, "tpot_s": 0.5,
          "throughput_tok_s": 2.0, "peak_ram_gb": 4.0, "peak_vram_gb": 0.0,
          "est_energy_wh": 3.0, "failed": False}
    bad = {"label": "baseline", "quantization": "fp16", "failed": True}
    (results_dir / "airllm_fp16_short.json").write_text(json.dumps(ok), encoding="utf-8")
    (results_dir / "baseline_short.json").write_text(json.dumps(bad), encoding="utf-8")
    (results_dir / "hardware.json").write_text(json.dumps({"os": "x"}), encoding="utf-8")


def test_load_results_skips_hardware(tmp_path: Path) -> None:
    _write_results(tmp_path / "results")
    rows = load_results(str(tmp_path / "results"))
    labels = {r["label"] for r in rows}
    assert labels == {"airllm-fp16", "baseline"}  # hardware.json excluded


def test_markdown_table_has_rows(tmp_path: Path) -> None:
    _write_results(tmp_path / "results")
    table = markdown_table(load_results(str(tmp_path / "results")))
    assert "airllm-fp16" in table
    assert "FAILED" in table


def test_make_figures_creates_files(tmp_path: Path, monkeypatch) -> None:
    _write_results(tmp_path / "results")
    monkeypatch.chdir(tmp_path)
    rows = load_results("results")
    paths = make_figures(rows, Scenario())
    assert any("break_even" in p for p in paths)
    assert any("throughput" in p for p in paths)  # ok row present
    for p in paths:
        assert Path(p).exists()


def test_analyze_writes_summary(tmp_path: Path, monkeypatch) -> None:
    _write_results(tmp_path / "results")
    monkeypatch.chdir(tmp_path)
    out = analyze(Scenario(), results_dir="results")
    assert "Break-even" in out
    assert (tmp_path / "results" / "summary_table.md").exists()
