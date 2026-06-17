"""Unit tests for the SDK facade (runners mocked — no torch/airllm/ollama)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from airllm_bench.sdk import sdk as sdk_mod
from airllm_bench.sdk.sdk import AirLLMBenchSDK
from airllm_bench.services.metrics import RunMetrics
from airllm_bench.shared.config import Config


@pytest.fixture
def sdk(config_dir: Path, tmp_path: Path, monkeypatch) -> AirLLMBenchSDK:
    monkeypatch.chdir(tmp_path)  # results/ written under tmp
    return AirLLMBenchSDK(Config(config_dir))


def _fake_metrics(label: str) -> RunMetrics:
    m = RunMetrics(label=label, model="m", quantization="fp16")
    m.output_tokens, m.ttft_s, m.total_gen_s = 4, 1.0, 3.0
    m.finalize()
    return m


def test_resolve_model_from_config(sdk: AirLLMBenchSDK) -> None:
    assert sdk.resolve_model() == "Qwen/Qwen2.5-7B-Instruct"


def test_resolve_model_auto_select(config_dir: Path, tmp_path: Path, monkeypatch) -> None:
    setup = json.loads((config_dir / "setup.json").read_text(encoding="utf-8"))
    setup["model"]["auto_select"] = True
    (config_dir / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    model = AirLLMBenchSDK(Config(config_dir)).resolve_model()
    assert "Qwen" in model


def test_save_hardware_writes_file(sdk: AirLLMBenchSDK, tmp_path: Path) -> None:
    sdk.save_hardware()
    assert (tmp_path / "results" / "hardware.json").exists()


def test_recommend_model_text(sdk: AirLLMBenchSDK) -> None:
    assert "Recommended model" in sdk.recommend_model()


def test_run_baseline_saves_record(sdk: AirLLMBenchSDK, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sdk_mod, "run_baseline", lambda *a, **k: _fake_metrics("baseline"))
    m = sdk.run_baseline("short")
    assert not m.failed
    saved = json.loads((tmp_path / "results" / "baseline_short.json").read_text("utf-8"))
    assert saved["label"] == "baseline"


def test_run_airllm_saves_record(sdk: AirLLMBenchSDK, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sdk_mod, "run_airllm", lambda *a, **k: _fake_metrics("airllm-fp16"))
    sdk.run_airllm("fp16", "short")
    assert (tmp_path / "results" / "airllm_fp16_short.json").exists()


def test_run_ollama_saves_record(sdk: AirLLMBenchSDK, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sdk_mod, "run_ollama", lambda *a, **k: _fake_metrics("ollama-q4"))
    sdk.run_ollama("q4", "short")
    assert (tmp_path / "results" / "ollama_q4_short.json").exists()


def test_run_records_prompt(sdk: AirLLMBenchSDK, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sdk_mod, "run_ollama", lambda *a, **k: _fake_metrics("ollama-q4"))
    m = sdk.run_ollama("q4", "medium")
    assert m.prompt == "medium"
    saved = json.loads((tmp_path / "results" / "ollama_q4_medium.json").read_text("utf-8"))
    assert saved["prompt"] == "medium"


def test_input_length_study_runs_all_prompts(sdk: AirLLMBenchSDK, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(sdk_mod, "run_ollama",
                        lambda *a, **k: calls.append(a) or _fake_metrics("ollama-q4"))
    results = sdk.run_input_length_study()
    assert len(results) == 3  # short, medium, long_context


def test_scenario_from_config(sdk: AirLLMBenchSDK) -> None:
    assert sdk.scenario().api.in_tokens == 600


def test_analyze_runs(sdk: AirLLMBenchSDK, tmp_path: Path) -> None:
    out = sdk.analyze()
    assert "Break-even" in out
    assert (tmp_path / "results" / "summary_table.md").exists()
