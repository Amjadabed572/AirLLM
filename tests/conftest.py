"""Shared pytest fixtures (guidelines §6.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from airllm_bench.services.metrics import RunMetrics
from airllm_bench.services.prompts import Prompt
from airllm_bench.shared.config import Config

_SETUP = {
    "version": "1.00",
    "model": {"model_id": "Qwen/Qwen2.5-7B-Instruct", "auto_select": False},
    "experiment": {
        "quant_sweep": ["fp16"],
        "ollama_quant_sweep": ["q4", "q8"],
        "prompt_names": ["short"],
        "avg_power_w": 15.0,
        "layer_shards_path": "layer_shards",
    },
    "economics": {
        "api": {"price_in_per_mtok": 3.0, "price_out_per_mtok": 15.0,
                "in_tokens": 600, "out_tokens": 300},
        "onprem": {"hardware_cost_usd": 2500.0, "lifetime_years": 3.0,
                   "maintenance_usd_per_year": 150.0, "energy_wh_per_request": 12.0,
                   "electricity_usd_per_kwh": 0.17},
        "cloud_gpu": {"enabled": False, "gpu_hourly_usd": 1.2, "seconds_per_request": 8.0},
        "period_years": 1.0,
    },
}
_RATE_LIMITS = {"version": "1.00", "services": {"default": {"requests_per_minute": 30}}}


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A temporary, valid config/ directory."""
    (tmp_path / "setup.json").write_text(json.dumps(_SETUP), encoding="utf-8")
    (tmp_path / "rate_limits.json").write_text(json.dumps(_RATE_LIMITS), encoding="utf-8")
    return tmp_path


@pytest.fixture
def cfg(config_dir: Path) -> Config:
    """A Config bound to the temporary config dir."""
    return Config(config_dir)


@pytest.fixture
def sample_prompt() -> Prompt:
    """A tiny prompt for runner-independent tests."""
    return Prompt(name="t", text="hello", max_new_tokens=4)


@pytest.fixture
def sample_metrics() -> RunMetrics:
    """A populated RunMetrics record."""
    m = RunMetrics(label="airllm-fp16", model="m", quantization="fp16")
    m.prompt_tokens, m.output_tokens, m.ttft_s, m.total_gen_s = 10, 5, 1.0, 5.0
    m.finalize()
    return m
