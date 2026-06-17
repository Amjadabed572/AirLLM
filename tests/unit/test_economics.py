"""Unit tests for the economic models."""
from __future__ import annotations

import numpy as np

from airllm_bench.services.economics import (
    APIParams,
    CloudGPUParams,
    OnPremParams,
    Scenario,
    summarize,
)


def test_api_cost_per_request() -> None:
    api = APIParams(price_in_per_mtok=3.0, price_out_per_mtok=15.0,
                    in_tokens=1_000_000, out_tokens=0)
    assert api.cost_per_request() == 3.0


def test_api_cache_discount_lowers_cost() -> None:
    base = APIParams(cached_input_fraction=0.0)
    cached = APIParams(cached_input_fraction=1.0, cache_price_multiplier=0.1)
    assert cached.cost_per_request() < base.cost_per_request()


def test_onprem_variable_cost() -> None:
    op = OnPremParams(energy_wh_per_request=1000.0, electricity_usd_per_kwh=0.2)
    assert op.variable_cost_per_request() == 0.2


def test_cloud_cost_per_request() -> None:
    cloud = CloudGPUParams(gpu_hourly_usd=3600.0, seconds_per_request=1.0)
    assert cloud.cost_per_request() == 1.0


def test_break_even_positive() -> None:
    scn = Scenario()
    be = scn.break_even()
    assert be is not None and be > 0


def test_break_even_none_when_api_cheaper() -> None:
    scn = Scenario(api=APIParams(in_tokens=0, out_tokens=0))
    assert scn.break_even() is None


def test_curves_include_cloud_when_enabled() -> None:
    scn = Scenario(cloud=CloudGPUParams(enabled=True))
    curves = scn.curves(np.array([1.0, 10.0]))
    assert "Cloud GPU (rented)" in curves


def test_from_config_roundtrip(cfg) -> None:
    scn = Scenario.from_config(cfg.section("economics"))
    assert scn.api.in_tokens == 600
    assert scn.onprem.hardware_cost_usd == 2500.0


def test_summarize_mentions_break_even() -> None:
    assert "Break-even" in summarize(Scenario())
    assert "NEVER" in summarize(Scenario(api=APIParams(in_tokens=0, out_tokens=0)))


def test_summarize_includes_cloud_line() -> None:
    text = summarize(Scenario(cloud=CloudGPUParams(enabled=True)))
    assert "Cloud GPU" in text
