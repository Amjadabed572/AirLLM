"""Economic analysis (Task 5.5) -- the break-even between local and API.

Two independent cost models plus optional scenarios:
  1) API     = requests * (in*price_in + out*price_out), with optional
               prompt/context-caching discount on the fixed repeated prefix.
  2) On-Prem = amortized CAPEX + OPEX (electricity from measured energy + upkeep).
  3) Cloud GPU (optional) = gpu_hourly_rate * seconds_per_request.

All assumptions come from config/setup.json (no hardcoded values) so the analysis
is transparent and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class APIParams:
    """Third-party API pricing (USD per 1M tokens) and request shape."""

    price_in_per_mtok: float = 3.0
    price_out_per_mtok: float = 15.0
    in_tokens: int = 600
    out_tokens: int = 300
    cached_input_fraction: float = 0.0
    cache_price_multiplier: float = 0.1

    def cost_per_request(self) -> float:
        """USD per request, applying any prompt-cache discount on the prefix."""
        pin = self.price_in_per_mtok / 1e6
        pout = self.price_out_per_mtok / 1e6
        cached = self.in_tokens * self.cached_input_fraction
        fresh = self.in_tokens - cached
        in_cost = fresh * pin + cached * pin * self.cache_price_multiplier
        return in_cost + self.out_tokens * pout


@dataclass
class OnPremParams:
    """Local hardware CAPEX/OPEX assumptions."""

    hardware_cost_usd: float = 2500.0
    lifetime_years: float = 3.0
    maintenance_usd_per_year: float = 150.0
    energy_wh_per_request: float = 12.0
    electricity_usd_per_kwh: float = 0.17

    def amortized_fixed_for_period(self, years: float) -> float:
        """CAPEX share for the period plus maintenance."""
        capex = self.hardware_cost_usd * (years / self.lifetime_years)
        return capex + self.maintenance_usd_per_year * years

    def variable_cost_per_request(self) -> float:
        """Electricity cost per request from measured energy."""
        return (self.energy_wh_per_request / 1000.0) * self.electricity_usd_per_kwh


@dataclass
class CloudGPUParams:
    """Optional rented-GPU pricing."""

    enabled: bool = False
    gpu_hourly_usd: float = 1.2
    seconds_per_request: float = 8.0

    def cost_per_request(self) -> float:
        """USD per request for a rented GPU."""
        return self.gpu_hourly_usd * (self.seconds_per_request / 3600.0)


@dataclass
class Scenario:
    """Bundles the three cost models and computes curves + break-even."""

    api: APIParams = field(default_factory=APIParams)
    onprem: OnPremParams = field(default_factory=OnPremParams)
    cloud: CloudGPUParams = field(default_factory=CloudGPUParams)
    period_years: float = 1.0

    @classmethod
    def from_config(cls, economics: dict[str, Any]) -> Scenario:
        """Build a Scenario from the config/setup.json `economics` section."""
        return cls(
            api=APIParams(**economics.get("api", {})),
            onprem=OnPremParams(**economics.get("onprem", {})),
            cloud=CloudGPUParams(**economics.get("cloud_gpu", {})),
            period_years=float(economics.get("period_years", 1.0)),
        )

    def curves(self, volumes: np.ndarray) -> dict[str, np.ndarray]:
        """Cumulative USD vs. request volume for each option."""
        api = volumes * self.api.cost_per_request()
        fixed = self.onprem.amortized_fixed_for_period(self.period_years)
        onprem = fixed + volumes * self.onprem.variable_cost_per_request()
        out = {"API (third-party)": api, "On-Prem (local)": onprem}
        if self.cloud.enabled:
            out["Cloud GPU (rented)"] = volumes * self.cloud.cost_per_request()
        return out

    def break_even(self) -> float | None:
        """Requests at which On-Prem becomes cheaper than API (None if never)."""
        api_pr = self.api.cost_per_request()
        var_pr = self.onprem.variable_cost_per_request()
        if api_pr <= var_pr:
            return None
        fixed = self.onprem.amortized_fixed_for_period(self.period_years)
        return fixed / (api_pr - var_pr)


def summarize(scn: Scenario) -> str:
    """Plain-text summary of the cost models and the break-even result."""
    be = scn.break_even()
    fixed = scn.onprem.amortized_fixed_for_period(scn.period_years)
    lines = [
        f"API cost / request:        ${scn.api.cost_per_request():.5f}",
        f"On-Prem variable / request:${scn.onprem.variable_cost_per_request():.5f}",
        f"On-Prem fixed (period):    ${fixed:.2f}",
    ]
    if scn.cloud.enabled:
        lines.append(f"Cloud GPU / request:       ${scn.cloud.cost_per_request():.5f}")
    if be is None:
        lines.append("Break-even: NEVER -- API is cheaper at every volume.")
    else:
        lines.append(f"Break-even volume: ~{be:,.0f} requests over the period.")
    return "\n".join(lines)
