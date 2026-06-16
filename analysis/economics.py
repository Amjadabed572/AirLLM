"""Economic analysis (Task 5.5) -- the break-even between local and API.

Two independent cost models, plus two optional scenarios, exactly as required:

  1) API cost     = requests * (in_tokens*price_in + out_tokens*price_out)
                    with optional Prompt/Context-Caching discount on the
                    fixed, repeated part of the prompt (PagedAttention-style
                    providers charge much less for cached prefix tokens).
  2) On-Prem cost = amortized CAPEX (hardware over its lifetime)
                    + OPEX (electricity from measured energy/request + upkeep).
  3) Cloud GPU    (optional) = gpu_hourly_rate * seconds_per_request / 3600.

All assumptions are explicit and editable so the analysis is transparent and
reproducible (a hard requirement). Prices below are *editable placeholders* --
replace with the real numbers you cite in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class APIParams:
    # USD per 1M tokens -> stored per-token. Replace with the provider you cite.
    price_in_per_mtok: float = 3.00     # e.g. a mid-tier API input price
    price_out_per_mtok: float = 15.00   # output usually costs more
    in_tokens: int = 600
    out_tokens: int = 300
    # Prompt caching: fraction of input that is a fixed, repeated prefix, and
    # the discount applied to those cached tokens (e.g. cached reads ~10% price).
    cached_input_fraction: float = 0.0
    cache_price_multiplier: float = 0.1

    def cost_per_request(self) -> float:
        pin = self.price_in_per_mtok / 1e6
        pout = self.price_out_per_mtok / 1e6
        cached = self.in_tokens * self.cached_input_fraction
        fresh = self.in_tokens - cached
        in_cost = fresh * pin + cached * pin * self.cache_price_multiplier
        return in_cost + self.out_tokens * pout


@dataclass
class OnPremParams:
    hardware_cost_usd: float = 2500.0   # your machine / GPU box CAPEX
    lifetime_years: float = 3.0
    maintenance_usd_per_year: float = 150.0
    energy_wh_per_request: float = 12.0  # from your measured est_energy_wh
    electricity_usd_per_kwh: float = 0.17

    def amortized_fixed_for_period(self, years: float) -> float:
        capex = self.hardware_cost_usd * (years / self.lifetime_years)
        return capex + self.maintenance_usd_per_year * years

    def variable_cost_per_request(self) -> float:
        kwh = self.energy_wh_per_request / 1000.0
        return kwh * self.electricity_usd_per_kwh


@dataclass
class CloudGPUParams:
    enabled: bool = False
    gpu_hourly_usd: float = 1.20
    seconds_per_request: float = 8.0

    def cost_per_request(self) -> float:
        return self.gpu_hourly_usd * (self.seconds_per_request / 3600.0)


@dataclass
class Scenario:
    api: APIParams = field(default_factory=APIParams)
    onprem: OnPremParams = field(default_factory=OnPremParams)
    cloud: CloudGPUParams = field(default_factory=CloudGPUParams)
    period_years: float = 1.0  # window over which CAPEX is amortized

    def curves(self, volumes: np.ndarray) -> dict[str, np.ndarray]:
        api = volumes * self.api.cost_per_request()
        fixed = self.onprem.amortized_fixed_for_period(self.period_years)
        onprem = fixed + volumes * self.onprem.variable_cost_per_request()
        out = {"API (third-party)": api, "On-Prem (local)": onprem}
        if self.cloud.enabled:
            out["Cloud GPU (rented)"] = volumes * self.cloud.cost_per_request()
        return out

    def break_even(self) -> float | None:
        """Requests at which On-Prem becomes cheaper than API. None if never."""
        api_pr = self.api.cost_per_request()
        var_pr = self.onprem.variable_cost_per_request()
        if api_pr <= var_pr:
            return None  # API per-request already below on-prem variable cost
        fixed = self.onprem.amortized_fixed_for_period(self.period_years)
        return fixed / (api_pr - var_pr)


def summarize(scn: Scenario) -> str:
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
        lines.append("Break-even: NEVER -- API is cheaper at every volume "
                     "(its per-request price is below on-prem's variable cost).")
    else:
        lines.append(f"Break-even volume: ~{be:,.0f} requests over the period. "
                     f"Above this, On-Prem wins.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(Scenario()))
