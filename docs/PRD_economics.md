# PRD — Economic analysis mechanism (On-Prem vs API)

## Theory

Two cost structures are compared as a function of request volume V:

- **API:** linear — `cost = V · (in·price_in + out·price_out)`, with an optional
  prompt/context-caching discount on the fixed, repeated prefix (cached prefix
  tokens are billed at a fraction of the normal rate → shifts break-even rightward).
- **On-Prem:** fixed + linear — amortized hardware CAPEX over its lifetime
  (for the period) + maintenance, plus a small per-request electricity term derived
  from the **measured** energy/request.
- **Cloud GPU (optional):** linear — `gpu_hourly_rate · seconds_per_request`.

The **break-even** volume is where On-Prem becomes cheaper than the API:
`break_even = fixed / (api_per_req − onprem_variable_per_req)` (None if the API's
per-request price is already below on-prem's variable cost).

## Requirements

- All assumptions read from `config/setup.json → economics` (no hardcoded values).
- Energy/request must come from the **measured** runs, not guessed.
- Output a cumulative-cost-vs-volume figure with the break-even annotated.

## I/O contract

- **Input:** the `economics` config section (api/onprem/cloud_gpu/period_years).
- **Output:** `Scenario` → cost curves, `break_even()` value, `summarize()` text,
  and `figures/break_even.png`.
- **Setup:** prices/tariff/lifetime are editable; cite the provider in the report.

## Success criteria & edge cases

- A finite, sensible break-even with the stated assumptions (~157k requests,
  using the measured 0.21 Wh/request from the warm Q4 run).
- Caching fraction = 1.0 lowers API cost; in/out tokens = 0 → break-even None.
- Recommendation states *when* each option wins, including non-cost factors
  (privacy, data security, offline) that can favour On-Prem regardless of volume.
