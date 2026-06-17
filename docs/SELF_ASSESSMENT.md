# Self-Assessment & Compliance Matrix (EX05)

Mapping of the assignment (EX05) and the "Guidelines for Writing Professional
Software at the Highest Level of Excellence" to concrete evidence in this repo.
**Suggested self-score: 88 / 100.**

## A. EX05 assignment deliverables

| Requirement | Status | Evidence |
|---|---|---|
| Hardware spec + model justification (5.1) | ✅ | `results/hardware.json`, README §1 |
| Baseline direct run + bottleneck (5.2) | ✅ | `results/baseline_short.json` (OS-killed OOM) |
| AirLLM integration + layer-streaming (5.3) | ✅ | `results/airllm_fp16_short.json` (3.6 GB, 144 s/tok) |
| Quantization comparison (5.3) | ✅ | Ollama Q4/Q8 GGUF (`results/ollama_*`) |
| Measurement: TTFT/TPOT/throughput/RAM/energy (5.4) | ✅ | `results/summary_table.md`, figures |
| Economic analysis On-Prem vs API + break-even (5.5) | ✅ | `figures/break_even.png`, `services/economics.py` |
| Inference-concept analysis: prefill/decode, roofline, paging (5.6) | ✅ | report §5, `figures/roofline.png` |
| Original extension (5.7) | ✅ | dual-engine quantization (GGUF) + TTFT-vs-input study |
| Report **is** a README with embedded figures/tables/screenshots (§8) | ✅ | `README.md` (figures + tables + `figures/screenshots/`) |
| Clear reproduction instructions | ✅ | README "Reproduce" (`airllm-bench` CLI) |

## B. Professional-software guidelines

| Section | Item | Status | Evidence |
|---|---|---|---|
| §2 | README + docs/ (PRD, PLAN, TODO, per-mechanism PRDs) | ✅ | `docs/` |
| §3 | Modular layout, files ≤150 code lines, docstrings, DRY | ✅ | `src/airllm_bench/`, ruff-clean |
| §4 | SDK architecture (single entry point) | ✅ | `sdk/sdk.py`, README diagram |
| §4.2 | OOP, no code duplication | ✅ | services + shared, mix- free reuse |
| §5 | API Gatekeeper | ⚠️ N/A (documented) | `docs/PLAN.md §4` — no live API calls ($0 design) |
| §6 | Tests ≥85% coverage, edge cases | ✅ | 59 tests, **93%** (`tests/`, `tests_passing.png`) |
| §7.1 | ruff — zero violations | ✅ | `uv run ruff check .` passes |
| §7.2–7.4 | No hardcoded values, secrets via env, `.env-example`, `.gitignore` | ✅ | `config/*.json`, `shared/config.py`, `.env-example` |
| §8 | uv-only, version tracking, prompt log | ✅ | `uv.lock`, `shared/version.py` (1.00), `docs/PROMPT_LOG.md` |
| §9 | Parameter study, analysis notebook, visualization | ✅ | `airllm-bench study`, `notebooks/analysis.ipynb`, figures |
| §11 | Cost/budget analysis | ✅ | economics module + break-even |
| §12 | Maintainability / extension points | ⚠️ partial | config-driven + SDK seams; no formal plugin hooks |
| §13 | ISO/IEC 25010 quality attributes | ✅ | addressed across reliability/maintainability/portability |
| §14 | Package organization (`__init__`, `__version__`, relative imports) | ✅ | package exports + console script |
| §16 | Building-block design (Input/Output/Setup documented) | ✅ | docstrings on services/dataclasses |
| §General | Git history, LICENSE, credits | ✅ | clean history, `LICENSE`, README "License & Credits" |

## C. Honest gaps (why not 95+)

- **API Gatekeeper / rate-limiting / queue (§5):** genuinely not applicable —
  no live third-party API calls are made; documented as N/A rather than
  implemented for show.
- **Formal plugin/extension points (§12):** the design is config-driven and
  SDK-seamed, but there are no `lifecycle hooks`/middleware.
- **Single hardware sample:** results are from one 8 GB laptop (the assigned
  machine); no cross-hardware sweep.
- **Output-quality scoring:** quantization quality is discussed qualitatively,
  not scored with a benchmark (out of scope per the brief, which de-emphasizes
  output quality).

## D. Why 88

Every mandatory deliverable and the great majority of excellence criteria are met
with real, reproducible evidence (93% test coverage, ruff-clean, SDK architecture,
full docs, real measurements, economic + parameter studies, bilingual report).
The deductions reflect the few genuinely-not-applicable or partial items above,
documented honestly rather than faked.
