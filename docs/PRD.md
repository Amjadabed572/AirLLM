# PRD — Product Requirements Document (EX05)

**Project:** Running a Massive LLM Locally — AirLLM, Quantization & Performance Benchmarking
**Course:** L08 · Dr. Yoram Segal · June 2026

## 1. Context & problem

Large language models often exceed the RAM/VRAM of modest hardware, so they cannot
be loaded the naive way. This project proves, on a real 8 GB laptop, *when* an
LLM that does not fit can still be run locally (via AirLLM layer-streaming and via
GGUF quantization), at what performance cost, and *when* doing so is economically
rational versus a third-party API. The target user is an engineer deciding between
on-prem and API deployment for an LLM workload.

## 2. Goals & measurable success criteria (KPIs)

| KPI | Target |
|---|---|
| Baseline bottleneck demonstrated | Naive FP16 load fails deterministically (recorded) |
| AirLLM enables the model | FP16 generation completes and is measured (TTFT/TPOT/throughput) |
| Quantization comparison | Real Q4 vs Q8 numbers (peak RAM, TPOT, throughput) |
| Economic analysis | Break-even volume computed from transparent, config-driven assumptions |
| Reproducibility | One command regenerates every table & figure from raw JSON |
| Quality gate | ruff clean, ≥90% test coverage, all files ≤150 code lines |

## 3. Functional requirements

- FR1 — Detect & persist the host hardware spec (CPU/RAM/GPU/VRAM/disk).
- FR2 — Recommend a model from detected RAM + free disk, with justification.
- FR3 — Run a naive baseline and capture its failure (or metrics).
- FR4 — Run AirLLM (FP16) layer-streaming and measure it.
- FR5 — Run GGUF Q4/Q8 via Ollama and measure them.
- FR6 — Compute API vs On-Prem (and optional Cloud-GPU) cost curves + break-even.
- FR7 — Emit a comparison table and all figures from raw `results/*.json`.

## 4. Non-functional requirements

- NFR1 — SDK architecture: all logic reachable through one facade (`AirLLMBenchSDK`).
- NFR2 — No hardcoded tunables; everything from `config/*.json` or env vars.
- NFR3 — `uv`-only dependency/runtime management; pinned, lockable.
- NFR4 — ruff-clean, ≥90% test coverage, files ≤150 code lines.
- NFR5 — No secrets in code or git; `.env-example` documents placeholders.

## 5. User stories

- As an engineer, I run `airllm-bench all` and get tables + figures proving the
  bottleneck, the AirLLM/quantization wins, and the cost break-even.
- As a grader, I open the repo and reproduce every number from a single command.

## 6. Assumptions, dependencies, out of scope

- **Assumptions:** Python 3.10–3.12; SSD with ≥30 GB free; public Qwen model.
- **Dependencies:** torch, transformers (<4.45), airllm, optimum (<2.0), Ollama (optional).
- **Out of scope:** training/fine-tuning at scale, multi-GPU, a hosted web UI,
  live calls to paid APIs (economics uses published prices on paper — $0 design).

## 7. Milestones

1. Environment + model download · 2. Baseline + AirLLM + Ollama runs ·
3. Analysis + economics · 4. Report (README + reports/). See `docs/TODO.md`.
