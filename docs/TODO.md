# TODO — Task tracking (EX05)

Status: ☐ not started · ◐ in progress · ☑ done

## Phase 1 — Environment & models
- ☑ uv venv + `uv pip install -e .` (pinned AirLLM-compatible chain)
- ☑ Detect & persist real hardware (`results/hardware.json`)
- ☑ Resolve/justify model choice (Qwen2.5-7B for 8 GB / 40 GB SSD)
- ☑ Download model to HF cache (~15 GB)

## Phase 2 — Runs & measurement
- ☑ Baseline: deterministic memory-capacity failure recorded (OOM kill captured)
- ☑ AirLLM FP16 layer-streaming run → `results/airllm_fp16_short.json` (129 s TTFT, 144 s/tok, 3.6 GB)
- ☑ Ollama GGUF Q4 run → `results/ollama_q4_short.json` (4.49 tok/s, 2.2 GB, 0.18 Wh)
- ☑ Ollama GGUF Q8 run → `results/ollama_q8_short.json` (0.03 tok/s — overflows RAM)
- ☑ Parameter study: Ollama Q4 across short/medium/long_context (TTFT-vs-length) → `airllm-bench study`

## Phase 3 — Analysis & economics
- ☑ Economic models + break-even (config-driven)
- ☑ Roofline & break-even figures
- ☑ Throughput / peak-RAM / TTFT-vs-TPOT figures (from the real runs)
- ☐ (optional) Fill measured energy/request into `config/setup.json` economics

## Phase 4 — Report & quality
- ☑ README.md (structure, reproduce, pitfalls)
- ☑ reports/report.md (deep-dive technical report)
- ☑ docs/ suite (PRD, PLAN, TODO, per-mechanism PRDs, prompt log)
- ☑ ruff clean · ☑ ≥90% test coverage (97%) · ☑ files ≤150 code lines
- ☑ Measured numbers filled into README + reports (EN+HE) tables
- ☑ Run screenshots embedded (`figures/screenshots/`)
- ☑ LICENSE + Credits, architecture diagram, analysis notebook

**Definition of done (each run task):** a real `results/*.json` exists, the figure
regenerates from it via `airllm-bench analyze`, and the number is cited in the report.
