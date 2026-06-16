# TODO — Task tracking (EX05)

Status: ☐ not started · ◐ in progress · ☑ done

## Phase 1 — Environment & models
- ☑ uv venv + `uv pip install -e .` (pinned AirLLM-compatible chain)
- ☑ Detect & persist real hardware (`results/hardware.json`)
- ☑ Resolve/justify model choice (Qwen2.5-7B for 8 GB / 40 GB SSD)
- ☑ Download model to HF cache (~15 GB)

## Phase 2 — Runs & measurement
- ☑ Baseline: deterministic memory-capacity failure recorded (OOM kill captured)
- ◐ AirLLM FP16 layer-streaming run → `results/airllm_fp16_short.json`
- ◐ Ollama GGUF Q4 run → `results/ollama_q4_short.json`
- ◐ Ollama GGUF Q8 run → `results/ollama_q8_short.json`
- ☐ (optional) medium / long_context prompts for TTFT-vs-length sweep

## Phase 3 — Analysis & economics
- ☑ Economic models + break-even (config-driven)
- ☑ Roofline & break-even figures
- ◐ Throughput / peak-RAM / TTFT-vs-TPOT figures (need the runs above)
- ☐ Fill measured energy/request into `config/setup.json` economics

## Phase 4 — Report & quality
- ☑ README.md / README.he.md (structure, reproduce, pitfalls)
- ☑ reports/report.md / report.he.md (framework + pending-result markers)
- ☑ docs/ suite (PRD, PLAN, TODO, per-mechanism PRDs, prompt log)
- ☑ ruff clean · ☑ ≥85% test coverage (93%) · ☑ files ≤150 code lines
- ☐ Paste final measured numbers into report tables once runs complete

**Definition of done (each run task):** a real `results/*.json` exists, the figure
regenerates from it via `airllm-bench analyze`, and the number is cited in the report.
