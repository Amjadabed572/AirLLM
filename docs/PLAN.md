# PLAN — Architecture & Planning (EX05)

## 1. Architecture overview (C4, text form)

**Context.** A single-user CLI experiment harness that benchmarks local LLM
inference on the host machine and computes a local-vs-API cost analysis.

**Containers.** One Python package (`airllm_bench`) + config files + a results/
figures store. No network services; no live external APIs.

**Components (layered, SDK-fronted — guidelines §4.1):**

```
Consumers (CLI: airllm_bench.main  ·  notebooks  ·  future GUI/REST)
        │  (only ever call the SDK)
        ▼
   SDK facade  (airllm_bench.sdk.sdk.AirLLMBenchSDK)   ← single entry point
        │
        ▼
   Domain services (airllm_bench.services.*)
     hardware · model_selector · prompts · metrics
     baseline_runner · airllm_runner · ollama_runner
     economics · plots · analyze
        │
        ▼
   Infrastructure: torch/transformers/airllm, Ollama, file I/O, matplotlib
```

**Shared:** `shared.config.Config` (config manager), `shared.version`
(version tracking), `constants` (immutable maps/enums).

## 2. Key flows

- **Hardware** → `detect()` → `results/hardware.json`.
- **Baseline (5.2)** → naive load with a 6 GiB RAM budget and offload forbidden →
  deterministic memory-capacity failure, recorded (provisional record survives a
  hard OS kill).
- **AirLLM (5.3)** → `AutoModel` streams one layer at a time from the SSD (FP16).
- **Quantization (5.3)** → Ollama GGUF Q4/Q8 on CPU (see `docs/PRD_quantization.md`).
- **Analyze (5.4/5.5)** → table + figures + economic break-even from config.

## 3. Architectural decisions (ADRs)

- **ADR-1: SDK facade over a flat module API.** Rationale: one tested entry point,
  swappable consumers. Trade-off: a thin extra layer; accepted for testability.
- **ADR-2: Ollama/GGUF for the quantization track.** Rationale: AirLLM's q4/q8 need
  bitsandbytes + CUDA ≥7.5; the GTX 950M (Maxwell, 2 GB) cannot. GGUF on CPU yields
  real Q4/Q8 numbers. Alternative (AirLLM compression) rejected as infeasible here.
- **ADR-3: Baseline uses `max_memory` + no offload.** Rationale: a bare
  `device_map="auto"` silently disk-offloads on CPU-only torch, masking the
  bottleneck. Forbidding offload yields a fast, deterministic failure.
- **ADR-4: All tunables in `config/*.json`.** Rationale: guidelines §7.2; keeps the
  analysis transparent/reproducible.

## 4. API Gatekeeper — documented as **N/A** (guidelines §5)

The rubric's centralized API Gatekeeper governs *live* third-party API calls
(rate limiting, queue, retry, logging). **This project makes no live external API
calls**: the economic analysis multiplies *published* per-token prices on paper (a
deliberate $0 design). A runtime gatekeeper would have nothing to guard.
`config/rate_limits.json` is retained for completeness and would configure a
gatekeeper if a live provider were ever wired in.

## 5. Concurrency (guidelines §15)

- `MemorySampler` uses a background **thread** (I/O-bound polling of RSS) — correct
  choice over multiprocessing for a sampler that mostly sleeps.
- The baseline streams tokens on a worker thread while the main thread times them.
- Ollama runs in its own server **process**; we sample whole-system RAM as a
  documented approximation.

## 6. Data schemas / contracts

- `RunMetrics` (results/*.json): label, model, quantization, prompt/output tokens,
  ttft_s, tpot_s, total_gen_s, throughput_tok_s, peak_ram_gb, peak_vram_gb,
  est_energy_wh, failed, failure_reason, per_token_latencies_s.
- `HardwareInfo` (results/hardware.json): os, cpu_model, cores, ram, gpu, vram,
  disk, python_version.
