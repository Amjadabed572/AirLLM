# Prompt Engineering Log (guidelines §8.3)

This project was built with Vibe Coding — a human architect directing an AI agent.
This log records the significant prompts, their intent, and how the approach
evolved. It is a faithful summary, not a verbatim transcript.

## 1. Bootstrap & hardware-aware planning
- **Intent:** turn the EX05 brief into a runnable project on *this* machine.
- **Prompt (summary):** "Implement EX05 from scratch." → first detected the host
  (i7-7500U, 8 GB RAM, GTX 950M 2 GB, 40 GB SATA SSD) and let the hardware drive
  every choice (model size, fp16-only AirLLM, CPU GGUF for quantization).
- **Lesson:** detect hardware first; let constraints pick the model, not the reverse.

## 2. Integrity fix — remove fabricated benchmarks
- **Intent:** a prior scaffold had placeholder "results" presented as real.
- **Action:** deleted fake `results/*.json` + derived figures; regenerated only the
  *computed* economics/roofline figures and a real `hardware.json`.
- **Lesson:** never present invented numbers as measurements; mark pending data.

## 3. Dependency-chain debugging (AirLLM 2.11)
- **Symptoms:** `No module named 'optimum.bettertransformer'`, then
  `cannot import name 'is_tf_available'`, then `No module named 'sentencepiece'`.
- **Resolution:** pin `transformers>=4.44,<4.45`, `optimum<2.0`, add `sentencepiece`.
- **Lesson:** AirLLM lags the latest transformers; pin the whole chain.

## 4. Baseline that *proves* the bottleneck
- **Iteration 1:** `device_map="auto"` silently disk-offloaded → unusably slow, not a
  clean failure. **Iteration 2:** add `max_memory={"cpu":"6GiB"}` + no offload →
  deterministic failure. **Iteration 3:** OS hard-killed the process mid-load →
  write a provisional failure record so the kill is still captured.
- **Lesson:** make the negative result deterministic *and* recordable.

## 5. Professional-guidelines refactor
- **Intent:** apply Dr. Segal's excellence rubric.
- **Action:** restructured to `src/airllm_bench/{sdk,services,shared}`, added a config
  manager (no hardcoded values), version tracking, docs/ suite, ruff-clean lint,
  and a pytest suite at 93% coverage. Documented the API Gatekeeper as N/A (no live
  API calls — $0 design).
- **Lesson:** an SDK facade + config-driven design makes the work testable and
  reproducible.
