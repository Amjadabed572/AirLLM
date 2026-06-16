# PRD — Quantization comparison mechanism (GGUF via Ollama)

## Theory

Quantization stores each weight in fewer bits (FP16 → 8-bit → 4-bit), cutting the
bytes moved per token. Because decode is **memory-bandwidth-bound**, fewer bytes →
faster decode and lower peak RAM — until precision loss degrades output quality
(the accuracy "red line"). We compare FP16-class vs **Q8 (`q8_0`)** vs
**Q4 (`q4_K_M`)**.

## Why Ollama/GGUF (not AirLLM compression) on this machine

AirLLM's q4/q8 route through `bitsandbytes`, which requires a CUDA GPU of compute
capability ≥ 7.5 with several GB VRAM. The target GPU is a Maxwell **GTX 950M
(compute 5.0, 2 GB)** — it cannot run bitsandbytes quantization. **This limitation
is itself a documented result.** GGUF (llama.cpp) quantizes and runs on the **CPU**,
so it provides the real Q4/Q8 numbers. The assignment lists Ollama and GGUF as
in-scope tools; this is the project's required original extension (Task 5.7).

## I/O contract

- **Input:** quant label (q4/q8), `Prompt`, avg power.
- **Output:** `RunMetrics` from Ollama's native ns counters (TTFT = load +
  prompt_eval; TPOT = eval_duration/(eval_count-1)) → `results/ollama_<quant>_<prompt>.json`.
- **Setup:** Ollama app installed + `uv pip install ollama`; model auto-pulled.

## Success criteria

- Both Q4 and Q8 complete on CPU and produce real numbers.
- Expected (to confirm with data): Q4 < Q8 for peak RAM and TPOT; quality degrades
  Q8 → Q4.

## Edge cases

- Ollama not installed → `RunMetrics.failed` with a clear reason (no crash).
- Peak RAM is whole-system delta (Ollama is a separate process) — documented caveat.
