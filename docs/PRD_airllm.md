# PRD — AirLLM layer-streaming mechanism

## Theory

A transformer is a stack of N near-identical layers. Standard inference holds all
N in memory at once. **AirLLM** keeps only the *active* layer resident: it pre-
splits the model into per-layer shards on disk, then for each forward pass loads
layer i, computes, frees it, and loads layer i+1. Peak memory drops from "whole
model" to "one layer + activations + KV-cache". This is the OS virtual-memory /
paging idea applied to model weights — the binding constraint moves from RAM size
to **disk read bandwidth**.

## Requirements

- Use the general `airllm.AutoModel` so the architecture is matched automatically
  (avoids the Qwen `Class mismatch` error).
- Write shards to a configurable `layer_shards_saving_path` (config / `AIRLLM_SHARDS`)
  so they do not flood the OS drive.
- FP16 only on the target machine (q4/q8 need bitsandbytes + CUDA ≥7.5 — see
  `docs/PRD_quantization.md`).

## I/O contract

- **Input:** model_id, `Prompt`, quant label (fp16), shards path, avg power.
- **Output:** `RunMetrics` (TTFT≈one full forward, TPOT from generate, throughput,
  peak RAM/VRAM, energy) → `results/airllm_<quant>_<prompt>.json`.
- **Setup:** AirLLM splits the model on first use (one-time ~15 GB disk write).

## Expected behaviour & success criteria

- Generation **completes** where the naive baseline failed → AirLLM enables the model.
- Decode is **disk-I/O-bound** (tens of seconds/token on a SATA SSD): each token
  streams ~15 GB. Success = a real, recorded TTFT/TPOT/throughput, however slow.

## Edge cases

- Out of disk during sharding → fail fast; point shards at a larger drive.
- Transformers 5.x / optimum 2.x break AirLLM 2.11 → pinned (`docs/PRD` NFR3).
