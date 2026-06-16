"""Experiment configuration -- edit this one file for your machine."""
from __future__ import annotations

import os

# --- Model -------------------------------------------------------------------
# 7B chosen: its FP16 footprint far exceeds usable RAM (naive load fails ->
# bottleneck demonstrated), while shards fit the 42 GB disk. Justify in report.
MODEL_ID: str | None = "Qwen/Qwen2.5-7B-Instruct"

# --- AirLLM shard storage ----------------------------------------------------
# Reads AIRLLM_SHARDS env var; falls back to a folder in the project.
LAYER_SHARDS_PATH: str = os.environ.get(
    "AIRLLM_SHARDS", os.path.join(os.getcwd(), "layer_shards")
)

# --- Quantization sweep ------------------------------------------------------
# AirLLM's q4/q8 go through bitsandbytes, which needs a CUDA GPU with compute
# capability >= 7.5 and several GB VRAM. This machine's GTX 950M (Maxwell 5.0,
# 2 GB) cannot run it, so AirLLM is FP16-only here. That limitation is itself a
# documented result (see the quantization research question).
QUANT_SWEEP: list[str] = ["fp16"]

# The real quantization comparison runs on the CPU via Ollama/GGUF (llama.cpp),
# which needs no capable GPU. Requires the Ollama app + `uv pip install ollama`.
# See experiments/run_ollama.py.
OLLAMA_QUANT_SWEEP: list[str] = ["q4", "q8"]

# --- Energy estimate ---------------------------------------------------------
# Assumed average package power in watts for the energy figure. Document this.
# ~15 W is reasonable for a dual-core mobile CPU under load; adjust if you have
# a wall meter reading.
AVG_POWER_W: float = 15.0

# --- Which prompts to run ----------------------------------------------------
# Start with just the short prompt to confirm the pipeline works, then add
# "medium" and "long_context" once you've seen tokens come out.
PROMPT_NAMES: list[str] = ["short"]