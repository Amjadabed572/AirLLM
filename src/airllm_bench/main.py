"""CLI entry point (thin consumer of the SDK; no business logic here).

    uv run airllm-bench hardware        # detect + save results/hardware.json
    uv run airllm-bench model           # show the recommended model
    uv run airllm-bench baseline        # naive load (expected to OOM)
    uv run airllm-bench airllm          # AirLLM FP16 layer-streaming sweep
    uv run airllm-bench ollama          # GGUF Q4/Q8 quantization sweep
    uv run airllm-bench analyze         # tables + figures + economics
    uv run airllm-bench all             # baseline -> airllm -> ollama -> analyze
"""
from __future__ import annotations

import argparse

from airllm_bench.sdk.sdk import AirLLMBenchSDK
from airllm_bench.shared.config import Config


def _status(metrics) -> str:
    if metrics.failed:
        return f"FAILED: {metrics.failure_reason}"
    return (f"TTFT={metrics.ttft_s:.2f}s  TPOT={metrics.tpot_s * 1000:.1f}ms  "
            f"tok/s={metrics.throughput_tok_s:.2f}  peakRAM={metrics.peak_ram_gb:.1f}GB")


def _run_baseline(sdk: AirLLMBenchSDK) -> None:
    for name in sdk.config.get("experiment.prompt_names", ["short"]):
        print(f"[baseline] {name} ...", flush=True)
        print("   " + _status(sdk.run_baseline(name)))


def _run_airllm(sdk: AirLLMBenchSDK) -> None:
    for quant in sdk.config.get("experiment.quant_sweep", ["fp16"]):
        for name in sdk.config.get("experiment.prompt_names", ["short"]):
            print(f"[airllm] {quant}/{name} ...", flush=True)
            print("   " + _status(sdk.run_airllm(quant, name)))


def _run_ollama(sdk: AirLLMBenchSDK) -> None:
    for quant in sdk.config.get("experiment.ollama_quant_sweep", ["q4", "q8"]):
        for name in sdk.config.get("experiment.prompt_names", ["short"]):
            print(f"[ollama] {quant}/{name} ...", flush=True)
            print("   " + _status(sdk.run_ollama(quant, name)))


def main(argv: list[str] | None = None) -> int:
    """Parse the sub-command and dispatch to the SDK."""
    parser = argparse.ArgumentParser(prog="airllm-bench", description=__doc__)
    parser.add_argument(
        "command",
        choices=["hardware", "model", "baseline", "airllm", "ollama", "analyze", "all"],
    )
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args(argv)
    sdk = AirLLMBenchSDK(Config(args.config_dir))

    if args.command == "hardware":
        print(sdk.save_hardware().pretty())
    elif args.command == "model":
        print(sdk.recommend_model())
    elif args.command == "baseline":
        _run_baseline(sdk)
    elif args.command == "airllm":
        _run_airllm(sdk)
    elif args.command == "ollama":
        _run_ollama(sdk)
    elif args.command == "analyze":
        print(sdk.analyze())
    elif args.command == "all":
        _run_baseline(sdk)
        _run_airllm(sdk)
        _run_ollama(sdk)
        print(sdk.analyze())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
