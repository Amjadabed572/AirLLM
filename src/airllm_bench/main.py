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


def _force_utf8_stdout() -> None:
    """Tables contain unicode (✓, —); make stdout UTF-8 so the CLI doesn't crash
    on a non-UTF-8 console (e.g. a Hebrew-locale cmd using cp1255)."""
    import contextlib
    import sys

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


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


def _run_study(sdk: AirLLMBenchSDK) -> None:
    for m in sdk.run_input_length_study():
        print(f"[study] {m.label}/{m.prompt} ...\n   " + _status(m))


def main(argv: list[str] | None = None) -> int:
    """Parse the sub-command and dispatch to the SDK."""
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(prog="airllm-bench", description=__doc__)
    parser.add_argument(
        "command",
        choices=["hardware", "model", "baseline", "airllm", "ollama", "study",
                 "analyze", "all"],
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
    elif args.command == "study":
        _run_study(sdk)
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
