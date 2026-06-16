"""airllm_bench — run a massive LLM locally and benchmark it (EX05).

Public surface: import the SDK facade and the version.
"""
from __future__ import annotations

from airllm_bench.sdk.sdk import AirLLMBenchSDK
from airllm_bench.shared.version import __version__

__all__ = ["AirLLMBenchSDK", "__version__"]
