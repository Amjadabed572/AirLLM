"""Shared utilities: configuration manager and version tracking."""
from __future__ import annotations

from airllm_bench.shared.config import Config
from airllm_bench.shared.version import __version__

__all__ = ["Config", "__version__"]
