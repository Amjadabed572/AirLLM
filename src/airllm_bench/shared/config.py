"""Configuration manager.

All tunable values are read from config/*.json through this manager — never
hardcoded in source. Secrets are read only from environment variables, never
from these files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from airllm_bench.shared.version import COMPATIBLE_CONFIG_VERSIONS


class Config:
    """Loads and exposes JSON configuration with dotted-key access.

    Input:  config_dir (path holding setup.json / rate_limits.json / ...).
    Output: typed values via get("a.b.c", default).
    Setup:  reads files lazily on first access; validates version on load.
    """

    def __init__(self, config_dir: str | os.PathLike[str] = "config") -> None:
        self._dir = Path(config_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def _load(self, name: str) -> dict[str, Any]:
        if name not in self._cache:
            path = self._dir / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing config file: {path}")
            with path.open(encoding="utf-8") as handle:
                self._cache[name] = json.load(handle)
        return self._cache[name]

    def get(self, dotted_key: str, default: Any = None, *, file: str = "setup") -> Any:
        """Return a nested value, e.g. get("experiment.avg_power_w")."""
        node: Any = self._load(file)
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, dotted_key: str, *, file: str = "setup") -> dict[str, Any]:
        """Return a whole sub-section as a dict (empty if absent)."""
        value = self.get(dotted_key, {}, file=file)
        return dict(value) if isinstance(value, dict) else {}

    def validate_versions(self) -> None:
        """Fail fast if a config file's version is incompatible with the code."""
        for name in ("setup", "rate_limits"):
            version = str(self._load(name).get("version", "missing"))
            if version not in COMPATIBLE_CONFIG_VERSIONS:
                raise ValueError(
                    f"config/{name}.json version {version!r} is not compatible "
                    f"with code (supports {COMPATIBLE_CONFIG_VERSIONS})."
                )

    @staticmethod
    def secret(env_name: str, default: str | None = None) -> str | None:
        """Read a secret from the environment only — never from config files."""
        return os.environ.get(env_name, default)
