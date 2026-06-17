"""Single source of truth for the package version.

Code, config files, and rate-limit config all start at 1.00 and rise only on
meaningful change. The app validates config-version compatibility at startup
(see shared.config.Config.validate_versions).
"""
from __future__ import annotations

__version__ = "1.00"

# Config versions this code is known to be compatible with.
COMPATIBLE_CONFIG_VERSIONS: tuple[str, ...] = ("1.00",)
