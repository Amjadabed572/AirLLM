"""Unit tests for the configuration manager."""
from __future__ import annotations

import json

import pytest

from airllm_bench.shared.config import Config


def test_get_nested_value(cfg: Config) -> None:
    assert cfg.get("model.model_id") == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.get("experiment.avg_power_w") == 15.0


def test_get_missing_returns_default(cfg: Config) -> None:
    assert cfg.get("model.nope", "fallback") == "fallback"
    assert cfg.get("a.b.c.d") is None


def test_section_returns_dict(cfg: Config) -> None:
    api = cfg.section("economics.api")
    assert api["in_tokens"] == 600


def test_section_absent_is_empty(cfg: Config) -> None:
    assert cfg.section("nothing.here") == {}


def test_validate_versions_ok(cfg: Config) -> None:
    cfg.validate_versions()  # should not raise


def test_validate_versions_rejects_bad(tmp_path) -> None:
    (tmp_path / "setup.json").write_text(json.dumps({"version": "9.99"}), encoding="utf-8")
    (tmp_path / "rate_limits.json").write_text(json.dumps({"version": "1.00"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not compatible"):
        Config(tmp_path).validate_versions()


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Config(tmp_path).get("x.y")


def test_secret_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("MY_SECRET", "s3cr3t")
    assert Config.secret("MY_SECRET") == "s3cr3t"
    assert Config.secret("ABSENT", "def") == "def"
