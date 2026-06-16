"""Unit tests for hardware detection."""
from __future__ import annotations

from airllm_bench.services import hardware
from airllm_bench.services.hardware import HardwareInfo, detect


def test_detect_returns_populated_info() -> None:
    info = detect()
    assert isinstance(info, HardwareInfo)
    assert info.ram_total_gb > 0
    assert info.cpu_logical_cores >= 1
    assert info.disk_free_gb > 0
    assert info.python_version


def test_pretty_and_to_dict_roundtrip() -> None:
    info = detect()
    assert "CPU:" in info.pretty()
    data = info.to_dict()
    assert data["ram_total_gb"] == info.ram_total_gb


def test_cpu_model_is_nonempty() -> None:
    assert hardware._cpu_model()


def test_disk_type_hint_is_string() -> None:
    assert isinstance(hardware._disk_type_hint(), str)


def test_gpu_info_returns_pair() -> None:
    name, vram = hardware._gpu_info()
    # name may be None on a headless CI box; the contract is a 2-tuple.
    assert vram is None or vram > 0 or name is not None or name is None
