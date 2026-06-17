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


class _CP:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def test_cpu_model_fallback(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "processor", lambda: "")
    assert hardware._cpu_model()  # machine() or 'unknown'


def test_gpu_from_nvidia_smi_parses(monkeypatch) -> None:
    monkeypatch.setattr(hardware.subprocess, "run",
                        lambda *a, **k: _CP("NVIDIA GeForce GTX 950M, 2048\n", 0))
    name, vram = hardware._gpu_from_nvidia_smi()
    assert "950M" in name
    assert round(vram, 1) == 2.0


def test_gpu_from_nvidia_smi_missing(monkeypatch) -> None:
    def boom(*a, **k):
        raise OSError("not found")

    monkeypatch.setattr(hardware.subprocess, "run", boom)
    assert hardware._gpu_from_nvidia_smi() is None


def test_gpu_from_wmi_prefers_nvidia(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    csv = "Node,AdapterRAM,Name\nPC,1073741824,Intel HD\nPC,2147483648,NVIDIA GTX 950M\n"
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: _CP(csv, 0))
    name, _ = hardware._gpu_from_wmi()
    assert "NVIDIA" in name


def test_gpu_from_wmi_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    assert hardware._gpu_from_wmi() is None


def test_disk_type_hint_windows_ssd(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: _CP("SSD\n", 0))
    assert "SSD" in hardware._disk_type_hint()


def test_disk_type_hint_linux(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    assert "lsblk" in hardware._disk_type_hint()


def test_gpu_info_uses_first_source(monkeypatch) -> None:
    monkeypatch.setattr(hardware, "_gpu_from_torch", lambda: None)
    monkeypatch.setattr(hardware, "_gpu_from_nvidia_smi", lambda: ("GPU-X", 1.5))
    assert hardware._gpu_info() == ("GPU-X", 1.5)


def test_gpu_info_all_none(monkeypatch) -> None:
    monkeypatch.setattr(hardware, "_gpu_from_torch", lambda: None)
    monkeypatch.setattr(hardware, "_gpu_from_nvidia_smi", lambda: None)
    monkeypatch.setattr(hardware, "_gpu_from_wmi", lambda: None)
    assert hardware._gpu_info() == (None, None)


def test_gpu_from_nvidia_smi_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: _CP("", 1))
    assert hardware._gpu_from_nvidia_smi() is None


def test_gpu_from_wmi_bad_vram(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    csv = "Node,AdapterRAM,Name\nPC,notanumber,NVIDIA GTX 950M\n"
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: _CP(csv, 0))
    name, vram = hardware._gpu_from_wmi()
    assert "NVIDIA" in name
    assert vram is None


def test_disk_type_hint_windows_exception(monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")

    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(hardware.subprocess, "run", boom)
    assert "confirm" in hardware._disk_type_hint()


def test_detect_to_dict_has_all_fields() -> None:
    data = detect().to_dict()
    for key in ("os", "cpu_model", "ram_total_gb", "disk_free_gb", "python_version"):
        assert key in data
