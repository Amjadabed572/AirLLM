"""Hardware auto-detection (Task 5.1).

Generates the exact spec sheet the report requires, straight from the machine the
experiment runs on. GPU detection falls back torch -> nvidia-smi -> Windows WMI so
the adapter is captured even before torch is installed; disk class is detected via
Get-PhysicalDisk on Windows (SSD vs HDD sets the AirLLM decode ceiling).
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass

import psutil


@dataclass
class HardwareInfo:
    """Detected machine specification (a building-block value object)."""

    os: str
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: str | None
    vram_total_gb: float | None
    disk_free_gb: float
    disk_type_hint: str
    python_version: str

    def pretty(self) -> str:
        """Format the spec for console output."""
        gpu = self.gpu_name or "None (CPU-only)"
        vram = f"{self.vram_total_gb:.1f} GB" if self.vram_total_gb else "N/A"
        return (
            f"OS:            {self.os}\n"
            f"CPU:           {self.cpu_model}\n"
            f"Cores:         {self.cpu_physical_cores} physical / "
            f"{self.cpu_logical_cores} logical\n"
            f"RAM:           {self.ram_total_gb:.1f} GB total "
            f"({self.ram_available_gb:.1f} GB free now)\n"
            f"GPU:           {gpu}\n"
            f"VRAM:          {vram}\n"
            f"Disk free:     {self.disk_free_gb:.1f} GB ({self.disk_type_hint})\n"
            f"Python:        {self.python_version}"
        )

    def to_dict(self) -> dict:
        """Serialise for results/hardware.json."""
        return asdict(self)


def _cpu_model() -> str:
    name = platform.processor()
    if name:
        return name
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine() or "unknown"


def _gpu_from_torch() -> tuple[str, float] | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return props.name, props.total_memory / (1024**3)
    except ImportError:
        pass
    return None


def _gpu_from_nvidia_smi() -> tuple[str, float] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, mem_mib = out.stdout.strip().splitlines()[0].split(",")
            return name.strip(), float(mem_mib) / 1024.0
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _gpu_from_wmi() -> tuple[str, float | None] | None:
    if platform.system() != "Windows":
        return None
    try:
        out = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM",
             "/format:csv"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    best: tuple[str, float | None] | None = None
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[2] or parts[1] == "AdapterRAM":
            continue
        name = parts[2]
        try:
            vram: float | None = int(parts[1]) / (1024**3)
        except ValueError:
            vram = None
        if best is None or "nvidia" in name.lower() or "amd" in name.lower():
            best = (name, vram)
    return best


def _gpu_info() -> tuple[str | None, float | None]:
    """Returns (gpu_name, vram_gb) via the most accurate available source."""
    for source in (_gpu_from_torch, _gpu_from_nvidia_smi, _gpu_from_wmi):
        result = source()
        if result is not None:
            return result
    return None, None


def _disk_type_hint() -> str:
    """Best-effort SSD/HDD detection — disk class sets the AirLLM decode ceiling."""
    if platform.system() != "Windows":
        return "confirm with `lsblk -d -o name,rota` (rota=0 means SSD)"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-PhysicalDisk | Select-Object -First 1 -ExpandProperty MediaType)"],
            capture_output=True, text=True, timeout=15,
        )
        if out.stdout.strip():
            return f"{out.stdout.strip()} (Get-PhysicalDisk)"
    except (OSError, subprocess.SubprocessError):
        pass
    return "confirm SSD/NVMe vs HDD in Task Manager > Performance"


def detect() -> HardwareInfo:
    """Probe the current machine and return its full specification."""
    vm = psutil.virtual_memory()
    gpu_name, vram = _gpu_info()
    free = shutil.disk_usage(".").free / (1024**3)
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        cpu_model=_cpu_model(),
        cpu_physical_cores=psutil.cpu_count(logical=False) or 0,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_gb=vm.total / (1024**3),
        ram_available_gb=vm.available / (1024**3),
        gpu_name=gpu_name,
        vram_total_gb=vram,
        disk_free_gb=free,
        disk_type_hint=_disk_type_hint(),
        python_version=platform.python_version(),
    )
