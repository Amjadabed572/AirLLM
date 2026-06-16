"""Hardware auto-detection.

Generates the exact spec sheet Task 5.1 (hardware documentation) requires,
straight from the machine the experiment runs on. No manual typing, no guessing.

Run standalone:  python -m src.hardware
"""
from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict, dataclass

import psutil


@dataclass
class HardwareInfo:
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


def _cpu_model() -> str:
    # platform.processor() is often empty on Linux; fall back to /proc/cpuinfo.
    name = platform.processor()
    if name:
        return name
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine() or "unknown"


def _gpu_info() -> tuple[str | None, float | None]:
    """Returns (gpu_name, vram_gb). Lazy torch import so this module
    stays importable on machines without torch installed yet."""
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return props.name, props.total_memory / (1024**3)


def _disk_type_hint(path: str = ".") -> str:
    """Best-effort SSD/HDD hint. Exact detection is OS-specific; we flag
    that the user should confirm. Most modern laptops are NVMe/SSD."""
    system = platform.system()
    if system == "Windows":
        return "confirm SSD/NVMe vs HDD in Task Manager > Performance"
    return "confirm with `lsblk -d -o name,rota` (rota=0 means SSD)"


def detect() -> HardwareInfo:
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


def main() -> None:
    info = detect()
    print(info.pretty())
    with open("results/hardware.json", "w") as f:
        json.dump(asdict(info), f, indent=2)
    print("\nSaved -> results/hardware.json")


if __name__ == "__main__":
    main()
