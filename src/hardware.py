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
    """Returns (gpu_name, vram_gb). Prefers torch (gives exact usable VRAM),
    then falls back to nvidia-smi, then to Windows WMI so the GPU is captured
    even before torch is installed."""
    # 1) torch (most accurate — reports the CUDA device the runs will use)
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return props.name, props.total_memory / (1024**3)
    except ImportError:
        pass

    # 2) nvidia-smi (present when an NVIDIA driver is installed, no torch needed)
    try:
        import subprocess

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

    # 3) Windows WMI fallback (captures the adapter even without NVIDIA tooling)
    if platform.system() == "Windows":
        try:
            import subprocess

            out = subprocess.run(
                ["wmic", "path", "win32_VideoController",
                 "get", "name,AdapterRAM", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            best_name, best_vram = None, None
            for line in out.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3 or not parts[2] or parts[1] == "AdapterRAM":
                    continue
                name = parts[2]
                try:
                    vram = int(parts[1]) / (1024**3)
                except ValueError:
                    vram = None
                # Prefer a discrete NVIDIA/AMD adapter over integrated graphics.
                if best_name is None or "nvidia" in name.lower() or "amd" in name.lower():
                    best_name, best_vram = name, vram
            if best_name:
                # WMI AdapterRAM is a 32-bit field: it caps/garbles VRAM > 4 GB,
                # so treat it as a hint only.
                return best_name, best_vram
        except (OSError, subprocess.SubprocessError):
            pass

    return None, None


def _disk_type_hint(path: str = ".") -> str:
    """Best-effort SSD/HDD detection. Disk class matters here: AirLLM is
    disk-I/O-bound, so SSD vs HDD directly sets the decode ceiling."""
    system = platform.system()
    if system == "Windows":
        try:
            import subprocess

            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-PhysicalDisk | Select-Object -First 1 -ExpandProperty MediaType)"],
                capture_output=True, text=True, timeout=15,
            )
            media = out.stdout.strip()
            if media:
                return f"{media} (Get-PhysicalDisk)"
        except (OSError, subprocess.SubprocessError):
            pass
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
