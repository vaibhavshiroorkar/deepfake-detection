from __future__ import annotations

import importlib.metadata
import os
import platform as platform_module
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "mlflow",
    "timm",
    "transformers",
    "av",
    "opencv-python",
    "facenet-pytorch",
    "scikit-learn",
)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    started_at_utc: str
    git_commit: str
    git_dirty: bool
    python_version: str
    platform: str
    packages: dict[str, str | None]
    cpu: str
    gpu: str | None
    gpu_memory_mib: int | None
    available_memory_mib: int | None
    ffmpeg_version: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at_utc": self.started_at_utc,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "packages": dict(self.packages),
            "cpu": self.cpu,
            "gpu": self.gpu,
            "gpu_memory_mib": self.gpu_memory_mib,
            "available_memory_mib": self.available_memory_mib,
            "ffmpeg_version": self.ffmpeg_version,
        }


def capture_runtime(root: Path) -> RuntimeSnapshot:
    git_commit = _command_output(["git", "rev-parse", "HEAD"], root)
    git_status = _command_output(["git", "status", "--porcelain"], root)
    gpu, gpu_memory_mib = _gpu_details()
    cpu = platform_module.processor() or platform_module.machine() or "unknown"
    return RuntimeSnapshot(
        started_at_utc=datetime.now(UTC).isoformat(),
        git_commit=git_commit or "unavailable",
        git_dirty=bool(git_status),
        python_version=sys.version,
        platform=platform_module.platform(),
        packages={name: _package_version(name) for name in _PACKAGE_NAMES},
        cpu=cpu,
        gpu=gpu,
        gpu_memory_mib=gpu_memory_mib,
        available_memory_mib=_available_memory_mib(),
        ffmpeg_version=_ffmpeg_version(root),
    )


def seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except (ImportError, OSError):
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _command_output(command: list[str], root: Path) -> str | None:
    try:
        process = subprocess.run(  # noqa: S603
            command,
            cwd=root,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return None
    output = process.stdout.strip()
    return output or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_details() -> tuple[str | None, int | None]:
    try:
        import torch
    except (ImportError, OSError):
        return None, None
    if not torch.cuda.is_available():
        return None, None
    try:
        properties = torch.cuda.get_device_properties(0)
        return torch.cuda.get_device_name(0), properties.total_memory // (1024**2)
    except (RuntimeError, ValueError):
        return None, None


def _available_memory_mib() -> int | None:
    if os.name == "nt":
        return _windows_available_memory_mib()
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or available_pages < 0:
        return None
    return (page_size * available_pages) // (1024**2)


def _windows_available_memory_mib() -> int | None:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.available_physical // (1024**2)
    except (AttributeError, OSError):
        return None


def _ffmpeg_version(root: Path) -> str | None:
    output = _command_output(["ffmpeg", "-version"], root)
    return output.splitlines()[0] if output else None
