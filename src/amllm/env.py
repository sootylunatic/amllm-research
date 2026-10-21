"""Record the hardware/software environment of a run (backlog AM-001)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return None


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def source_hash() -> str:
    """SHA-256 over the amllm package sources: identifies the code version without git."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


# modules that determine training and evaluation results (not reporting/profiling tools)
TRAIN_CODE = ("config.py", "data.py", "model.py", "routing.py", "train.py")


def train_code_hash() -> str:
    """SHA-256 over the modules that determine training/evaluation results."""
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode() + (Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:16]


def sanitize(text: str) -> str:
    """Strip machine-specific absolute paths (working directory, home) before results are published."""
    text = text.replace(str(Path.cwd()) + "/", "")
    return text.replace(str(Path.home()), "~")


def collect() -> dict:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": p.name, "vram_gb": round(p.total_memory / 2**30, 2),
                         "capability": f"{p.major}.{p.minor}"})
    return {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "os": platform.platform(), "cpu": _cpu_name(),
        "ram_gb": _ram_gb(), "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "driver": (_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or "").split("\n")[0] or None,
        "gpus": gpus, "visible_device": torch.cuda.get_device_name(0) if gpus else "cpu",
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]), "source_sha256": source_hash(), "train_code_sha256": train_code_hash(),
        "command": sanitize(" ".join(sys.argv)),
    }


if __name__ == "__main__":
    info = collect()
    out = Path("results/environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
