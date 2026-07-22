from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Some managed Windows environments do not expose physical-core metadata to
    # joblib.  An explicit, slightly lower logical-core limit avoids its
    # unavailable physical-core probe while leaving one core responsive.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 1) - 1)))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def gpu_is_visible() -> bool:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return completed.returncode == 0 and bool(completed.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def timed_predict_proba(model: Any, x: Any, repeats: int = 5) -> float:
    if len(x) == 0:
        return float("nan")
    n = min(1000, len(x))
    sample = x.iloc[:n] if hasattr(x, "iloc") else x[:n]
    model.predict_proba(sample)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(sample)
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed) * 1000.0 * (1000.0 / n))


def copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
