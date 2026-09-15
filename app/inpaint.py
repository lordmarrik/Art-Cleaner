"""Runs IOPaint's batch mode as a subprocess."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config


def run_inpaint(image_dir: Path, mask_dir: Path, out_dir: Path, log_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "iopaint", "run",
        "--model=lama",
        f"--device={config.ART_DEVICE}",
        f"--image={image_dir}",
        f"--mask={mask_dir}",
        f"--output={out_dir}",
        f"--model-dir={config.ART_ROOT / 'models'}",
    ]
    with open(log_path, "ab") as log:
        log.write((" ".join(cmd) + "\n").encode())
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError(f"iopaint run failed (exit {proc.returncode}):\n{tail}")
