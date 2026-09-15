"""Runtime configuration from environment variables."""
import os
from pathlib import Path

ART_ROOT = Path(os.environ.get("ART_ROOT", "/workspace/art-cleaner"))
ART_DEVICE = os.environ.get("ART_DEVICE", "cuda")
ART_OCR_LANGS = [s.strip() for s in os.environ.get("ART_OCR_LANGS", "en").split(",") if s.strip()]
SYNC_JOBS = os.environ.get("ART_SYNC_JOBS", "0") == "1"
IOPAINT_PORT = int(os.environ.get("IOPAINT_PORT", "8080"))
APP_PORT = int(os.environ.get("ART_PORT", "8000"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

SUBDIRS = ("in", "masks", "out", "models", "stage", "preview", "status")


def ensure_dirs() -> None:
    for d in SUBDIRS:
        (ART_ROOT / d).mkdir(parents=True, exist_ok=True)
