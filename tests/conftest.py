import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def art_root(tmp_path, monkeypatch):
    """Point the app at a temp folder and run jobs synchronously."""
    root = tmp_path / "art"
    monkeypatch.setenv("ART_ROOT", str(root))
    monkeypatch.setenv("ART_SYNC_JOBS", "1")
    monkeypatch.setenv("ART_DEVICE", "cpu")
    from app import config
    monkeypatch.setattr(config, "ART_ROOT", root)
    monkeypatch.setattr(config, "SYNC_JOBS", True)
    monkeypatch.setattr(config, "ART_DEVICE", "cpu")
    config.ensure_dirs()
    return root


@pytest.fixture()
def fake_engines(monkeypatch):
    """Stand-ins for the GPU pieces: fake OCR boxes and an inpaint that just copies files."""
    from app import detect, inpaint

    def fake_polys(path):
        name = Path(path).name
        if "notext" in name:
            return []
        return [[(10, 10), (60, 10), (60, 30), (10, 30)]]

    def fake_inpaint(image_dir, mask_dir, out_dir, log_path):
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in Path(image_dir).iterdir():
            shutil.copy2(p, Path(out_dir) / (p.stem + ".png"))
        Path(log_path).write_text("fake inpaint ok\n")

    monkeypatch.setattr(detect, "text_polygons", fake_polys)
    monkeypatch.setattr(inpaint, "run_inpaint", fake_inpaint)
    return fake_polys, fake_inpaint
