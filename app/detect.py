"""Text detection via EasyOCR. Imported lazily so the app loads without torch."""
from __future__ import annotations

from pathlib import Path

from . import config

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr  # heavy import, GPU
        model_dir = config.ART_ROOT / "models" / "easyocr"
        model_dir.mkdir(parents=True, exist_ok=True)
        _reader = easyocr.Reader(
            config.ART_OCR_LANGS,
            gpu=(config.ART_DEVICE == "cuda"),
            model_storage_directory=str(model_dir),
            download_enabled=True,
        )
    return _reader


def _to_polygons(horizontal_list, free_list):
    polys = []
    for box in horizontal_list or []:
        x_min, x_max, y_min, y_max = box
        polys.append([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    for quad in free_list or []:
        polys.append([(float(x), float(y)) for x, y in quad])
    return polys


def text_polygons(image_path: str | Path) -> list[list[tuple[float, float]]]:
    """Return a list of 4-point polygons around detected text (detection only, no reading)."""
    reader = get_reader()
    horizontal, free = reader.detect(str(image_path))
    # EasyOCR returns one entry per input image.
    h = horizontal[0] if horizontal else []
    f = free[0] if free else []
    return _to_polygons(h, f)
