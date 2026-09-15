"""Mask building for Art-Cleaner.

Pure numpy + Pillow. This module must never import torch, cv2, easyocr or
iopaint so it can be unit-tested on any machine without a GPU.

Masks are uint8 HxW arrays where 255 = "remove this" and 0 = "keep".
IOPaint expects exactly that: white-on-black PNGs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MAX_DILATE = 64
Point = tuple[float, float]


@dataclass
class Params:
    auto_text: bool = True
    rect: bool = False
    rect_x: float = 70.0
    rect_y: float = 90.0
    rect_w: float = 30.0
    rect_h: float = 10.0
    dilate_px: int = 8

    def validate(self) -> None:
        if not (self.auto_text or self.rect):
            raise ValueError("Turn on at least one mode: auto text or fixed rectangle.")
        if self.rect:
            rect_pct_to_px(100, 100, self.rect_x, self.rect_y, self.rect_w, self.rect_h)
        if not (0 <= int(self.dilate_px) <= MAX_DILATE):
            raise ValueError(f"dilate_px must be between 0 and {MAX_DILATE}.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        def b(v):
            return str(v).lower() in ("1", "true", "yes", "on")
        return cls(
            auto_text=b(d.get("auto_text", True)),
            rect=b(d.get("rect", False)),
            rect_x=float(d.get("rect_x", 70)),
            rect_y=float(d.get("rect_y", 90)),
            rect_w=float(d.get("rect_w", 30)),
            rect_h=float(d.get("rect_h", 10)),
            dilate_px=int(float(d.get("dilate_px", 8))),
        )


def rect_pct_to_px(width: int, height: int, x_pct: float, y_pct: float,
                   w_pct: float, h_pct: float) -> tuple[int, int, int, int]:
    """Convert a rectangle given in percent of width/height to pixel box (x0, y0, x1, y1)."""
    for name, v in (("x", x_pct), ("y", y_pct), ("w", w_pct), ("h", h_pct)):
        if not (0 <= v <= 100):
            raise ValueError(f"rect_{name} must be between 0 and 100 (got {v}).")
    if w_pct <= 0 or h_pct <= 0:
        raise ValueError("rect_w and rect_h must be greater than 0.")
    x0 = int(round(width * x_pct / 100.0))
    y0 = int(round(height * y_pct / 100.0))
    x1 = int(round(width * (x_pct + w_pct) / 100.0))
    y1 = int(round(height * (y_pct + h_pct) / 100.0))
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    return x0, y0, x1, y1


def _blank(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.zeros((h, w), dtype=np.uint8)


def rect_mask(size: tuple[int, int], rects_px: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    m = _blank(size)
    for x0, y0, x1, y1 in rects_px:
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 255
    return m


def polygons_to_mask(size: tuple[int, int], polygons: Sequence[Sequence[Point]]) -> np.ndarray:
    w, h = size
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        pts = [(min(max(float(x), 0), w - 1), min(max(float(y), 0), h - 1)) for x, y in poly]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.array(img, dtype=np.uint8)


def union(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("union needs at least one mask")
    out = np.zeros_like(masks[0])
    for m in masks:
        out = np.maximum(out, m)
    return out


def dilate(mask: np.ndarray, n_px: int) -> np.ndarray:
    n = int(n_px)
    if n <= 0:
        return mask.copy()
    img = Image.fromarray(mask, mode="L").filter(ImageFilter.MaxFilter(2 * n + 1))
    return np.array(img, dtype=np.uint8)


def is_empty(mask: np.ndarray) -> bool:
    return not bool(mask.any())


def build_mask(size: tuple[int, int], params: Params,
               text_polygons: Sequence[Sequence[Point]] | None = None) -> np.ndarray:
    """final = dilate(union(auto_mask, rect_mask), dilate_px)."""
    w, h = size
    parts = []
    if params.auto_text:
        parts.append(polygons_to_mask(size, text_polygons or []))
    if params.rect:
        box = rect_pct_to_px(w, h, params.rect_x, params.rect_y, params.rect_w, params.rect_h)
        parts.append(rect_mask(size, [box]))
    if not parts:
        return _blank(size)
    return dilate(union(*parts), params.dilate_px)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    Image.fromarray(mask, mode="L").save(str(path), format="PNG")


def load_mask(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def overlay(image: Image.Image, mask: np.ndarray, color=(255, 0, 0),
            alpha: float = 0.5, max_side: int = 1024) -> Image.Image:
    """Red translucent fill over the masked area, downscaled for phone display."""
    base = image.convert("RGB")
    m = Image.fromarray(mask, mode="L")
    if m.size != base.size:
        m = m.resize(base.size, Image.NEAREST)
    tint = Image.new("RGB", base.size, color)
    a = m.point(lambda v: int(255 * alpha) if v > 127 else 0)
    out = Image.composite(tint, base, a)
    out.thumbnail((max_side, max_side))
    return out
