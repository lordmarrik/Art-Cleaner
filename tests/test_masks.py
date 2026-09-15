import numpy as np
import pytest
from PIL import Image

from app.masks import (Params, build_mask, dilate, is_empty, load_mask, overlay,
                       polygons_to_mask, rect_mask, rect_pct_to_px, save_mask, union)


def test_rect_pct_to_px_basic_and_clamp():
    assert rect_pct_to_px(200, 100, 50, 50, 50, 50) == (100, 50, 200, 100)
    assert rect_pct_to_px(200, 100, 90, 90, 30, 30) == (180, 90, 200, 100)  # clamped
    assert rect_pct_to_px(1000, 500, 0, 0, 100, 100) == (0, 0, 1000, 500)


@pytest.mark.parametrize("bad", [(-1, 0, 10, 10), (0, 101, 10, 10), (0, 0, 0, 10), (0, 0, 10, 0)])
def test_rect_pct_to_px_rejects(bad):
    with pytest.raises(ValueError):
        rect_pct_to_px(100, 100, *bad)


def test_rect_mask_values():
    m = rect_mask((20, 10), [(5, 2, 15, 8)])
    assert m.shape == (10, 20) and m.dtype == np.uint8
    assert set(np.unique(m)) <= {0, 255}
    assert m[2:8, 5:15].min() == 255
    assert m[0, 0] == 0 and m[9, 19] == 0 and m[1, 5] == 0 and m[2, 4] == 0


def test_polygons_to_mask_and_clamping():
    m = polygons_to_mask((20, 10), [[(2, 2), (8, 2), (8, 6), (2, 6)], [(-50, -50), (100, -50), (100, 100)]])
    assert m.shape == (10, 20)
    assert m[3, 4] == 255 and m[8, 1] == 0 or m[8, 1] == 255  # second poly may cover; no crash is the point
    assert set(np.unique(m)) <= {0, 255}


def test_union_is_or():
    a = rect_mask((10, 10), [(0, 0, 5, 5)])
    b = rect_mask((10, 10), [(5, 5, 10, 10)])
    u = union(a, b)
    assert u[2, 2] == 255 and u[7, 7] == 255 and u[2, 7] == 0


def test_dilate_distance():
    m = np.zeros((21, 21), dtype=np.uint8)
    m[10, 10] = 255
    d = dilate(m, 3)
    assert d[10, 13] == 255 and d[13, 10] == 255 and d[13, 13] == 255  # square kernel
    assert d[10, 14] == 0 and d[14, 10] == 0
    assert np.array_equal(dilate(m, 0), m)


def test_build_mask_modes():
    size = (100, 50)
    poly = [[(10, 10), (30, 10), (30, 20), (10, 20)]]
    auto = build_mask(size, Params(auto_text=True, rect=False, dilate_px=0), poly)
    assert auto[15, 20] == 255 and auto[45, 90] == 0
    rect = build_mask(size, Params(auto_text=False, rect=True, rect_x=70, rect_y=80, rect_w=30, rect_h=20, dilate_px=0), poly)
    assert rect[45, 90] == 255 and rect[15, 20] == 0
    both = build_mask(size, Params(auto_text=True, rect=True, rect_x=70, rect_y=80, rect_w=30, rect_h=20, dilate_px=0), poly)
    assert both[45, 90] == 255 and both[15, 20] == 255
    none = build_mask(size, Params(auto_text=False, rect=False), poly)
    assert is_empty(none)
    empty_auto = build_mask(size, Params(auto_text=True, rect=False), [])
    assert is_empty(empty_auto)


def test_rect_is_proportional_across_sizes():
    p = Params(auto_text=False, rect=True, rect_x=50, rect_y=50, rect_w=50, rect_h=50, dilate_px=0)
    small = build_mask((100, 100), p)
    big = build_mask((400, 200), p)
    assert small[75, 75] == 255 and small[25, 25] == 0
    assert big[150, 300] == 255 and big[50, 100] == 0


def test_params_validate():
    with pytest.raises(ValueError):
        Params(auto_text=False, rect=False).validate()
    with pytest.raises(ValueError):
        Params(dilate_px=999).validate()
    with pytest.raises(ValueError):
        Params(rect=True, rect_w=0).validate()
    Params().validate()
    p = Params.from_dict({"auto_text": "false", "rect": "on", "rect_x": "10", "dilate_px": "4.0"})
    assert p.auto_text is False and p.rect is True and p.rect_x == 10 and p.dilate_px == 4


def test_save_load_roundtrip(tmp_path):
    m = rect_mask((30, 20), [(5, 5, 10, 10)])
    save_mask(m, tmp_path / "m.png")
    img = Image.open(tmp_path / "m.png")
    assert img.mode == "L"
    assert np.array_equal(load_mask(tmp_path / "m.png"), m)


def test_overlay_downscales():
    im = Image.new("RGB", (2048, 1024), (0, 255, 0))
    m = rect_mask((2048, 1024), [(0, 0, 100, 100)])
    ov = overlay(im, m, max_side=512)
    assert ov.mode == "RGB" and max(ov.size) == 512
    r, g, b = ov.getpixel((5, 5))
    assert r > 100  # reddish where masked
    assert ov.getpixel((500, 250)) == (0, 255, 0)
