import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image


def _png_bytes(size=(120, 80), color=(50, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("folder/a.png", _png_bytes())
        zf.writestr("folder/b_notext.png", _png_bytes())
        zf.writestr("__MACOSX/._a.png", b"junk")
        zf.writestr(".DS_Store", b"junk")
        zf.writestr("readme.txt", b"not an image")
    return buf.getvalue()


@pytest.fixture()
def client(art_root, fake_engines):
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_import_without_gpu_libs():
    import app.main  # noqa: F401  (must import even where torch/easyocr are absent)


def test_index_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and "Art-Cleaner" in r.text
    h = client.get("/health").json()
    assert h["ok"] is True and h["device"] == "cpu"


def test_preview_then_clean_flow(client, art_root):
    files = [("files", ("art.zip", _zip_bytes(), "application/zip")),
             ("files", ("c.jpg", _png_bytes(), "image/jpeg"))]
    r = client.post("/jobs", files=files, data={"auto_text": "true", "dilate_px": "2", "action": "preview"})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    assert r.json()["count"] == 3
    st = client.get(f"/jobs/{jid}").json()
    assert st["state"] == "done" and st["action"] == "preview"
    assert len(st["previews"]) == 3
    assert st["empty_mask_files"] == ["b_notext.png"]
    assert sorted(p.name for p in (art_root / "masks" / jid).iterdir()) == ["a.png", "b_notext.png", "c.png"]
    assert client.get(st["previews"][0]).status_code == 200
    assert client.get(f"/jobs/{jid}/download").status_code == 404

    r = client.post(f"/jobs/{jid}/run", json={"action": "clean", "params": {"auto_text": True, "dilate_px": 2}})
    assert r.status_code == 200
    st = client.get(f"/jobs/{jid}").json()
    assert st["state"] == "done" and st["download_url"]
    d = client.get(f"/jobs/{jid}/download")
    assert d.status_code == 200 and d.headers["content-type"] == "application/zip"
    names = sorted(zipfile.ZipFile(io.BytesIO(d.content)).namelist())
    assert names == ["a.png", "b_notext.png", "c.png"]  # passthrough kept its original name+ext

    assert any(j["job_id"] == jid for j in client.get("/jobs").json())
    assert client.delete(f"/jobs/{jid}").status_code == 200
    assert client.get(f"/jobs/{jid}").status_code == 404
    assert not (art_root / "in" / jid).exists()


def test_rect_only_clean(client):
    r = client.post("/jobs", files=[("files", ("x.png", _png_bytes(), "image/png"))],
                    data={"auto_text": "false", "rect": "true", "rect_x": "70", "rect_y": "90",
                          "rect_w": "30", "rect_h": "10", "dilate_px": "0", "action": "clean"})
    assert r.status_code == 200, r.text
    st = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert st["state"] == "done" and st["empty_mask_files"] == []


def test_validation_errors(client):
    png = [("files", ("x.png", _png_bytes(), "image/png"))]
    assert client.post("/jobs", files=png, data={"auto_text": "false", "rect": "false"}).status_code == 422
    assert client.post("/jobs", files=png, data={"rect": "true", "rect_w": "0"}).status_code == 422
    assert client.post("/jobs", files=png, data={"dilate_px": "500"}).status_code == 422
    assert client.post("/jobs", files=png, data={"action": "explode"}).status_code == 422
    assert client.post("/jobs", data={"source_path": "/etc"}).status_code == 400
    assert client.post("/jobs", data={"source_path": "/workspace/does-not-exist"}).status_code == 400
    assert client.post("/jobs", files=[("files", ("n.txt", b"hi", "text/plain"))]).status_code == 400
    assert client.get("/jobs/nope").status_code == 404


def test_same_stem_different_extension_kept_apart(client, art_root):
    files = [("files", ("foo.png", _png_bytes(), "image/png")),
             ("files", ("foo.jpg", _png_bytes(), "image/jpeg"))]
    r = client.post("/jobs", files=files, data={"auto_text": "true", "action": "clean"})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    stems = sorted(p.stem for p in (art_root / "in" / jid).iterdir())
    assert stems == ["foo", "foo_1"]
    assert sorted(p.name for p in (art_root / "masks" / jid).iterdir()) == ["foo.png", "foo_1.png"]
    d = client.get(f"/jobs/{jid}/download")
    assert sorted(zipfile.ZipFile(io.BytesIO(d.content)).namelist()) == ["foo.png", "foo_1.png"]


def test_busy_rejects_and_leaves_no_orphans(client, art_root, monkeypatch):
    from app import jobs
    monkeypatch.setattr(jobs, "_running", {"other": True})
    before = {p.name for p in (art_root / "in").iterdir()}
    r = client.post("/jobs", files=[("files", ("x.png", _png_bytes(), "image/png"))], data={"auto_text": "true"})
    assert r.status_code == 409
    assert {p.name for p in (art_root / "in").iterdir()} == before

    # the race case: start_job itself raises Busy after ingest -> inputs are cleaned up
    monkeypatch.setattr(jobs, "_running", {})
    def boom(*a, **k):
        raise jobs.Busy("busy")
    monkeypatch.setattr(jobs, "start_job", boom)
    r = client.post("/jobs", files=[("files", ("x.png", _png_bytes(), "image/png"))], data={"auto_text": "true"})
    assert r.status_code == 409
    assert {p.name for p in (art_root / "in").iterdir()} == before


def test_delete_running_job_refused(client, monkeypatch):
    from app import jobs
    r = client.post("/jobs", files=[("files", ("x.png", _png_bytes(), "image/png"))], data={"auto_text": "true"})
    jid = r.json()["job_id"]
    monkeypatch.setattr(jobs, "_running", {jid: True})
    assert client.delete(f"/jobs/{jid}").status_code == 409
    monkeypatch.setattr(jobs, "_running", {})
    assert client.delete(f"/jobs/{jid}").status_code == 200


def test_blank_or_garbage_numbers_are_422_not_500(client):
    png = [("files", ("x.png", _png_bytes(), "image/png"))]
    assert client.post("/jobs", files=png, data={"rect": "true", "rect_x": "", "dilate_px": ""}).status_code == 200
    assert client.post("/jobs", files=png, data={"dilate_px": "abc"}).status_code == 422
    assert client.post("/jobs", files=png, data={"rect": "true", "rect_w": "wide"}).status_code == 422
    r = client.post("/jobs", files=png, data={"auto_text": "true"})
    jid = r.json()["job_id"]
    assert client.post(f"/jobs/{jid}/run", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post(f"/jobs/{jid}/run", json=["nope"]).status_code == 422
    assert client.post(f"/jobs/{jid}/run", json={"action": "preview", "params": {"dilate_px": "x"}}).status_code == 422
    assert client.post(f"/jobs/{jid}/run", json={"action": "preview", "params": "str"}).status_code == 422
