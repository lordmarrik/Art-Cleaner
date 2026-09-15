"""Job pipeline: ingest -> masks -> previews -> inpaint -> zip. State lives in status/<id>.json."""
from __future__ import annotations

import json
import random
import shutil
import string
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import config
from .masks import Params, build_mask, is_empty, load_mask, overlay, save_mask


class Busy(Exception):
    pass


def is_running(job_id: str) -> bool:
    with _lock:
        return job_id in _running


def any_running() -> bool:
    with _lock:
        return bool(_running)


_lock = threading.Lock()
_running: dict[str, bool] = {}


def new_job_id() -> str:
    tag = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + tag


def _dir(kind: str, job_id: str) -> Path:
    return config.ART_ROOT / kind / job_id


def status_path(job_id: str) -> Path:
    return config.ART_ROOT / "status" / f"{job_id}.json"


def read_status(job_id: str) -> dict | None:
    p = status_path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_status(job_id: str, **updates) -> dict:
    st = read_status(job_id) or {"job_id": job_id, "started_at": time.time()}
    st.update(updates)
    st["updated_at"] = time.time()
    status_path(job_id).parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path(job_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    tmp.replace(status_path(job_id))
    return st


def list_jobs() -> list[dict]:
    d = config.ART_ROOT / "status"
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            st = json.loads(p.read_text())
        except Exception:
            continue
        out.append({k: st.get(k) for k in ("job_id", "state", "stage", "total", "done_count", "action", "updated_at")})
    return out


def _safe_name(name: str) -> str:
    base = Path(name).name
    keep = "".join(c if (c.isalnum() or c in "-_. ") else "_" for c in base).strip()
    return keep or "image"


def _stem_taken(folder: Path, stem: str) -> bool:
    """Masks, previews and outputs are keyed by stem, so stems must be unique (foo.jpg vs foo.png)."""
    return any(p.stem == stem for p in folder.iterdir() if p.is_file())


def _unique(dest: Path) -> Path:
    folder, stem, suf = dest.parent, dest.stem, dest.suffix
    if not _stem_taken(folder, stem):
        return dest
    for i in range(1, 10000):
        cand = f"{stem}_{i}"
        if not _stem_taken(folder, cand):
            return dest.with_name(f"{cand}{suf}")
    raise RuntimeError("too many duplicate names")


def _is_image(name: str) -> bool:
    p = Path(name)
    if p.name.startswith(".") or "__MACOSX" in p.parts:
        return False
    return p.suffix.lower() in config.IMAGE_EXTS


def ingest(job_id: str, uploads: list[tuple[str, Path]] | None = None,
           source_path: str | None = None) -> int:
    """uploads: list of (original_filename, temp_path). source_path: folder under /workspace."""
    in_dir = _dir("in", job_id)
    in_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, tmp in uploads or []:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(tmp) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not _is_image(info.filename):
                        continue
                    dest = _unique(in_dir / _safe_name(info.filename))
                    with zf.open(info) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    count += 1
        elif _is_image(name):
            dest = _unique(in_dir / _safe_name(name))
            shutil.move(str(tmp), dest)
            count += 1
    if source_path:
        src = Path(source_path).resolve()
        allowed = Path("/workspace").resolve()
        if allowed not in src.parents and src != allowed:
            raise ValueError("Folder must be inside /workspace.")
        if not src.is_dir():
            raise ValueError(f"Folder not found: {source_path}")
        for p in sorted(src.iterdir()):
            if p.is_file() and _is_image(p.name):
                shutil.copy2(p, _unique(in_dir / _safe_name(p.name)))
                count += 1
    return count


def _images(job_id: str) -> list[Path]:
    d = _dir("in", job_id)
    return sorted(p for p in d.iterdir() if p.is_file() and _is_image(p.name)) if d.exists() else []


def build_masks(job_id: str, params: Params) -> list[str]:
    imgs = _images(job_id)
    mask_dir = _dir("masks", job_id)
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir(parents=True)
    empty: list[str] = []
    write_status(job_id, state="detecting" if params.auto_text else "masking",
                 total=len(imgs), done_count=0)
    for i, p in enumerate(imgs):
        write_status(job_id, current_file=p.name, done_count=i,
                     stage=f"{'Detecting text' if params.auto_text else 'Masking'} {i + 1} / {len(imgs)}")
        with Image.open(p) as im:
            size = im.size
        polys = None
        if params.auto_text:
            from .detect import text_polygons  # lazy: GPU
            polys = text_polygons(p)
        m = build_mask(size, params, polys)
        if is_empty(m):
            empty.append(p.name)
        save_mask(m, mask_dir / f"{p.stem}.png")
    write_status(job_id, done_count=len(imgs), empty_mask_files=empty)
    return empty


def make_previews(job_id: str) -> list[str]:
    imgs = _images(job_id)
    prev_dir = _dir("preview", job_id)
    if prev_dir.exists():
        shutil.rmtree(prev_dir)
    prev_dir.mkdir(parents=True)
    names = []
    write_status(job_id, state="previewing", total=len(imgs), done_count=0)
    for i, p in enumerate(imgs):
        write_status(job_id, done_count=i, stage=f"Preview {i + 1} / {len(imgs)}", current_file=p.name)
        m = load_mask(_dir("masks", job_id) / f"{p.stem}.png")
        with Image.open(p) as im:
            ov = overlay(im, m)
        out = prev_dir / f"{p.stem}.jpg"
        ov.save(out, quality=80)
        names.append(out.name)
    write_status(job_id, done_count=len(imgs),
                 previews=[f"/jobs/{job_id}/preview/{n}" for n in names])
    return names


def _watch_progress(job_id: str, out_dir: Path, total: int, stop: threading.Event) -> None:
    while not stop.wait(1.0):
        n = len(list(out_dir.glob("*"))) if out_dir.exists() else 0
        write_status(job_id, done_count=n, stage=f"Inpainting {n} / {total}")


def run_clean(job_id: str) -> Path:
    from .inpaint import run_inpaint  # lazy
    imgs = _images(job_id)
    mask_dir = _dir("masks", job_id)
    stage_dir = _dir("stage", job_id)
    out_dir = _dir("out", job_id)
    for d in (stage_dir, out_dir):
        if d.exists():
            shutil.rmtree(d)
    (stage_dir / "img").mkdir(parents=True)
    (stage_dir / "mask").mkdir(parents=True)
    out_dir.mkdir(parents=True)
    to_paint = 0
    for p in imgs:
        m_path = mask_dir / f"{p.stem}.png"
        if m_path.exists() and not is_empty(load_mask(m_path)):
            shutil.copy2(p, stage_dir / "img" / p.name)
            shutil.copy2(m_path, stage_dir / "mask" / m_path.name)
            to_paint += 1
        else:
            shutil.copy2(p, out_dir / p.name)  # nothing to remove: pass through unchanged
    write_status(job_id, state="inpainting", total=to_paint, done_count=0,
                 stage=f"Inpainting 0 / {to_paint}")
    if to_paint:
        stop = threading.Event()
        painted_dir = stage_dir / "painted"
        t = threading.Thread(target=_watch_progress, args=(job_id, painted_dir, to_paint, stop), daemon=True)
        t.start()
        try:
            run_inpaint(stage_dir / "img", stage_dir / "mask", painted_dir,
                        config.ART_ROOT / "status" / f"{job_id}.log")
        finally:
            stop.set()
            t.join(timeout=2)
        for p in sorted(painted_dir.glob("*")):
            shutil.move(str(p), out_dir / p.name)
    write_status(job_id, state="zipping", stage="Zipping results")
    zip_path = config.ART_ROOT / "out" / f"{job_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(out_dir.iterdir()):
            if p.is_file():
                zf.write(p, p.name)
    return zip_path


def run_job(job_id: str, action: str, params: Params) -> None:
    try:
        write_status(job_id, action=action, params=params.to_dict(), state="queued", error=None)
        build_masks(job_id, params)
        make_previews(job_id)
        if action == "clean":
            run_clean(job_id)
            write_status(job_id, download_url=f"/jobs/{job_id}/download")
        write_status(job_id, state="done", stage="Done")
    except Exception as e:  # noqa: BLE001
        write_status(job_id, state="error", stage="Error", error=str(e))
    finally:
        with _lock:
            _running.pop(job_id, None)


def start_job(job_id: str, action: str, params: Params) -> None:
    params.validate()
    with _lock:
        if _running:
            raise Busy("Another job is still running. Wait for it to finish.")
        _running[job_id] = True
    write_status(job_id, action=action, state="queued", stage="Queued", total=len(_images(job_id)), done_count=0)
    if config.SYNC_JOBS:
        run_job(job_id, action, params)
    else:
        threading.Thread(target=run_job, args=(job_id, action, params), daemon=True).start()


def delete_job(job_id: str) -> None:
    for kind in ("in", "masks", "out", "stage", "preview"):
        d = _dir(kind, job_id)
        if d.exists():
            shutil.rmtree(d)
    for p in (config.ART_ROOT / "out" / f"{job_id}.zip", status_path(job_id),
              config.ART_ROOT / "status" / f"{job_id}.log"):
        if p.exists():
            p.unlink()
