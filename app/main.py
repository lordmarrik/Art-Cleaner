"""Art-Cleaner web app (FastAPI). Phone-first UI on port 8000; IOPaint UI on 8080."""
from __future__ import annotations

import shutil
import socket
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, jobs
from .masks import Params

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Art-Cleaner")
config.ensure_dirs()
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _iopaint_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", config.IOPAINT_PORT), timeout=0.3):
            return True
    except OSError:
        return False


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.get("/health")
def health():
    return {"ok": True, "root": str(config.ART_ROOT), "device": config.ART_DEVICE,
            "iopaint_ui_up": _iopaint_up(), "iopaint_port": config.IOPAINT_PORT}


def _params_from_form(form) -> Params:
    if not hasattr(form, "get"):
        raise HTTPException(422, "params must be an object")
    raw = {k: form.get(k) for k in ("auto_text", "rect", "rect_x", "rect_y", "rect_w", "rect_h",
                                     "dilate_px", "merge_px", "band")
           if form.get(k) not in (None, "")}
    try:
        p = Params.from_dict(raw)
        p.validate()
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Check the numbers you entered: {e}")
    return p


def _check_action(action: str) -> str:
    if action not in ("preview", "clean"):
        raise HTTPException(422, "action must be 'preview' or 'clean'")
    return action


@app.post("/jobs")
async def create_job(request: Request,
                     files: list[UploadFile] = File(default=[]),
                     source_path: str | None = Form(default=None),
                     action: str = Form(default="preview")):
    form = await request.form()
    params = _params_from_form(form)
    action = _check_action(action)
    if jobs.any_running():
        raise HTTPException(409, "Another job is still running. Wait for it to finish.")
    job_id = jobs.new_job_id()
    tmpdir = Path(tempfile.mkdtemp(prefix="upl-"))
    uploads: list[tuple[str, Path]] = []
    try:
        for f in files:
            if not f.filename:
                continue
            dest = tmpdir / Path(f.filename).name
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)  # stream to disk, never into memory
            uploads.append((f.filename, dest))
        try:
            n = jobs.ingest(job_id, uploads, source_path or None)
        except ValueError as e:
            raise HTTPException(400, str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if n == 0:
        jobs.delete_job(job_id)
        raise HTTPException(400, "No images found. Upload a zip or image files (jpg, png, webp).")
    try:
        jobs.start_job(job_id, action, params)
    except jobs.Busy as e:
        jobs.delete_job(job_id)  # don't leave orphaned uploads on the volume
        raise HTTPException(409, str(e))
    return {"job_id": job_id, "count": n}


@app.get("/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    st = jobs.read_status(job_id)
    if st is None:
        raise HTTPException(404, "job not found")
    return st


@app.post("/jobs/{job_id}/run")
async def rerun_job(job_id: str, request: Request):
    if jobs.read_status(job_id) is None:
        raise HTTPException(404, "job not found")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(422, "body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(422, "body must be a JSON object")
    action = _check_action(body.get("action", "preview"))
    params = _params_from_form(body.get("params") or {})
    try:
        jobs.start_job(job_id, action, params)
    except jobs.Busy as e:
        raise HTTPException(409, str(e))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/preview/{name}")
def preview(job_id: str, name: str):
    p = config.ART_ROOT / "preview" / job_id / Path(name).name
    if not p.is_file():
        raise HTTPException(404, "preview not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    p = config.ART_ROOT / "out" / f"{job_id}.zip"
    st = jobs.read_status(job_id)
    if st is None or st.get("state") != "done" or not p.is_file():
        raise HTTPException(404, "results not ready")
    return FileResponse(str(p), media_type="application/zip", filename=f"cleaned-{job_id}.zip")


@app.delete("/jobs/{job_id}")
def delete(job_id: str):
    if jobs.read_status(job_id) is None:
        raise HTTPException(404, "job not found")
    if jobs.is_running(job_id):
        raise HTTPException(409, "This job is still running. Wait for it to finish, then delete it.")
    jobs.delete_job(job_id)
    return JSONResponse({"deleted": job_id})
