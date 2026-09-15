# Art-Cleaner — project notes

Phone-operated text/watermark remover for the user's own artwork, packaged
as a Docker image for RunPod GPU pods. LaMa (via IOPaint) does the
inpainting, EasyOCR finds text. The user is a non-programmer working from
a phone: keep replies short, do the work, don't hand them commands.

## Layout

- `app/masks.py` — mask building. Pure numpy + Pillow. **Never import
  torch, cv2, easyocr or iopaint here.** This is the unit-tested core.
- `app/detect.py` — EasyOCR wrapper (lazy import). `text_polygons(path)`.
- `app/inpaint.py` — runs `iopaint run` as a subprocess. Tests monkeypatch it.
- `app/jobs.py` — pipeline: ingest → masks → previews → inpaint → zip.
  State is a JSON file per job under `<ART_ROOT>/status/`.
- `app/main.py` — FastAPI routes. `app/static/index.html` — the phone UI.
- `start.sh` — boots IOPaint UI (port 8080) and the app (port 8000).
- `Dockerfile` — `pytorch/pytorch` CUDA runtime base + `requirements.txt`.
  torch/torchvision are deliberately NOT in requirements.txt.
- `.github/workflows/build-image.yml` builds and pushes
  `ghcr.io/lordmarrik/art-cleaner:latest` on pushes to the default
  branch that touch the app, Dockerfile, start script or requirements.

## Data layout on the pod

`/workspace/art-cleaner/{in,masks,out,stage,preview,status,models}`,
each of in/masks/out/stage/preview subdivided by job id. Override with
`ART_ROOT`. `ART_DEVICE=cpu` for local runs. `ART_SYNC_JOBS=1` runs jobs
inline (tests).

## Verify

```
pip install -r requirements-dev.txt && pytest
```
Runs on CPU with no torch installed. Anything touching the GPU (model
download, CUDA, IOPaint flags, EasyOCR output shape) is only proven on a
real pod; the README's first-run steps are that checklist.

## Rules

- Keep `masks.py` torch-free. GPU code stays behind lazy imports.
- Masks are white-on-black PNGs named `<image stem>.png` (IOPaint matches by stem).
- Preview and Clean must use the same `build_mask()`.
- Deploy = push to the default branch. Don't bake models into the image.
