# Art-Cleaner

Removes text and watermarks from **your own artwork** so you can build clean
training datasets. Runs on a RunPod GPU pod. You use it from your phone's browser.

Only use this on art you own or have permission to edit.

## What it does

1. You upload a zip of images (or point it at a folder on the pod).
2. It finds text automatically, and/or masks a fixed spot you choose
   (for a watermark that sits in the same corner on every image).
3. **Preview** shows in red what will be removed. Nothing changes yet.
4. **Clean** fills those areas in with the LaMa inpainting model.
5. You download a zip of the cleaned images (lossless PNGs).

There is also a **brush editor** (IOPaint) for touch-ups on anything the
automatic pass missed. Paint over the mark with your finger, tap, done.

## One-time setup

### A. Make the image pullable

The Docker image is built automatically by GitHub Actions and stored at
`ghcr.io/lordmarrik/art-cleaner-runpod:latest`. Because this repo is public, the
package normally comes out public too. Check once: on GitHub, open your
profile → **Packages** → **art-cleaner-runpod**. If it says **Private**, either:

- **Easiest:** **Package settings** → **Change visibility** → **Public**.
  One time only.
- **Or keep it private:** in RunPod go to **Settings → Container Registry
  Credentials**, add `ghcr.io` with your GitHub username and a personal
  access token that has the `read:packages` permission.

### B. Create the RunPod template

RunPod → **Templates → New Template**:

| Field | Value |
|---|---|
| Container image | `ghcr.io/lordmarrik/art-cleaner-runpod:latest` |
| Container disk | 20 GB |
| Volume disk | 20 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8000,8080` |

Leave the start command blank. The image knows how to start itself.

### C. Deploy a pod

**Pods → Deploy**, pick any GPU with 8 GB or more of memory, choose the
template above, deploy. Attaching a **network volume** at `/workspace` is
optional but means the models and your files survive between pods.

## Using it from your phone

1. On the pod page tap **Connect → HTTP Service [Port 8000]**.
   The first start takes a few minutes while about 300 MB of models
   download. After that they live on the volume and it's fast.
2. Pick your zip. Choose **Find text automatically** and/or **Fixed spot**.
3. Tap **Preview masks**. Check the red areas. Adjust the numbers or the
   margin and preview again if needed.
4. Tap **Clean with these masks**. Watch the progress bar.
5. Tap **Download cleaned zip**.

Images where nothing was found are passed through unchanged, and the page
tells you which ones so you can handle them in the brush editor.

For touch-ups: **Open brush editor** (that's port 8080), load the image,
paint over the mark, save.

Phone tab got killed mid-job? Just reopen the page. Jobs run on the pod, and
**Recent jobs** lets you pick them back up.

## Tips

- Big zips over a phone connection can time out. Split into a few smaller
  zips, or upload to the pod's `/workspace` some other way and type the
  folder path instead.
- **Safety margin** (dilate) of 6 to 12 px is usual. Bigger fills cleaner but
  eats more of the surrounding art.
- Stop the pod when you're done. You only pay while it runs.

## Updating

Push a change and GitHub Actions rebuilds the image
(look under the repo's **Actions** tab). Then redeploy the pod so it pulls
the new `latest`.

## Troubleshooting

- **"Not ready" page from RunPod:** the app hasn't started yet. Wait a
  minute and refresh.
- **Yellow "engine warming up" banner:** the brush editor is still loading
  the model. Uploading works; cleaning waits for it.
- **Something failed:** the error shows on the page. Pod logs are under
  the pod's **Logs** button, and the batch log is at
  `/workspace/art-cleaner/status/<job id>.log`.
- Quick check that the app is alive: open `/health` on port 8000.

## For developers

```
pip install -r requirements-dev.txt
pytest
ART_DEVICE=cpu ART_ROOT=/tmp/art uvicorn app.main:app --port 8000
```

Tests never need a GPU. `app/masks.py` is torch-free by design; the GPU
pieces (`detect.py`, `inpaint.py`) are imported lazily.
