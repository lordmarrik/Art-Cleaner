#!/usr/bin/env bash
# Boots both services inside the RunPod pod.
set -e
ROOT="${ART_ROOT:-/workspace/art-cleaner}"
for d in in masks out models stage preview status; do mkdir -p "$ROOT/$d"; done
export HF_HOME="$ROOT/models/hf" TORCH_HOME="$ROOT/models/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

POD="${RUNPOD_POD_ID:-localhost}"
echo "=== Art-Cleaner ==="
echo "Web app:      https://${POD}-${ART_PORT:-8000}.proxy.runpod.net"
echo "Brush editor: https://${POD}-${IOPAINT_PORT:-8080}.proxy.runpod.net"
echo "Data root:    $ROOT"

# IOPaint's own UI for manual touch-ups (downloads LaMa on first run into the volume).
python -m iopaint start --model=lama --device="${ART_DEVICE:-cuda}" --host=0.0.0.0 \
  --port="${IOPAINT_PORT:-8080}" --model-dir="$ROOT/models" >"$ROOT/status/iopaint.log" 2>&1 &

cd /app
exec uvicorn app.main:app --host 0.0.0.0 --port "${ART_PORT:-8000}"
