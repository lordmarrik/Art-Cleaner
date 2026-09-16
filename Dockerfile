# Art-Cleaner: LaMa inpainting + EasyOCR text detection, phone-first web UI, for RunPod GPU pods.
# Base image already contains Python, torch, torchvision and the CUDA runtime.
# CUDA 12.8 build: its torch ships kernels for every current RunPod GPU, including
# RTX 50-series / Blackwell (sm_120). The older cuda12.1 tag failed on those with
# "CUDA error: no kernel image is available for execution on the device".
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    ART_ROOT=/workspace/art-cleaner \
    ART_DEVICE=cuda \
    ART_PORT=8000 \
    IOPAINT_PORT=8080

# opencv (pulled in by easyocr/iopaint) needs libGL; git is not needed.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app/ /app/app/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000 8080
CMD ["bash", "/app/start.sh"]
