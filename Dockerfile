# syntax=docker/dockerfile:1.6
#
# Inference image for the three-style LoRA stylizer.
#
#   - CUDA 12.4 runtime (the cu128 torch wheels bring their own CUDA libs).
#   - Python 3.12 (matches the host venv).
#   - LoRA weights are NOT baked in — mount ./loras as a volume at runtime.
#   - HF cache is NOT baked in — mount it as a named volume for persistence.
#
# Build / run:
#   docker compose build
#   docker compose up
#
# Requires NVIDIA Container Toolkit on the host (Linux native, or Windows
# with WSL2 backend in Docker Desktop) for GPU passthrough.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Python 3.12 from deadsnakes PPA + curl for the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /app

# Install Python deps first — separate layer so requirements changes don't
# bust the (very large) torch download every code edit.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Application code
COPY serve.py .
COPY configs/ ./configs/

# Bake the trained LoRAs into the image so the container is self-contained.
# Recipients can still override by mounting a different ./loras as a volume
# (volume mount takes precedence over baked-in files).
COPY loras/ ./loras/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "serve.py"]
