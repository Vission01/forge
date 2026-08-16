# syntax=docker/dockerfile:1
# FORGE — vLLM gateway & model manager.
#   build:  docker build -t forge .
#   run:    docker run -d -p 9090:9090 -v forge-data:/data forge
#
# Base: official vLLM image (CUDA 12.x, Python 3.12, vLLM pre-installed).
# This provides the 'vllm' binary for serving + all CUDA/Blackwell kernels.
FROM vllm/vllm-openai:latest

USER root

# curl is needed by the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install gateway deps (vLLM is already in the base image).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source (src/ + the static dashboard inside the package).
COPY src ./src

# Non-root user; /data holds registry + state + model weights.
RUN useradd --create-home --uid 10001 forge \
    && mkdir -p /data \
    && chown -R forge:forge /data /app
USER forge

ENV PYTHONUNBUFFERED=1 \
    FORGE_HOST=0.0.0.0 \
    FORGE_PORT=9090 \
    FORGE_INTERNAL_PORT=18080 \
    FORGE_DATA_DIR=/data

VOLUME ["/data"]
EXPOSE 9090

# /health never blocks on model state, so it is a safe liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9090/health || exit 1

# The ASGI entrypoint.
ENTRYPOINT ["python3", "-m", "uvicorn", "forge.app:app"]
CMD ["--app-dir", "src", "--host", "0.0.0.0", "--port", "9090"]
