# FORGE — Deployment Guide

FORGE is a FastAPI gateway that sits in front of vLLM, providing an OpenAI-compatible API with automatic model downloading, loading, idle-unload, and a web dashboard.

## Prerequisites

- **Docker** with Compose v2+
- **NVIDIA GPU** with recent drivers (535+)
- **nvidia-container-toolkit** installed and configured
- Minimum **24 GB VRAM** for 27B-class models (32 GB recommended)

### Install nvidia-container-toolkit (if not already)

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify with: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`

---

## Quick Start

```bash
# 1. Create a directory and download the compose file
mkdir forge && cd forge
curl -O https://raw.githubusercontent.com/Vission01/forge/main/deploy/docker-compose.yml

# 2. Start (pulls the pre-built image automatically)
docker compose up -d

# 3. Verify
curl http://localhost:9090/health
# → {"status":"ok","state":"IDLE","active_alias":null}
```

The dashboard is available at `http://<server-ip>:9090/`

> **Building from source instead?** Clone the full repo and use the root `docker-compose.yml`:
> ```bash
> git clone https://github.com/Vission01/forge.git
> cd forge
> docker compose up -d --build
> ```

---

## Register a Model

Create a model manifest via the admin API:

```bash
curl -X POST http://localhost:9090/admin/v1/registry \
  -H "Content-Type: application/json" \
  -d '{
    "alias": "qwen38-27b-nvfp4",
    "source": {"repo": "unsloth/Qwen3.8-27B-NVFP4"},
    "served_model_name": "qwen3.8-27b",
    "engine_args": {
      "kv-cache-dtype": "fp8_e4m3",
      "max-model-len": "65536",
      "gpu-memory-utilization": "0.93",
      "enforce-eager": "true",
      "enable-auto-tool-choice": "true",
      "tool-call-parser": "hermes"
    },
    "idle_timeout_seconds": 600
  }'
```

### Engine Args Reference

| Arg | Description |
|-----|-------------|
| `kv-cache-dtype` | `fp8_e4m3` halves KV cache memory — fits 2x more context |
| `max-model-len` | Max context window in tokens (model supports up to 262144) |
| `gpu-memory-utilization` | Fraction of VRAM to use (0.93 = 93%) |
| `enforce-eager` | Skip CUDA graph capture — **required on tight VRAM** (saves ~800MB) |
| `enable-auto-tool-choice` | Enable tool calling (needed for Open WebUI, n8n agents) |
| `tool-call-parser` | `hermes` for Qwen3.x models |
| `chat-template` | Path to a Jinja2 template inside the container (see Thinking section) |

### Boolean Flags

Engine args with values `true`, `yes`, `1`, or empty string are treated as boolean flags (emitted as `--flag` with no value). All others are emitted as `--key value`.

---

## Activate / Unload a Model

```bash
# Load model into GPU (downloads weights on first use)
curl -X POST http://localhost:9090/admin/v1/activate \
  -H "Content-Type: application/json" \
  -d '{"alias": "qwen38-27b-nvfp4"}'

# Unload model from GPU (frees VRAM)
curl -X POST http://localhost:9090/admin/v1/unload
```

You can also use the **Activate / Unload** toggle button in the dashboard.

---

## Test Inference

```bash
curl -X POST http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Connecting to Other Services

### Open WebUI

1. Connect Forge to Open WebUI's Docker network:
   ```bash
   docker network connect <open-webui-network> forge
   ```
2. In Open WebUI → Admin → Settings → Connections → **OpenAI API**:
   - **Base URL:** `http://forge:9090/v1`
   - **API Key:** any string (e.g. `sk-forge`) — Forge doesn't check it unless `FORGE_API_KEY` is set

### n8n

1. Connect Forge to n8n's Docker network:
   ```bash
   docker network connect <n8n-network> forge
   ```
2. In n8n, create an **OpenAI API** credential:
   - **Base URL:** `http://forge:9090/v1`
   - **API Key:** any non-empty string
   - **Model:** `qwen3.8-27b`

### Any OpenAI-compatible client

Use `http://<server-ip>:9090/v1` as the base URL. All standard `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/models` endpoints are supported.

---

## Persisting Docker Networks Across Restarts

To automatically join external networks on `docker compose up`, uncomment the `networks` sections in `docker-compose.yml` and update the network names to match your environment.

---

## Disabling Thinking (Qwen3.x)

Qwen3.8 is a reasoning model that outputs `<think>...</think>` blocks by default. To suppress thinking:

1. After the model weights download, create a modified chat template inside the container:
   ```bash
   docker exec forge bash -c '
     cp /data/models/unsloth__Qwen3.8-27B-NVFP4/chat_template.jinja \
        /data/models/unsloth__Qwen3.8-27B-NVFP4/chat_template_no_think.jinja
     sed -i "s/if enable_thinking is undefined or enable_thinking is true/if enable_thinking is defined and enable_thinking is true/" \
        /data/models/unsloth__Qwen3.8-27B-NVFP4/chat_template_no_think.jinja
     sed -i "s/if enable_thinking is defined and enable_thinking is false/if enable_thinking is not defined or enable_thinking is false/" \
        /data/models/unsloth__Qwen3.8-27B-NVFP4/chat_template_no_think.jinja
   '
   ```
2. Add to your manifest's `engine_args`:
   ```json
   "chat-template": "/data/models/unsloth__Qwen3.8-27B-NVFP4/chat_template_no_think.jinja"
   ```
3. Unload and re-activate the model.

---

## Security

- **Admin API** (`/admin/v1/*`) is **unauthenticated by default**. Set `FORGE_API_KEY` in a `.env` file to require a Bearer token.
- **CORS** is `allow_origins=["*"]` — restrict in production if Forge is exposed to the internet.
- The `/v1/*` inference endpoints do **not** check `FORGE_API_KEY` — they pass through whatever auth the client sends.

### Setting an Admin Key

```bash
echo 'FORGE_API_KEY=your-secret-key-here' > .env
docker compose restart
```

### Gated Models (HuggingFace)

For models that require HF authentication:

```bash
echo 'FORGE_HF_TOKEN=hf_your_token_here' >> .env
docker compose restart
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FORGE_HOST` | `0.0.0.0` | Bind address |
| `FORGE_PORT` | `9090` | API + dashboard port |
| `FORGE_INTERNAL_PORT` | `18080` | vLLM subprocess port (internal) |
| `FORGE_DATA_DIR` | `/data` | Registry, state, model weights |
| `FORGE_API_KEY` | *(unset)* | Admin API bearer token |
| `FORGE_HF_TOKEN` | *(unset)* | HuggingFace auth token |
| `FORGE_STARTUP_TIMEOUT_SECONDS` | `600` | Max wait for vLLM to start (increase for first load) |
| `FORGE_HEALTH_POLL_INTERVAL_SECONDS` | `2` | Health check polling interval |
| `FORGE_LOG_LEVEL` | `info` | Log level: debug, info, warning, error |

---

## VRAM Sizing Guide

| Model | Quantization | VRAM Required | Recommended `max-model-len` | Notes |
|-------|-------------|---------------|----------------------------|-------|
| Qwen3.8-27B | NVFP4 | ~29 GB | 4K–65K | Needs `enforce-eager` on 32GB cards |
| Qwen3-14B | NVFP4 | ~14 GB | 65K+ | Comfortable on 24GB+ |
| Qwen3-8B | NVFP4 | ~8 GB | 65K+ | Fits easily on 16GB+ |
| Any model | BF16 | ~2x params | Varies | Only for 48GB+ cards |

If you get OOM errors:
1. Add `"enforce-eager": "true"` (saves ~800MB from CUDA graph capture)
2. Lower `max-model-len`
3. Lower `gpu-memory-utilization`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe (always 200) |
| GET | `/v1/models` | List registered models |
| GET | `/v1/models/{id}` | Single model detail |
| POST | `/v1/chat/completions` | Chat inference (streaming + non-streaming) |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Embeddings |
| GET | `/admin/v1/status` | Current state, active model, VRAM |
| GET | `/admin/v1/stats` | Uptime, request counts, VRAM usage |
| GET | `/admin/v1/registry` | List all manifests |
| POST | `/admin/v1/registry` | Create a manifest |
| DELETE | `/admin/v1/registry/{alias}` | Delete a manifest |
| POST | `/admin/v1/activate` | Load a model into GPU |
| POST | `/admin/v1/unload` | Unload model from GPU |
| POST | `/admin/v1/pull` | Download weights (streamed ndjson progress) |
