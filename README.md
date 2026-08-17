# FORGE

A FastAPI gateway for vLLM — manage and serve LLMs with an OpenAI-compatible API and web dashboard.

![Dashboard](https://img.shields.io/badge/dashboard-port%209090-blue)
![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20compatible-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

- **OpenAI-compatible API** — drop-in replacement at `/v1/chat/completions`, `/v1/models`, etc.
- **Web dashboard** — manage models, monitor GPU/VRAM, activate/unload, full CRUD on manifests
- **Auto-download** — register a HuggingFace model, Forge pulls the weights on first use
- **Idle unload** — automatically frees GPU memory after configurable idle timeout
- **One model at a time** — clean state machine (IDLE → DOWNLOADING → LOADING → READY)
- **Tool calling** — supports `tool_choice: "auto"` for agent workflows
- **GPU monitoring** — real-time VRAM usage, GPU utilization %, temperature
- **VRAM pre-flight check** — fails fast with a clear message if a model won't fit, instead of waiting 20 minutes for OOM
- **Version tracking** — dashboard shows current vLLM version with update-available badge
- **API key management** — generate/clear keys from the dashboard, secured by a master password

## Quick Deploy

```bash
mkdir forge && cd forge
curl -O https://raw.githubusercontent.com/Vission01/forge/main/deploy/docker-compose.yml

# (Recommended) Set a master password for API key management
echo 'FORGE_MASTER_PASSWORD=choose-a-strong-password' > .env

docker compose up -d
```

**Requirements:** Docker, NVIDIA GPU, [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Then open `http://localhost:9090` for the dashboard.

## Security

- Admin endpoints (`/admin/v1/*`) are open by default. Generate an API key from the dashboard to lock them down.
- API key generation requires a **master password** (`FORGE_MASTER_PASSWORD` env var) — without it, the generate/clear buttons are disabled.
- Inference endpoints (`/v1/*`) are always open and pass through client auth to vLLM.

## Full Documentation

See [deploy/DEPLOY.md](deploy/DEPLOY.md) for complete setup instructions, model registration, connecting to Open WebUI / n8n, VRAM sizing, security configuration, and API reference.

## Building from Source

```bash
git clone https://github.com/Vission01/forge.git
cd forge
docker compose up -d --build
```
