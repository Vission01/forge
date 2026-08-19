"""Management endpoints under /admin/v1/* (§8.3).

FastAPI default {"detail": ...} error shape is used consistently across this
namespace (MUST be internally consistent per §8.4).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from forge.catalog import get_catalog, get_catalog_entry
from forge.downloader import ProgressChannel, download_manifest
from forge.registry import Manifest, Registry
from forge.stats import RequestCounter, query_vram_mb, query_gpu_info

router = APIRouter()


def _st(request: Request):
    return (
        request.app.state.config,
        request.app.state.registry,
        request.app.state.lifecycle,
        request.app.state.counter,
    )


# ---- API key persistence ---- #

def _api_key_path(cfg) -> str:
    return os.path.join(cfg.data_dir, "state", "api_key.json")


def load_persisted_api_key(cfg) -> Optional[str]:
    """Load API key from disk on startup. Returns None if not found."""
    p = _api_key_path(cfg)
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("api_key")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _persist_api_key(cfg, key: Optional[str]) -> None:
    """Write API key to disk for persistence across restarts."""
    p = _api_key_path(cfg)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"api_key": key}, f)
        os.replace(tmp, p)
    except OSError:
        pass  # best-effort; env var still works as fallback


# =============== registry ===============

class _PullBody:
    pass


@router.get("/admin/v1/registry")
async def get_registry(request: Request):
    _, reg, _, _ = _st(request)
    return JSONResponse(status_code=200, content=reg.list_with_downloaded())


@router.post("/admin/v1/registry", status_code=201)
async def create_registry(request: Request):
    _, reg, lc, _ = _st(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    try:
        m = Manifest(**body)
    except Exception as exc:
        msg = exc.errors()[0]["msg"] if hasattr(exc, "errors") else str(exc)
        raise HTTPException(status_code=400, detail=msg)

    if reg.exists(m.alias):
        raise HTTPException(status_code=409, detail=f"alias {m.alias!r} already exists")
    try:
        reg.save(m)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # If the alias currently owns the active slot, an unload is required before
    # the manifest can be deleted; for CREATE, validate cross-uniqueness only.
    return JSONResponse(status_code=201, content=m.model_dump())


@router.delete("/admin/v1/registry/{alias}", status_code=204)
async def delete_registry(alias: str, request: Request):
    _, reg, lc, _ = _st(request)
    if not reg.exists(alias):
        raise HTTPException(status_code=404, detail=f"unknown alias {alias!r}")
    # If the alias is the active model, unload first (§8.3 DELETE semantics)
    async with lc._state_lock:
        active_alias = lc.alias
        active_state = lc.state
    if active_alias == alias and active_state in ("READY", "LOADING", "DOWNLOADING"):
        await lc.unload_forced()
    reg.delete(alias)
    # do NOT delete weights — they may be shared by other manifests (§8.3)
    return None


@router.patch("/admin/v1/registry/{alias}")
async def patch_registry(alias: str, request: Request):
    """Update mutable fields of an existing manifest.

    Supported fields: idle_timeout_seconds, engine_args, served_model_name,
    source (repo, file).
    """
    _, reg, lc, _ = _st(request)
    manifest = reg.get(alias)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"unknown alias {alias!r}")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    changed = False
    if "idle_timeout_seconds" in body:
        val = body["idle_timeout_seconds"]
        if not isinstance(val, int) or val < 30:
            raise HTTPException(status_code=400, detail="idle_timeout_seconds must be an integer >= 30")
        manifest.idle_timeout_seconds = val
        changed = True
    if "engine_args" in body:
        val = body["engine_args"]
        if not isinstance(val, dict):
            raise HTTPException(status_code=400, detail="engine_args must be an object")
        # Validate all values are strings
        for k, v in val.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise HTTPException(status_code=400,
                    detail=f"engine_args keys and values must be strings")
        manifest.engine_args = val
        changed = True
    if "served_model_name" in body:
        val = body["served_model_name"]
        if not isinstance(val, str) or not val.strip():
            raise HTTPException(status_code=400, detail="served_model_name must be a non-empty string")
        manifest.served_model_name = val
        changed = True
    if "source" in body:
        val = body["source"]
        if not isinstance(val, dict) or "repo" not in val:
            raise HTTPException(status_code=400, detail="source must be an object with 'repo'")
        from forge.registry import Source
        try:
            manifest.source = Source(**val)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid source: {exc}")
        changed = True
    if not changed:
        raise HTTPException(status_code=400,
            detail="no supported fields to update (supported: idle_timeout_seconds, engine_args, served_model_name, source)")
    try:
        reg.save(manifest)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(status_code=200, content=manifest.model_dump())


@router.get("/admin/v1/status")
async def get_status(request: Request):
    _, _, lc, _ = _st(request)
    return JSONResponse(status_code=200, content=lc.status())


@router.get("/admin/v1/version")
async def get_version(request: Request):
    """Return current and latest vLLM version (cached 1 hour)."""
    import httpx as _httpx
    import time as _time
    global _version_cache_time, _version_cache_data
    now = _time.monotonic()
    if now - _version_cache_time < 3600 and _version_cache_data:
        return JSONResponse(status_code=200, content=_version_cache_data)
    current = None
    try:
        import vllm
        current = vllm.__version__
    except Exception:
        pass
    latest = None
    try:
        async with _httpx.AsyncClient(timeout=5) as c:
            r = await c.get("https://pypi.org/pypi/vllm/json")
            if r.status_code == 200:
                latest = r.json().get("info", {}).get("version")
    except Exception:
        pass
    update_available = bool(current and latest and current != latest)
    data = {
        "vllm_current": current,
        "vllm_latest": latest,
        "update_available": update_available,
    }
    _version_cache_time = now
    _version_cache_data = data
    return JSONResponse(status_code=200, content=data)


_version_cache_time: float = 0.0
_version_cache_data: dict = {}


@router.post("/admin/v1/api-key")
async def set_api_key(request: Request):
    """Generate or set an API key at runtime. Requires master password.
    Body: {"password": "..."} to auto-generate, or {"password": "...", "key": "..."} to set a specific key.
    """
    cfg, _, _, _ = _st(request)
    if not cfg.master_password:
        raise HTTPException(status_code=403,
            detail="API key management is disabled. Set FORGE_MASTER_PASSWORD in the environment to enable it.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    supplied_pw = body.get("password", "")
    import secrets as _sec
    if not supplied_pw or not _sec.compare_digest(str(supplied_pw), cfg.master_password):
        raise HTTPException(status_code=403, detail="Invalid master password.")
    if body.get("key"):
        new_key = str(body["key"])
    else:
        new_key = "forge-" + _sec.token_urlsafe(32)
    cfg.api_key = new_key
    _persist_api_key(cfg, new_key)
    return JSONResponse(status_code=200, content={"api_key": new_key})


@router.delete("/admin/v1/api-key")
async def clear_api_key(request: Request):
    """Remove the API key (open admin access). Requires master password in body."""
    cfg, _, _, _ = _st(request)
    if not cfg.master_password:
        raise HTTPException(status_code=403,
            detail="API key management is disabled. Set FORGE_MASTER_PASSWORD in the environment to enable it.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    supplied_pw = body.get("password", "")
    import secrets as _sec
    if not supplied_pw or not _sec.compare_digest(str(supplied_pw), cfg.master_password):
        raise HTTPException(status_code=403, detail="Invalid master password.")
    cfg.api_key = None
    _persist_api_key(cfg, None)
    return JSONResponse(status_code=200, content={"api_key": None})


@router.post("/admin/v1/unload")
async def post_unload(request: Request):
    _, _, lc, _ = _st(request)
    code, body = await lc.unload_forced()
    return JSONResponse(status_code=code, content=body)


@router.post("/admin/v1/activate")
async def post_activate(request: Request):
    from forge.lifecycle import ST_IDLE, ST_DOWNLOADING, ST_LOADING, ST_UNLOADING
    _, reg, lc, _ = _st(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    alias = body.get("alias") if isinstance(body, dict) else None
    if not isinstance(alias, str) or not alias:
        raise HTTPException(status_code=400, detail="'alias' (string) is required")

    # Immediate-feedback contract (§6.5) — checked BEFORE the shared path:
    if not reg.exists(alias):
        import logging as _lg
        _lg.getLogger("forge.admin").info("activate rejected-404 alias=%s", alias)
        raise HTTPException(status_code=404, detail=f"unknown alias {alias!r}")
    async with lc._state_lock:
        if lc.state in (ST_DOWNLOADING, ST_LOADING, ST_UNLOADING) and lc.alias != alias:
            import logging as _lg
            _lg.getLogger("forge.admin").info(
                "activate rejected-409 alias=%s busy=%s", alias, lc.alias)
            raise HTTPException(
                status_code=409,
                detail=(f"busy: state={lc.state}, active_alias={lc.alias!r}"))
        if lc.state == "READY" and lc.alias == alias:
            import logging as _lg
            _lg.getLogger("forge.admin").info("activate idempotent-noop alias=%s", alias)
            return JSONResponse(status_code=200,
                                content={"state": "READY", "alias": alias})

    import logging as _lg
    _lg.getLogger("forge.admin").info("activate accepted alias=%s", alias)
    # Non-blocking: launch loading in background, return 202 immediately.
    # The dashboard polls /admin/v1/status to track progress.
    import asyncio
    async def _bg_load():
        try:
            await lc.ensure_ready(alias)
        except Exception as exc:
            _lg.getLogger("forge.admin").error("activate load failed alias=%s: %s", alias, exc)
    asyncio.create_task(_bg_load())
    return JSONResponse(status_code=202, content={"state": lc.state, "alias": alias})


@router.get("/admin/v1/stats")
async def get_stats(request: Request):
    cfg, reg, lc, cnt = _st(request)
    total, since_load = await cnt.snapshot()
    gpu = await query_gpu_info()
    body = {
        "state": lc.state,
        "active_alias": lc.alias if lc.state == "READY" else None,
        "served_model_name": lc.served_model_name if lc.state == "READY" else None,
        "loaded_at": lc.loaded_at if lc.state == "READY" else None,
        "uptime_seconds": _uptime(lc) if lc.state == "READY" else None,
        "requests_served_total": total,
        "requests_served_since_load": since_load if lc.state == "READY" else 0,
        "vram_used_mb": gpu.get("vram_used_mb"),
        "vram_total_mb": gpu.get("vram_total_mb"),
        "gpu_utilization_pct": gpu.get("gpu_utilization_pct"),
        "gpu_temp_c": gpu.get("gpu_temp_c"),
        "gpu_name": gpu.get("gpu_name"),
    }
    return JSONResponse(status_code=200, content=body)


def _uptime(lc) -> Optional[int]:
    if not lc.loaded_at:
        return None
    from datetime import datetime, timezone
    try:
        t = datetime.strptime(lc.loaded_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int((datetime.now(timezone.utc) - t).total_seconds())
    except Exception:
        return None


# =============== pull ===============

@router.post("/admin/v1/pull")
async def post_pull(request: Request):
    cfg, reg, lc, _ = _st(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    alias = body.get("alias") if isinstance(body, dict) else None
    if not isinstance(alias, str) or not alias:
        raise HTTPException(status_code=400, detail="'alias' (string) is required")
    manifest = reg.get(alias)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"unknown alias {alias!r}")

    import logging as _lg
    _lg.getLogger("forge.admin").info("pull requested alias=%s", alias)

    channel = ProgressChannel()
    channel.bind_loop(asyncio.get_running_loop())
    task = None
    loop = asyncio.get_running_loop()

    async def start_download() -> None:
        try:
            await download_manifest(cfg, reg, manifest, channel)
        except Exception as exc:  # MUST NOT crash forged (§13)
            channel.push({"status": "error", "message": str(exc)})
        finally:
            channel.push(None)  # sentinel: end of stream

    async def gen():
        nonlocal task
        task = asyncio.create_task(start_download())
        while True:
            event = await channel.get()
            if event is None:
                break
            yield json.dumps(event) + "\n"
        await task

    return StreamingResponse(
        gen(),
        status_code=200,
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============== models/resolve ===============

_HF_REPO_RE = re.compile(r"^https?://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(/.*)?$")
_HF_FILE_RE = re.compile(
    r"^https?://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/"
    r"(?:blob|resolve)/([A-Za-z0-9_.-]+)/(.+)$"
)


def _sanitize_alias(s: str) -> str:
    out = s.lower()
    out = re.sub(r"[^a-z0-9._-]", "-", out)
    out = re.sub(r"-{2,}", "-", out)
    return out.strip("-")


def _resolve_input(url: str) -> Optional[dict]:
    url = url.strip()
    if not url:
        return None
    m = _HF_FILE_RE.match(url)
    if m:
        org, name, _branch, path = m.groups()
        file = path
        repo = f"{org}/{name}"
    else:
        m = _HF_REPO_RE.match(url)
        if m:
            org, name, _rest = m.groups()
            repo = f"{org}/{name}"
            file = None
        else:
            if "/" in url and not url.startswith("http"):
                parts = url.split("/", 1)
                if len(parts) == 2 and all(p.strip() for p in parts):
                    repo = url
                    file = None
                else:
                    return None
            else:
                return None

    full = f"{repo}/{file}" if file else repo
    suggested_alias = _sanitize_alias(full) or "model"
    suggested_served = _sanitize_alias(repo.split("/", 1)[1]) or "model"
    return {
        "repo": repo,
        "file": file,
        "suggested_alias": suggested_alias,
        "suggested_served_model_name": suggested_served,
    }


@router.post("/admin/v1/models/resolve")
async def post_resolve(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    url = body.get("url") if isinstance(body, dict) else None
    if not isinstance(url, str):
        raise HTTPException(status_code=400,
                            detail="Could not parse a Hugging Face repo from the given URL/string")
    result = _resolve_input(url)
    if result is None:
        raise HTTPException(status_code=400,
                            detail="Could not parse a Hugging Face repo from the given URL/string")
    return JSONResponse(status_code=200, content=result)


# =============== catalog ===============

@router.get("/admin/v1/catalog")
async def get_catalog_list(request: Request):
    """Return the built-in model catalog (curated for RTX 5090 / 32 GB)."""
    _, reg, _, _ = _st(request)
    catalog = get_catalog()
    # Annotate each entry with whether it's already registered
    registered_repos = {m.source.repo for m in reg.all().values()}
    for entry in catalog:
        entry["registered"] = entry["repo"] in registered_repos
    return JSONResponse(status_code=200, content=catalog)


# =============== storage ===============

@router.get("/admin/v1/storage")
async def get_storage(request: Request):
    """Return total weight storage used + per-model breakdown."""
    _, reg, lc, _ = _st(request)
    models = []
    for m in reg.all().values():
        info = reg.weights_info(m)
        is_active = (lc.alias == m.alias and lc.state in ("READY", "LOADING", "DOWNLOADING"))
        models.append({
            "alias": m.alias,
            "served_model_name": m.served_model_name,
            "repo": m.source.repo,
            "downloaded": info["present"],
            "size_mb": info["size_mb"],
            "active": is_active,
        })
    models.sort(key=lambda x: -(x["size_mb"] or 0))
    total_mb = reg.total_weights_size_mb()
    return JSONResponse(status_code=200, content={
        "total_size_mb": total_mb,
        "total_size_gb": round(total_mb / 1024, 1) if total_mb else 0,
        "models": models,
    })


@router.delete("/admin/v1/storage/{alias}")
async def delete_weights(alias: str, request: Request):
    """Delete downloaded weights for a model. Unloads first if active."""
    _, reg, lc, _ = _st(request)
    manifest = reg.get(alias)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"unknown alias {alias!r}")
    # If active, unload first
    async with lc._state_lock:
        is_active = (lc.alias == alias and lc.state in ("READY", "LOADING", "DOWNLOADING"))
    if is_active:
        await lc.unload_forced()
    deleted = reg.delete_weights(manifest)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no weights found for alias {alias!r}")
    return JSONResponse(status_code=200, content={
        "deleted": True,
        "alias": alias,
    })
