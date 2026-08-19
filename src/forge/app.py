"""FastAPI application assembly (§8: /v1/* + /admin/v1/* + static dashboard).

Auth (§8.3.3):
  - /admin/v1/* requires ``Authorization: Bearer <FORGE_API_KEY>`` when a key
    is configured; otherwise 401 {"detail": "Invalid API key."}
  - /v1/*: FORGE_API_KEY is NOT consulted (clients send their own upstream
    key, forwarded verbatim to vLLM).
"""
from __future__ import annotations

import logging
import secrets
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from forge import admin_routes, proxy_routes, stub_routes
from forge.config import ForgeConfig, load_config
from forge.lifecycle import LifecycleManager
from forge.registry import Registry
from forge.stats import RequestCounter
from forge.admin_routes import load_persisted_api_key

STATIC_DIR = Path(__file__).parent / "static"


def create_app(cfg: ForgeConfig | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="FORGE", version="0.1.0", docs_url=None, redoc_url=None)
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    log = logging.getLogger("forge.app")

    # CORS: the dashboard is served same-origin but cross-PORT access is a
    # documented config pattern (FORGE_PORT != browser-visible port).
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["*"])

    registry = Registry(cfg.data_dir)
    counter = RequestCounter()
    lifecycle = LifecycleManager(cfg, registry, counter)

    # Load persisted API key (if no env-var key is set)
    if not cfg.api_key:
        persisted = load_persisted_api_key(cfg)
        if persisted:
            cfg.api_key = persisted

    app.state.config = cfg
    app.state.registry = registry
    app.state.lifecycle = lifecycle
    app.state.counter = counter

    @app.on_event("startup")
    async def _startup():
        await lifecycle.start_background()
        log.info("FORGE listening on %s:%s (vLLM internal port %s), data_dir=%s",
                 cfg.host, cfg.port, cfg.internal_port, cfg.data_dir)
        # Log GPU info on startup
        from forge.stats import query_gpu_info
        gpu = await query_gpu_info()
        if gpu:
            free = gpu['vram_total_mb'] - gpu['vram_used_mb']
            log.info("GPU: %s — VRAM %s/%s MB (%s MB free), %s°C",
                     gpu.get('gpu_name', '?'),
                     gpu.get('vram_used_mb', '?'),
                     gpu.get('vram_total_mb', '?'),
                     free,
                     gpu.get('gpu_temp_c', '?'))
        else:
            log.warning("GPU: nvidia-smi unavailable — cannot detect GPU")

    @app.on_event("shutdown")
    async def _shutdown():
        await lifecycle.stop_background()
        from forge.proxy import close_shared_client
        await close_shared_client()

    @app.middleware("http")
    async def admin_auth(request, call_next):
        if cfg.api_key and request.url.path.startswith("/admin/"):
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied, f"Bearer {cfg.api_key}"):
                return JSONResponse(status_code=401,
                                    content={"detail": "Invalid API key."})
        return await call_next(request)

    # Root health — always available, never depends on model state
    @app.get("/health")
    async def health():
        return {"status": "ok", "state": lifecycle.state,
                "active_alias": lifecycle.alias}

    # Routers (explicit routes; take precedence over the root static mount)
    app.include_router(stub_routes.router)
    app.include_router(proxy_routes.router)
    app.include_router(admin_routes.router)

    # Static dashboard at / — single static dir, minimal, no build step
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True),
                  name="static")
        app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True),
                  name="ui")
        log.info("dashboard served at / and /ui")

    return app


app = create_app()
