"""/v1/* reverse-proxy logic (§8.2).

Strategy: resolve model -> guarantee READY (shared lifecycle path) -> forward
byte-for-byte (headers minus Host, body unchanged) -> stream response as-is.

GET /v1/models and /v1/models/{id} are synthesized from the registry (special
case, §8.2) — they are NOT proxied to vLLM.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from forge.errors import openai_error
from forge.lifecycle import LifecycleManager, LoadFailed
from forge.registry import Registry
from forge.stats import RequestCounter

log = logging.getLogger("forge.proxy")

router = APIRouter()

MODELS_PATH_RE = re.compile(r"^/v1/models/([^/]+)$")

# Module-level shared httpx client — created lazily on first use,
# closed on app shutdown.  Avoids creating a new TCP connection per request.
_shared_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """Return the module-level shared httpx client, creating it if needed."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(1800.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=120,
            ),
        )
    return _shared_client


async def close_shared_client() -> None:
    """Shut down the shared httpx client (call from app shutdown)."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


def _ctx(request: Request) -> tuple:
    return (
        request.app.state.config,
        request.app.state.registry,
        request.app.state.lifecycle,
        request.app.state.counter,
    )


def _hdrs(request: Request) -> dict:
    h = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    if "transfer-encoding" in h:
        del h["transfer-encoding"]
    if "content-length" in h:
        del h["content-length"]
    return h


async def extract_model(request: Request) -> Optional[str]:
    try:
        body = await request.body()
        if not body:
            return None
        data = json.loads(body.decode("utf-8"))
        if isinstance(data, dict):
            m = data.get("model")
            if isinstance(m, str) and m:
                return m
    except Exception:
        return None
    return None


async def _resolve_for_post(request: Request, registry: Registry):
    """Resolve the model name in a POST body to a manifest.

    Returns (manifest, openai_404_response|None). MUST NOT forward unknown
    models to vLLM (§8.2).
    """
    model = await extract_model(request)
    if model is None:
        return None, JSONResponse(
            status_code=400,
            content=openai_error(
                "Missing 'model' field in request body.", "invalid_request_error",
                param="model", code="missing_model",
            ),
        )
    manifest = registry.find_by_served(model)
    if manifest is None:
        return None, JSONResponse(
            status_code=404,
            content=openai_error(
                f"The model '{model}' does not exist or you don't have access to it.",
                "not_found_error", param="model", code="model_not_found",
            ),
        )
    return manifest, None


async def proxy_post(
    request: Request,
    lifecycle: LifecycleManager,
    registry: Registry,
    counter: RequestCounter,
    path: str,
    *,
    passthrough: bool = False,
) -> Response:
    """Forward a POST to the upstream vLLM.

    If passthrough=True, skip model resolution (for routes like
    /v1/audio/transcriptions and /v1/responses/{id}/cancel where the body
    has no JSON 'model' field). The currently loaded model is used as-is.
    """
    if passthrough:
        alias = lifecycle.alias
        if alias is None or lifecycle.state != "READY":
            return JSONResponse(
                status_code=503,
                content=openai_error(
                    "No model is currently loaded. Activate one first.",
                    "server_error", code="no_model_loaded",
                ),
            )
        manifest = registry.get(alias)
        if manifest is None:
            return JSONResponse(
                status_code=503,
                content=openai_error(
                    "Active model manifest not found.",
                    "server_error", code="model_unavailable",
                ),
            )
    else:
        manifest, err = await _resolve_for_post(request, registry)
        if err is not None:
            return err
        alias = manifest.alias

    lifecycle.note_request_start(alias)
    _stream_owns_cleanup = False
    try:
        try:
            await lifecycle.ensure_ready(alias)
        except LoadFailed as exc:
            log.warning("proxy %s: model load failed for %s: %s", path, alias, exc)
            return JSONResponse(
                status_code=503,
                content=openai_error(
                    f"The model is temporarily overloaded or failed to load: {exc}",
                    "server_error", param="model", code="model_unavailable",
                ),
            )

        body = await request.body()
        base = lifecycle.proxy_base
        assert base is not None
        url = base + path
        target_url = f"{url}"
        client = _get_client()
        try:
            req = client.build_request(
                "POST", target_url, content=body, headers=_hdrs(request)
            )
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            log.error("proxy %s: upstream error: %s", path, exc)
            return JSONResponse(
                status_code=502,
                content=openai_error(
                    "Failed to proxy request to the inference backend.",
                    "server_error", code="proxy_error",
                ),
            )

        await counter.inc_total()
        await counter.inc_since_load()
        lifecycle.touch(alias)

        ct = upstream.headers.get("content-type", "")
        if "text/event-stream" in ct:
            async def gen():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                except Exception as exc:
                    log.error("proxy %s: stream read error for model=%s: %s",
                              path, manifest.served_model_name, exc)
                finally:
                    await upstream.aclose()
                    lifecycle.note_request_end()
                    log.debug("proxy %s: %s %s model=%s -> %s (stream, client closed)",
                              "POST", path, alias, manifest.served_model_name,
                              upstream.status_code)
            _stream_owns_cleanup = True
            log.debug("proxy %s: %s %s model=%s -> %s (stream)",
                      "POST", path, alias, manifest.served_model_name,
                      upstream.status_code)
            return StreamingResponse(
                gen(),
                status_code=upstream.status_code,
                media_type="text/event-stream",
                headers={
                    k: v for k, v in upstream.headers.items()
                    if k.lower() in ("cache-control", "x-request-id")
                },
            )

        chunks = []
        async for c in upstream.aiter_bytes():
            chunks.append(c)
        await upstream.aclose()
        payload = b"".join(chunks)
        log.debug("proxy %s: %s %s model=%s -> %s", "POST", path, alias,
                  manifest.served_model_name, upstream.status_code)
        return Response(
            content=payload,
            status_code=upstream.status_code,
            media_type=ct or "application/json",
            headers={
                k: v for k, v in upstream.headers.items()
                if k.lower() in ("x-request-id",)
            },
        )
    finally:
        if not _stream_owns_cleanup:
            lifecycle.note_request_end()


async def proxy_get_passthrough(
    request: Request,
    lifecycle: LifecycleManager,
    counter: RequestCounter,
    path: str,
) -> Response:
    """GET routes that pass through untouched (currently unneeded — /v1/models
    is synthesized). Kept for spec completeness."""
    base = lifecycle.proxy_base
    if base is None:
        return JSONResponse(
            status_code=503,
            content=openai_error(
                "No model is currently loaded. Activate one first.",
                "server_error", code="no_model_loaded",
            ),
        )
    url = base + path
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.get(url, headers=_hdrs(request))
    await counter.inc_total()
    return Response(
        content=r.content, status_code=r.status_code, media_type="application/json"
    )
