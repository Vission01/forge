"""Route wiring for §8.2 proxied endpoints (chat, completions, embeddings,
responses + sub-routes, audio/transcriptions) and the synthesized
GET /v1/models special case.
"""
from __future__ import annotations

import re
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from forge.errors import openai_error
from forge.proxy import proxy_get_passthrough, proxy_post

router = APIRouter()

_MODELS_DETAIL_RE = re.compile(r"^/v1/models/([^/]+)$")


def _v1_models_list(registry) -> JSONResponse:
    from fastapi.responses import JSONResponse as JR
    data = []
    for d in registry.list_with_downloaded():
        data.append({
            "id": d["served_model_name"],
            "object": "model",
            "created": registry.mtime(d["alias"]),
            "owned_by": "forge",
        })
    return JR(status_code=200, content={"object": "list", "data": data})


@router.get("/v1/models")
async def list_models(request: Request):
    return _v1_models_list(request.app.state.registry)


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str, request: Request):
    registry = request.app.state.registry
    for d in registry.list_with_downloaded():
        if d["served_model_name"] == model_id:
            return JSONResponse(status_code=200, content={
                "id": d["served_model_name"],
                "object": "model",
                "created": registry.mtime(d["alias"]),
                "owned_by": "forge",
            })
    return JSONResponse(status_code=404, content=openai_error(
        f"The model '{model_id}' does not exist or you don't have access to it.",
        "not_found_error", param="model", code="model_not_found",
    ))


# ---- POST inference endpoints (proxied) ---- #

@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    cfg = request.app.state.config
    reg = request.app.state.registry
    lc = request.app.state.lifecycle
    cnt = request.app.state.counter
    return await proxy_post(request, lc, reg, cnt, "/v1/chat/completions")


@router.post("/v1/completions")
async def completions(request: Request):
    reg = request.app.state.registry
    lc = request.app.state.lifecycle
    cnt = request.app.state.counter
    return await proxy_post(request, lc, reg, cnt, "/v1/completions")


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    reg = request.app.state.registry
    lc = request.app.state.lifecycle
    cnt = request.app.state.counter
    return await proxy_post(request, lc, reg, cnt, "/v1/embeddings")


@router.post("/v1/responses")
async def responses_create(request: Request):
    reg = request.app.state.registry
    lc = request.app.state.lifecycle
    cnt = request.app.state.counter
    return await proxy_post(request, lc, reg, cnt, "/v1/responses")


# ---- /v1/responses/{id} and cancel ---- #
_RESPONSES_ID_RE = re.compile(r"^/v1/responses/([^/]+)(/cancel)?$")


@router.get("/v1/responses/{response_id}")
async def responses_get(response_id: str, request: Request):
    reg = request.app.state.registry
    lc = request.app.state.lifecycle
    cnt = request.app.state.counter
    return await proxy_get_passthrough(request, lc, cnt, f"/v1/responses/{response_id}")


@router.post("/v1/responses/{response_id}")
async def responses_update(response_id: str, request: Request):
    reg = request.app.state.registry
    lc = request.app.state.lifecycle
    cnt = request.app.state.counter
    return await proxy_post(request, lc, reg, cnt, f"/v1/responses/{response_id}")


@router.post("/v1/responses/{response_id}/cancel")
async def responses_cancel(response_id: str, request: Request):
    reg = request.app.state.registry
    lc = request.app.state.lifecycle
    cnt = request.app.state.counter
    return await proxy_post(request, lc, reg, cnt, f"/v1/responses/{response_id}/cancel", passthrough=True)


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    reg = request.app.state.registry
    lc = request.app.state.lifecycle
    cnt = request.app.state.counter
    return await proxy_post(request, lc, reg, cnt, "/v1/audio/transcriptions", passthrough=True)
