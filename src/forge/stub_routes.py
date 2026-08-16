"""/v1/* not-implemented stub routes (§8.2.1).

These endpoints have NO vLLM backing. They MUST exist and return exactly the
501 not_implemented_error body — nothing is proxied, and there is deliberately
no partial implementation in this module.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from forge.errors import not_implemented_error

router = APIRouter()


def _stub_501() -> JSONResponse:
    return JSONResponse(status_code=501, content=not_implemented_error())


@router.post("/v1/moderations")
async def stub_moderations(request: Request):  # noqa: ARG001
    return _stub_501()


@router.post("/v1/images/generations")
async def stub_images_generations(request: Request):  # noqa: ARG001
    return _stub_501()


@router.post("/v1/images/edits")
async def stub_images_edits(request: Request):  # noqa: ARG001
    return _stub_501()


@router.post("/v1/videos")
async def stub_videos(request: Request):  # noqa: ARG001
    return _stub_501()


@router.post("/v1/files")
async def stub_files(request: Request):  # noqa: ARG001
    return _stub_501()


@router.post("/v1/fine_tuning/jobs")
async def stub_fine_tuning_jobs(request: Request):  # noqa: ARG001
    return _stub_501()
