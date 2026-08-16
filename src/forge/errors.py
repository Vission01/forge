"""OpenAI-shaped error body helpers (§8.4).

Every response under /v1/* that is an error MUST use the exact shape below.
The sole sanctioned exception: not_implemented_error(), whose "type" field
uses the FORGE-specific value "not_implemented_error".
"""
from __future__ import annotations
from typing import Optional


def openai_error(message: str, type_: str,
                 param: Optional[str] = None,
                 code: Optional[str] = None) -> dict:
    """Canonical OpenAI error body. type_ in {invalid_request_error, not_found_error, server_error}."""
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


_NOT_IMPL_MSG = (
    "This endpoint is not implemented. FORGE proxies vLLM, which has no backing "
    "capability for this (no image/video generation, moderation classifier, "
    "file storage, or fine-tuning backend)."
)


def not_implemented_error() -> dict:
    """Exact §8.2.1 body. The only FORGE-specific value allowed under /v1/*."""
    return {
        "error": {
            "message": _NOT_IMPL_MSG,
            "type": "not_implemented_error",
            "param": None,
            "code": "unsupported_capability",
        }
    }
