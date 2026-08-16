"""Environment-variable configuration for forged.

All values are read at process start and validated via pydantic.
Malformed values (e.g. a non-integer port) fail fast per §12.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from pydantic import BaseModel, field_validator, ValidationError


class ForgeConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9090
    internal_port: int = 18080
    data_dir: str = "/data"
    api_key: Optional[str] = None
    startup_timeout_seconds: int = 600
    health_poll_interval_seconds: int = 2
    hf_token: Optional[str] = None
    log_level: str = "info"

    @field_validator("port", "internal_port", mode="before")
    @classmethod
    def _coerce_port(cls, v: object) -> int:
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError(f"Port must be an integer, got {v!r}")

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        v = v.lower()
        if v not in {"debug", "info", "warning", "error"}:
            raise ValueError(f"log_level must be debug|info|warning|error, got {v!r}")
        return v


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def load_config() -> ForgeConfig:
    """Read FORGE_* env vars and build a ForgeConfig, dying fast on bad values."""
    try:
        return ForgeConfig(
            host=_env("FORGE_HOST", "0.0.0.0"),
            port=_env("FORGE_PORT", "9090"),
            internal_port=_env("FORGE_INTERNAL_PORT", "18080"),
            data_dir=_env("FORGE_DATA_DIR", "/data"),
            api_key=_env("FORGE_API_KEY"),
            startup_timeout_seconds=_env("FORGE_STARTUP_TIMEOUT_SECONDS", "600"),
            health_poll_interval_seconds=_env("FORGE_HEALTH_POLL_INTERVAL_SECONDS", "2"),
            hf_token=_env("FORGE_HF_TOKEN"),
            log_level=_env("FORGE_LOG_LEVEL", "info"),
        )
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(l) for l in err["loc"])
            print(f"[forge] FATAL: config error: {loc}: {err['msg']}", file=sys.stderr)
        sys.exit(1)
