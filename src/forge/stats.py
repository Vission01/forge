"""nvidia-smi wrapper + in-memory request counters (§8.3 stats payload).

The exact nvidia-smi command is mandated by §8.3 — no pynvml, no gpustat.
Counters are in-memory only; reset on restart is intentional, not an oversight.

GPU info is cached with a configurable TTL to avoid spawning nvidia-smi
on every dashboard poll (default 5 s).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Tuple

log = logging.getLogger("forge.stats")

# ---- GPU info cache ----

_gpu_cache_time: float = 0.0
_gpu_cache_data: dict = {}
_GPU_CACHE_TTL: float = 5.0  # seconds


async def query_gpu_info(ttl: float = _GPU_CACHE_TTL) -> dict:
    """Return GPU info dict; empty dict if nvidia-smi unavailable.

    Results are cached for *ttl* seconds so the dashboard's 3-second
    poll cycle does not fork nvidia-smi on every tick.

    Keys: vram_used_mb, vram_total_mb, gpu_utilization_pct, gpu_temp_c, gpu_name
    """
    global _gpu_cache_time, _gpu_cache_data
    now = time.monotonic()
    if now - _gpu_cache_time < ttl:
        return _gpu_cache_data

    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,name",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            log.debug("nvidia-smi exit code %d", proc.returncode)
            return {}
        lines = [ln for ln in stdout.decode().splitlines() if ln.strip()]
        if not lines:
            return {}
        parts = [p.strip() for p in lines[0].split(",")]
        data = {
            "vram_used_mb": int(parts[0]),
            "vram_total_mb": int(parts[1]),
            "gpu_utilization_pct": int(parts[2]),
            "gpu_temp_c": int(parts[3]),
            "gpu_name": parts[4] if len(parts) > 4 else None,
        }
        _gpu_cache_time = now
        _gpu_cache_data = data
        return data
    except Exception as exc:
        log.debug("nvidia-smi unavailable: %s", exc)
        return {}


def estimate_weights_mb(weights_dir) -> Optional[int]:
    """Estimate model weight size in MB by summing weight files on disk."""
    from pathlib import Path
    import os as _os
    d = Path(weights_dir)
    if not d.is_dir():
        return None
    _WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ggml"}
    total = 0
    for root, dirs, files in _os.walk(d):
        dirs[:] = [dd for dd in dirs if not dd.startswith(".")]
        for f in files:
            if any(f.endswith(ext) for ext in _WEIGHT_EXTS):
                try:
                    total += _os.path.getsize(_os.path.join(root, f))
                except OSError:
                    pass
    return int(total / (1024 * 1024)) if total > 0 else None


async def check_vram_fit(weights_dir, overhead_mb: int = 2048) -> dict:
    """Pre-flight VRAM check. Returns dict with fit assessment.

    overhead_mb: estimated overhead for CUDA context, KV cache init, etc.
    Returns: {fits: bool, weights_mb, free_mb, total_mb, gpu_name, message}
    """
    gpu = await query_gpu_info()
    if not gpu:
        return {"fits": True, "message": "GPU info unavailable — skipping VRAM check"}

    weights_mb = estimate_weights_mb(weights_dir)
    if weights_mb is None:
        return {"fits": True, "message": "No weight files found — skipping VRAM check"}

    free_mb = gpu["vram_total_mb"] - gpu["vram_used_mb"]
    needed_mb = weights_mb + overhead_mb

    result = {
        "weights_mb": weights_mb,
        "overhead_mb": overhead_mb,
        "needed_mb": needed_mb,
        "free_mb": free_mb,
        "total_mb": gpu["vram_total_mb"],
        "used_mb": gpu["vram_used_mb"],
        "gpu_name": gpu.get("gpu_name"),
    }

    if needed_mb > free_mb:
        result["fits"] = False
        result["message"] = (
            f"Model weights ({weights_mb:,} MB) + overhead ({overhead_mb:,} MB) = "
            f"{needed_mb:,} MB needed, but only {free_mb:,} MB free "
            f"on {gpu.get('gpu_name', 'GPU')} ({gpu['vram_used_mb']:,} / {gpu['vram_total_mb']:,} MB used). "
            f"Free up VRAM or choose a smaller model."
        )
    else:
        result["fits"] = True
        result["message"] = (
            f"VRAM OK: {weights_mb:,} MB weights + {overhead_mb:,} MB overhead = "
            f"{needed_mb:,} MB needed, {free_mb:,} MB free."
        )

    return result


# Backward compat wrapper
async def query_vram_mb() -> Tuple[Optional[int], Optional[int]]:
    info = await query_gpu_info()
    return (info.get("vram_used_mb"), info.get("vram_total_mb"))


class RequestCounter:
    """In-memory counters (persisted nowhere — §8.3)."""

    def __init__(self) -> None:
        self._total = 0
        self._since_load = 0

    async def inc_total(self) -> None:
        self._total += 1

    async def inc_since_load(self) -> None:
        self._since_load += 1

    async def reset_since_load(self) -> None:
        self._since_load = 0

    async def snapshot(self) -> Tuple[int, int]:
        return (self._total, self._since_load)
