"""nvidia-smi wrapper + in-memory request counters (§8.3 stats payload).

The exact nvidia-smi command is mandated by §8.3 — no pynvml, no gpustat.
Counters are in-memory only; reset on restart is intentional, not an oversight.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

log = logging.getLogger("forge.stats")


async def query_gpu_info() -> dict:
    """Return GPU info dict; empty dict if nvidia-smi unavailable.

    Keys: vram_used_mb, vram_total_mb, gpu_utilization_pct, gpu_temp_c, gpu_name
    """
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
        return {
            "vram_used_mb": int(parts[0]),
            "vram_total_mb": int(parts[1]),
            "gpu_utilization_pct": int(parts[2]),
            "gpu_temp_c": int(parts[3]),
            "gpu_name": parts[4] if len(parts) > 4 else None,
        }
    except Exception as exc:
        log.debug("nvidia-smi unavailable: %s", exc)
        return {}


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
