"""nvidia-smi wrapper + in-memory request counters (§8.3 stats payload).

The exact nvidia-smi command is mandated by §8.3 — no pynvml, no gpustat.
Counters are in-memory only; reset on restart is intentional, not an oversight.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

log = logging.getLogger("forge.stats")


async def query_vram_mb() -> Tuple[Optional[int], Optional[int]]:
    """Return (used_mb, total_mb); (None, None) if the command fails or is unavailable."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            log.debug("nvidia-smi exit code %d", proc.returncode)
            return (None, None)
        lines = [ln for ln in stdout.decode().splitlines() if ln.strip()]
        if not lines:
            return (None, None)
        used_s, total_s = lines[0].split(",")
        return (int(used_s.strip()), int(total_s.strip()))
    except Exception as exc:  # noqa: BLE001 - spec: MUST not error the endpoint
        log.debug("nvidia-smi unavailable: %s", exc)
        return (None, None)


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
