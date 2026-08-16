"""Weight acquisition via huggingface_hub (§7).

MUST use huggingface_hub (resumable downloads, split files, auth) — not a
manual HTTP client. Files land under /data/models/<org>__<name>/ using
snapshot_download (whole repo) or hf_hub_download (single file).

Progress is reported as dict events (mirroring the §7 ndjson lines) via a
thread-safe push function, decoupled from who consumes them (admin /pull
streams them; the implicit /v1/* trigger discards them).
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Awaitable, Callable, Optional

from huggingface_hub import hf_hub_download, snapshot_download

from forge.config import ForgeConfig
from forge.registry import Manifest, Registry

log = logging.getLogger("forge.downloader")

PushFn = Callable[[dict], Awaitable[None]]
_SENTINEL: Optional[dict] = None  # internal: use None to close the queue


class ProgressChannel:
    """Thread-safe queue of progress events consumed by an ndjson streamer."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def push(self, event: Optional[dict]) -> None:
        """Thread-safe: may be called from huggingface_hub worker threads."""
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._queue.put(event), self._loop)
        # else: loop not running (tests) — drop

    async def get(self) -> Optional[dict]:
        return await self._queue.get()


class _ForgeTqdm:
    """Minimal tqdm-class shim: reports (desc, done, total) via a push function.

    Newer huggingface_hub (>=0.24) uses tqdm as a context manager,
    so we must implement __enter__/__exit__.
    """

    def __init__(self, push: Callable[[dict], None], *args, **kwargs) -> None:
        self.total = kwargs.get("total") or (args[1] if len(args) > 1 and args[1] else None)
        self.desc = kwargs.get("desc") or (args[0] if args else "")
        self.n = int(kwargs.get("initial", 0))
        self._push = push

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def update(self, n: int = 1, **kwargs) -> int:
        self.n += int(n)
        try:
            self._push({
                "status": "downloading",
                "file": self.desc or None,
                "downloaded_bytes": self.n,
                "total_bytes": self.total,
            })
        except Exception:  # progress is best-effort; never fail the download
            pass
        return int(n)

    def close(self) -> None:
        pass

    def set_description(self, desc: str = "", **kwargs) -> None:
        self.desc = desc

    def clear(self, **kwargs) -> None:
        pass

    def refresh(self, **kwargs) -> None:
        pass

    def reset(self, total=None, **kwargs) -> None:
        if total is not None:
            self.total = total
        self.n = 0


def _make_push(channel: Optional[ProgressChannel]) -> Callable[[dict], None]:
    def push(event: dict) -> None:
        if channel is None:
            return
        channel.push(event)
    return push


def _resolve_token(cfg: ForgeConfig) -> Optional[str]:
    return cfg.hf_token or None


async def download_manifest(
    cfg: ForgeConfig,
    registry: Registry,
    manifest: Manifest,
    channel: Optional[ProgressChannel] = None,
) -> None:
    """Download weights for `manifest` into the registry's models dir.

    Resolves via huggingface_hub. Single file (manifest.source.file set) →
    hf_hub_download; whole repo → snapshot_download. Raises on failure.
    """
    local_dir = registry.weights_dir(manifest)
    local_dir.mkdir(parents=True, exist_ok=True)
    if channel is not None:
        channel.bind_loop(asyncio.get_running_loop())
    push = _make_push(channel)
    token = _resolve_token(cfg)

    loop = asyncio.get_running_loop()

    def _work() -> str:
        if manifest.source.file:
            path = hf_hub_download(
                repo_id=manifest.source.repo,
                filename=manifest.source.file,
                local_dir=str(local_dir),
                token=token,
                tqdm_class=functools.partial(_ForgeTqdm, push),
            )
            return str(path)
        path = snapshot_download(
            repo_id=manifest.source.repo,
            local_dir=str(local_dir),
            token=token,
            tqdm_class=functools.partial(_ForgeTqdm, push),
        )
        return str(path)

    resolved = await loop.run_in_executor(None, _work)
    log.info("download complete alias=%s resolved_path=%s", manifest.alias, resolved)
    push({"status": "complete", "alias": manifest.alias})
