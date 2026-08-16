"""Model lifecycle manager: state machine, idle-unload, activate (§6).

States (verbatim, used in /admin/v1/status):
  IDLE, DOWNLOADING, LOADING, READY, UNLOADING, ERROR

Exactly one vLLM subprocess at a time. Both the implicit /v1/* trigger and
POST /admin/v1/activate funnel into the same public() entry point
(ensure_ready) — there is no separate "UI load" implementation (§6.5).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import httpx

from forge.config import ForgeConfig
from forge.registry import Manifest, Registry

log = logging.getLogger("forge.lifecycle")

ST_IDLE = "IDLE"
ST_DOWNLOADING = "DOWNLOADING"
ST_LOADING = "LOADING"
ST_READY = "READY"
ST_UNLOADING = "UNLOADING"
ST_ERROR = "ERROR"

ACTIVE_STATES = {ST_DOWNLOADING, ST_LOADING, ST_READY}
BUSY_STATES = {ST_DOWNLOADING, ST_LOADING, ST_UNLOADING}

_SIGTERM_GRACE_SECONDS = 15  # §6.4


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class BusyError(Exception):
    def __init__(self, state: str, alias: str) -> None:
        self.state = state
        self.alias = alias
        super().__init__(f"busy in {state} for {alias!r}")


class LoadFailed(Exception):
    pass


def _launch_args(manifest: Manifest, registry: Registry, cfg: ForgeConfig) -> list:
    """Build the exact `vllm serve ...` argv (§6.3 launch contract)."""
    if manifest.source.file:
        target = str(registry.weights_dir(manifest) / manifest.source.file)
    else:
        target = str(registry.weights_dir(manifest))
    args = [
        "serve",
        target,
        "--host", "127.0.0.1",
        "--port", str(cfg.internal_port),
        "--served-model-name", manifest.served_model_name,
    ]
    # Boolean flags (value is empty string or "true"/"yes"/"1") get appended
    # as just --flag; others get --key value.
    _BOOL_TRUE = {"", "true", "yes", "1"}
    for key, val in manifest.engine_args.items():  # manifest order preserved
        args.append(f"--{key}")
        if val.lower() not in _BOOL_TRUE:
            args.append(val)
    return args


class LifecycleManager:
    def __init__(self, cfg: ForgeConfig, registry: Registry, counter) -> None:
        self.cfg = cfg
        self.registry = registry
        self.counter = counter

        self.state = ST_IDLE
        self.alias: Optional[str] = None
        self.served_model_name: Optional[str] = None
        self.pid: Optional[int] = None
        self.port: Optional[int] = None
        self.loaded_at: Optional[str] = None
        self.last_request_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self._queue_depth = 0

        self._op_lock = asyncio.Lock()   # single operation slot (§13)
        self._state_lock = asyncio.Lock()  # guards state bookkeeping
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_lines: list = []
        self._stop = asyncio.Event()
        self._tasks: list = []
        self._http = httpx.AsyncClient(timeout=None)

    # ================= state bookkeeping =================

    def _set(self, state: str, **fields) -> None:
        log.info("state transition %s -> %s", self.state, state)
        self.state = state
        for k, v in fields.items():
            setattr(self, k, v)
        asyncio.get_event_loop().create_task(self._write_state_file())

    async def _write_state_file(self) -> None:
        """Write-through mirror to /data/state/active.json (§6.2)."""
        try:
            state_dir = os.path.join(self.cfg.data_dir, "state")
            os.makedirs(state_dir, exist_ok=True)
            payload = {
                "state": self.state,
                "alias": self.alias,
                "served_model_name": self.served_model_name,
                "pid": self.pid,
                "port": self.port,
                "loaded_at": self.loaded_at,
                "last_request_at": self.last_request_at,
            }
            p = os.path.join(state_dir, "active.json")
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, p)
        except Exception as exc:  # observability only — never break the request
            log.warning("could not write state file: %s", exc)

    async def _stderr_pump(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while True:
            raw = await proc.stderr.readline()
            if not raw:
                break
            line = raw.decode(errors="replace")
            os.write(1, b"[vllm] " + line.encode())
            async with self._state_lock:
                self._stderr_lines.append(line)
                if len(self._stderr_lines) > 500:
                    self._stderr_lines = self._stderr_lines[-350:]

    # ================= core: ensure a model is READY (§6.5 single path) =================

    async def ensure_ready(self, alias: str) -> None:
        """Download if needed, load, wait for health — then proxyable."""
        manifest = self.registry.get(alias)
        if manifest is None:
            raise LookupError(f"unknown alias {alias!r}")

        async with self._op_lock:
            if self.state == ST_READY and self.alias == alias:
                return  # idempotent: already active

            if self.state == ST_READY and self.alias != alias:
                await self._unload("swap")

            if self.state in (ST_ERROR, ST_LOADING, ST_UNLOADING) and self._proc is not None:
                # settle any dying/lingering subprocess before (re)launching
                await self._kill(self._proc)
                self._proc = None

            # ---- DOWNLOADING ---- #
            if not self.registry.weights_present(manifest):
                self._set(ST_DOWNLOADING, alias=alias,
                          served_model_name=manifest.served_model_name)
                from forge.downloader import download_manifest
                try:
                    await download_manifest(self.cfg, self.registry, manifest)
                except Exception as exc:
                    self.last_error = f"download failed: {exc}"
                    self._set(ST_ERROR, alias=alias)
                    raise LoadFailed(f"download failed: {exc}") from exc

            # ---- LOADING ---- #
            self._set(ST_LOADING, alias=alias,
                      served_model_name=manifest.served_model_name)
            await self._launch(manifest)
            healthy = await self._wait_healthy()
            if not healthy:
                await self._kill(self._proc)
                self._proc = None
                tail = self.stderr_tail or "(no stderr captured)"
                self.last_error = (
                    f"vLLM failed to start (timeout {self.cfg.startup_timeout_seconds}s): {tail}"
                )
                self._set(ST_ERROR, alias=alias)
                raise LoadFailed(self.last_error)

            # ---- READY ---- #
            self.loaded_at = _utcnow()
            self.last_request_at = _utcnow()
            self.port = self.cfg.internal_port
            self._set(ST_READY)
            await self.counter.reset_since_load()

    # ================= launch / health / kill =================

    async def _launch(self, manifest: Manifest) -> None:
        prog = "vllm"
        args = _launch_args(manifest, self.registry, self.cfg)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                prog, *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            self.last_error = (
                f"Failed to start vLLM subprocess: {exc}. "
                f"Is 'vllm' installed and on the PATH?"
            )
            self._set(ST_ERROR, alias=manifest.alias)
            raise LoadFailed(self.last_error) from exc
        self.pid = self._proc.pid
        asyncio.get_event_loop().create_task(self._stderr_pump(self._proc))

    async def _wait_healthy(self) -> bool:
        deadline = time.monotonic() + self.cfg.startup_timeout_seconds
        url = f"http://127.0.0.1:{self.cfg.internal_port}/health"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.returncode is not None:
                await asyncio.sleep(0.3)  # let stderr drain
                return False
            try:
                r = await self._http.get(url, timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(self.cfg.health_poll_interval_seconds)
        return False

    async def _kill(self, proc: Optional[asyncio.subprocess.Process]) -> None:
        """SIGTERM, wait 15s, SIGKILL (§6.4)."""
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=_SIGTERM_GRACE_SECONDS)
        except asyncio.TimeoutError:
            log.warning("unload: still alive after 15s, sending SIGKILL")
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass

    async def _unload(self, reason: str) -> None:
        """Unload the active subprocess back to IDLE. Caller holds op lock."""
        self._set(ST_UNLOADING)
        await self._kill(self._proc)
        self._proc = None
        self.alias = None
        self.served_model_name = None
        self.loaded_at = None
        self.port = None
        self.last_request_at = None
        self.pid = None
        self._set(ST_IDLE)
        log.info("unload complete (%s)", reason)

    # ================= explicit admin triggers (§6.5, §8.3) =================

    async def activate(self, alias: str) -> Tuple[int, dict]:
        """POST /admin/v1/activate → (http_status, body).

        200 idempotent no-op (already active)
        202 accepted, transitioning
        404 unknown alias
        409 busy (DOWNLOADING/LOADING/UNLOADING) with a different alias
        503 the attempted load failed
        """
        if self.registry.get(alias) is None:
            return 404, {"detail": f"unknown alias {alias!r}"}

        async with self._state_lock:
            if self.state == ST_READY and self.alias == alias:
                log.info("activate: idempotent-noop alias=%s", alias)
                return 200, {"state": self.state, "alias": alias}
            if self.state in (ST_DOWNLOADING, ST_LOADING, ST_UNLOADING) \
                    and self.alias != alias:
                log.info("activate: rejected-409 alias=%s busy_with=%s state=%s",
                         alias, self.alias, self.state)
                return 409, {
                    "detail": (f"busy: state={self.state}, "
                               f"active_alias={self.alias!r}; "
                               f"requested {alias!r}")
                }

        # READY+same already handled; READY+different → swap (valid, 202)
        log.info("activate: accepted alias=%s", alias)
        try:
            await self.ensure_ready(alias)
        except LookupError:
            return 404, {"detail": f"unknown alias {alias!r}"}
        except LoadFailed as exc:
            return 503, {"detail": str(exc)}
        except Exception as exc:  # unexpected — surface, don't crash the API
            log.exception("activate: unexpected failure for %s", alias)
            return 500, {"detail": f"internal error: {exc}"}
        return 202, {"state": self.state, "alias": alias}

    async def unload_forced(self) -> Tuple[int, dict]:
        """POST /admin/v1/unload: 202 if unloaded, 200 {"state":"IDLE"} if idle."""
        async with self._op_lock:
            async with self._state_lock:
                if self.state == ST_IDLE or self._proc is None:
                    return 200, {"state": ST_IDLE}
                was = self.state
            await self._unload("forced")
        return 202, {"state": self.state, "was": was}

    # ================= read endpoints =================

    def status(self) -> dict:
        d = {
            "state": self.state,
            "alias": self.alias,
            "served_model_name": self.served_model_name,
            "pid": self.pid,
            "port": self.port,
            "loaded_at": self.loaded_at,
            "last_request_at": self.last_request_at,
            "queue_depth": self._queue_depth,
        }
        if self.state == ST_ERROR:
            d["last_error"] = self.last_error or self.stderr_tail or "(no stderr captured)"
        return d

    @property
    def stderr_tail(self) -> str:
        return "".join(self._stderr_lines[-80:])

    # ================= request bookkeeping (proxy path) =================

    def note_request_start(self, alias: str) -> None:
        self._queue_depth += 1
        self.touch(alias)

    def note_request_end(self) -> None:
        self._queue_depth = max(0, self._queue_depth - 1)

    def touch(self, alias: str) -> None:
        if self.state == ST_READY and self.alias == alias:
            self.last_request_at = _utcnow()

    @property
    def proxy_base(self) -> Optional[str]:
        if self.state == ST_READY:
            return f"http://127.0.0.1:{self.cfg.internal_port}"
        return None

    # ================= background tasks =================

    async def start_background(self) -> None:
        self._tasks.append(asyncio.create_task(self._idle_loop(), name="forge-idle"))
        self._tasks.append(asyncio.create_task(self._crash_loop(), name="forge-crash"))

    async def stop_background(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await self._kill(self._proc)
        self._proc = None
        try:
            await self._http.aclose()
        except Exception:
            pass

    async def _idle_loop(self) -> None:
        """Tick every 10s; unload READY models past idle_timeout_seconds (§6.4)."""
        while not self._stop.is_set():
            await asyncio.sleep(10)
            if self._stop.is_set():
                return
            try:
                await self._idle_check()
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("idle check failed")

    async def _idle_check(self) -> None:
        async with self._op_lock:
            if self.state != ST_READY or self.alias is None:
                return
            manifest = self.registry.get(self.alias)
            if manifest is None or not self.last_request_at:
                return
            age = (datetime.now(timezone.utc) - _parse_ts(self.last_request_at)).total_seconds()
            if age > manifest.idle_timeout_seconds:
                log.info("idle timeout reached (%.0fs > %ds) for %s",
                         age, manifest.idle_timeout_seconds, self.alias)
                await self._unload("idle")

    async def _crash_loop(self) -> None:
        """Detect unexpected subprocess exit → ERROR state (§13, §6.1)."""
        while not self._stop.is_set():
            await asyncio.sleep(2)
            if self._stop.is_set():
                return
            proc = self._proc
            if proc is None or proc.returncode is None:
                continue
            async with self._op_lock:
                if self.state in (ST_READY, ST_LOADING):
                    self.last_error = (
                        f"vLLM exited with code {proc.returncode}\n"
                        f"{self.stderr_tail or '(no stderr captured)'}"
                    )
                    self._set(ST_ERROR)
                    log.error("vLLM crashed: %s", self.last_error)
