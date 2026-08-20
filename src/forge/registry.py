"""Model registry: manifest load/validate/CRUD (§5).

Pydantic v2 validation on every read. Manifests at <data>/registry/<alias>.yaml.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, field_validator

log = logging.getLogger("forge.registry")

ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class Source(BaseModel):
    repo: str
    file: Optional[str] = None

    @field_validator("repo")
    @classmethod
    def _repo(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source.repo MUST be a non-empty string")
        return v


class Manifest(BaseModel):
    alias: str
    source: Source
    served_model_name: str
    engine_args: Dict[str, str] = {}
    idle_timeout_seconds: int = 600

    @field_validator("alias")
    @classmethod
    def _alias(cls, v: str) -> str:
        if not ALIAS_RE.match(v):
            raise ValueError(f"alias {v!r} does not match ^[a-z0-9][a-z0-9._-]*$")
        return v

    @field_validator("served_model_name")
    @classmethod
    def _served(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("served_model_name MUST be a non-empty string")
        return v

    @field_validator("engine_args", mode="before")
    @classmethod
    def _engine(cls, v: object) -> Dict[str, str]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("engine_args MUST be an object (dict)")
        out: Dict[str, str] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                raise ValueError(f"engine_args keys MUST be str, got {k!r}")
            if not isinstance(val, str):
                raise ValueError(f"engine_args[{k!r}] MUST be a string, got {type(val).__name__}")
            out[k] = val
        return out

    @field_validator("idle_timeout_seconds")
    @classmethod
    def _idle(cls, v: int) -> int:
        if v < 30:
            raise ValueError("idle_timeout_seconds MUST be >= 30")
        return v


class Registry:
    """In-process view over the registry directory."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.registry_dir = os.path.join(data_dir, "registry")
        os.makedirs(self.registry_dir, exist_ok=True)

    # ---- loading ---- #

    def _stem(self, path: Path) -> str:
        return path.stem  # filename without .yaml extension

    def load(self, path: Path) -> Manifest:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"manifest {path.name} is not a YAML mapping")
        m = Manifest(**raw)
        # Filename stem MUST equal alias (spec §5.1)
        if self._stem(path) != m.alias:
            raise ValueError(
                f"filename stem {self._stem(path)!r} != alias {m.alias!r}"
            )
        return m

    def all(self) -> Dict[str, Manifest]:
        result: Dict[str, Manifest] = {}
        rdir = Path(self.registry_dir)
        if not rdir.is_dir():
            return result
        for f in sorted(rdir.glob("*.yaml")):
            try:
                m = self.load(f)
                result[m.alias] = m
            except Exception as exc:  # §12: one bad manifest must not crash startup
                log.warning("Skipping invalid manifest %s: %s", f.name, exc)
        return result

    def get(self, alias: str) -> Optional[Manifest]:
        p = self._path(alias)
        if not p.is_file():
            return None
        try:
            return self.load(p)
        except Exception as exc:
            log.warning("Manifest %s invalid: %s", p.name, exc)
            return None

    def find_by_served(self, name: str) -> Optional[Manifest]:
        for m in self.all().values():
            if m.served_model_name == name:
                return m
        return None

    def list_with_downloaded(self) -> List[dict]:
        """GET /admin/v1/registry amended shape — computed 'downloaded' field (§8.3)."""
        out: List[dict] = []
        for m in self.all().values():
            d = m.model_dump()
            info = self.weights_info(m)
            d["downloaded"] = info["present"]
            d["size_mb"] = info["size_mb"]
            out.append(d)
        # Stable order by alias
        out.sort(key=lambda x: x["alias"])
        return out

    def _path(self, alias: str) -> Path:
        return Path(self.registry_dir) / f"{alias}.yaml"

    # ---- save / delete ---- #

    def save(self, manifest: Manifest) -> Path:
        self._check_uniqueness(manifest)
        p = self._path(manifest.alias)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = manifest.model_dump()
        p.write_text(yaml.dump(data, sort_keys=False, default_flow_style=False))
        return p

    def delete(self, alias: str) -> bool:
        p = self._path(alias)
        if p.is_file():
            p.unlink()
            return True
        return False

    def exists(self, alias: str) -> bool:
        return self._path(alias).is_file()

    def mtime(self, alias: str) -> int:
        p = self._path(alias)
        try:
            return int(p.stat().st_mtime)
        except OSError:
            return 0

    # ---- helpers ---- #

    def _check_uniqueness(self, candidate: Manifest) -> None:
        for m in self.all().values():
            if m.alias == candidate.alias:
                # same alias = overwrite path, allowed (POST is create-only; caller enforces 409)
                continue
            if m.served_model_name == candidate.served_model_name:
                raise ValueError(
                    f"served_model_name {candidate.served_model_name!r} is already "
                    f"used by alias {m.alias!r}"
                )

    def weights_dir(self, manifest: Manifest) -> Path:
        org, _, name = manifest.source.repo.partition("/")
        return Path(self.data_dir) / "models" / f"{org}__{name}"

    _WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ggml"}

    def weights_info(self, manifest: Manifest) -> dict:
        """Return {"present": bool, "size_mb": int|None} in one walk."""
        d = self.weights_dir(manifest)
        if not d.is_dir():
            return {"present": False, "size_mb": None}
        total = 0
        found = False
        try:
            for root, dirs, files in os.walk(d):
                dirs[:] = [dd for dd in dirs if not dd.startswith(".")]
                for f in files:
                    if any(f.endswith(ext) for ext in self._WEIGHT_EXTS):
                        found = True
                        try:
                            total += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
        except OSError:
            return {"present": False, "size_mb": None}
        size_mb = int(total / (1024 * 1024)) if total > 0 else None
        return {"present": found, "size_mb": size_mb}

    def weights_present(self, manifest: Manifest) -> bool:
        return self.weights_info(manifest)["present"]

    def weights_size_mb(self, manifest: Manifest) -> int | None:
        return self.weights_info(manifest)["size_mb"]

    def delete_weights(self, manifest: Manifest) -> bool:
        """Delete downloaded weights for a manifest. Returns True if deleted."""
        import shutil
        d = self.weights_dir(manifest)
        if not d.is_dir():
            return False
        try:
            shutil.rmtree(d)
            log.info("deleted weights at %s", d)
            return True
        except OSError as exc:
            log.warning("failed to delete weights at %s: %s", d, exc)
            return False

    def total_weights_size_mb(self) -> int:
        """Total size of all downloaded model weights in MB."""
        models_dir = Path(self.data_dir) / "models"
        if not models_dir.is_dir():
            return 0
        total = 0
        try:
            for root, dirs, files in os.walk(models_dir):
                dirs[:] = [dd for dd in dirs if not dd.startswith(".")]
                for f in files:
                    if any(f.endswith(ext) for ext in self._WEIGHT_EXTS):
                        try:
                            total += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
        except OSError:
            pass
        return int(total / (1024 * 1024))
