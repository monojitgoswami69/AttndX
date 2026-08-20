"""
Central configuration loader for gen2.
Loads YAML config, provides typed access, validates values at startup.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Resolve gen2 root (this file is at gen2/config/__init__.py)
GEN2_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(p: str) -> Path:
    """Resolve a path relative to the repo root if not absolute."""
    path = Path(p)
    if path.is_absolute():
        return path
    # Paths in config like "gen2/data/..." resolve relative to repo root
    if str(path).startswith("gen2/"):
        return GEN2_ROOT.parent / path
    return GEN2_ROOT / path


class Config:
    """Typed configuration accessor. Validates at load time."""

    _data: dict = {}
    _loaded: bool = False

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Config":
        if config_path is None:
            config_path = GEN2_ROOT / "config" / "default.yaml"
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            cls._data = yaml.safe_load(f)
        cls._loaded = True
        cls._validate()
        return cls

    @classmethod
    def _validate(cls):
        """Validate critical config values at load time."""
        cfg = cls._data
        errors = []
        if cfg["embedding"]["dimension"] != 512:
            errors.append("embedding.dimension must be 512 for ArcFace")
        if cfg["alignment"]["output_size"] != [112, 112]:
            errors.append("alignment.output_size must be [112, 112] for ArcFace")
        if len(cfg["alignment"]["template"]) != 5:
            errors.append("alignment.template must have exactly 5 points")
        if cfg["recognition"]["accept_threshold"] < cfg["recognition"]["reject_threshold"]:
            errors.append("recognition.accept_threshold must be >= reject_threshold")
        if cfg["enrollment"]["min_samples"] > cfg["enrollment"]["max_samples"]:
            errors.append("enrollment.min_samples must be <= max_samples")
        if errors:
            raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))

    @classmethod
    def get(cls, *keys: str, default: Any = None) -> Any:
        """Get a nested config value by dotted path keys."""
        if not cls._loaded:
            cls.load()
        val = cls._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    @classmethod
    def raw(cls) -> dict:
        """Return the raw config dict."""
        if not cls._loaded:
            cls.load()
        return cls._data

    # ─── Path helpers ───

    @classmethod
    def data_dir(cls) -> Path:
        return _resolve_path(cls.get("paths", "data_dir"))

    @classmethod
    def model_dir(cls) -> Path:
        return _resolve_path(cls.get("paths", "model_dir"))

    @classmethod
    def captures_dir(cls) -> Path:
        return _resolve_path(cls.get("paths", "captures_dir"))

    @classmethod
    def snapshots_dir(cls) -> Path:
        return _resolve_path(cls.get("paths", "snapshots_dir"))

    @classmethod
    def eval_dir(cls) -> Path:
        return _resolve_path(cls.get("paths", "eval_dir"))

    @classmethod
    def biometric_db_path(cls) -> Path:
        return _resolve_path(cls.get("paths", "biometric_db"))

    @classmethod
    def attendance_db_path(cls) -> Path:
        return _resolve_path(cls.get("paths", "attendance_db"))

    @classmethod
    def model_path(cls, key: str) -> Path:
        """Resolve a model file path. key: 'detector' | 'embedder' | 'liveness'."""
        filename = cls.get("models", key)
        p = cls.model_dir() / filename
        if not p.exists():
            # Fallback: look in parent repo's models/ dir (read-only, not modified)
            fallback = GEN2_ROOT.parent / "models" / filename
            if fallback.exists():
                return fallback
        return p

    @classmethod
    def ensure_directories(cls):
        """Create all required data directories."""
        for d in [cls.data_dir(), cls.model_dir(), cls.captures_dir(),
                  cls.snapshots_dir(), cls.eval_dir()]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def pipeline_version_string(cls) -> str:
        """Return a composite version string for embedding compatibility."""
        pv = cls.get("pipeline_version")
        return f"{pv['detector']}|{pv['alignment']}|{pv['embedder']}|{pv['preprocessing']}"
