from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(name: str) -> dict[str, Any]:
    """Load a named config file from configs/. name e.g. 'data', 'models'."""
    return load_yaml(CONFIGS_DIR / f"{name}.yaml")


class Config:
    """Lazy-loaded project configuration."""

    _cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def get(cls, name: str) -> dict[str, Any]:
        if name not in cls._cache:
            cls._cache[name] = load_config(name)
        return cls._cache[name]

    @classmethod
    def data(cls) -> dict[str, Any]:
        return cls.get("data")

    @classmethod
    def features(cls) -> dict[str, Any]:
        return cls.get("features")

    @classmethod
    def models(cls) -> dict[str, Any]:
        return cls.get("models")

    @classmethod
    def experiment(cls) -> dict[str, Any]:
        return cls.get("experiment")
