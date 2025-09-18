"""Utilities for locating configuration and data pack resources."""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory containing configuration assets."""

    return Path(__file__).resolve().parents[3]


def config_root() -> Path:
    """Return the default directory that stores experiment configuration manifests."""

    return project_root() / "config"


def data_packs_root() -> Path:
    """Return the default directory that stores dataset packs and split CSVs."""

    return project_root() / "data_packs"


def resolve_config_path(path: str | Path) -> Path:
    """Resolve ``path`` relative to :func:`config_root` when it is not absolute."""

    path = Path(path)
    if path.is_absolute():
        return path
    return config_root() / path


def resolve_data_pack_path(path: str | Path) -> Path:
    """Resolve ``path`` relative to :func:`data_packs_root` when it is not absolute."""

    path = Path(path)
    if path.is_absolute():
        return path
    return data_packs_root() / path


__all__ = [
    "config_root",
    "data_packs_root",
    "project_root",
    "resolve_config_path",
    "resolve_data_pack_path",
]
