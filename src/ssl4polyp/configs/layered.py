"""Utilities for layered YAML experiment configuration loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Tuple

import yaml

from . import config_root, resolve_config_path


def _ensure_yaml_suffix(path: Path) -> Path:
    if path.suffix:
        return path
    return path.with_suffix(".yaml")


def _resolve_reference(reference: str | Path, anchor: Path | None = None) -> Path:
    candidate = Path(reference)
    candidate = _ensure_yaml_suffix(candidate)
    if candidate.is_absolute():
        return candidate
    if anchor is not None:
        anchored = (anchor.parent / candidate).resolve()
        if anchored.exists():
            return anchored
    resolved = resolve_config_path(candidate)
    if resolved.exists():
        return resolved
    # Fall back to plain candidate relative to config root even if missing; callers
    # will raise when attempting to read it, which offers a clearer error message.
    return (config_root() / candidate).resolve()


def _deep_merge(base: MutableMapping[str, Any], updates: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    result: MutableMapping[str, Any] = deepcopy(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], MutableMapping) and isinstance(value, MutableMapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[assignment]
        else:
            result[key] = deepcopy(value)
    return result


def _load_recursive(path: Path, stack: Tuple[Path, ...]) -> Tuple[Dict[str, Any], List[Path]]:
    if path in stack:
        chain = " -> ".join(str(p) for p in stack + (path,))
        raise ValueError(f"Cyclic defaults detected while loading configs: {chain}")

    with open(path, "r") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults = raw.pop("defaults", [])
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]

    merged: Dict[str, Any] = {}
    sources: List[Path] = []
    for default in defaults:
        default_path = _resolve_reference(default, anchor=path)
        default_cfg, default_sources = _load_recursive(default_path, stack + (path,))
        merged = _deep_merge(merged, default_cfg)
        sources.extend(default_sources)

    merged = _deep_merge(merged, raw)
    sources.append(path)
    return merged, sources


def load_layered_config(reference: str | Path) -> Dict[str, Any]:
    """Load ``reference`` resolving ``defaults`` recursively."""

    path = _resolve_reference(reference)
    config, sources = _load_recursive(path, tuple())
    # Attach provenance for downstream logging/debugging.
    config.setdefault("__sources__", [str(p) for p in sources])
    return config


def resolve_model_entries(entries: Iterable[Any]) -> List[Dict[str, Any]]:
    """Resolve model references declared inside an experiment configuration."""

    resolved: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, (str, Path)):
            data = load_layered_config(entry)
        else:
            data = deepcopy(entry)
        if "model" in data:
            resolved.append(deepcopy(data["model"]))
        else:
            resolved.append(deepcopy(data))
    return resolved


def extract_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = deepcopy(config.get("dataset", {}))
    if not dataset:
        raise ValueError("Experiment configuration must define a dataset section via defaults or overrides.")
    return dataset

