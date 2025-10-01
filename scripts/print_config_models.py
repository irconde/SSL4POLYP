#!/usr/bin/env python3
"""Print the canonical model keys for an experiment configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from ssl4polyp.configs.layered import load_layered_config, resolve_model_entries
except ModuleNotFoundError as exc:  # pragma: no cover - dependency issues
    if exc.name in {"ssl4polyp", "yaml"}:
        raise SystemExit(
            "Unable to import required dependencies. Ensure PyYAML is installed and "
            "the repository is available on PYTHONPATH before running this helper."
        ) from exc
    raise


def _normalize_model_keys(raw: Iterable[dict]) -> List[str]:
    keys: List[str] = []
    for entry in raw:
        key = entry.get("key")
        if not key:
            raise ValueError(f"Model entry missing 'key': {entry!r}")
        keys.append(str(key))
    return keys


def _extract_models(config: dict) -> List[str]:
    entries = config.get("models", []) or []
    if not entries:
        raise ValueError("Configuration does not define any models.")

    resolved = resolve_model_entries(entries)
    if not resolved:
        raise ValueError("Configuration resolved to an empty model list.")

    return _normalize_model_keys(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        help=(
            "Experiment configuration reference (e.g., 'exp/exp1.yaml'). "
            "The path is resolved using the layered config loader."
        ),
    )
    args = parser.parse_args()

    cfg = load_layered_config(args.config)
    models = _extract_models(cfg)
    if not models:
        raise SystemExit("Configuration does not define any models.")

    print(" ".join(models))


if __name__ == "__main__":
    main()
