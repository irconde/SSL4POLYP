#!/usr/bin/env python3
"""Print the canonical seed list for an experiment configuration."""

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
    from ssl4polyp.configs.layered import load_layered_config
except ModuleNotFoundError as exc:  # pragma: no cover - dependency issues
    if exc.name == "ssl4polyp":
        raise SystemExit(
            "Unable to import 'ssl4polyp'. Ensure the repository is installed or "
            "the PYTHONPATH includes its root before running this helper."
        ) from exc
    raise


def _normalize_seeds(raw: Iterable[int | str]) -> List[int]:
    seeds: List[int] = []
    for item in raw:
        if isinstance(item, int):
            seeds.append(item)
        elif isinstance(item, str):
            parts = [p for p in item.replace(",", " ").split() if p]
            seeds.extend(int(p) for p in parts)
        else:
            raise TypeError(f"Unsupported seed value type: {type(item)!r}")
    return seeds


def _extract_seeds(config: dict) -> List[int]:
    if "seeds" in config and config["seeds"] is not None:
        raw = config["seeds"]
        if isinstance(raw, (int, str)):
            return _normalize_seeds([raw])
        return _normalize_seeds(raw)

    if "seed" in config and config["seed"] is not None:
        raw = config["seed"]
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            return _normalize_seeds(raw)  # type: ignore[arg-type]
        return _normalize_seeds([raw])

    protocol = config.get("protocol", {}) or {}
    for key in ("seeds", "subset_seeds"):
        if key in protocol and protocol[key] is not None:
            raw = protocol[key]
            if isinstance(raw, (int, str)):
                return _normalize_seeds([raw])
            return _normalize_seeds(raw)

    dataset = config.get("dataset", {}) or {}
    for key in ("seeds", "seed"):
        if key in dataset and dataset[key] is not None:
            raw = dataset[key]
            if isinstance(raw, (int, str)):
                return _normalize_seeds([raw])
            return _normalize_seeds(raw)

    raise ValueError("Configuration does not define a seed list.")


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
    seeds = _extract_seeds(cfg)
    if not seeds:
        raise SystemExit("Configuration does not define any seeds.")

    print(" ".join(str(seed) for seed in seeds))


if __name__ == "__main__":
    main()
