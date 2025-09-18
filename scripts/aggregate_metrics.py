#!/usr/bin/env python
"""Aggregate per-seed metric exports and compute summary statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def _load_metric_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".json" and p.is_file())


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty sample")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    fraction = pos - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _aggregate_metric(values: Iterable[float], ci: float, bootstrap: int, rng_seed: int) -> Mapping[str, float]:
    samples = [float(v) for v in values]
    n = len(samples)
    if n == 0:
        raise ValueError("Cannot aggregate an empty set of values")
    mean = float(statistics.fmean(samples))
    std = float(statistics.stdev(samples)) if n > 1 else 0.0
    if n == 1 or bootstrap <= 0:
        lower = upper = mean
    else:
        rng = random.Random(rng_seed)
        boot_means = []
        for _ in range(bootstrap):
            draw = [samples[rng.randrange(n)] for _ in range(n)]
            boot_means.append(float(statistics.fmean(draw)))
        boot_means.sort()
        alpha = (1.0 - ci) / 2.0
        lower = float(_quantile(boot_means, alpha))
        upper = float(_quantile(boot_means, 1.0 - alpha))
    return {"mean": mean, "std": std, "ci_lower": lower, "ci_upper": upper}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed metrics into summary statistics",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/classification"),
        help="Directory containing per-experiment seed metrics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for the aggregated CSV (defaults to metrics-root/summary.csv)",
    )
    parser.add_argument(
        "--metric-key",
        choices=["test", "val"],
        default="test",
        help="Which metric block to aggregate (default: test)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples to compute confidence intervals (default: 1000)",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals (default: 0.95)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=12345,
        help="Seed for the bootstrap RNG (default: 12345)",
    )
    args = parser.parse_args()

    metrics_root = args.metrics_root
    output_path = args.output if args.output is not None else metrics_root / "summary.csv"

    experiments: Dict[str, Dict[str, List[float]]] = {}
    seed_registry: Dict[str, List[int]] = {}

    if not metrics_root.exists():
        raise FileNotFoundError(f"Metrics root directory does not exist: {metrics_root}")

    for experiment_dir in sorted(p for p in metrics_root.iterdir() if p.is_dir()):
        metric_values: Dict[str, List[float]] = defaultdict(list)
        seeds: List[int] = []
        for metrics_path in _load_metric_files(experiment_dir):
            with open(metrics_path, "r") as handle:
                payload = json.load(handle)
            seed = int(payload.get("seed")) if payload.get("seed") is not None else None
            if seed is not None:
                seeds.append(seed)
            block = payload.get(args.metric_key, {})
            for metric_name, value in block.items():
                metric_values[metric_name].append(float(value))
        if metric_values:
            experiments[experiment_dir.name] = metric_values
            seed_registry[experiment_dir.name] = sorted(set(seeds))

    if not experiments:
        raise RuntimeError(f"No metrics found in {metrics_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "metric",
                "n",
                "mean",
                "std",
                "ci_lower",
                "ci_upper",
                "seeds",
            ],
        )
        writer.writeheader()
        for experiment, metrics in experiments.items():
            for metric_name, values in sorted(metrics.items()):
                summary = _aggregate_metric(values, args.ci, args.bootstrap, args.rng_seed)
                writer.writerow(
                    {
                        "experiment": experiment,
                        "metric": metric_name,
                        "n": len(values),
                        "mean": summary["mean"],
                        "std": summary["std"],
                        "ci_lower": summary["ci_lower"],
                        "ci_upper": summary["ci_upper"],
                        "seeds": ",".join(str(s) for s in seed_registry.get(experiment, [])),
                    }
                )


if __name__ == "__main__":
    main()
