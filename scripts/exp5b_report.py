#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ssl4polyp.classification.analysis.exp5b_report import (
    discover_runs,
    summarize_runs,
    write_severity_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Experiment 5B robustness metrics across runs.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing *_metrics.json files (default: results)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full summary JSON. If omitted, a compact summary is printed.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write the per-severity table as CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model identifiers to include. Defaults to all discovered models.",
    )
    return parser.parse_args()


def _stringify_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _emit_summary(summary: dict[str, object], models: Sequence[str]) -> None:
    model_block = summary.get("models")
    if not isinstance(model_block, dict):
        print("No models found in summary payload.")
        return
    print("Experiment 5B robustness summary for models: " + ", ".join(models))
    for model in models:
        entry = model_block.get(model)
        if not isinstance(entry, dict):
            continue
        ausc_block = entry.get("ausc")
        if not isinstance(ausc_block, dict):
            continue
        print(f"\nModel: {model}")
        for family, metrics in sorted(ausc_block.items()):
            if not isinstance(metrics, dict):
                continue
            metrics_display = []
            for metric, stats in sorted(metrics.items()):
                if not isinstance(stats, dict):
                    continue
                mean = stats.get("mean")
                std = stats.get("std")
                if mean is None:
                    continue
                if std is None or not isinstance(std, (int, float)):
                    metrics_display.append(f"{metric}={mean:.4f}")
                else:
                    metrics_display.append(f"{metric}={mean:.4f}±{std:.4f}")
            if metrics_display:
                print(f"  {family}: " + "; ".join(metrics_display))


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.expanduser()
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root '{runs_root}' does not exist")
    runs = discover_runs(runs_root, models=args.models)
    if not runs:
        print("No matching runs with perturbation summaries were found.")
        return
    summary = summarize_runs(runs)
    if args.output_json is not None:
        output_json = args.output_json.expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote summary JSON to {_stringify_path(output_json)}")
    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser()
        write_severity_csv(summary, output_csv)
        print(f"Wrote severity table CSV to {_stringify_path(output_csv)}")
    if args.output_json is None:
        _emit_summary(summary, sorted(runs.keys()))


if __name__ == "__main__":
    main()
