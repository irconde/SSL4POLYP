#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ssl4polyp.classification.analysis.exp2_report import (  # type: ignore[import]
    DEFAULT_PAIRED_MODELS,
    PRIMARY_METRICS,
    Exp2Summary,
    discover_runs,
    summarize_runs,
)
from ssl4polyp.classification.analysis.display import (  # type: ignore[import]
    PLACEHOLDER,
    format_ci,
    format_mean_std,
    format_signed,
)

_METRIC_LABELS = {
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "balanced_accuracy": "Balanced Acc",
    "mcc": "MCC",
}


def _format_model_block(model: str, summary: Exp2Summary) -> Iterable[str]:
    metrics = summary.model_metrics.get(model)
    if not metrics:
        return []
    lines = [f"Model: {model}"]
    for metric in PRIMARY_METRICS:
        aggregate = metrics.get(metric)
        if not aggregate:
            continue
        label = _METRIC_LABELS.get(metric, metric)
        mean_std = format_mean_std(aggregate.mean, aggregate.std)
        lines.append(f"  {label}: {mean_std} (n={aggregate.n})")
    return lines


def _format_delta_block(
    paired_models: Optional[Sequence[str]],
    summary: Exp2Summary,
) -> Iterable[str]:
    if not paired_models or len(paired_models) != 2:
        return []
    header = f"Paired deltas ({paired_models[0]} − {paired_models[1]})"
    lines = [header]
    for metric in PRIMARY_METRICS:
        delta = summary.paired_deltas.get(metric)
        if delta is None:
            continue
        label = _METRIC_LABELS.get(metric, metric)
        mean_text = format_signed(delta.mean)
        ci_text = (
            format_ci(delta.ci_lower, delta.ci_upper)
            if delta.ci_lower is not None and delta.ci_upper is not None
            else PLACEHOLDER
        )
        if ci_text != PLACEHOLDER:
            lines.append(f"  {label}: {mean_text} (95% CI: {ci_text})")
        else:
            lines.append(f"  {label}: {mean_text}")
        per_seed_parts = [
            f"s{seed}={format_signed(value)}" for seed, value in sorted(delta.per_seed.items())
        ]
        if per_seed_parts:
            lines.append("    per-seed: " + ", ".join(per_seed_parts))
    return lines


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise Experiment 2 results across seeds.")
    parser.add_argument(
        "root",
        nargs="?",
        default="checkpoints",
        help="Root directory to search for metrics JSON files (default: checkpoints)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional list of model names to include (defaults to all discovered models).",
    )
    parser.add_argument(
        "--paired-models",
        nargs=2,
        metavar=("TREATMENT", "BASELINE"),
        help="Models to compare for paired deltas (default: ssl_colon vs ssl_imnet).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap replicates for paired deltas (default: 2000).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=20240521,
        help="Seed for bootstrap resampling (default: 20240521).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write the summary as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root)
    paired_models = tuple(args.paired_models) if args.paired_models else DEFAULT_PAIRED_MODELS
    try:
        runs = discover_runs(root, models=args.models)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not runs:
        print("No runs discovered under", root)
        return 1
    summary = summarize_runs(
        runs,
        paired_models=paired_models,
        bootstrap=args.bootstrap,
        rng_seed=args.rng_seed,
    )
    blocks: list[str] = []
    for model in sorted(summary.model_metrics.keys()):
        blocks.extend(_format_model_block(model, summary))
    delta_lines = list(_format_delta_block(paired_models, summary))
    if delta_lines:
        blocks.append("")
        blocks.extend(delta_lines)
    print("\n".join(blocks))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
