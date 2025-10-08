#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_exp5c_module = importlib.import_module("ssl4polyp.classification.analysis.exp5c_report")
from ssl4polyp.classification.analysis.display import (  # type: ignore[import]
    PLACEHOLDER,
    format_ci,
    format_mean_std,
    format_signed,
)

DiscoverRunsFn = Callable[..., Mapping[str, Any]]
SummarizeRunsFn = Callable[..., Mapping[str, Any]]
WriteCsvFn = Callable[..., None]

discover_runs = cast(DiscoverRunsFn, _exp5c_module.discover_runs)
summarize_runs = cast(SummarizeRunsFn, _exp5c_module.summarize_runs)
write_performance_csv = cast(WriteCsvFn, _exp5c_module.write_performance_csv)
write_learning_curve_csv = cast(WriteCsvFn, _exp5c_module.write_learning_curve_csv)
write_pairwise_csv = cast(WriteCsvFn, _exp5c_module.write_pairwise_csv)
write_aulc_csv = cast(WriteCsvFn, _exp5c_module.write_aulc_csv)
write_sample_efficiency_csv = cast(WriteCsvFn, _exp5c_module.write_sample_efficiency_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Experiment 5C few-shot adaptation metrics across runs.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing *_last.metrics.json files (default: results)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full summary JSON.",
    )
    parser.add_argument(
        "--performance-csv",
        type=Path,
        default=None,
        help="Optional path to export aggregated performance statistics as CSV.",
    )
    parser.add_argument(
        "--learning-curve-csv",
        type=Path,
        default=None,
        help="Optional path to export learning-curve tables as CSV.",
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        default=None,
        help="Optional path to export pairwise delta summaries as CSV.",
    )
    parser.add_argument(
        "--aulc-csv",
        type=Path,
        default=None,
        help="Optional path to export area-under-learning-curve summaries as CSV.",
    )
    parser.add_argument(
        "--sample-efficiency-csv",
        type=Path,
        default=None,
        help="Optional path to export S@target sample-efficiency summaries as CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model identifiers to include. Defaults to all discovered models.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for paired comparisons (default: 1000).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=12345,
        help="Random seed used for bootstrap resampling (default: 12345).",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="ssl_imnet",
        help="Model used as the S@target reference (default: ssl_imnet).",
    )
    parser.add_argument(
        "--target-budget",
        type=int,
        default=500,
        help="Budget S used for the S@target reference (default: 500).",
    )
    return parser.parse_args()


def _stringify_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _format_mean_std(stats: Mapping[str, object]) -> str:
    mean = stats.get("mean") if isinstance(stats, Mapping) else None
    std = stats.get("std") if isinstance(stats, Mapping) else None
    if not isinstance(mean, (int, float)):
        return PLACEHOLDER
    return format_mean_std(mean, std if isinstance(std, (int, float)) else None)


def _emit_summary(summary: Mapping[str, object], models: Sequence[str]) -> None:
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), Mapping) else {}
    budgets = metadata.get("budgets") if isinstance(metadata, Mapping) else None
    budgets_seq = list(budgets) if isinstance(budgets, Sequence) else None
    performance = summary.get("performance") if isinstance(summary.get("performance"), Mapping) else {}
    print("Experiment 5C few-shot adaptation summary")
    if budgets_seq:
        print("Budgets (S): " + ", ".join(str(b) for b in budgets_seq))
    for metric in ("auprc", "f1"):
        print(f"\n{metric.upper()} mean±std by budget:")
        for model in models:
            per_budget = performance.get(model) if isinstance(performance, Mapping) else None
            if not isinstance(per_budget, Mapping):
                continue
            entries = []
            for budget in (budgets_seq or sorted(per_budget.keys())):
                stats_block = per_budget.get(budget)
                if not isinstance(stats_block, Mapping):
                    continue
                metrics_block = stats_block.get("metrics")
                if not isinstance(metrics_block, Mapping):
                    continue
                metric_stats = metrics_block.get(metric)
                if not isinstance(metric_stats, Mapping):
                    continue
                entries.append(f"S={budget}:{_format_mean_std(metric_stats)}")
            if entries:
                print(f"  {model}: " + ", ".join(entries))
    pairwise = summary.get("pairwise") if isinstance(summary.get("pairwise"), Mapping) else {}
    pairwise_metric = pairwise.get("auprc") if isinstance(pairwise, Mapping) else None
    if isinstance(pairwise_metric, Mapping):
        print("\nΔAUPRC (SSL-Colon minus baseline) with 95% CI:")
        for baseline, per_budget in pairwise_metric.items():
            if not isinstance(per_budget, Mapping):
                continue
            fragments = []
            for budget, stats in sorted(per_budget.items()):
                if not isinstance(stats, Mapping):
                    continue
                delta = stats.get("delta")
                ci = stats.get("ci") if isinstance(stats.get("ci"), Mapping) else None
                if not isinstance(delta, (int, float)):
                    continue
                delta_text = format_signed(delta)
                ci_text = PLACEHOLDER
                if isinstance(ci, Mapping):
                    lower = ci.get("lower") if isinstance(ci.get("lower"), (int, float)) else None
                    upper = ci.get("upper") if isinstance(ci.get("upper"), (int, float)) else None
                    if lower is not None and upper is not None:
                        ci_text = format_ci(lower, upper)
                if ci_text != PLACEHOLDER:
                    fragments.append(f"S={budget}:{delta_text} {ci_text}")
                else:
                    fragments.append(f"S={budget}:{delta_text}")
            if fragments:
                print(f"  {baseline}: " + ", ".join(fragments))
    sample_eff = summary.get("sample_efficiency") if isinstance(summary.get("sample_efficiency"), Mapping) else {}
    target_block = sample_eff.get("auprc") if isinstance(sample_eff, Mapping) else None
    if isinstance(target_block, Mapping):
        target_model = metadata.get("target_model") if isinstance(metadata, Mapping) else None
        target_budget = metadata.get("target_budget") if isinstance(metadata, Mapping) else None
        print("\nS@target (AUPRC target from {model} at S={budget}):".format(model=target_model, budget=target_budget))
        for model in models:
            stats = target_block.get(model)
            if not isinstance(stats, Mapping):
                continue
            budget = stats.get("budget")
            display = str(budget) if isinstance(budget, int) else "—"
            print(f"  {model}: {display}")


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.expanduser()
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root '{runs_root}' does not exist")
    runs = discover_runs(runs_root, models=args.models)
    if not runs:
        print("No matching runs with metrics were found.")
        return
    summary = summarize_runs(
        runs,
        bootstrap=max(0, args.bootstrap),
        rng_seed=args.rng_seed,
        target_model=args.target_model,
        target_budget=args.target_budget,
    )
    if args.output_json is not None:
        output_json = args.output_json.expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote summary JSON to {_stringify_path(output_json)}")
    if args.performance_csv is not None:
        path = args.performance_csv.expanduser()
        write_performance_csv(summary, path)
        print(f"Wrote performance table to {_stringify_path(path)}")
    if args.learning_curve_csv is not None:
        path = args.learning_curve_csv.expanduser()
        write_learning_curve_csv(summary, path)
        print(f"Wrote learning-curve table to {_stringify_path(path)}")
    if args.pairwise_csv is not None:
        path = args.pairwise_csv.expanduser()
        write_pairwise_csv(summary, path)
        print(f"Wrote pairwise delta table to {_stringify_path(path)}")
    if args.aulc_csv is not None:
        path = args.aulc_csv.expanduser()
        write_aulc_csv(summary, path)
        print(f"Wrote AULC summary to {_stringify_path(path)}")
    if args.sample_efficiency_csv is not None:
        path = args.sample_efficiency_csv.expanduser()
        write_sample_efficiency_csv(summary, path)
        print(f"Wrote sample-efficiency table to {_stringify_path(path)}")
    if args.output_json is None:
        _emit_summary(summary, sorted(runs.keys()))


if __name__ == "__main__":
    main()
