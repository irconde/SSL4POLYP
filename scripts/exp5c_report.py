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
write_test_composition_csv = cast(WriteCsvFn, _exp5c_module.write_test_composition_csv)
write_performance_csv = cast(WriteCsvFn, _exp5c_module.write_performance_csv)
write_gain_csv = cast(WriteCsvFn, _exp5c_module.write_gain_csv)
write_pairwise_csv = cast(WriteCsvFn, _exp5c_module.write_pairwise_csv)
write_aulc_csv = cast(WriteCsvFn, _exp5c_module.write_aulc_csv)


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
        "--t1-csv",
        type=Path,
        default=None,
        help="Optional path to export the T1 test composition table.",
    )
    parser.add_argument(
        "--t2-primary-csv",
        type=Path,
        default=None,
        help="Optional path to export the T2 primary-threshold performance table.",
    )
    parser.add_argument(
        "--t2-sensitivity-csv",
        type=Path,
        default=None,
        help="Optional path to export the sensitivity-threshold performance table (T6).",
    )
    parser.add_argument(
        "--t3-gain-csv",
        type=Path,
        default=None,
        help="Optional path to export the adaptation gain vs zero-shot table (T3).",
    )
    parser.add_argument(
        "--t4-primary-csv",
        type=Path,
        default=None,
        help="Optional path to export the primary post-adaptation delta table (T4).",
    )
    parser.add_argument(
        "--t4-sensitivity-csv",
        type=Path,
        default=None,
        help="Optional path to export the sensitivity post-adaptation delta table (T6).",
    )
    parser.add_argument(
        "--t5-aulc-csv",
        type=Path,
        default=None,
        help="Optional path to export the AULC and ΔAULC table (T5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to emit all standard Experiment 5C tables (T1–T5/T6).",
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
        default=2000,
        help="Number of bootstrap iterations for paired comparisons (default: 2000).",
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
    policies = summary.get("policies") if isinstance(summary.get("policies"), Mapping) else {}
    primary = policies.get("primary") if isinstance(policies, Mapping) else None
    performance = primary.get("performance") if isinstance(primary, Mapping) else {}
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
    pairwise = primary.get("pairwise") if isinstance(primary, Mapping) else {}
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
    gains = summary.get("gains") if isinstance(summary.get("gains"), Mapping) else {}
    if isinstance(gains, Mapping) and gains:
        print("\nAdaptation gain vs zero-shot (AUPRC) mean±std:")
        for model in sorted(gains.keys()):
            per_budget = gains.get(model)
            if not isinstance(per_budget, Mapping):
                continue
            fragments = []
            for budget, metrics in sorted(per_budget.items()):
                metric_stats = metrics.get("auprc") if isinstance(metrics, Mapping) else None
                if not isinstance(metric_stats, Mapping):
                    continue
                fragments.append(f"S={budget}:{_format_mean_std(metric_stats)}")
            if fragments:
                print(f"  {model}: " + ", ".join(fragments))


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
    if args.t1_csv is not None:
        path = args.t1_csv.expanduser()
        write_test_composition_csv(summary, path)
        print(f"Wrote T1 test composition table to {_stringify_path(path)}")
    if args.t2_primary_csv is not None:
        path = args.t2_primary_csv.expanduser()
        write_performance_csv(summary, path, policy="primary")
        print(f"Wrote T2 primary performance table to {_stringify_path(path)}")
    if args.t2_sensitivity_csv is not None:
        path = args.t2_sensitivity_csv.expanduser()
        write_performance_csv(summary, path, policy="sensitivity")
        print(f"Wrote sensitivity performance table to {_stringify_path(path)}")
    if args.t3_gain_csv is not None:
        path = args.t3_gain_csv.expanduser()
        write_gain_csv(summary, path)
        print(f"Wrote T3 adaptation gain table to {_stringify_path(path)}")
    if args.t4_primary_csv is not None:
        path = args.t4_primary_csv.expanduser()
        write_pairwise_csv(summary, path, policy="primary")
        print(f"Wrote T4 primary delta table to {_stringify_path(path)}")
    if args.t4_sensitivity_csv is not None:
        path = args.t4_sensitivity_csv.expanduser()
        write_pairwise_csv(summary, path, policy="sensitivity")
        print(f"Wrote sensitivity delta table to {_stringify_path(path)}")
    if args.t5_aulc_csv is not None:
        path = args.t5_aulc_csv.expanduser()
        write_aulc_csv(summary, path, policy="primary")
        print(f"Wrote T5 AULC table to {_stringify_path(path)}")
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "T1 test composition": output_dir / "T1_test_comp.csv",
            "T2 primary performance": output_dir / "T2_perf_primary.csv",
            "T2 sensitivity performance": output_dir / "T2_perf_sensitivity.csv",
            "T3 gain": output_dir / "T3_gain_vs_zeroshot.csv",
            "T4 primary deltas": output_dir / "T4_postadapt_deltas.csv",
            "T4 sensitivity deltas": output_dir / "T4_postadapt_sensitivity.csv",
            "T5 AULC": output_dir / "T5_aulc_primary.csv",
        }
        write_test_composition_csv(summary, artifacts["T1 test composition"])
        write_performance_csv(summary, artifacts["T2 primary performance"], policy="primary")
        write_performance_csv(summary, artifacts["T2 sensitivity performance"], policy="sensitivity")
        write_gain_csv(summary, artifacts["T3 gain"])
        write_pairwise_csv(summary, artifacts["T4 primary deltas"], policy="primary")
        write_pairwise_csv(summary, artifacts["T4 sensitivity deltas"], policy="sensitivity")
        write_aulc_csv(summary, artifacts["T5 AULC"], policy="primary")
        for label, path in artifacts.items():
            print(f"Wrote {label} table to {_stringify_path(path)}")
    if args.output_json is None:
        _emit_summary(summary, sorted(runs.keys()))


if __name__ == "__main__":
    main()
