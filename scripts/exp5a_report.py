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

_exp5a_module = importlib.import_module("ssl4polyp.classification.analysis.exp5a_report")

DiscoverRunsFn = Callable[..., Mapping[str, Any]]
SummarizeRunsFn = Callable[..., Mapping[str, Any]]
WriteCsvFn = Callable[..., None]

discover_runs = cast(DiscoverRunsFn, _exp5a_module.discover_runs)
summarize_runs = cast(SummarizeRunsFn, _exp5a_module.summarize_runs)
write_performance_csv = cast(WriteCsvFn, _exp5a_module.write_performance_csv)
write_domain_shift_csv = cast(WriteCsvFn, _exp5a_module.write_domain_shift_csv)
write_seed_metrics_csv = cast(WriteCsvFn, _exp5a_module.write_seed_metrics_csv)
write_pairwise_csv = cast(WriteCsvFn, _exp5a_module.write_pairwise_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Experiment 5A PolypGen evaluation metrics across runs.",
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
        help="Optional path to write the aggregated summary JSON.",
    )
    parser.add_argument(
        "--performance-csv",
        type=Path,
        default=None,
        help="Optional path to export aggregated performance statistics as CSV.",
    )
    parser.add_argument(
        "--domain-shift-csv",
        type=Path,
        default=None,
        help="Optional path to export PolypGen − SUN domain-shift deltas as CSV.",
    )
    parser.add_argument(
        "--seed-metrics-csv",
        type=Path,
        default=None,
        help="Optional path to export per-seed metrics with bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        default=None,
        help="Optional path to export SSL-Colon vs baselines pairwise deltas.",
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
        help="Number of bootstrap iterations for confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=12345,
        help="Random seed used for bootstrap resampling (default: 12345).",
    )
    return parser.parse_args()


def _stringify_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _format_ci(ci: Mapping[str, Any] | None) -> str:
    if not isinstance(ci, Mapping):
        return "—"
    lower = ci.get("lower") if isinstance(ci.get("lower"), (int, float)) else None
    upper = ci.get("upper") if isinstance(ci.get("upper"), (int, float)) else None
    if lower is None or upper is None:
        return "—"
    return f"[{float(lower):+.3f},{float(upper):+.3f}]"


def _format_mean_std(stats: Mapping[str, Any] | None) -> str:
    if not isinstance(stats, Mapping):
        return "—"
    mean = stats.get("mean") if isinstance(stats.get("mean"), (int, float)) else None
    std = stats.get("std") if isinstance(stats.get("std"), (int, float)) else None
    if mean is None:
        return "—"
    if std is None:
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f}±{float(std):.3f}"


def _emit_summary(summary: Mapping[str, Any]) -> None:
    models_obj = summary.get("models")
    models_block: Mapping[str, Any]
    if isinstance(models_obj, Mapping):
        models_block = models_obj
    else:
        models_block = {}
    model_names = sorted(models_block.keys())
    print("Experiment 5A PolypGen evaluation summary")
    if not model_names:
        print("No models discovered.")
        return
    for model in model_names:
        payload = models_block.get(model)
        if not isinstance(payload, Mapping):
            continue
        performance = payload.get("performance") if isinstance(payload.get("performance"), Mapping) else {}
        domain_shift = payload.get("domain_shift") if isinstance(payload.get("domain_shift"), Mapping) else {}
        print(f"\nModel: {model}")
        if isinstance(performance, Mapping) and performance:
            print("  PolypGen metrics (mean±std across seeds):")
            for metric, stats in sorted(performance.items()):
                if isinstance(stats, Mapping):
                    print(f"    {metric}: {_format_mean_std(stats)}")
        if isinstance(domain_shift, Mapping) and domain_shift:
            print("  Domain-shift Δ (PolypGen − SUN) mean±std:")
            for metric, stats in sorted(domain_shift.items()):
                if isinstance(stats, Mapping):
                    print(f"    {metric}: {_format_mean_std(stats)}")
        seeds = payload.get("seeds") if isinstance(payload.get("seeds"), Mapping) else {}
        if not isinstance(seeds, Mapping) or not seeds:
            continue
        print("  Seed breakdown:")
        for seed, seed_payload in sorted(seeds.items()):
            if not isinstance(seed_payload, Mapping):
                continue
            metrics_block = seed_payload.get("metrics") if isinstance(seed_payload.get("metrics"), Mapping) else {}
            ci_block = seed_payload.get("ci") if isinstance(seed_payload.get("ci"), Mapping) else {}
            delta_block = seed_payload.get("delta") if isinstance(seed_payload.get("delta"), Mapping) else {}
            delta_ci_block = seed_payload.get("delta_ci") if isinstance(seed_payload.get("delta_ci"), Mapping) else {}
            print(f"    Seed {seed} (τ={seed_payload.get('tau')}):")
            for metric in ("auprc", "auroc", "f1", "recall"):
                value = metrics_block.get(metric) if isinstance(metrics_block, Mapping) else None
                ci = ci_block.get(metric) if isinstance(ci_block, Mapping) else None
                if isinstance(value, (int, float)):
                    print(f"      {metric}: {float(value):.3f} {_format_ci(ci)}")
            for metric in ("auprc", "f1"):
                delta_val = delta_block.get(metric) if isinstance(delta_block, Mapping) else None
                delta_ci = delta_ci_block.get(metric) if isinstance(delta_ci_block, Mapping) else None
                if isinstance(delta_val, (int, float)):
                    print(f"      Δ{metric}: {float(delta_val):+.3f} {_format_ci(delta_ci)}")
    pairwise = summary.get("pairwise") if isinstance(summary.get("pairwise"), Mapping) else {}
    if isinstance(pairwise, Mapping) and pairwise:
        print("\nPairwise Δ(SSL-Colon − baseline):")
        for metric, baselines in sorted(pairwise.items()):
            if not isinstance(baselines, Mapping):
                continue
            print(f"  Metric: {metric}")
            for baseline, payload in sorted(baselines.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_block = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else None
                summary_ci = payload.get("summary_ci") if isinstance(payload.get("summary_ci"), Mapping) else None
                delta_display = _format_mean_std(summary_block)
                ci_display = _format_ci(summary_ci)
                print(f"    {baseline}: {delta_display} {ci_display}")
                seeds_block = payload.get("seeds") if isinstance(payload.get("seeds"), Sequence) else None
                if not seeds_block:
                    continue
                fragments = []
                for entry in seeds_block:
                    if not isinstance(entry, Mapping):
                        continue
                    seed_id = entry.get("seed")
                    delta_val = entry.get("delta")
                    ci = entry.get("ci") if isinstance(entry.get("ci"), Mapping) else None
                    if isinstance(seed_id, int) and isinstance(delta_val, (int, float)):
                        fragments.append(
                            f"seed={seed_id}:{float(delta_val):+.3f} {_format_ci(ci)}"
                        )
                if fragments:
                    print("      " + ", ".join(fragments))


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
    if args.domain_shift_csv is not None:
        path = args.domain_shift_csv.expanduser()
        write_domain_shift_csv(summary, path)
        print(f"Wrote domain-shift table to {_stringify_path(path)}")
    if args.seed_metrics_csv is not None:
        path = args.seed_metrics_csv.expanduser()
        write_seed_metrics_csv(summary, path)
        print(f"Wrote per-seed metrics table to {_stringify_path(path)}")
    if args.pairwise_csv is not None:
        path = args.pairwise_csv.expanduser()
        write_pairwise_csv(summary, path)
        print(f"Wrote pairwise delta table to {_stringify_path(path)}")
    if args.output_json is None:
        _emit_summary(summary)


if __name__ == "__main__":
    main()
