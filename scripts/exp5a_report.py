#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ssl4polyp.classification.analysis.display import (  # type: ignore[import]
    PLACEHOLDER,
    format_ci,
    format_mean_std,
    format_scalar,
    format_signed,
)

import importlib

_exp5a_module = importlib.import_module("ssl4polyp.classification.analysis.exp5a_report")

DiscoverRunsFn = Callable[..., Mapping[str, Any]]
SummarizeRunsFn = Callable[..., Mapping[str, Any]]
WriteCsvFn = Callable[..., None]

discover_runs = cast(DiscoverRunsFn, _exp5a_module.discover_runs)
summarize_runs = cast(SummarizeRunsFn, _exp5a_module.summarize_runs)
write_performance_csv = cast(WriteCsvFn, _exp5a_module.write_performance_csv)
write_domain_shift_csv = cast(WriteCsvFn, _exp5a_module.write_domain_shift_csv)
write_composition_csv = cast(WriteCsvFn, _exp5a_module.write_composition_csv)
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
        "--composition-csv",
        type=Path,
        default=None,
        help="Optional path to export PolypGen composition counts as CSV.",
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
        default=2000,
        help="Number of bootstrap iterations for confidence intervals (default: 2000).",
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
        return PLACEHOLDER
    lower = ci.get("lower") if isinstance(ci.get("lower"), (int, float)) else None
    upper = ci.get("upper") if isinstance(ci.get("upper"), (int, float)) else None
    if lower is None or upper is None:
        return PLACEHOLDER
    return format_ci(lower, upper)


def _format_mean_std(stats: Mapping[str, Any] | None) -> str:
    if not isinstance(stats, Mapping):
        return PLACEHOLDER
    mean = stats.get("mean") if isinstance(stats.get("mean"), (int, float)) else None
    std = stats.get("std") if isinstance(stats.get("std"), (int, float)) else None
    if mean is None:
        return PLACEHOLDER
    return format_mean_std(mean, std)


def _emit_summary(summary: Mapping[str, Any]) -> None:
    composition = summary.get("composition") if isinstance(summary.get("composition"), Mapping) else None
    if isinstance(composition, Mapping) and composition:
        print("PolypGen test composition:")
        n_pos = composition.get("n_pos")
        n_neg = composition.get("n_neg")
        total = composition.get("total")
        prevalence = composition.get("prevalence")
        sha256 = composition.get("sha256")
        base_parts = []
        if isinstance(n_pos, (int, float)):
            base_parts.append(f"n_pos={int(n_pos)}")
        if isinstance(n_neg, (int, float)):
            base_parts.append(f"n_neg={int(n_neg)}")
        if isinstance(total, (int, float)):
            base_parts.append(f"total={int(total)}")
        if isinstance(prevalence, (int, float)):
            base_parts.append(f"prevalence={prevalence:.4f}")
        if isinstance(sha256, str):
            base_parts.append(f"sha256={sha256}")
        if base_parts:
            print("  " + ", ".join(base_parts))
        per_center = composition.get("per_center") if isinstance(composition.get("per_center"), Mapping) else None
        if isinstance(per_center, Mapping) and per_center:
            print("  Per-center counts:")
            for center, payload in sorted(per_center.items()):
                if not isinstance(payload, Mapping):
                    continue
                center_parts = [f"{center}:"]
                n_pos_center = payload.get("n_pos")
                n_neg_center = payload.get("n_neg")
                total_center = payload.get("total")
                prevalence_center = payload.get("prevalence")
                if isinstance(n_pos_center, (int, float)):
                    center_parts.append(f"n_pos={int(n_pos_center)}")
                if isinstance(n_neg_center, (int, float)):
                    center_parts.append(f"n_neg={int(n_neg_center)}")
                if isinstance(total_center, (int, float)):
                    center_parts.append(f"total={int(total_center)}")
                if isinstance(prevalence_center, (int, float)):
                    center_parts.append(f"prevalence={prevalence_center:.4f}")
                print("    " + " ".join(center_parts))
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
                    value_text = format_scalar(value)
                    ci_text = _format_ci(ci)
                    if ci_text != PLACEHOLDER:
                        print(f"      {metric}: {value_text} {ci_text}")
                    else:
                        print(f"      {metric}: {value_text}")
            for metric in ("auprc", "f1"):
                delta_val = delta_block.get(metric) if isinstance(delta_block, Mapping) else None
                delta_ci = delta_ci_block.get(metric) if isinstance(delta_ci_block, Mapping) else None
                if isinstance(delta_val, (int, float)):
                    delta_text = format_signed(delta_val)
                    ci_text = _format_ci(delta_ci)
                    if ci_text != PLACEHOLDER:
                        print(f"      Δ{metric}: {delta_text} {ci_text}")
                    else:
                        print(f"      Δ{metric}: {delta_text}")
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
                if ci_display != PLACEHOLDER:
                    print(f"    {baseline}: {delta_display} {ci_display}")
                else:
                    print(f"    {baseline}: {delta_display}")
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
                        delta_text = format_signed(delta_val)
                        ci_text = _format_ci(ci)
                        if ci_text != PLACEHOLDER:
                            fragments.append(f"seed={seed_id}:{delta_text} {ci_text}")
                        else:
                            fragments.append(f"seed={seed_id}:{delta_text}")
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
    if args.composition_csv is not None:
        path = args.composition_csv.expanduser()
        write_composition_csv(summary, path)
        print(f"Wrote composition table to {_stringify_path(path)}")
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
