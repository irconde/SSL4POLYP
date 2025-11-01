from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

from ssl4polyp.classification.analysis.exp4_report import collect_summary, render_report
from ssl4polyp.classification.analysis.result_loader import build_report_manifest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "Bootstrap iterations must be a positive integer (>= 1)."
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Experiment 4 learning-curve and delta summary report.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing *_last.metrics.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write the markdown report. If omitted, the report is printed.",
    )
    parser.add_argument(
        "--bootstrap",
        type=_positive_int,
        default=1000,
        help=(
            "Number of bootstrap iterations for confidence intervals (default: 1000). "
            "Must be a positive integer."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for bootstrap resampling (default: 12345).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional path to write a report manifest JSON capturing guardrail digests and output hashes."
        ),
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict guardrail enforcement (not recommended).",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help=(
            "Enable INFO-level logging to display bootstrap progress updates during report generation."
        ),
    )
    parser.set_defaults(strict=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_progress:
        logging.basicConfig(level=logging.INFO)
    runs, summary, loader = collect_summary(
        args.runs_root,
        bootstrap=args.bootstrap,
        rng_seed=args.seed,
        strict=args.strict,
    )
    if runs:
        report = render_report(summary)
    else:
        report = "No Experiment 4 runs found.\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.manifest is not None:
        validated_seeds = summary.get("validated_seeds") if summary else None
        seed_groups = summary.get("seed_groups") if summary else None
        manifest = build_report_manifest(
            output_path=args.output,
            loader=loader,
            runs=loader.loaded_runs,
            rng_seed=args.seed,
            bootstrap=args.bootstrap,
            validated_seeds=validated_seeds,
            seed_groups=seed_groups if isinstance(seed_groups, dict) else None,
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
