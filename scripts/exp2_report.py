#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from ssl4polyp.classification.analysis.exp2_report import (  # type: ignore[import]
    DEFAULT_BOOTSTRAP,
    DEFAULT_RNG_SEED,
    collect_summary,
    render_markdown,
    write_csv_tables,
    build_manifest,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Experiment 2 SUN report (SSL-ImNet vs SSL-Colon)."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory containing *_last.metrics.json files (default: checkpoints).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help=f"Number of bootstrap iterations for paired deltas (default: {DEFAULT_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help=f"Random seed for bootstrap resampling (default: {DEFAULT_RNG_SEED}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the markdown report. If omitted, the report is printed.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        help="Optional directory to write CSV tables (T1/T2/T3 and appendix).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to dump the summary payload as JSON.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path to write a report manifest capturing guardrail digests.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict guardrail enforcement (not recommended).",
    )
    parser.set_defaults(strict=True)
    return parser.parse_args(argv)


def _collect_extra_outputs(*paths: Optional[Path]) -> Iterable[Path]:
    for path in paths:
        if path is None:
            continue
        if path.exists():
            yield path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    runs, summary, loader = collect_summary(
        args.runs_root,
        bootstrap=max(0, args.bootstrap),
        rng_seed=args.seed,
        strict=args.strict,
    )
    if not runs:
        print(f"No Experiment 2 runs found under {args.runs_root}")
        return 0
    if summary is None:
        print("Unable to construct Experiment 2 summary.")
        return 1

    report_text = render_markdown(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text, encoding="utf-8")
    else:
        print(report_text)

    tables: Dict[str, Path] = {}
    if args.tables_dir:
        tables = write_csv_tables(summary, args.tables_dir)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")

    extra_outputs = list(
        _collect_extra_outputs(
            args.output,
            args.json,
            *(tables.values()),
        )
    )

    if args.manifest:
        build_manifest(
            summary,
            loader=loader,
            manifest_path=args.manifest,
            output_path=args.output,
            extra_outputs=extra_outputs,
            rng_seed=args.seed,
            bootstrap=max(0, args.bootstrap),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
