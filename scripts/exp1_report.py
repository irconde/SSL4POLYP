#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ssl4polyp.classification.analysis.exp1_report import (  # type: ignore[import]
    DEFAULT_BOOTSTRAP,
    DEFAULT_RNG_SEED,
    Exp1Summary,
    build_manifest,
    discover_runs,
    render_markdown,
    summarize_runs,
    write_csv_tables,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Experiment 1 report (SUP-ImNet vs SSL-ImNet on SUN).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing *_last.metrics.json files (default: results)",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to write the markdown report. If omitted, the report is printed to stdout.",
    )
    parser.add_argument(
        "--output-csv-dir",
        type=Path,
        default=None,
        help="Optional directory to emit CSV tables (T1–T3).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write the manifest JSON with provenance and digests.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help=f"Number of bootstrap replicates for paired deltas (default: {DEFAULT_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help=f"Random seed for bootstrap resampling (default: {DEFAULT_RNG_SEED}).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the structured summary JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    runs_root = args.runs_root.expanduser()
    try:
        runs, loader = discover_runs(runs_root, return_loader=True)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        summary = summarize_runs(
            runs,
            bootstrap=max(0, args.bootstrap),
            rng_seed=args.rng_seed,
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    markdown_text = render_markdown(summary)
    if args.output_markdown is not None:
        output_path = args.output_markdown.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_text, encoding="utf-8")
    else:
        print(markdown_text, end="")

    extra_outputs = []
    if args.output_csv_dir is not None:
        csv_dir = args.output_csv_dir.expanduser()
        extra_outputs.extend(write_csv_tables(summary, csv_dir))

    if args.json is not None:
        json_path = args.json.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = summary.as_dict()
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        extra_outputs.append(json_path)

    if args.manifest is not None:
        manifest_path = args.manifest.expanduser()
        build_manifest(
            summary,
            loader=loader,
            manifest_path=manifest_path,
            output_path=args.output_markdown.expanduser() if args.output_markdown else None,
            extra_outputs=extra_outputs,
            rng_seed=args.rng_seed,
            bootstrap=args.bootstrap,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
