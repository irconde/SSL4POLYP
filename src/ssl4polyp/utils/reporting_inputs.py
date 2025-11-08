"""Utilities for copying experiment artifacts into reporting directories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from ssl4polyp.configs.layered import load_layered_config


@dataclass(frozen=True)
class ReportingCopyResult:
    """Details about a copied artifact."""

    source: Path
    destination: Path


class ReportingInputsError(RuntimeError):
    """Raised when reporting inputs cannot be resolved."""


def _expand_path(value: Path) -> Path:
    return value.expanduser().resolve()


def _load_reporting_subdir(config_path: Path) -> Optional[str]:
    """Return the reporting inputs subdirectory declared in ``config_path``."""

    try:
        config = load_layered_config(str(config_path))
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise ReportingInputsError(f"Experiment config not found: {config_path}") from exc

    reporting_block = config.get("reporting")
    if isinstance(reporting_block, Mapping):
        subdir = reporting_block.get("inputs_subdir")
        if isinstance(subdir, str) and subdir.strip():
            return subdir.strip()
    legacy_subdir = config.get("reporting_inputs_subdir")
    if isinstance(legacy_subdir, str) and legacy_subdir.strip():
        return legacy_subdir.strip()
    return None


def _prefer_metrics(run_dir: Path) -> Sequence[Path]:
    last_candidates = list(run_dir.glob("*_last.metrics.json"))
    other_candidates = list(run_dir.glob("*.metrics.json"))
    unique: List[Path] = []
    for candidate in sorted(last_candidates + other_candidates):
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _load_metrics_payload(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ReportingInputsError(f"Metrics file {path} is not valid JSON") from exc
    except OSError as exc:  # pragma: no cover - defensive
        raise ReportingInputsError(f"Failed to read metrics file {path}") from exc
    if not isinstance(payload, Mapping):
        raise ReportingInputsError(f"Metrics file {path} does not contain a JSON object")
    return payload


def _resolve_outputs_path(metrics_path: Path, payload: Mapping[str, object]) -> Optional[Path]:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        rel_path = provenance.get("test_outputs_csv")
        if isinstance(rel_path, str) and rel_path.strip():
            candidate = Path(rel_path)
            if not candidate.is_absolute():
                candidate = metrics_path.parent / candidate
            return candidate
    # Fallback to conventional naming when provenance is missing.
    name = metrics_path.name
    if name.endswith("_last.metrics.json"):
        stem = name[: -len("_last.metrics.json")]
    elif name.endswith(".metrics.json"):
        stem = name[: -len(".metrics.json")]
    else:
        stem = metrics_path.stem
    candidate = metrics_path.with_name(f"{stem}_test_outputs.csv")
    if candidate.exists():
        return candidate
    csv_candidates = list(metrics_path.parent.glob("*_test_outputs.csv"))
    if len(csv_candidates) == 1:
        return csv_candidates[0]
    return None


def copy_reporting_inputs(
    run_dir: Path,
    reporting_root: Path,
    *,
    reporting_subdir: str,
) -> Sequence[ReportingCopyResult]:
    """Copy metrics and test outputs from ``run_dir`` into ``reporting_root``."""

    run_dir = _expand_path(run_dir)
    if not run_dir.is_dir():
        raise ReportingInputsError(f"Run directory does not exist: {run_dir}")
    reporting_root = _expand_path(reporting_root)
    destination_dir = reporting_root / reporting_subdir
    destination_dir.mkdir(parents=True, exist_ok=True)

    metrics_candidates = _prefer_metrics(run_dir)
    if not metrics_candidates:
        raise ReportingInputsError(
            f"No metrics exports were found in run directory {run_dir}"
        )

    errors: List[str] = []
    copies: List[ReportingCopyResult] = []
    for metrics_path in metrics_candidates:
        if not metrics_path.exists():  # pragma: no cover - defensive
            continue
        try:
            payload = _load_metrics_payload(metrics_path)
        except ReportingInputsError as exc:
            errors.append(str(exc))
            continue
        outputs_path = _resolve_outputs_path(metrics_path, payload)
        if outputs_path is None or not outputs_path.exists():
            errors.append(
                f"Test outputs CSV corresponding to {metrics_path.name} was not found"
            )
            continue
        dest_metrics = destination_dir / metrics_path.name
        dest_outputs = destination_dir / outputs_path.name
        shutil.copy2(metrics_path, dest_metrics)
        shutil.copy2(outputs_path, dest_outputs)
        copies.append(ReportingCopyResult(metrics_path, dest_metrics))
        copies.append(ReportingCopyResult(outputs_path, dest_outputs))
        break
    if not copies:
        joined_errors = "; ".join(errors) if errors else "unknown reason"
        raise ReportingInputsError(
            f"Failed to copy reporting inputs from {run_dir}: {joined_errors}"
        )
    return copies


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy *_last.metrics.json (or fallback metrics exports) and their corresponding "
            "*_test_outputs.csv files into results/reporting_inputs."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Output directory for a finished experiment run.",
    )
    parser.add_argument(
        "--exp-config",
        type=Path,
        help="Experiment configuration used for the run (for resolving reporting paths).",
    )
    parser.add_argument(
        "--reporting-root",
        type=Path,
        default=Path("results/reporting_inputs"),
        help="Root directory where reporting inputs are stored.",
    )
    parser.add_argument(
        "--reporting-subdir",
        type=str,
        default=None,
        help="Override the reporting inputs subdirectory (defaults to config value).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output.",
    )
    return parser.parse_args(argv)


def _resolve_reporting_subdir(
    *,
    explicit: Optional[str],
    config_path: Optional[Path],
) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if config_path is None:
        raise ReportingInputsError(
            "--reporting-subdir was not provided and experiment config is unavailable"
        )
    subdir = _load_reporting_subdir(config_path)
    if not subdir:
        raise ReportingInputsError(
            f"Experiment config {config_path} does not declare reporting.inputs_subdir"
        )
    return subdir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        reporting_subdir = _resolve_reporting_subdir(
            explicit=args.reporting_subdir,
            config_path=args.exp_config,
        )
        copies = copy_reporting_inputs(
            args.run_dir,
            args.reporting_root,
            reporting_subdir=reporting_subdir,
        )
    except ReportingInputsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        for entry in copies:
            print(f"Copied {entry.source} -> {entry.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
