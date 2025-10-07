from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

# Metrics used for retention/drop calculations and AUSC summaries.
RETENTION_METRICS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
)


@dataclass(frozen=True)
class TagInfo:
    """Metadata describing how a perturbation tag maps to a severity family."""

    family: str
    raw_severity: float
    normalized_severity: float = 0.0


@dataclass
class RunPerturbationResult:
    """Structured record representing a single run's perturbation metrics."""

    model: str
    seed: int
    tau: Optional[float]
    metrics: Dict[str, float]
    perturbations: Dict[str, Dict[str, float]]
    provenance: Dict[str, object]
    path: Path


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _parse_fixed_point(value: str) -> float:
    text = value.strip().lower().replace("p", ".")
    try:
        numeric = float(text)
    except ValueError:
        return float("nan")
    return numeric


def _parse_tag(tag: str) -> TagInfo:
    label = str(tag).strip()
    if not label:
        return TagInfo(family="unknown", raw_severity=float("nan"))
    if label == "clean":
        return TagInfo(family="baseline", raw_severity=0.0, normalized_severity=0.0)
    if label.lower() == "all-perturbed":
        return TagInfo(family="aggregate", raw_severity=float("nan"), normalized_severity=float("nan"))
    if label.startswith("blur_sigma_"):
        suffix = label.split("blur_sigma_", 1)[1]
        level = _parse_fixed_point(suffix)
        return TagInfo(family="gaussian_blur", raw_severity=level)
    if label.startswith("jpeg_q_"):
        suffix = label.split("jpeg_q_", 1)[1]
        quality = _coerce_float(suffix)
        if quality is None:
            return TagInfo(family="jpeg", raw_severity=float("nan"))
        # Higher severity for lower quality; invert around 100.
        return TagInfo(family="jpeg", raw_severity=100.0 - quality)
    if label.startswith("bc_b") and "_c" in label:
        # brightness/contrast e.g. bc_b0p8_c0p6
        try:
            brightness_token, contrast_token = label.split("_c", 1)
            brightness = _parse_fixed_point(brightness_token.split("bc_b", 1)[1])
            contrast = _parse_fixed_point(contrast_token)
            if math.isnan(brightness) or math.isnan(contrast):
                level = float("nan")
            else:
                # Treat severity as the deficit from unity of the weaker factor.
                deficit = 1.0 - min(brightness, contrast)
                level = max(0.0, deficit)
        except (IndexError, ValueError):
            level = float("nan")
        return TagInfo(family="brightness_contrast", raw_severity=level)
    if label.startswith("occ_a"):
        suffix = label.split("occ_a", 1)[1]
        area = _parse_fixed_point(suffix)
        return TagInfo(family="occlusion", raw_severity=area)
    return TagInfo(family=label, raw_severity=float("nan"), normalized_severity=float("nan"))


def _build_tag_catalog(runs: Mapping[str, Mapping[int, RunPerturbationResult]]) -> Dict[str, TagInfo]:
    catalog: Dict[str, TagInfo] = {}
    families: DefaultDict[str, List[float]] = defaultdict(list)
    for run_map in runs.values():
        for run in run_map.values():
            for tag in run.perturbations:
                if tag not in catalog:
                    catalog[tag] = _parse_tag(tag)
                info = catalog[tag]
                if info.family not in {"baseline", "aggregate", "unknown"} and not math.isnan(info.raw_severity):
                    families[info.family].append(info.raw_severity)
    # Ensure clean is available even if absent from perturbations map.
    if "clean" not in catalog:
        catalog["clean"] = TagInfo(family="baseline", raw_severity=0.0, normalized_severity=0.0)
    for tag, info in list(catalog.items()):
        if info.family in {"baseline", "aggregate", "unknown"}:
            continue
        levels = families.get(info.family)
        if not levels:
            catalog[tag] = TagInfo(family=info.family, raw_severity=info.raw_severity, normalized_severity=float("nan"))
            continue
        min_level = min(levels)
        max_level = max(levels)
        if math.isnan(info.raw_severity):
            normalized = float("nan")
        elif max_level > min_level:
            normalized = (info.raw_severity - min_level) / (max_level - min_level)
        else:
            normalized = 1.0
        catalog[tag] = TagInfo(
            family=info.family,
            raw_severity=info.raw_severity,
            normalized_severity=max(0.0, normalized),
        )
    return catalog


def load_run(metrics_path: Path) -> RunPerturbationResult:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance_raw = payload.get("provenance") or {}
    provenance: Dict[str, object] = dict(provenance_raw) if isinstance(provenance_raw, Mapping) else {}
    model_name = str(provenance.get("model") or metrics_path.stem.split("__", 1)[0])
    seed_value = _coerce_int(payload.get("seed"))
    if seed_value is None:
        seed_value = _coerce_int(provenance.get("train_seed"))
    if seed_value is None:
        raise ValueError(f"Metrics file '{metrics_path}' does not specify a seed")
    test_primary_raw = payload.get("test_primary") or {}
    metrics: Dict[str, float] = {}
    for key, value in test_primary_raw.items():
        numeric = _coerce_float(value)
        if numeric is not None:
            metrics[key] = numeric
    tau_value = metrics.get("tau")
    pert_block = payload.get("test_perturbations") or {}
    per_tag_raw = pert_block.get("per_tag") if isinstance(pert_block, Mapping) else None
    if not isinstance(per_tag_raw, Mapping) or not per_tag_raw:
        raise ValueError(f"Metrics file '{metrics_path}' does not contain test_perturbations.per_tag")
    perturbations: Dict[str, Dict[str, float]] = {}
    for tag, stats in per_tag_raw.items():
        if not isinstance(stats, Mapping):
            continue
        sanitized: Dict[str, float] = {}
        for key, value in stats.items():
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            sanitized[str(key)] = numeric
        if sanitized:
            perturbations[str(tag)] = sanitized
    if not perturbations:
        raise ValueError(f"Metrics file '{metrics_path}' does not provide numeric perturbation metrics")
    return RunPerturbationResult(
        model=model_name,
        seed=int(seed_value),
        tau=tau_value,
        metrics=metrics,
        perturbations=perturbations,
        provenance=provenance,
        path=metrics_path,
    )


def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[int, RunPerturbationResult]]:
    model_filter = {str(m) for m in models} if models else None
    runs: DefaultDict[str, Dict[int, RunPerturbationResult]] = defaultdict(dict)
    for metrics_path in sorted(root.rglob("*.metrics.json")):
        try:
            run = load_run(metrics_path)
        except (OSError, ValueError):
            continue
        if model_filter and run.model not in model_filter:
            continue
        runs[run.model][run.seed] = run
    return {model: dict(seed_map) for model, seed_map in runs.items()}


def _compute_stats(values: Sequence[float]) -> Dict[str, float]:
    numeric = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not numeric:
        return {}
    mean = float(np.mean(numeric))
    std = float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(numeric)}


def _deduplicate_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda item: (item[0], item[1]))
    deduplicated: List[Tuple[float, float]] = []
    last_severity: Optional[float] = None
    for severity, value in ordered:
        if last_severity is not None and abs(severity - last_severity) < 1e-12:
            deduplicated[-1] = (severity, value)
        else:
            deduplicated.append((severity, value))
            last_severity = severity
    return deduplicated


def _normalised_trapz(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    width = float(xs[-1] - xs[0])
    if width <= 0.0:
        return None
    area = float(np.trapz(ys, xs))
    return area / width


def _safe_numeric(value: Any, default: float = float("inf")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def _flatten_stats(row: MutableMapping[str, object], prefix: str, stats_map: Mapping[str, Any]) -> None:
    for metric, stats in stats_map.items():
        if not isinstance(stats, Mapping):
            continue
        mean = stats.get("mean")
        std = stats.get("std")
        count = stats.get("n")
        row[f"{prefix}{metric}_mean"] = mean
        row[f"{prefix}{metric}_std"] = std
        row[f"{prefix}{metric}_n"] = int(count) if count is not None else None


def _summarize_model(
    model: str,
    runs: Mapping[int, RunPerturbationResult],
    tag_catalog: Mapping[str, TagInfo],
    *,
    metrics: Sequence[str],
) -> Dict[str, object]:
    per_tag_values: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_tag_retention: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_tag_delta: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_tag_drop: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ausc_acc: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ausc_retention_acc: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for run in runs.values():
        clean_metrics = run.perturbations.get("clean", {})
        for tag, stats in run.perturbations.items():
            for key, value in stats.items():
                numeric = _coerce_float(value)
                if numeric is not None:
                    per_tag_values[tag][key].append(numeric)
        retention_seed_tracker: Dict[str, bool] = {metric: False for metric in metrics}
        for metric in metrics:
            baseline_value = _coerce_float(clean_metrics.get(metric))
            if baseline_value is None:
                continue
            per_tag_retention["clean"][metric].append(1.0)
            per_tag_drop["clean"][metric].append(0.0)
            per_tag_delta["clean"][metric].append(0.0)
            retention_seed_tracker[metric] = True
        for tag, stats in run.perturbations.items():
            if tag == "clean":
                continue
            base = clean_metrics
            for metric in metrics:
                metric_value = _coerce_float(stats.get(metric))
                baseline_value = _coerce_float(base.get(metric))
                if metric_value is None or baseline_value is None or not math.isfinite(baseline_value) or baseline_value == 0.0:
                    continue
                retention_value = metric_value / baseline_value
                per_tag_retention[tag][metric].append(retention_value)
                per_tag_drop[tag][metric].append(1.0 - retention_value)
                per_tag_delta[tag][metric].append(metric_value - baseline_value)
        # Compute AUSC per family for this run.
        family_points: DefaultDict[str, DefaultDict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for tag, stats in run.perturbations.items():
            info = tag_catalog.get(tag)
            if info is None or info.family in {"baseline", "aggregate", "unknown"}:
                continue
            severity = info.normalized_severity
            if severity is None or math.isnan(severity):
                continue
            for metric in metrics:
                numeric = _coerce_float(stats.get(metric))
                if numeric is None:
                    continue
                family_points[info.family][metric].append((severity, numeric))
        clean_cache: Dict[str, float] = {
            metric: val
            for metric in metrics
            if (val := _coerce_float(clean_metrics.get(metric))) is not None and math.isfinite(val)
        }
        for family, family_metric_points in family_points.items():
            for metric, points in family_metric_points.items():
                clean_value = clean_cache.get(metric)
                if clean_value is None:
                    continue
                series = _deduplicate_points([(0.0, clean_value)] + points)
                ausc_value = _normalised_trapz(series)
                if ausc_value is not None and math.isfinite(ausc_value):
                    ausc_acc[family][metric].append(ausc_value)
                if clean_value == 0.0:
                    continue
                retention_series = _deduplicate_points([(0.0, 1.0)] + [(sev, val / clean_value) for sev, val in points])
                retention_ausc = _normalised_trapz(retention_series)
                if retention_ausc is not None and math.isfinite(retention_ausc):
                    ausc_retention_acc[family][metric].append(retention_ausc)

    per_tag_summary: Dict[str, Dict[str, Any]] = {}
    for tag in sorted(per_tag_values.keys() | per_tag_retention.keys() | per_tag_delta.keys()):
        metrics_block = {
            key: stats
            for key in sorted(per_tag_values[tag].keys())
            if (stats := _compute_stats(per_tag_values[tag][key]))
        }
        retention_block = {
            key: stats
            for key in sorted(per_tag_retention[tag].keys())
            if (stats := _compute_stats(per_tag_retention[tag][key]))
        }
        drop_block = {
            key: stats
            for key in sorted(per_tag_drop[tag].keys())
            if (stats := _compute_stats(per_tag_drop[tag][key]))
        }
        delta_block = {
            key: stats
            for key in sorted(per_tag_delta[tag].keys())
            if (stats := _compute_stats(per_tag_delta[tag][key]))
        }
        per_tag_summary[tag] = {
            "metrics": metrics_block,
            "retention": retention_block,
            "drop": drop_block,
            "delta": delta_block,
        }

    severity_rows: List[Dict[str, object]] = []
    for tag, summary in per_tag_summary.items():
        info = tag_catalog.get(tag, TagInfo(family="unknown", raw_severity=float("nan"), normalized_severity=float("nan")))
        row: Dict[str, object] = {
            "model": model,
            "tag": tag,
            "family": info.family,
            "severity": info.raw_severity,
            "severity_normalized": info.normalized_severity,
        }
        _flatten_stats(row, "", summary.get("metrics", {}))
        _flatten_stats(row, "retention_", summary.get("retention", {}))
        _flatten_stats(row, "drop_", summary.get("drop", {}))
        _flatten_stats(row, "delta_", summary.get("delta", {}))
        severity_rows.append(row)

    families_summary: Dict[str, Dict[str, object]] = {}
    clean_entry = per_tag_summary.get("clean")
    for family in sorted({info.family for info in tag_catalog.values() if info.family not in {"baseline", "aggregate", "unknown"}}):
        tag_entries: List[Dict[str, object]] = []
        if clean_entry:
            tag_entries.append(
                {
                    "tag": "clean",
                    "severity": 0.0,
                    "severity_normalized": 0.0,
                    "metrics": clean_entry.get("metrics", {}),
                    "retention": clean_entry.get("retention", {}),
                    "drop": clean_entry.get("drop", {}),
                    "delta": clean_entry.get("delta", {}),
                }
            )
        for tag, summary in per_tag_summary.items():
            info = tag_catalog.get(tag)
            if info is None or info.family != family:
                continue
            tag_entries.append(
                {
                    "tag": tag,
                    "severity": info.raw_severity,
                    "severity_normalized": info.normalized_severity,
                    "metrics": summary.get("metrics", {}),
                    "retention": summary.get("retention", {}),
                    "drop": summary.get("drop", {}),
                    "delta": summary.get("delta", {}),
                }
            )
        if tag_entries:
            tag_entries.sort(
                key=lambda entry: (
                    _safe_numeric(entry.get("severity_normalized")),
                    str(entry.get("tag")),
                )
            )
            families_summary[family] = {"tags": tag_entries}

    ausc_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for family, metric_series_defaultdict in ausc_acc.items():
        family_entry: Dict[str, Dict[str, float]] = {}
        metric_series_map: Dict[str, List[float]] = dict(metric_series_defaultdict)
        for metric, values in metric_series_map.items():
            stats = _compute_stats(values)
            if stats:
                family_entry[metric] = stats
        retention_map: Dict[str, List[float]] = dict(ausc_retention_acc.get(family, {}))
        for metric, values in retention_map.items():
            stats = _compute_stats(values)
            if stats:
                family_entry[f"{metric}_retention"] = stats
        if family_entry:
            ausc_summary[family] = family_entry

    return {
        "per_tag": per_tag_summary,
        "families": families_summary,
        "severity_rows": severity_rows,
        "ausc": ausc_summary,
    }


def summarize_runs(
    runs: Mapping[str, Mapping[int, RunPerturbationResult]],
    *,
    metrics: Sequence[str] = RETENTION_METRICS,
) -> Dict[str, object]:
    tag_catalog = _build_tag_catalog(runs)
    models_summary: Dict[str, Dict[str, object]] = {}
    severity_rows: List[Dict[str, object]] = []
    for model, model_runs in sorted(runs.items()):
        summary = _summarize_model(model, model_runs, tag_catalog, metrics=metrics)
        models_summary[model] = {
            "per_tag": summary["per_tag"],
            "families": summary["families"],
            "ausc": summary["ausc"],
        }
        summary_rows = summary.get("severity_rows", [])
        if isinstance(summary_rows, list):
            severity_rows.extend([row for row in summary_rows if isinstance(row, dict)])
    catalog_serialized = {
        tag: {
            "family": info.family,
            "severity": info.raw_severity,
            "severity_normalized": info.normalized_severity,
        }
        for tag, info in sorted(tag_catalog.items())
    }
    return {
        "tag_catalog": catalog_serialized,
        "models": models_summary,
        "severity_rows": severity_rows,
    }


def write_severity_csv(summary: Mapping[str, object], output_path: Path) -> None:
    rows = summary.get("severity_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Summary does not contain severity_rows data")
    fieldnames = sorted({key for row in rows if isinstance(row, Mapping) for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            writer.writerow(row)


__all__ = [
    "RunPerturbationResult",
    "discover_runs",
    "load_run",
    "summarize_runs",
    "write_severity_csv",
]
