from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .common_loader import (
    CommonFrame,
    _extract_metrics,
    get_default_loader,
    load_common_run,
    load_outputs_csv,
)
from .result_loader import GuardrailViolation, ResultLoader
from .common_metrics import (
    ClusterSet,
    build_cluster_set,
    compute_binary_metrics,
    sample_cluster_ids,
    _clean_text,
    _coerce_float,
    _coerce_int,
)
from .seed_checks import SeedValidationResult, ensure_expected_seeds

PRIMARY_METRICS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "mcc",
)
PAIRWISE_BASELINES: Tuple[str, ...] = ("sup_imnet", "ssl_imnet")
CI_LEVEL = 0.95
EXPECTED_SEEDS: Tuple[int, ...] = (13, 29, 47)


@dataclass(frozen=True)
class EvalFrame:
    frame_id: str
    prob: float
    label: int
    center_id: Optional[str]
    sequence_id: Optional[str]
    origin: Optional[str]


@dataclass
class Exp5ARun:
    model: str
    seed: int
    tau: float
    metrics: Dict[str, float]
    delta: Dict[str, float]
    sun_metrics: Dict[str, float]
    frames: Dict[str, EvalFrame]
    sun_tau: Optional[float]
    sun_frames: Optional[Dict[str, EvalFrame]]
    provenance: Dict[str, Any]
    metrics_path: Path


def _frames_to_eval(frames: Iterable[CommonFrame]) -> Dict[str, EvalFrame]:
    mapping: Dict[str, EvalFrame] = {}
    for frame in frames:
        center_id = _clean_text(frame.row.get("center_id")) or _clean_text(frame.row.get("origin"))
        sequence_id = _clean_text(frame.row.get("sequence_id"))
        origin = _clean_text(frame.row.get("origin"))
        mapping[frame.frame_id] = EvalFrame(
            frame_id=frame.frame_id,
            prob=frame.prob,
            label=frame.label,
            center_id=center_id,
            sequence_id=sequence_id,
            origin=origin,
        )
    return mapping


def _resolve_relative_path(base: Path, candidate: str) -> Path:
    candidate_path = Path(candidate).expanduser()
    if candidate_path.is_absolute():
        return candidate_path
    return (base.parent / candidate_path).resolve()


def _as_frames_metrics(frames: Mapping[str, EvalFrame], frame_ids: Sequence[str], tau: float) -> Dict[str, float]:
    subset = [frames[fid] for fid in frame_ids if fid in frames]
    if not subset:
        return {metric: float("nan") for metric in PRIMARY_METRICS}
    probs = np.array([entry.prob for entry in subset], dtype=float)
    labels = np.array([entry.label for entry in subset], dtype=int)
    return compute_binary_metrics(probs, labels, tau)


def _build_cluster_set(frames: Mapping[str, EvalFrame]) -> ClusterSet:
    def positive_key(record: EvalFrame) -> Optional[str]:
        if record.center_id and record.sequence_id:
            return f"pos_center_seq::{record.center_id}::{record.sequence_id}"
        if record.center_id:
            return f"pos_center::{record.center_id}"
        if record.sequence_id:
            return f"pos_sequence::{record.sequence_id}"
        return None

    def negative_key(record: EvalFrame) -> Optional[str]:
        if record.sequence_id and record.center_id:
            return f"neg_center_seq::{record.center_id}::{record.sequence_id}"
        if record.sequence_id:
            return f"neg_sequence::{record.sequence_id}"
        if record.center_id:
            return f"neg_center::{record.center_id}"
        return None

    return build_cluster_set(
        frames.values(),
        is_positive=lambda record: record.label == 1,
        record_id=lambda record: record.frame_id,
        positive_key=positive_key,
        negative_key=negative_key,
    )


def _ci_bounds(values: Sequence[float], *, level: float = CI_LEVEL) -> Optional[Dict[str, float]]:
    if not values:
        return None
    array = np.array(values, dtype=float)
    if array.size == 0:
        return None
    lower_pct = (1.0 - level) / 2.0 * 100.0
    upper_pct = (1.0 + level) / 2.0 * 100.0
    lower = float(np.percentile(array, lower_pct))
    upper = float(np.percentile(array, upper_pct))
    return {"lower": lower, "upper": upper}


def _compute_stats(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    array = np.array(values, dtype=float)
    if array.size == 0:
        return None
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {"mean": mean, "std": std, "n": float(array.size)}


def _load_parent_payload(provenance: Mapping[str, Any], metrics_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[Path]]:
    parent_block = provenance.get("parent_run") if isinstance(provenance, Mapping) else None
    if not isinstance(parent_block, Mapping):
        return None, None, None
    metrics_info = parent_block.get("metrics") if isinstance(parent_block, Mapping) else None
    payload = None
    parent_metrics_path = None
    if isinstance(metrics_info, Mapping):
        path_entry = metrics_info.get("path")
        if isinstance(path_entry, str) and path_entry:
            parent_metrics_path = _resolve_relative_path(metrics_path, path_entry)
        payload = metrics_info.get("payload") if isinstance(metrics_info.get("payload"), Mapping) else None
    outputs_info = parent_block.get("outputs") if isinstance(parent_block, Mapping) else None
    parent_outputs_path = None
    if isinstance(outputs_info, Mapping):
        path_entry = outputs_info.get("path")
        if isinstance(path_entry, str) and path_entry:
            parent_outputs_path = _resolve_relative_path(metrics_path, path_entry)
    return payload, parent_metrics_path, parent_outputs_path


def _derive_delta(
    polyp_metrics: Mapping[str, float],
    sun_metrics: Mapping[str, float],
    metrics: Sequence[str] = ("recall", "f1", "auprc", "auroc"),
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for metric in metrics:
        polyp_value = _coerce_float(polyp_metrics.get(metric))
        sun_value = _coerce_float(sun_metrics.get(metric))
        if polyp_value is None or sun_value is None:
            continue
        deltas[metric] = float(polyp_value - sun_value)
    return deltas


def load_run(
    metrics_path: Path,
    *,
    loader: Optional[ResultLoader] = None,
) -> Exp5ARun:
    active_loader = loader or get_default_loader()
    base_run = load_common_run(metrics_path, loader=active_loader)
    payload = base_run.payload
    provenance = dict(base_run.provenance)
    test_metrics = dict(base_run.primary_metrics)
    delta_block = (
        payload.get("domain_shift_delta")
        if isinstance(payload.get("domain_shift_delta"), Mapping)
        else None
    )
    parent_payload, parent_metrics_path, parent_outputs_path = _load_parent_payload(
        provenance, metrics_path
    )
    sun_metrics = _extract_metrics(
        parent_payload.get("test_primary") if isinstance(parent_payload, Mapping) else None
    )
    sun_tau = _coerce_float(sun_metrics.get("tau")) if sun_metrics else None
    sun_frames: Optional[Dict[str, EvalFrame]] = None
    if parent_metrics_path and parent_metrics_path.exists():
        try:
            parent_run = load_common_run(parent_metrics_path, loader=active_loader)
        except (OSError, ValueError, GuardrailViolation):
            parent_run = None
        else:
            sun_metrics = dict(parent_run.primary_metrics)
            sun_tau = parent_run.tau
            sun_frames = _frames_to_eval(parent_run.frames)
    if sun_frames is None and parent_outputs_path and parent_outputs_path.exists():
        if sun_tau is not None:
            try:
                parent_frames, _ = load_outputs_csv(parent_outputs_path, tau=float(sun_tau))
            except (OSError, ValueError):
                sun_frames = None
            else:
                sun_frames = _frames_to_eval(parent_frames)
    if not delta_block:
        delta_metrics = _derive_delta(test_metrics, sun_metrics)
    else:
        metrics_subblock = (
            delta_block.get("metrics")
            if isinstance(delta_block.get("metrics"), Mapping)
            else None
        )
        delta_metrics = {
            key: float(value)
            for key, value in (metrics_subblock or {}).items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value))
        }
    frames = _frames_to_eval(base_run.frames)
    return Exp5ARun(
        model=base_run.model,
        seed=base_run.seed,
        tau=base_run.tau,
        metrics=test_metrics,
        delta=delta_metrics,
        sun_metrics=sun_metrics,
        frames=frames,
        sun_tau=float(sun_tau) if isinstance(sun_tau, float) else None,
        sun_frames=sun_frames,
        provenance=provenance,
        metrics_path=metrics_path,
    )


def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    loader: Optional[ResultLoader] = None,
) -> Dict[str, Dict[int, Exp5ARun]]:
    root = root.expanduser()
    metrics_paths = {
        path
        for path in root.rglob("*.metrics.json")
        if path.is_file() and not path.name.endswith("_best.metrics.json")
    }
    runs: Dict[str, Dict[int, Exp5ARun]] = {}
    model_filter = {name.lower() for name in models} if models else None
    active_loader = loader or get_default_loader()
    for metrics_path in sorted(metrics_paths):
        try:
            run = load_run(metrics_path, loader=active_loader)
        except (ValueError, FileNotFoundError, GuardrailViolation) as exc:
            raise RuntimeError(f"Failed to load metrics from {metrics_path}") from exc
        if model_filter and run.model.lower() not in model_filter:
            continue
        runs.setdefault(run.model, {})[run.seed] = run
    return runs


def _bootstrap_metrics(
    run: Exp5ARun,
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: int,
) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {metric: [] for metric in metrics}
    if bootstrap <= 0:
        return results
    clusters = _build_cluster_set(run.frames)
    rng = np.random.default_rng(rng_seed)
    for _ in range(bootstrap):
        sample_ids = sample_cluster_ids(clusters, rng)
        if not sample_ids:
            continue
        metric_values = _as_frames_metrics(run.frames, sample_ids, run.tau)
        for metric in metrics:
            value = metric_values.get(metric)
            if value is None or not math.isfinite(float(value)):
                continue
            results[metric].append(float(value))
    return results


def _bootstrap_domain_shift(
    run: Exp5ARun,
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: int,
) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {metric: [] for metric in metrics}
    if bootstrap <= 0:
        return results
    if run.sun_frames is None or run.sun_tau is None:
        return results
    polyp_clusters = _build_cluster_set(run.frames)
    sun_clusters = _build_cluster_set(run.sun_frames)
    rng = np.random.default_rng(rng_seed)
    for _ in range(bootstrap):
        polyp_ids = sample_cluster_ids(polyp_clusters, rng)
        sun_ids = sample_cluster_ids(sun_clusters, rng)
        if not polyp_ids or not sun_ids:
            continue
        polyp_values = _as_frames_metrics(run.frames, polyp_ids, run.tau)
        sun_values = _as_frames_metrics(run.sun_frames, sun_ids, run.sun_tau)
        for metric in metrics:
            polyp_metric = polyp_values.get(metric)
            sun_metric = sun_values.get(metric)
            if polyp_metric is None or sun_metric is None:
                continue
            if not (math.isfinite(float(polyp_metric)) and math.isfinite(float(sun_metric))):
                continue
            results[metric].append(float(polyp_metric) - float(sun_metric))
    return results


def _bootstrap_pairwise(
    colon_run: Exp5ARun,
    baseline_run: Exp5ARun,
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
) -> List[float]:
    if bootstrap <= 0:
        return []
    clusters = _build_cluster_set(colon_run.frames)
    rng = np.random.default_rng(rng_seed)
    replicates: List[float] = []
    for _ in range(bootstrap):
        sample_ids = sample_cluster_ids(clusters, rng)
        if not sample_ids:
            continue
        colon_metrics = _as_frames_metrics(colon_run.frames, sample_ids, colon_run.tau)
        baseline_metrics = _as_frames_metrics(baseline_run.frames, sample_ids, baseline_run.tau)
        colon_value = colon_metrics.get(metric)
        baseline_value = baseline_metrics.get(metric)
        if colon_value is None or baseline_value is None:
            continue
        if not (math.isfinite(float(colon_value)) and math.isfinite(float(baseline_value))):
            continue
        replicates.append(float(colon_value) - float(baseline_value))
    return replicates


def summarize_runs(
    runs_by_model: Mapping[str, Mapping[int, Exp5ARun]],
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
) -> Dict[str, Any]:
    if not runs_by_model:
        seed_validation = SeedValidationResult((), MappingProxyType({}))
    else:
        seed_validation = ensure_expected_seeds(
            runs_by_model,
            expected_seeds=EXPECTED_SEEDS,
            context="Experiment 5A",
        )
    summary: Dict[str, Any] = {
        "metadata": {
            "metrics": list(PRIMARY_METRICS),
            "delta_metrics": ["recall", "f1", "auprc", "auroc"],
            "bootstrap": int(max(0, bootstrap)),
            "ci_level": CI_LEVEL,
        },
        "models": {},
        "pairwise": {},
        "validated_seeds": list(seed_validation.expected_seeds),
        "seed_groups": {
            key: list(values) for key, values in seed_validation.observed_seeds.items()
        },
    }
    model_entries: Dict[str, Any] = {}
    for model, runs in sorted(runs_by_model.items()):
        seeds_entry: Dict[int, Any] = {}
        metric_accumulators: DefaultDict[str, List[float]] = defaultdict(list)
        delta_accumulators: DefaultDict[str, List[float]] = defaultdict(list)
        for seed, run in sorted(runs.items()):
            metrics_ci_replicates = _bootstrap_metrics(
                run,
                metrics=PRIMARY_METRICS,
                bootstrap=bootstrap,
                rng_seed=rng_seed + seed,
            )
            delta_ci_replicates = _bootstrap_domain_shift(
                run,
                metrics=("recall", "f1", "auprc", "auroc"),
                bootstrap=bootstrap,
                rng_seed=rng_seed + seed * 7,
            )
            metrics_ci = {
                metric: _ci_bounds(values)
                for metric, values in metrics_ci_replicates.items()
                if values
            }
            delta_ci = {
                metric: _ci_bounds(values)
                for metric, values in delta_ci_replicates.items()
                if values
            }
            seeds_entry[int(seed)] = {
                "metrics": run.metrics,
                "ci": metrics_ci,
                "delta": run.delta,
                "delta_ci": delta_ci,
                "tau": run.tau,
                "sun_metrics": run.sun_metrics,
            }
            for metric in PRIMARY_METRICS:
                value = _coerce_float(run.metrics.get(metric))
                if value is not None:
                    metric_accumulators[metric].append(value)
            for metric in ("recall", "f1", "auprc", "auroc"):
                value = _coerce_float(run.delta.get(metric))
                if value is not None:
                    delta_accumulators[metric].append(value)
        performance_summary: Dict[str, Dict[str, float]] = {}
        for metric, values in metric_accumulators.items():
            stats = _compute_stats(values)
            if stats:
                performance_summary[metric] = stats
        delta_summary: Dict[str, Dict[str, float]] = {}
        for metric, values in delta_accumulators.items():
            stats = _compute_stats(values)
            if stats:
                delta_summary[metric] = stats
        model_entries[model] = {
            "seeds": seeds_entry,
            "performance": performance_summary,
            "domain_shift": delta_summary,
        }
    summary["models"] = model_entries

    colon_runs = runs_by_model.get("ssl_colon", {})
    if colon_runs:
        pairwise_summary: Dict[str, Dict[str, Any]] = {}
        for metric in ("auprc", "f1"):
            metric_entry: Dict[str, Any] = {}
            for baseline in PAIRWISE_BASELINES:
                baseline_runs = runs_by_model.get(baseline, {})
                if not baseline_runs:
                    continue
                ensure_expected_seeds(
                    {
                        "ssl_colon": colon_runs,
                        baseline: baseline_runs,
                    },
                    expected_seeds=seed_validation.expected_seeds,
                    context=f"Experiment 5A pairwise ({baseline})",
                )
                rows: List[Dict[str, Any]] = []
                deltas: List[float] = []
                aggregate_replicates: List[float] = []
                for seed in seed_validation.expected_seeds:
                    colon_run = colon_runs.get(seed)
                    baseline_run = baseline_runs.get(seed)
                    if colon_run is None or baseline_run is None:
                        continue
                    colon_value = _coerce_float(colon_run.metrics.get(metric))
                    baseline_value = _coerce_float(baseline_run.metrics.get(metric))
                    if colon_value is None or baseline_value is None:
                        continue
                    delta_value = float(colon_value - baseline_value)
                    deltas.append(delta_value)
                    replicates = _bootstrap_pairwise(
                        colon_run,
                        baseline_run,
                        metric=metric,
                        bootstrap=bootstrap,
                        rng_seed=rng_seed + seed * 11,
                    )
                    rows.append(
                        {
                            "seed": int(seed),
                            "delta": delta_value,
                            "ci": _ci_bounds(replicates),
                        }
                    )
                    aggregate_replicates.extend(replicates)
                metric_stats = _compute_stats(deltas) if deltas else None
                summary_ci = _ci_bounds(aggregate_replicates)
                payload: Dict[str, Any] = {"seeds": rows}
                if metric_stats:
                    payload["summary"] = metric_stats
                if summary_ci:
                    payload["summary_ci"] = summary_ci
                metric_entry[baseline] = payload
            if metric_entry:
                pairwise_summary[metric] = metric_entry
        summary["pairwise"] = pairwise_summary

    return summary


def write_performance_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    models_block = summary.get("models")
    if not isinstance(models_block, Mapping):
        raise ValueError("Summary payload does not include models block")
    rows: List[Dict[str, Any]] = []
    for model, payload in models_block.items():
        if not isinstance(payload, Mapping):
            continue
        performance = payload.get("performance")
        if not isinstance(performance, Mapping):
            continue
        for metric, stats in performance.items():
            if not isinstance(stats, Mapping):
                continue
            row = {"model": model, "metric": metric}
            row.update({key: stats.get(key) for key in ("mean", "std", "n")})
            rows.append(row)
    if not rows:
        raise ValueError("Summary payload does not include performance data")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_domain_shift_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    models_block = summary.get("models")
    if not isinstance(models_block, Mapping):
        raise ValueError("Summary payload does not include models block")
    rows: List[Dict[str, Any]] = []
    for model, payload in models_block.items():
        if not isinstance(payload, Mapping):
            continue
        domain_shift = payload.get("domain_shift")
        if not isinstance(domain_shift, Mapping):
            continue
        for metric, stats in domain_shift.items():
            if not isinstance(stats, Mapping):
                continue
            row = {"model": model, "metric": metric}
            row.update({key: stats.get(key) for key in ("mean", "std", "n")})
            rows.append(row)
    if not rows:
        raise ValueError("Summary payload does not include domain shift data")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_seed_metrics_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    models_block = summary.get("models")
    if not isinstance(models_block, Mapping):
        raise ValueError("Summary payload does not include models block")
    rows: List[Dict[str, Any]] = []
    for model, payload in models_block.items():
        if not isinstance(payload, Mapping):
            continue
        seeds = payload.get("seeds")
        if not isinstance(seeds, Mapping):
            continue
        for seed, seed_payload in seeds.items():
            if not isinstance(seed_payload, Mapping):
                continue
            metrics_obj = seed_payload.get("metrics")
            metrics_block = metrics_obj if isinstance(metrics_obj, Mapping) else {}
            ci_obj = seed_payload.get("ci")
            ci_block = ci_obj if isinstance(ci_obj, Mapping) else {}
            for metric, value in metrics_block.items():
                if not isinstance(value, (int, float)):
                    continue
                row: Dict[str, Any] = {
                    "model": model,
                    "seed": seed,
                    "metric": metric,
                    "kind": "performance",
                    "value": float(value),
                }
                ci_entry = ci_block.get(metric) if isinstance(ci_block, Mapping) else None
                if isinstance(ci_entry, Mapping):
                    row["ci_lower"] = ci_entry.get("lower")
                    row["ci_upper"] = ci_entry.get("upper")
                rows.append(row)
            delta_obj = seed_payload.get("delta")
            delta_block = delta_obj if isinstance(delta_obj, Mapping) else {}
            delta_ci_obj = seed_payload.get("delta_ci")
            delta_ci_block = delta_ci_obj if isinstance(delta_ci_obj, Mapping) else {}
            for metric, value in delta_block.items():
                if not isinstance(value, (int, float)):
                    continue
                row = {
                    "model": model,
                    "seed": seed,
                    "metric": metric,
                    "kind": "domain_shift",
                    "value": float(value),
                }
                ci_entry = delta_ci_block.get(metric) if isinstance(delta_ci_block, Mapping) else None
                if isinstance(ci_entry, Mapping):
                    row["ci_lower"] = ci_entry.get("lower")
                    row["ci_upper"] = ci_entry.get("upper")
                rows.append(row)
    if not rows:
        raise ValueError("Summary payload does not include per-seed metrics")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    pairwise_block = summary.get("pairwise")
    if not isinstance(pairwise_block, Mapping):
        raise ValueError("Summary payload does not include pairwise block")
    rows: List[Dict[str, Any]] = []
    for metric, baselines in pairwise_block.items():
        if not isinstance(baselines, Mapping):
            continue
        for baseline, payload in baselines.items():
            if not isinstance(payload, Mapping):
                continue
            seeds_block = payload.get("seeds")
            seeds_list: Sequence[Any] = seeds_block if isinstance(seeds_block, Sequence) else ()
            summary_block = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else None
            summary_ci = payload.get("summary_ci") if isinstance(payload.get("summary_ci"), Mapping) else None
            if summary_block:
                summary_row: Dict[str, Any] = {
                    "metric": metric,
                    "baseline": baseline,
                    "seed": "mean",
                    "delta": summary_block.get("mean"),
                    "std": summary_block.get("std"),
                    "n": summary_block.get("n"),
                }
                if summary_ci:
                    summary_row["ci_lower"] = summary_ci.get("lower")
                    summary_row["ci_upper"] = summary_ci.get("upper")
                rows.append(summary_row)
            for entry in seeds_list:
                if not isinstance(entry, Mapping):
                    continue
                row = {
                    "metric": metric,
                    "baseline": baseline,
                    "seed": entry.get("seed"),
                    "delta": entry.get("delta"),
                }
                ci_block = entry.get("ci")
                if isinstance(ci_block, Mapping):
                    row["ci_lower"] = ci_block.get("lower")
                    row["ci_upper"] = ci_block.get("upper")
                rows.append(row)
    if not rows:
        raise ValueError("Summary payload does not include pairwise data")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "Exp5ARun",
    "discover_runs",
    "load_run",
    "summarize_runs",
    "write_performance_csv",
    "write_domain_shift_csv",
    "write_seed_metrics_csv",
    "write_pairwise_csv",
    "EXPECTED_SEEDS",
]
