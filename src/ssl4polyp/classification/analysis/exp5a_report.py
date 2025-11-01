from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from reporting.threshold_specs import THRESHOLD_SPECS

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
from .seed_checks import ensure_expected_seeds

PRIMARY_METRICS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "mcc",
    "loss",
)
PAIRWISE_METRICS: Tuple[str, ...] = PRIMARY_METRICS
PAIRWISE_BASELINES: Tuple[str, ...] = ("sup_imnet", "ssl_imnet")
CI_LEVEL = 0.95
EXPECTED_SEEDS: Tuple[int, ...] = (13, 29, 47)


@dataclass(frozen=True)
class EvalFrame:
    frame_id: str
    prob: float
    label: int
    case_id: Optional[str]
    center_id: Optional[str]
    sequence_id: Optional[str]
    origin: Optional[str]


@dataclass(frozen=True)
class CompositionStats:
    n_pos: int
    n_neg: int
    total: int
    prevalence: float
    per_center: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "total": self.total,
            "prevalence": self.prevalence,
        }
        if self.per_center:
            payload["per_center"] = {
                key: dict(value) for key, value in self.per_center.items()
            }
        return payload


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
    composition: CompositionStats
    outputs_sha256: str
    experiment: Optional[str]


def _get_loader(*, strict: bool = True) -> ResultLoader:
    return get_default_loader(exp_id="exp5a", strict=strict)


def _frames_to_eval(frames: Iterable[CommonFrame]) -> Dict[str, EvalFrame]:
    mapping: Dict[str, EvalFrame] = {}
    for frame in frames:
        case_id = _clean_text(frame.row.get("case_id"))
        if case_id is None:
            case_id = _clean_text(frame.row.get("sequence_id"))
        if case_id is None:
            case_id = _clean_text(frame.case_id)
        center_id = _clean_text(frame.row.get("center_id")) or _clean_text(frame.row.get("origin"))
        sequence_id = _clean_text(frame.row.get("sequence_id"))
        origin = _clean_text(frame.row.get("origin"))
        mapping[frame.frame_id] = EvalFrame(
            frame_id=frame.frame_id,
            prob=frame.prob,
            label=frame.label,
            case_id=case_id,
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


def _build_cluster_set(frames: Mapping[str, EvalFrame], *, domain: str) -> ClusterSet:
    domain_key = domain.lower()

    def positive_key(record: EvalFrame) -> Optional[str]:
        if domain_key == "sun":
            if record.case_id:
                return f"pos_case::{record.case_id}"
            if record.sequence_id:
                return f"pos_sequence::{record.sequence_id}"
            return None
        if domain_key == "polypgen":
            center = record.center_id or record.origin
            if center:
                return f"pos_center::{center}"
            if record.sequence_id:
                return f"pos_sequence::{record.sequence_id}"
            if record.case_id:
                return f"pos_case::{record.case_id}"
            return None
        return None

    def negative_key(record: EvalFrame) -> Optional[str]:
        if domain_key == "sun":
            if record.case_id:
                return f"neg_case::{record.case_id}"
            if record.sequence_id:
                return f"neg_sequence::{record.sequence_id}"
            return None
        center = record.center_id or record.origin
        if center:
            return f"neg_center::{center}"
        if record.sequence_id:
            return f"neg_sequence::{record.sequence_id}"
        if record.case_id:
            return f"neg_case::{record.case_id}"
        return None

    return build_cluster_set(
        frames.values(),
        is_positive=lambda record: record.label == 1,
        record_id=lambda record: record.frame_id,
        positive_key=positive_key,
        negative_key=negative_key,
    )


def _compute_composition(
    metrics: Mapping[str, float],
    frames: Mapping[str, EvalFrame],
) -> CompositionStats:
    def _require_int(value: Optional[float], name: str) -> int:
        numeric = _coerce_float(value)
        if numeric is None:
            raise ValueError(f"Missing {name} in test_primary metrics")
        return int(round(float(numeric)))

    n_pos = _require_int(metrics.get("n_pos"), "n_pos")
    n_neg = _require_int(metrics.get("n_neg"), "n_neg")
    total = n_pos + n_neg
    if total <= 0:
        raise ValueError("Invalid composition: zero total examples")
    if len(frames) != total:
        raise ValueError(
            "Frame count does not match n_pos + n_neg"
        )
    prevalence = float(n_pos) / float(total)
    stored_prevalence = _coerce_float(metrics.get("prevalence"))
    if stored_prevalence is not None and not math.isclose(
        stored_prevalence, prevalence, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError("Prevalence mismatch between frames and metrics")

    tp = _coerce_float(metrics.get("tp"))
    fp = _coerce_float(metrics.get("fp"))
    tn = _coerce_float(metrics.get("tn"))
    fn = _coerce_float(metrics.get("fn"))
    if all(value is not None for value in (tp, fp, tn, fn)):
        confusion_total = int(round(float(tp) + float(fp) + float(tn) + float(fn)))
        if confusion_total != total:
            raise ValueError("Confusion matrix counts do not sum to n_pos + n_neg")

    per_center_counts: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"n_pos": 0, "n_neg": 0})
    for record in frames.values():
        center_key = record.center_id or record.origin or record.sequence_id or record.case_id
        if not center_key:
            continue
        label_key = "n_pos" if record.label == 1 else "n_neg"
        per_center_counts[center_key][label_key] += 1
    per_center: Dict[str, Mapping[str, float]] = {}
    for center, counts in per_center_counts.items():
        center_total = counts["n_pos"] + counts["n_neg"]
        if center_total <= 0:
            continue
        per_center[center] = MappingProxyType(
            {
                "n_pos": counts["n_pos"],
                "n_neg": counts["n_neg"],
                "total": center_total,
                "prevalence": float(counts["n_pos"]) / float(center_total),
            }
        )

    return CompositionStats(
        n_pos=n_pos,
        n_neg=n_neg,
        total=total,
        prevalence=prevalence,
        per_center=MappingProxyType(dict(per_center)),
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


def _infer_parent_experiment(
    parent_payload: Optional[Mapping[str, Any]], metrics_path: Path
) -> Optional[str]:
    def _extract(block: Mapping[str, Any]) -> Optional[str]:
        run_block = block.get("run") if isinstance(block.get("run"), Mapping) else None
        if not isinstance(run_block, Mapping):
            return None
        for key in ("exp", "experiment"):
            candidate = _clean_text(run_block.get(key)) if key in run_block else None
            if candidate:
                return candidate.lower()
        return None

    if isinstance(parent_payload, Mapping):
        inferred = _extract(parent_payload)
        if inferred:
            return inferred
    try:
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(raw, Mapping):
        inferred = _extract(raw)
        if inferred:
            return inferred
    return None


def _resolve_parent_loader(
    parent_payload: Optional[Mapping[str, Any]], metrics_path: Path
) -> Optional[ResultLoader]:
    exp_id = _infer_parent_experiment(parent_payload, metrics_path)
    if not exp_id or exp_id == "exp5a":
        return None
    if exp_id not in THRESHOLD_SPECS:
        return None
    return get_default_loader(exp_id=exp_id)


def _derive_delta(
    polyp_metrics: Mapping[str, float],
    sun_metrics: Mapping[str, float],
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    metric_list = tuple(metrics) if metrics is not None else PRIMARY_METRICS
    deltas: Dict[str, float] = {}
    for metric in metric_list:
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
    active_loader = loader or _get_loader()
    base_run = load_common_run(metrics_path, loader=active_loader)
    payload = base_run.payload
    provenance = dict(base_run.provenance)
    run_block = payload.get("run") if isinstance(payload, Mapping) else None
    experiment_name = (
        _clean_text(run_block.get("exp")) if isinstance(run_block, Mapping) else None
    )
    if experiment_name is None:
        raise ValueError("Metrics payload must include run.exp for Experiment 5A")
    if experiment_name.lower() != "exp5a":
        raise ValueError(
            f"Metrics payload at {metrics_path} reports run.exp='{experiment_name}', expected 'exp5a'"
        )
    test_metrics = dict(base_run.primary_metrics)
    delta_block = (
        payload.get("domain_shift_delta")
        if isinstance(payload.get("domain_shift_delta"), Mapping)
        else None
    )
    parent_payload, parent_metrics_path, parent_outputs_path = _load_parent_payload(
        provenance, metrics_path
    )
    sun_metrics: Dict[str, float] = {}
    if isinstance(parent_payload, Mapping):
        extracted_metrics = _extract_metrics(parent_payload.get("test_primary"))
        if extracted_metrics:
            sun_metrics = dict(extracted_metrics)
    sun_tau = _coerce_float(sun_metrics.get("tau")) if sun_metrics else None
    sun_frames: Optional[Dict[str, EvalFrame]] = None
    base_tau = _coerce_float(base_run.tau)
    if parent_metrics_path and parent_metrics_path.exists():
        parent_run = None
        try:
            parent_run = load_common_run(parent_metrics_path, loader=active_loader)
        except GuardrailViolation:
            fallback_loader = _resolve_parent_loader(parent_payload, parent_metrics_path)
            if fallback_loader is not None:
                try:
                    parent_run = load_common_run(parent_metrics_path, loader=fallback_loader)
                except (OSError, ValueError, GuardrailViolation):
                    parent_run = None
        except (OSError, ValueError):
            parent_run = None
        if parent_run is not None:
            sun_metrics = dict(parent_run.primary_metrics)
            sun_tau = _coerce_float(parent_run.tau)
            sun_frames = _frames_to_eval(parent_run.frames)
    if not sun_metrics:
        raise ValueError(
            f"Missing SUN baseline metrics for Experiment 5A run at {metrics_path}"
        )
    if sun_frames is None and parent_outputs_path and parent_outputs_path.exists():
        if sun_tau is not None:
            try:
                parent_frames, _ = load_outputs_csv(parent_outputs_path, tau=float(sun_tau))
            except (OSError, ValueError):
                sun_frames = None
            else:
                sun_frames = _frames_to_eval(parent_frames)
    if sun_tau is None:
        raise ValueError(
            f"SUN baseline tau missing for Experiment 5A run at {metrics_path}"
        )
    if (
        base_tau is not None
        and not math.isclose(
            float(base_tau), float(sun_tau), rel_tol=1e-9, abs_tol=1e-6
        )
    ):
        raise ValueError(
            "Experiment 5A run uses a threshold tau that differs from the SUN baseline"
        )
    if sun_frames is None:
        raise ValueError(
            f"SUN baseline outputs missing for Experiment 5A run at {metrics_path}"
        )
    computed_delta = _derive_delta(test_metrics, sun_metrics)
    if not delta_block:
        delta_metrics = computed_delta
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
        for metric, value in computed_delta.items():
            delta_metrics.setdefault(metric, value)
    frames = _frames_to_eval(base_run.frames)
    composition = _compute_composition(test_metrics, frames)
    outputs_sha = _clean_text(provenance.get("test_outputs_csv_sha256"))
    if outputs_sha is None:
        raise ValueError(
            f"Missing provenance.test_outputs_csv_sha256 for run at {metrics_path}"
        )
    return Exp5ARun(
        model=base_run.model,
        seed=base_run.seed,
        tau=base_run.tau,
        metrics=test_metrics,
        delta=delta_metrics,
        sun_metrics=sun_metrics,
        frames=frames,
        sun_tau=float(sun_tau),
        sun_frames=sun_frames,
        provenance=provenance,
        metrics_path=metrics_path,
        composition=composition,
        outputs_sha256=outputs_sha,
        experiment=experiment_name,
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
    active_loader = loader or _get_loader()
    for metrics_path in sorted(metrics_paths):
        try:
            run = load_run(metrics_path, loader=active_loader)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load metrics from {metrics_path} (missing per-frame outputs)"
            ) from exc
        except (ValueError, GuardrailViolation) as exc:
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
    clusters = _build_cluster_set(run.frames, domain="polypgen")
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


def _joint_sample_cluster_ids(
    polyp_clusters: ClusterSet,
    sun_clusters: ClusterSet,
    rng: np.random.Generator,
) -> Tuple[List[str], List[str]]:
    def sample_with_uniform(clusters: Tuple[Tuple[str, ...], ...], uniforms: np.ndarray, count: int) -> List[str]:
        if count == 0:
            return []
        if uniforms.size < count:
            raise ValueError("Insufficient uniform samples for cluster resampling")
        indices = np.floor(uniforms[:count] * count).astype(int)
        indices = np.clip(indices, 0, count - 1)
        sampled: List[str] = []
        for idx in indices:
            sampled.extend(clusters[idx])
        return sampled

    pos_count = max(len(polyp_clusters.positives), len(sun_clusters.positives))
    neg_count = max(len(polyp_clusters.negatives), len(sun_clusters.negatives))
    pos_uniforms = rng.random(pos_count) if pos_count else np.empty(0, dtype=float)
    neg_uniforms = rng.random(neg_count) if neg_count else np.empty(0, dtype=float)

    polyp_ids: List[str] = []
    sun_ids: List[str] = []
    polyp_ids.extend(
        sample_with_uniform(polyp_clusters.positives, pos_uniforms, len(polyp_clusters.positives))
    )
    polyp_ids.extend(
        sample_with_uniform(polyp_clusters.negatives, neg_uniforms, len(polyp_clusters.negatives))
    )
    sun_ids.extend(sample_with_uniform(sun_clusters.positives, pos_uniforms, len(sun_clusters.positives)))
    sun_ids.extend(sample_with_uniform(sun_clusters.negatives, neg_uniforms, len(sun_clusters.negatives)))
    return polyp_ids, sun_ids


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
    polyp_clusters = _build_cluster_set(run.frames, domain="polypgen")
    sun_clusters = _build_cluster_set(run.sun_frames, domain="sun")
    rng = np.random.default_rng(rng_seed)
    for _ in range(bootstrap):
        polyp_ids, sun_ids = _joint_sample_cluster_ids(polyp_clusters, sun_clusters, rng)
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


def _bootstrap_domain_shift_summary(
    runs: Mapping[int, Exp5ARun],
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: int,
) -> Dict[str, List[float]]:
    summary_replicates: Dict[str, List[float]] = {metric: [] for metric in metrics}
    if bootstrap <= 0 or not runs:
        return summary_replicates
    prepared: Dict[int, Tuple[Exp5ARun, ClusterSet, ClusterSet]] = {}
    for seed, run in runs.items():
        if run.sun_frames is None or run.sun_tau is None:
            continue
        prepared[seed] = (
            run,
            _build_cluster_set(run.frames, domain="polypgen"),
            _build_cluster_set(run.sun_frames, domain="sun"),
        )
    if not prepared:
        return summary_replicates
    rng = np.random.default_rng(rng_seed)
    for _ in range(bootstrap):
        draw_totals: DefaultDict[str, List[float]] = defaultdict(list)
        for run, polyp_clusters, sun_clusters in prepared.values():
            polyp_ids, sun_ids = _joint_sample_cluster_ids(polyp_clusters, sun_clusters, rng)
            if not polyp_ids or not sun_ids:
                continue
            polyp_values = _as_frames_metrics(run.frames, polyp_ids, run.tau)
            sun_values = _as_frames_metrics(run.sun_frames, sun_ids, run.sun_tau)
            for metric in metrics:
                polyp_metric = _coerce_float(polyp_values.get(metric))
                sun_metric = _coerce_float(sun_values.get(metric))
                if polyp_metric is None or sun_metric is None:
                    continue
                draw_totals[metric].append(float(polyp_metric - sun_metric))
        for metric, values in draw_totals.items():
            if values:
                summary_replicates[metric].append(float(np.mean(values)))
    return summary_replicates


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
    clusters = _build_cluster_set(colon_run.frames, domain="polypgen")
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


def _bootstrap_pairwise_summary(
    colon_runs: Mapping[int, Exp5ARun],
    baseline_runs: Mapping[int, Exp5ARun],
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
) -> List[float]:
    if bootstrap <= 0:
        return []
    prepared: Dict[int, Tuple[Exp5ARun, Exp5ARun, ClusterSet]] = {}
    for seed, colon_run in colon_runs.items():
        baseline_run = baseline_runs.get(seed)
        if colon_run is None or baseline_run is None:
            continue
        prepared[seed] = (
            colon_run,
            baseline_run,
            _build_cluster_set(colon_run.frames, domain="polypgen"),
        )
    if not prepared:
        return []
    rng = np.random.default_rng(rng_seed)
    replicates: List[float] = []
    for _ in range(bootstrap):
        draw_values: List[float] = []
        for colon_run, baseline_run, clusters in prepared.values():
            sample_ids = sample_cluster_ids(clusters, rng)
            if not sample_ids:
                continue
            colon_metrics = _as_frames_metrics(colon_run.frames, sample_ids, colon_run.tau)
            baseline_metrics = _as_frames_metrics(baseline_run.frames, sample_ids, baseline_run.tau)
            colon_value = _coerce_float(colon_metrics.get(metric))
            baseline_value = _coerce_float(baseline_metrics.get(metric))
            if colon_value is None or baseline_value is None:
                continue
            draw_values.append(float(colon_value - baseline_value))
        if draw_values:
            replicates.append(float(np.mean(draw_values)))
    return replicates


def summarize_runs(
    runs_by_model: Mapping[str, Mapping[int, Exp5ARun]],
    *,
    bootstrap: int = 2000,
    rng_seed: int = 12345,
) -> Dict[str, Any]:
    if not runs_by_model:
        raise ValueError("Experiment 5A summary requires at least one discovered run")
    seed_validation = ensure_expected_seeds(
        runs_by_model,
        expected_seeds=EXPECTED_SEEDS,
        context="Experiment 5A",
    )
    summary: Dict[str, Any] = {
        "metadata": {
            "metrics": list(PRIMARY_METRICS),
            "delta_metrics": list(PRIMARY_METRICS),
            "pairwise_metrics": list(PAIRWISE_METRICS),
            "bootstrap": int(max(0, bootstrap)),
            "ci_level": CI_LEVEL,
        },
        "models": {},
        "pairwise": {},
        "validated_seeds": list(seed_validation.expected_seeds),
        "seed_groups": {
            key: list(values) for key, values in seed_validation.observed_seeds.items()
        },
        "composition": {},
    }
    all_runs: List[Exp5ARun] = [run for runs in runs_by_model.values() for run in runs.values()]
    if all_runs:
        reference = all_runs[0]
        reference_comp = reference.composition
        reference_sha = reference.outputs_sha256
        for run in all_runs[1:]:
            if run.outputs_sha256 != reference_sha:
                raise ValueError(
                    "Mismatched test_outputs_csv_sha256 across Experiment 5A runs"
                )
            comp = run.composition
            if comp.n_pos != reference_comp.n_pos or comp.n_neg != reference_comp.n_neg:
                raise ValueError("PolypGen composition mismatch across runs")
            if comp.total != reference_comp.total:
                raise ValueError("PolypGen total count mismatch across runs")
            if not math.isclose(
                comp.prevalence,
                reference_comp.prevalence,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("PolypGen prevalence mismatch across runs")
            ref_centers = {key: dict(value) for key, value in reference_comp.per_center.items()}
            comp_centers = {key: dict(value) for key, value in comp.per_center.items()}
            if set(ref_centers) != set(comp_centers):
                raise ValueError("Per-center composition mismatch across runs")
            for center, ref_counts in ref_centers.items():
                comp_counts = comp_centers.get(center)
                if comp_counts is None:
                    raise ValueError("Missing per-center counts in one of the runs")
                for key in ("n_pos", "n_neg", "total"):
                    if int(ref_counts[key]) != int(comp_counts.get(key, -1)):
                        raise ValueError("Per-center counts mismatch across runs")
                if not math.isclose(
                    float(ref_counts.get("prevalence", 0.0)),
                    float(comp_counts.get("prevalence", 0.0)),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError("Per-center prevalence mismatch across runs")
        composition_payload = reference_comp.to_dict()
        composition_payload["sha256"] = reference_sha
        summary["composition"] = composition_payload
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
                metrics=PRIMARY_METRICS,
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
            for metric in PRIMARY_METRICS:
                value = _coerce_float(run.delta.get(metric))
                if value is not None:
                    delta_accumulators[metric].append(value)
        delta_summary_ci_replicates = _bootstrap_domain_shift_summary(
            runs,
            metrics=PRIMARY_METRICS,
            bootstrap=bootstrap,
            rng_seed=rng_seed + len(model) * 17,
        )
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
        delta_summary_ci = {
            metric: _ci_bounds(values)
            for metric, values in delta_summary_ci_replicates.items()
            if values
        }
        model_entries[model] = {
            "seeds": seeds_entry,
            "performance": performance_summary,
            "domain_shift": delta_summary,
            "domain_shift_ci": delta_summary_ci,
        }
    summary["models"] = model_entries

    colon_runs = runs_by_model.get("ssl_colon", {})
    if colon_runs:
        pairwise_summary: Dict[str, Dict[str, Any]] = {}
        for metric in PAIRWISE_METRICS:
            metric_entry: Dict[str, Any] = {}
            for baseline in PAIRWISE_BASELINES:
                baseline_runs = runs_by_model.get(baseline, {})
                if not baseline_runs:
                    raise ValueError(
                        f"Experiment 5A pairwise summary requires runs for baseline '{baseline}'"
                    )
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
                metric_stats = _compute_stats(deltas) if deltas else None
                summary_replicates = _bootstrap_pairwise_summary(
                    colon_runs,
                    baseline_runs,
                    metric=metric,
                    bootstrap=bootstrap,
                    rng_seed=rng_seed + len(metric) * 29 + PAIRWISE_BASELINES.index(baseline) * 31,
                )
                summary_ci = _ci_bounds(summary_replicates)
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
        domain_shift_ci = payload.get("domain_shift_ci") if isinstance(payload.get("domain_shift_ci"), Mapping) else {}
        for metric, stats in domain_shift.items():
            if not isinstance(stats, Mapping):
                continue
            row = {"model": model, "metric": metric}
            row.update({key: stats.get(key) for key in ("mean", "std", "n")})
            if isinstance(domain_shift_ci, Mapping):
                ci_block = domain_shift_ci.get(metric)
                if isinstance(ci_block, Mapping):
                    row["ci_lower"] = ci_block.get("lower")
                    row["ci_upper"] = ci_block.get("upper")
            rows.append(row)
    if not rows:
        raise ValueError("Summary payload does not include domain shift data")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_composition_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    composition = summary.get("composition")
    if not isinstance(composition, Mapping) or not composition:
        raise ValueError("Summary payload does not include composition data")
    rows: List[Dict[str, Any]] = []
    overall_row: Dict[str, Any] = {
        "scope": "overall",
        "n_pos": composition.get("n_pos"),
        "n_neg": composition.get("n_neg"),
        "total": composition.get("total"),
        "prevalence": composition.get("prevalence"),
    }
    if "sha256" in composition:
        overall_row["sha256"] = composition.get("sha256")
    rows.append(overall_row)
    per_center = composition.get("per_center")
    if isinstance(per_center, Mapping):
        for center, payload in per_center.items():
            if not isinstance(payload, Mapping):
                continue
            row = {
                "scope": "center",
                "center_id": center,
                "n_pos": payload.get("n_pos"),
                "n_neg": payload.get("n_neg"),
                "total": payload.get("total"),
                "prevalence": payload.get("prevalence"),
            }
            rows.append(row)
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
    "write_composition_csv",
    "write_seed_metrics_csv",
    "write_pairwise_csv",
    "EXPECTED_SEEDS",
    "PRIMARY_METRICS",
    "PAIRWISE_METRICS",
]
