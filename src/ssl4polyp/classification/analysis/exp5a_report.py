from __future__ import annotations

import csv
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    centerless_frames: Tuple[str, ...]

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
        if self.centerless_frames:
            payload["centerless_frames"] = list(self.centerless_frames)
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
    centers: Mapping[str, Tuple[str, ...]]
    center_ids: Tuple[str, ...]
    centerless_frames: Tuple[str, ...]
    sun_tau: Optional[float]
    sun_frames: Optional[Dict[str, EvalFrame]]
    sun_case_index: Optional[Mapping[str, Tuple[str, ...]]]
    sun_case_ids: Tuple[str, ...]
    sun_case_missing: Tuple[str, ...]
    provenance: Dict[str, Any]
    metrics_path: Path
    composition: CompositionStats
    outputs_sha256: Optional[str]
    experiment: Optional[str]


def _get_loader(*, strict: bool = True) -> ResultLoader:
    return get_default_loader(exp_id="exp5a", strict=strict)


def _extract_center_from_path(value: Optional[object]) -> Optional[str]:
    text = _clean_text(value)
    if not text:
        return None
    try:
        candidate = Path(text)
    except (TypeError, ValueError):
        return None
    parts = candidate.parts
    if not parts:
        return None
    for part in parts:
        cleaned = _clean_text(part)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if "center" in lowered or "centre" in lowered:
            return cleaned
    stem = candidate.stem
    if stem:
        tokens = re.split(r"[\\/_.-]+", stem)
        for token in tokens:
            cleaned = _clean_text(token)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if "center" in lowered or "centre" in lowered:
                return cleaned
    return None


def _extract_center_from_row(row: Mapping[str, Any]) -> Optional[str]:
    explicit_keys = (
        "center_id",
        "center",
        "centre",
        "center_name",
        "centre_name",
    )
    for key in explicit_keys:
        if key in row:
            center = _clean_text(row.get(key))
            if center:
                return center
    path_keys = (
        "path",
        "image_path",
        "img_path",
        "image",
        "filepath",
        "file_path",
        "source",
        "source_path",
    )
    for key in path_keys:
        if key in row:
            center = _extract_center_from_path(row.get(key))
            if center:
                return center
    frame_id = _clean_text(row.get("frame_id"))
    if frame_id:
        tokens = re.split(r"[\\/_.-]+", frame_id)
        for token in tokens:
            cleaned = _clean_text(token)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if "center" in lowered or "centre" in lowered:
                return cleaned
    origin = _clean_text(row.get("origin"))
    if origin:
        return origin
    return None


def _group_frames(
    frames: Mapping[str, EvalFrame],
    *,
    key_fn: Callable[[EvalFrame], Optional[str]],
) -> Tuple[Dict[str, Tuple[str, ...]], Tuple[str, ...]]:
    grouped: DefaultDict[str, List[str]] = defaultdict(list)
    missing: List[str] = []
    for frame_id, record in frames.items():
        key = key_fn(record)
        if key:
            grouped[key].append(frame_id)
        else:
            missing.append(frame_id)
    frozen = {key: tuple(values) for key, values in grouped.items()}
    return frozen, tuple(missing)


def _index_by_center(frames: Mapping[str, EvalFrame]) -> Tuple[Dict[str, Tuple[str, ...]], Tuple[str, ...]]:
    return _group_frames(frames, key_fn=lambda record: record.center_id or record.origin)


def _index_by_case(frames: Mapping[str, EvalFrame]) -> Tuple[Dict[str, Tuple[str, ...]], Tuple[str, ...]]:
    return _group_frames(frames, key_fn=lambda record: record.case_id or record.sequence_id)


def _flatten_index(
    index: Mapping[str, Sequence[str]],
    keys: Sequence[str],
) -> List[str]:
    frame_ids: List[str] = []
    for key in keys:
        entries = index.get(key)
        if not entries:
            continue
        frame_ids.extend(entries)
    return frame_ids


def _sample_keys(
    keys: Sequence[str],
    rng: np.random.Generator,
) -> List[str]:
    """Sample keys with replacement (same length as keys)."""
    if not keys:
        return []
    draw_indices = rng.integers(0, len(keys), size=len(keys))
    return [keys[int(i)] for i in draw_indices]


def _sample_index(
    index: Mapping[str, Sequence[str]],
    keys: Sequence[str],
    rng: np.random.Generator,
) -> List[str]:
    if not keys:
        return []
    draw_indices = rng.integers(0, len(keys), size=len(keys))
    sampled: List[str] = []
    for draw in draw_indices:
        key = keys[int(draw)]
        entries = index.get(key)
        if not entries:
            return []
        sampled.extend(entries)
    return sampled


def _frames_to_eval(frames: Iterable[CommonFrame]) -> Dict[str, EvalFrame]:
    mapping: Dict[str, EvalFrame] = {}
    for frame in frames:
        case_id = _clean_text(frame.row.get("case_id"))
        if case_id is None:
            case_id = _clean_text(frame.row.get("sequence_id"))
        if case_id is None:
            case_id = _clean_text(frame.case_id)
        center_id = _extract_center_from_row(frame.row)
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
    centers: Mapping[str, Tuple[str, ...]],
    centerless_frames: Sequence[str],
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
    for center_key, frame_ids in centers.items():
        counts = per_center_counts[center_key]
        for frame_id in frame_ids:
            record = frames.get(frame_id)
            if record is None:
                continue
            label_key = "n_pos" if record.label == 1 else "n_neg"
            counts[label_key] += 1
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
        centerless_frames=tuple(centerless_frames),
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
    center_index_raw, centerless_frames = _index_by_center(frames)
    centers = MappingProxyType({key: tuple(value) for key, value in center_index_raw.items()})
    center_ids = tuple(sorted(centers.keys()))
    composition = _compute_composition(test_metrics, frames, centers, centerless_frames)
    outputs_sha = _clean_text(
        provenance.get("test_outputs_csv_sha256")
        or provenance.get("test_csv_sha256")
    )
    sun_case_index: Optional[Mapping[str, Tuple[str, ...]]] = None
    sun_case_ids: Tuple[str, ...] = ()
    sun_case_missing: Tuple[str, ...] = ()
    if sun_frames is not None:
        case_index_raw, case_missing = _index_by_case(sun_frames)
        sun_case_index = MappingProxyType({key: tuple(value) for key, value in case_index_raw.items()})
        sun_case_ids = tuple(sorted(sun_case_index.keys()))
        sun_case_missing = tuple(case_missing)
    return Exp5ARun(
        model=base_run.model,
        seed=base_run.seed,
        tau=base_run.tau,
        metrics=test_metrics,
        delta=delta_metrics,
        sun_metrics=sun_metrics,
        frames=frames,
        centers=centers,
        center_ids=center_ids,
        centerless_frames=tuple(centerless_frames),
        sun_tau=float(sun_tau),
        sun_frames=sun_frames,
        sun_case_index=sun_case_index,
        sun_case_ids=sun_case_ids,
        sun_case_missing=sun_case_missing,
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
    centers: Sequence[str],
) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {metric: [] for metric in metrics}
    if bootstrap <= 0:
        return results
    if not centers:
        return results
    if not run.centers:
        return results
    rng = np.random.default_rng(rng_seed)
    for _ in range(bootstrap):
        sample_ids = _sample_index(run.centers, centers, rng)
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
    centers: Sequence[str],
    sun_cases: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {metric: [] for metric in metrics}
    if bootstrap <= 0:
        return results
    if run.sun_frames is None or run.sun_tau is None:
        return results
    if not centers:
        return results
    if not run.centers:
        return results
    if run.sun_case_index is None:
        return results
    sun_case_keys: Sequence[str]
    if sun_cases is not None:
        sun_case_keys = tuple(sun_cases)
    else:
        sun_case_keys = run.sun_case_ids
    if not sun_case_keys:
        return results
    rng = np.random.default_rng(rng_seed)
    for _ in range(bootstrap):
        polyp_ids = _sample_index(run.centers, centers, rng)
        if not polyp_ids:
            continue
        sun_ids = _sample_index(run.sun_case_index, sun_case_keys, rng)
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
    centers: Sequence[str],
) -> Dict[str, List[float]]:
    summary_replicates: Dict[str, List[float]] = {metric: [] for metric in metrics}
    if bootstrap <= 0 or not runs:
        return summary_replicates
    if not centers:
        return summary_replicates
    prepared: Dict[int, Tuple[Exp5ARun, Sequence[str]]] = {}
    for seed, run in runs.items():
        if run.sun_frames is None or run.sun_tau is None:
            continue
        if run.sun_case_index is None:
            continue
        prepared[seed] = (
            run,
            run.sun_case_ids,
        )
    if not prepared:
        return summary_replicates
    rng = np.random.default_rng(rng_seed)
    for _ in range(bootstrap):
        draw_totals: DefaultDict[str, List[float]] = defaultdict(list)
        for run, sun_case_ids in prepared.values():
            polyp_ids = _sample_index(run.centers, centers, rng)
            if not polyp_ids:
                continue
            sun_ids = _sample_index(run.sun_case_index or {}, sun_case_ids, rng)
            if not sun_ids:
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


def _resolve_center_alignment(runs: Sequence[Exp5ARun]) -> Tuple[Tuple[str, ...], List[str]]:
    if not runs:
        return tuple(), []
    warnings_list: List[str] = []
    missing_center_message = (
        "Uncertainty not reported: PolypGen lacks case-level identifiers; frame-level bootstrap would overstate precision."
    )
    if any(run.centerless_frames for run in runs):
        if missing_center_message not in warnings_list:
            warnings_list.append(missing_center_message)
        return tuple(), warnings_list
    center_sets: List[set[str]] = []
    for run in runs:
        centers = set(run.centers.keys())
        if not centers:
            if missing_center_message not in warnings_list:
                warnings_list.append(missing_center_message)
            return tuple(), warnings_list
        center_sets.append(centers)
    common = set.intersection(*center_sets) if center_sets else set()
    no_overlap_message = "Uncertainty not reported: PolypGen center overlap across runs is empty; bootstrap disabled."
    if not common:
        if no_overlap_message not in warnings_list:
            warnings_list.append(no_overlap_message)
        return tuple(), warnings_list
    mismatched: List[str] = []
    for center in sorted(common):
        counts = {len(run.centers.get(center, ())) for run in runs}
        if len(counts) > 1:
            mismatched.append(center)
    mismatch_message: Optional[str] = None
    if mismatched:
        mismatch_message = (
            "PolypGen per-center frame counts differ across runs for: "
            + ", ".join(mismatched)
            + "; excluding from bootstrap."
        )
        warnings_list.append(mismatch_message)
        common -= set(mismatched)
    empty_after_filter_message = (
        "Uncertainty not reported: No PolypGen centers remain after alignment; bootstrap disabled."
    )
    if not common:
        if empty_after_filter_message not in warnings_list:
            warnings_list.append(empty_after_filter_message)
        return tuple(), warnings_list
    reference = set(next(iter(center_sets)))
    if any(center_set != reference for center_set in center_sets):
        coverage_message = (
            "PolypGen center coverage differs across runs; restricting bootstrap to the shared centers."
        )
        if coverage_message not in warnings_list:
            warnings_list.append(coverage_message)
    return tuple(sorted(common)), warnings_list


def _bootstrap_pairwise(
    colon_run: Exp5ARun,
    baseline_run: Exp5ARun,
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
    centers: Sequence[str],
) -> List[float]:
    if bootstrap <= 0:
        return []
    if not centers:
        return []
    if not colon_run.centers or not baseline_run.centers:
        return []
    rng = np.random.default_rng(rng_seed)
    replicates: List[float] = []
    for _ in range(bootstrap):
        # Paired resampling: same sampled centers for both runs
        sampled_centers = _sample_keys(centers, rng)
        colon_ids = _flatten_index(colon_run.centers, sampled_centers)
        baseline_ids = _flatten_index(baseline_run.centers, sampled_centers)
        if not colon_ids or not baseline_ids:
            continue
        colon_metrics = _as_frames_metrics(colon_run.frames, colon_ids, colon_run.tau)
        baseline_metrics = _as_frames_metrics(baseline_run.frames, baseline_ids, baseline_run.tau)
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
    centers: Sequence[str],
) -> List[float]:
    if bootstrap <= 0:
        return []
    if not centers:
        return []
    prepared: Dict[int, Tuple[Exp5ARun, Exp5ARun]] = {}
    for seed, colon_run in colon_runs.items():
        baseline_run = baseline_runs.get(seed)
        if colon_run is None or baseline_run is None:
            continue
        if not colon_run.centers or not baseline_run.centers:
            continue
        prepared[seed] = (
            colon_run,
            baseline_run,
        )
    if not prepared:
        return []
    rng = np.random.default_rng(rng_seed)
    replicates: List[float] = []
    for _ in range(bootstrap):
        draw_values: List[float] = []
        # One shared center sample per replicate; reuse across all seeds
        sampled_centers = _sample_keys(centers, rng)
        for colon_run, baseline_run in prepared.values():
            colon_ids = _flatten_index(colon_run.centers, sampled_centers)
            baseline_ids = _flatten_index(baseline_run.centers, sampled_centers)
            if not colon_ids or not baseline_ids:
                continue
            colon_metrics = _as_frames_metrics(colon_run.frames, colon_ids, colon_run.tau)
            baseline_metrics = _as_frames_metrics(baseline_run.frames, baseline_ids, baseline_run.tau)
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
    bootstrap: int = 1000,
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
        unique_shas = {
            run.outputs_sha256 for run in all_runs if run.outputs_sha256
        }
        if len(unique_shas) > 1:
            raise ValueError(
                "Mismatched test CSV SHA256 digest across Experiment 5A runs"
            )
        reference_sha = next(iter(unique_shas)) if unique_shas else None
        for run in all_runs[1:]:
            if (
                reference_sha
                and run.outputs_sha256
                and run.outputs_sha256 != reference_sha
            ):
                raise ValueError(
                    "Mismatched test CSV SHA256 digest across Experiment 5A runs"
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
        composition_payload = reference_comp.to_dict()
        if reference_sha is not None:
            composition_payload["sha256"] = reference_sha
        summary["composition"] = composition_payload
    centers_for_bootstrap, center_warnings = _resolve_center_alignment(all_runs)
    effective_bootstrap = bootstrap if centers_for_bootstrap else 0
    summary["metadata"]["bootstrap"] = int(max(0, effective_bootstrap))
    summary["metadata"]["center_bootstrap"] = {
        "enabled": bool(centers_for_bootstrap),
        "centers": list(centers_for_bootstrap),
        "n_centers": len(centers_for_bootstrap),
        "warnings": list(center_warnings),
    }
    for message in center_warnings:
        warnings.warn(message)
    model_entries: Dict[str, Any] = {}
    recomputed_metrics_cache: Dict[str, Dict[int, Dict[str, float]]] = {}
    for model, runs in sorted(runs_by_model.items()):
        seeds_entry: Dict[int, Any] = {}
        metric_accumulators: DefaultDict[str, List[float]] = defaultdict(list)
        delta_accumulators: DefaultDict[str, List[float]] = defaultdict(list)
        model_recomputed_metrics: Dict[int, Dict[str, float]] = {}
        for seed, run in sorted(runs.items()):
            if centers_for_bootstrap:
                frame_ids = _flatten_index(run.centers, centers_for_bootstrap)
            else:
                frame_ids = list(run.frames.keys())
            polyp_metrics = _as_frames_metrics(run.frames, frame_ids, run.tau)
            polyp_payload = dict(polyp_metrics)
            if run.tau is not None and "tau" not in polyp_payload:
                polyp_payload["tau"] = float(run.tau)
            if run.sun_frames is None or run.sun_tau is None:
                raise ValueError(
                    "Experiment 5A bootstrap requires SUN baseline per-frame outputs"
                )
            sun_frame_ids = list(run.sun_frames.keys())
            sun_metrics = _as_frames_metrics(run.sun_frames, sun_frame_ids, run.sun_tau)
            sun_payload = dict(sun_metrics)
            if run.sun_tau is not None and "tau" not in sun_payload:
                sun_payload["tau"] = float(run.sun_tau)
            delta_metrics = _derive_delta(
                polyp_payload, sun_payload, metrics=PRIMARY_METRICS
            )
            model_recomputed_metrics[int(seed)] = dict(polyp_payload)
            metrics_ci_replicates = _bootstrap_metrics(
                run,
                metrics=PRIMARY_METRICS,
                bootstrap=effective_bootstrap,
                rng_seed=rng_seed + seed,
                centers=centers_for_bootstrap,
            )
            delta_ci_replicates = _bootstrap_domain_shift(
                run,
                metrics=PRIMARY_METRICS,
                bootstrap=effective_bootstrap,
                rng_seed=rng_seed + seed * 7,
                centers=centers_for_bootstrap,
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
                "metrics": polyp_payload,
                "ci": metrics_ci,
                "delta": dict(delta_metrics),
                "delta_ci": delta_ci,
                "tau": run.tau,
                "sun_metrics": sun_payload,
            }
            for metric in PRIMARY_METRICS:
                value = _coerce_float(polyp_payload.get(metric))
                if value is not None:
                    metric_accumulators[metric].append(value)
            for metric in PRIMARY_METRICS:
                value = _coerce_float(delta_metrics.get(metric))
                if value is not None:
                    delta_accumulators[metric].append(value)
        delta_summary_ci_replicates = _bootstrap_domain_shift_summary(
            runs,
            metrics=PRIMARY_METRICS,
            bootstrap=effective_bootstrap,
            rng_seed=rng_seed + len(model) * 17,
            centers=centers_for_bootstrap,
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
        recomputed_metrics_cache[model] = model_recomputed_metrics
    summary["models"] = model_entries

    colon_runs = runs_by_model.get("ssl_colon", {})
    if colon_runs:
        colon_points = recomputed_metrics_cache.get("ssl_colon", {})
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
                    colon_metrics = colon_points.get(int(seed))
                    baseline_metrics = recomputed_metrics_cache.get(baseline, {}).get(
                        int(seed)
                    )
                    colon_value = _coerce_float(
                        (colon_metrics or {}).get(metric) if colon_metrics else None
                    )
                    baseline_value = _coerce_float(
                        (baseline_metrics or {}).get(metric)
                        if baseline_metrics
                        else None
                    )
                    if colon_value is None or baseline_value is None:
                        continue
                    delta_value = float(colon_value - baseline_value)
                    deltas.append(delta_value)
                    replicates = _bootstrap_pairwise(
                        colon_run,
                        baseline_run,
                        metric=metric,
                        bootstrap=effective_bootstrap,
                        rng_seed=rng_seed + seed * 11,
                        centers=centers_for_bootstrap,
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
                    bootstrap=effective_bootstrap,
                    rng_seed=rng_seed + len(metric) * 29 + PAIRWISE_BASELINES.index(baseline) * 31,
                    centers=centers_for_bootstrap,
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
