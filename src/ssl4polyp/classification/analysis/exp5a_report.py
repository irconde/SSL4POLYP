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

from legacy.reporting.threshold_specs import THRESHOLD_SPECS

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
from .bootstrap_common import (
    DEFAULT_BOOTSTRAP,
    DEFAULT_RNG_SEED,
    ClusterSampler,
    bootstrap_replicates,
    bootstrap_scalar,
    metric_on_rows,
    percentile_ci,
    require_aligned,
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
    lower, upper = percentile_ci(values, level)
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
    if isinstance(run_block, Mapping):
        experiment_name = _clean_text(
            run_block.get("exp") or run_block.get("experiment")
        )
    else:
        experiment_name = None
    if experiment_name is None:
        raise ValueError(
            "Metrics payload must include run.exp or run.experiment for Experiment 5A"
        )
    if experiment_name.lower() != "exp5a":
        raise ValueError(
            f"Metrics payload at {metrics_path} reports run.exp/run.experiment="
            f"'{experiment_name}', expected 'exp5a'"
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


def _center_key(frame: EvalFrame) -> Optional[str]:
    return frame.center_id or frame.origin


def _polyp_sampler(
    frames: Mapping[str, EvalFrame], centers: Sequence[str]
) -> Tuple[ClusterSampler, List[str]]:
    """Class-stratified centre sampler over the frames of the shared centres."""

    keep = set(centers)
    ids = [fid for fid, frame in frames.items() if _center_key(frame) in keep]
    sampler = ClusterSampler(
        [frames[fid].label for fid in ids],
        [_center_key(frames[fid]) for fid in ids],
    )
    return sampler, ids


def _sun_sampler(frames: Mapping[str, EvalFrame]) -> Tuple[ClusterSampler, List[str]]:
    """Class-stratified case sampler over the SUN test frames."""

    ids = list(frames)
    sampler = ClusterSampler(
        [frames[fid].label for fid in ids],
        [frames[fid].case_id or frames[fid].sequence_id for fid in ids],
    )
    return sampler, ids


def _frame_arrays(
    frames: Mapping[str, EvalFrame], ids: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.array([frames[fid].prob for fid in ids], dtype=float),
        np.array([frames[fid].label for fid in ids], dtype=int),
    )


def _bootstrap_metrics(
    run: Exp5ARun,
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: int,
    centers: Sequence[str],
) -> Dict[str, List[float]]:
    if bootstrap <= 0 or not centers or not run.centers:
        return {metric: [] for metric in metrics}
    sampler, ids = _polyp_sampler(run.frames, centers)
    probs, labels = _frame_arrays(run.frames, ids)

    def statistic(rows: np.ndarray) -> Dict[str, Optional[float]]:
        return {m: metric_on_rows(probs, labels, run.tau, rows, m) for m in metrics}

    drawn = bootstrap_replicates(sampler, statistic, bootstrap=bootstrap, rng_seed=rng_seed)
    return {metric: drawn.get(metric, []) for metric in metrics}


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
    if bootstrap <= 0 or not centers or not run.centers:
        return results
    if run.sun_frames is None or run.sun_tau is None or run.sun_case_index is None:
        return results
    if not (sun_cases if sun_cases is not None else run.sun_case_ids):
        return results
    polyp_sampler, polyp_ids = _polyp_sampler(run.frames, centers)
    sun_sampler, sun_ids = _sun_sampler(run.sun_frames)
    polyp_probs, polyp_labels = _frame_arrays(run.frames, polyp_ids)
    sun_probs, sun_labels = _frame_arrays(run.sun_frames, sun_ids)

    def statistic(rows: Mapping[str, np.ndarray]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for m in metrics:
            a = metric_on_rows(polyp_probs, polyp_labels, run.tau, rows["polyp"], m)
            b = metric_on_rows(sun_probs, sun_labels, run.sun_tau, rows["sun"], m)
            out[m] = None if a is None or b is None else a - b
        return out

    drawn = bootstrap_replicates(
        {"polyp": polyp_sampler, "sun": sun_sampler},
        statistic,
        bootstrap=bootstrap,
        rng_seed=rng_seed,
    )
    return {metric: drawn.get(metric, []) for metric in metrics}


def _bootstrap_domain_shift_summary(
    runs: Mapping[int, Exp5ARun],
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: int,
    centers: Sequence[str],
) -> Dict[str, List[float]]:
    empty: Dict[str, List[float]] = {metric: [] for metric in metrics}
    if bootstrap <= 0 or not runs or not centers:
        return empty
    prepared = {
        seed: run
        for seed, run in runs.items()
        if run.sun_frames is not None and run.sun_tau is not None and run.sun_case_index is not None
    }
    if not prepared:
        return empty
    first = next(iter(prepared.values()))
    polyp_sampler, polyp_ids = _polyp_sampler(first.frames, centers)
    sun_sampler, sun_ids = _sun_sampler(first.sun_frames)
    require_aligned(
        polyp_ids,
        {f"seed {s}": _polyp_sampler(r.frames, centers)[1] for s, r in prepared.items()},
        context="Experiment 5A domain-shift bootstrap (PolypGen)",
    )
    require_aligned(
        sun_ids,
        {f"seed {s}": list(r.sun_frames) for s, r in prepared.items()},
        context="Experiment 5A domain-shift bootstrap (SUN)",
    )
    arrays = {
        seed: (
            _frame_arrays(run.frames, polyp_ids),
            _frame_arrays(run.sun_frames, sun_ids),
            run.tau,
            run.sun_tau,
        )
        for seed, run in prepared.items()
    }

    def statistic(rows: Mapping[str, np.ndarray]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for m in metrics:
            deltas: List[float] = []
            for (pp, pl), (sp, sl), tau, sun_tau in arrays.values():
                a = metric_on_rows(pp, pl, tau, rows["polyp"], m)
                b = metric_on_rows(sp, sl, sun_tau, rows["sun"], m)
                if a is not None and b is not None:
                    deltas.append(a - b)
            out[m] = float(np.mean(deltas)) if deltas else None
        return out

    drawn = bootstrap_replicates(
        {"polyp": polyp_sampler, "sun": sun_sampler},
        statistic,
        bootstrap=bootstrap,
        rng_seed=rng_seed,
    )
    return {metric: drawn.get(metric, []) for metric in metrics}


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
    if bootstrap <= 0 or not centers:
        return []
    if not colon_run.centers or not baseline_run.centers:
        return []
    sampler, ids = _polyp_sampler(colon_run.frames, centers)
    require_aligned(
        ids,
        {"baseline": _polyp_sampler(baseline_run.frames, centers)[1]},
        context="Experiment 5A pairwise bootstrap",
    )
    cp, cl = _frame_arrays(colon_run.frames, ids)
    bp, bl = _frame_arrays(baseline_run.frames, ids)

    def statistic(rows: np.ndarray) -> Optional[float]:
        a = metric_on_rows(cp, cl, colon_run.tau, rows, metric)
        b = metric_on_rows(bp, bl, baseline_run.tau, rows, metric)
        return None if a is None or b is None else a - b

    return bootstrap_scalar(sampler, statistic, bootstrap=bootstrap, rng_seed=rng_seed)


def _bootstrap_pairwise_summary(
    colon_runs: Mapping[int, Exp5ARun],
    baseline_runs: Mapping[int, Exp5ARun],
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
    centers: Sequence[str],
) -> List[float]:
    if bootstrap <= 0 or not centers:
        return []
    prepared = {
        seed: (colon_run, baseline_runs[seed])
        for seed, colon_run in colon_runs.items()
        if seed in baseline_runs and colon_run.centers and baseline_runs[seed].centers
    }
    if not prepared:
        return []
    first_colon = next(iter(prepared.values()))[0]
    sampler, ids = _polyp_sampler(first_colon.frames, centers)
    others: Dict[str, List[str]] = {}
    for seed, (colon_run, baseline_run) in prepared.items():
        others[f"colon@{seed}"] = _polyp_sampler(colon_run.frames, centers)[1]
        others[f"baseline@{seed}"] = _polyp_sampler(baseline_run.frames, centers)[1]
    require_aligned(ids, others, context="Experiment 5A pairwise summary bootstrap")
    arrays = {
        seed: (
            _frame_arrays(colon_run.frames, ids),
            _frame_arrays(baseline_run.frames, ids),
            colon_run.tau,
            baseline_run.tau,
        )
        for seed, (colon_run, baseline_run) in prepared.items()
    }

    def statistic(rows: np.ndarray) -> Optional[float]:
        deltas: List[float] = []
        for (cp, cl), (bp, bl), colon_tau, baseline_tau in arrays.values():
            a = metric_on_rows(cp, cl, colon_tau, rows, metric)
            b = metric_on_rows(bp, bl, baseline_tau, rows, metric)
            if a is not None and b is not None:
                deltas.append(a - b)
        return float(np.mean(deltas)) if deltas else None

    return bootstrap_scalar(sampler, statistic, bootstrap=bootstrap, rng_seed=rng_seed)


def compute_leave_one_center_out(
    runs_by_model: Mapping[str, Mapping[int, Exp5ARun]],
    *,
    centers: Sequence[str],
    metrics: Sequence[str] = PRIMARY_METRICS,
    target_model: str = "ssl_colon",
    baselines: Sequence[str] = PAIRWISE_BASELINES,
) -> Dict[str, Any]:
    """Recompute every model metric and the SSL-Colon deltas dropping one centre at a time.

    Values are the mean over training seeds.  ``pairwise[baseline][metric]`` reports the
    full-data delta, the delta without each centre, its range, the centre whose removal
    moves the delta the most, and whether any removal flips its sign.
    """

    if not centers:
        return {}
    keep_centers = list(centers)

    prepared: Dict[str, List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]] = {}
    for model, runs in runs_by_model.items():
        entries = []
        for run in runs.values():
            ids = [fid for fid, frame in run.frames.items() if _center_key(frame) in set(keep_centers)]
            probs, labels = _frame_arrays(run.frames, ids)
            keys = np.array([_center_key(run.frames[fid]) for fid in ids], dtype=object)
            entries.append((probs, labels, run.tau, keys))
        prepared[model] = entries

    def mean_metric(model: str, metric: str, excluded: Optional[str]) -> Optional[float]:
        values: List[float] = []
        for probs, labels, tau, keys in prepared.get(model, []):
            rows = np.flatnonzero(keys != excluded) if excluded is not None else np.arange(keys.size)
            value = metric_on_rows(probs, labels, tau, rows, metric)
            if value is not None:
                values.append(value)
        return float(np.mean(values)) if values else None

    result: Dict[str, Any] = {
        "centers": keep_centers,
        "metrics": list(metrics),
        "models": {},
        "pairwise": {},
    }
    for model in prepared:
        result["models"][model] = {
            "full": {m: mean_metric(model, m, None) for m in metrics},
            "excluding": {
                c: {m: mean_metric(model, m, c) for m in metrics} for c in keep_centers
            },
        }
    if target_model in prepared:
        for baseline in baselines:
            if baseline not in prepared:
                continue
            per_metric: Dict[str, Any] = {}
            for m in metrics:
                a, b = mean_metric(target_model, m, None), mean_metric(baseline, m, None)
                if a is None or b is None:
                    continue
                full = a - b
                by_center: Dict[str, float] = {}
                for c in keep_centers:
                    ac, bc = mean_metric(target_model, m, c), mean_metric(baseline, m, c)
                    if ac is not None and bc is not None:
                        by_center[c] = ac - bc
                if not by_center:
                    continue
                influential = max(by_center, key=lambda c: abs(by_center[c] - full))
                per_metric[m] = {
                    "full": full,
                    "by_center": by_center,
                    "min": min(by_center.values()),
                    "max": max(by_center.values()),
                    "most_influential": influential,
                    "sign_change": any(np.sign(v) != np.sign(full) for v in by_center.values() if v != 0)
                    if full != 0
                    else False,
                }
            if per_metric:
                result["pairwise"][baseline] = per_metric
    return result


def write_loco_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    block = summary.get("loco")
    if not isinstance(block, Mapping) or not block:
        raise ValueError("Summary payload does not include a leave-one-center-out block")
    rows: List[Dict[str, Any]] = []
    for model, payload in (block.get("models") or {}).items():
        for metric, value in (payload.get("full") or {}).items():
            rows.append({"kind": "model", "name": model, "metric": metric, "excluded_center": "none", "value": value})
        for center, values in (payload.get("excluding") or {}).items():
            for metric, value in values.items():
                rows.append({"kind": "model", "name": model, "metric": metric, "excluded_center": center, "value": value})
    for baseline, metrics_block in (block.get("pairwise") or {}).items():
        for metric, payload in metrics_block.items():
            rows.append({"kind": "delta_vs", "name": baseline, "metric": metric, "excluded_center": "none", "value": payload["full"]})
            for center, value in payload["by_center"].items():
                rows.append({"kind": "delta_vs", "name": baseline, "metric": metric, "excluded_center": center, "value": value})
    fieldnames = ["kind", "name", "metric", "excluded_center", "value"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(
    runs_by_model: Mapping[str, Mapping[int, Exp5ARun]],
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
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
                rng_seed=rng_seed,
                centers=centers_for_bootstrap,
            )
            delta_ci_replicates = _bootstrap_domain_shift(
                run,
                metrics=PRIMARY_METRICS,
                bootstrap=effective_bootstrap,
                rng_seed=rng_seed,
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
            rng_seed=rng_seed,
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
                        rng_seed=rng_seed,
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
                    rng_seed=rng_seed,
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
        loco = compute_leave_one_center_out(runs_by_model, centers=centers_for_bootstrap)
        if loco:
            summary["loco"] = loco

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
