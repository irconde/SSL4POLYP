"""Aggregation utilities for Experiment 5C few-shot adaptation results."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from reporting.metrics import bce_loss_from_csv

from .common_loader import CommonFrame, CommonRun, get_default_loader, load_common_run, load_outputs_csv
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

TARGET_MODEL = "ssl_colon"
BASELINE_MODELS: Tuple[str, ...] = ("ssl_imnet", "sup_imnet")
AGG_METRICS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "mcc",
    "loss",
)
VAL_METRICS: Tuple[str, ...] = ("auprc", "auroc", "loss")
PAIRWISE_METRICS: Tuple[str, ...] = ("auprc", "f1")
AULC_METRICS: Tuple[str, ...] = ("auprc", "f1")
EXPECTED_SEEDS: Tuple[int, ...] = (13, 29, 47)


def _get_loader(*, strict: bool = True) -> ResultLoader:
    return get_default_loader(exp_id="exp5c", strict=strict)


@dataclass(frozen=True)
class EvalFrame:
    frame_id: str
    prob: float
    label: int
    case_id: Optional[str]
    sequence_id: Optional[str]
    center_id: Optional[str]


@dataclass
class ZeroShotResult:
    tau: Optional[float]
    metrics: Dict[str, float]
    counts: Dict[str, int]
    frames: Dict[str, EvalFrame]
    path: Optional[Path]


@dataclass
class FewShotRun:
    model: str
    seed: int
    budget: int
    tau: float
    sensitivity_tau: Optional[float]
    primary_metrics: Dict[str, float]
    primary_counts: Dict[str, int]
    sensitivity_metrics: Dict[str, float]
    sensitivity_counts: Dict[str, int]
    val_metrics: Dict[str, float]
    provenance: Dict[str, Any]
    dataset: Dict[str, Any]
    frames: Dict[str, EvalFrame]
    zero_shot: Optional[ZeroShotResult]
    path: Path


def _frames_to_eval(frames: Iterable[CommonFrame]) -> Dict[str, EvalFrame]:
    mapping: Dict[str, EvalFrame] = {}
    for frame in frames:
        row = frame.row
        case_id = _clean_text(row.get("case_id")) or frame.case_id
        sequence_id = _clean_text(row.get("sequence_id"))
        center_id = _clean_text(row.get("center_id")) or _clean_text(row.get("origin"))
        mapping[frame.frame_id] = EvalFrame(
            frame_id=frame.frame_id,
            prob=frame.prob,
            label=frame.label,
            case_id=case_id,
            sequence_id=sequence_id,
            center_id=center_id,
        )
    return mapping


def _extract_metrics(block: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if not isinstance(block, Mapping):
        return {}
    metrics: Dict[str, float] = {}
    for key, value in block.items():
        numeric = _coerce_float(value)
        if numeric is not None:
            metrics[str(key)] = float(numeric)
    return metrics


def _extract_counts(block: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    if not isinstance(block, Mapping):
        return {}
    counts: Dict[str, int] = {}
    for key in ("tp", "fp", "tn", "fn", "n_pos", "n_neg"):
        numeric = _coerce_int(block.get(key))
        if numeric is not None:
            counts[key] = int(numeric)
    return counts


def _extract_tau(block: Optional[Mapping[str, Any]], fallback: Optional[float]) -> Optional[float]:
    if isinstance(block, Mapping):
        candidate = _coerce_float(block.get("tau"))
        if candidate is not None:
            return float(candidate)
    return float(fallback) if fallback is not None else None


def _normalize_budget(value: object, *, fallback: Optional[int] = None) -> int:
    numeric = _coerce_int(value)
    if numeric is not None and numeric > 0:
        return int(numeric)
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("s"):
            numeric = _coerce_int(text[1:])
            if numeric is not None and numeric > 0:
                return int(numeric)
    if fallback is not None:
        return int(fallback)
    raise ValueError(f"Unable to resolve few-shot budget from value {value!r}")


def load_run(
    metrics_path: Path,
    *,
    loader: Optional[ResultLoader] = None,
) -> FewShotRun:
    active_loader = loader or _get_loader()
    base_run = load_common_run(metrics_path, loader=active_loader)
    payload = base_run.payload
    provenance = dict(base_run.provenance)
    primary_block = payload.get("test_primary") if isinstance(payload.get("test_primary"), Mapping) else None
    primary_metrics = _extract_metrics(primary_block)
    primary_counts = _extract_counts(primary_block)
    tau = _extract_tau(primary_block, base_run.tau)
    sensitivity_block = payload.get("test_sensitivity") if isinstance(payload.get("test_sensitivity"), Mapping) else None
    sensitivity_metrics = _extract_metrics(sensitivity_block)
    sensitivity_counts = _extract_counts(sensitivity_block)
    sensitivity_tau = _extract_tau(sensitivity_block, None)
    val_block = payload.get("val") if isinstance(payload.get("val"), Mapping) else None
    val_metrics = _extract_metrics(val_block)
    dataset_block = payload.get("dataset") if isinstance(payload.get("dataset"), Mapping) else None
    dataset_summary = dict(dataset_block) if isinstance(dataset_block, Mapping) else {}
    budget_value = provenance.get("fewshot_budget")
    if budget_value is None:
        train_summary = dataset_summary.get("train") if isinstance(dataset_summary.get("train"), Mapping) else None
        if isinstance(train_summary, Mapping):
            budget_value = train_summary.get("fewshot_budget") or train_summary.get("budget") or train_summary.get("frames")
        if budget_value is None:
            pack_spec = provenance.get("train_pack") or (train_summary.get("pack_spec") if isinstance(train_summary, Mapping) else None)
            if isinstance(pack_spec, str) and "_s" in pack_spec:
                budget_value = pack_spec
    budget = _normalize_budget(budget_value, fallback=None)
    frames = _frames_to_eval(base_run.frames)
    zero_shot = _load_zero_shot(base_run, payload)
    return FewShotRun(
        model=base_run.model,
        seed=base_run.seed,
        budget=int(budget),
        tau=float(tau) if tau is not None else base_run.tau,
        sensitivity_tau=sensitivity_tau,
        primary_metrics=primary_metrics,
        primary_counts=primary_counts,
        sensitivity_metrics=sensitivity_metrics,
        sensitivity_counts=sensitivity_counts,
        val_metrics=val_metrics,
        provenance=provenance,
        dataset=dataset_summary,
        frames=frames,
        zero_shot=zero_shot,
        path=metrics_path,
    )


def _resolve_zero_shot_outputs_path(
    base_run: CommonRun,
    payload: Mapping[str, Any],
) -> Optional[Path]:
    candidates: List[Path] = []

    def _push(value: object) -> None:
        if isinstance(value, Path):
            path = value if value.is_absolute() else base_run.metrics_path.parent / value
            candidates.append(path)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return
            path_obj = Path(text)
            if not path_obj.is_absolute():
                path_obj = base_run.metrics_path.parent / path_obj
            candidates.append(path_obj)

    zero_block = payload.get("zero_shot")
    if isinstance(zero_block, Mapping):
        for key in ("outputs_csv", "outputs_path", "test_outputs_csv", "outputs"):
            _push(zero_block.get(key))
        provenance = zero_block.get("provenance")
        if isinstance(provenance, Mapping):
            for key in ("outputs_csv", "test_outputs_csv"):
                _push(provenance.get(key))

    provenance_block = base_run.payload.get("provenance")
    if isinstance(provenance_block, Mapping):
        for key in (
            "zero_shot_outputs",
            "zero_shot_outputs_csv",
            "test_zero_shot_outputs_csv",
        ):
            _push(provenance_block.get(key))
        for nested in provenance_block.values():
            if isinstance(nested, Mapping):
                for key in ("zero_shot_outputs_csv", "outputs_csv", "test_outputs_csv"):
                    _push(nested.get(key))

    for key in (
        "zero_shot_outputs",
        "zero_shot_outputs_csv",
        "test_zero_shot_outputs_csv",
    ):
        _push(payload.get(key))

    stem = base_run.metrics_path.stem
    base_name = stem[:-5] if stem.endswith("_last") else stem
    _push(base_run.metrics_path.with_name(f"{base_name}_zeroshot_outputs.csv"))
    _push(base_run.metrics_path.with_name(f"{stem}_zeroshot_outputs.csv"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_zero_shot(base_run: CommonRun, payload: Mapping[str, Any]) -> Optional[ZeroShotResult]:
    zero_block: Optional[Mapping[str, Any]] = None
    primary_zero = payload.get("test_primary_zero_shot")
    if isinstance(primary_zero, Mapping):
        zero_block = primary_zero
    else:
        zero_root = payload.get("zero_shot")
        if isinstance(zero_root, Mapping):
            for key in ("test_primary", "primary", "metrics"):
                candidate = zero_root.get(key)
                if isinstance(candidate, Mapping):
                    zero_block = candidate
                    break
            if zero_block is None:
                zero_block = zero_root
    metrics = _extract_metrics(zero_block)
    counts = _extract_counts(zero_block)
    tau = _extract_tau(zero_block, base_run.tau)
    outputs_path = _resolve_zero_shot_outputs_path(base_run, payload)
    frames: Dict[str, EvalFrame] = {}
    resolved_path: Optional[Path] = outputs_path
    if outputs_path is not None:
        try:
            zero_frames, _ = load_outputs_csv(
                outputs_path,
                tau=float(tau) if tau is not None else base_run.tau,
            )
        except (OSError, ValueError):
            resolved_path = None
        else:
            frames = _frames_to_eval(zero_frames)
    if resolved_path is not None:
        try:
            loss_value = bce_loss_from_csv(resolved_path)
        except (OSError, FileNotFoundError):
            loss_value = None
        else:
            existing_loss = _coerce_float(metrics.get("loss")) if metrics else None
            if loss_value is not None and math.isfinite(loss_value) and (
                existing_loss is None or not math.isfinite(existing_loss)
            ):
                metrics.setdefault("loss", float(loss_value))
    if not metrics and not counts and not frames:
        return None
    return ZeroShotResult(
        tau=float(tau) if tau is not None else None,
        metrics=metrics,
        counts=counts,
        frames=frames,
        path=resolved_path,
    )


def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    loader: Optional[ResultLoader] = None,
) -> Dict[str, Dict[int, Dict[int, FewShotRun]]]:
    model_filter = {str(m) for m in models} if models else None
    runs: DefaultDict[str, DefaultDict[int, Dict[int, FewShotRun]]] = defaultdict(lambda: defaultdict(dict))
    active_loader = loader or _get_loader()
    for metrics_path in sorted(root.rglob("*_last.metrics.json")):
        try:
            run = load_run(metrics_path, loader=active_loader)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load metrics from {metrics_path} (missing per-frame outputs)"
            ) from exc
        except (OSError, ValueError, GuardrailViolation):
            continue
        if model_filter and run.model not in model_filter:
            continue
        runs[run.model][run.budget][run.seed] = run
    return {
        model: {budget: dict(seed_map) for budget, seed_map in sorted(per_model.items())}
        for model, per_model in runs.items()
    }


def _collect_values(
    per_seed: Mapping[int, FewShotRun],
    metric: str,
    *,
    policy: str,
) -> Tuple[List[float], Dict[int, float]]:
    values: List[float] = []
    per_seed_values: Dict[int, float] = {}
    for seed, run in per_seed.items():
        metrics = _get_policy_metrics(run, policy)
        value = _coerce_float(metrics.get(metric)) if isinstance(metrics, Mapping) else None
        if value is None:
            continue
        values.append(float(value))
        per_seed_values[int(seed)] = float(value)
    return values, per_seed_values


def _collect_val_values(per_seed: Mapping[int, FewShotRun], metric: str) -> Tuple[List[float], Dict[int, float]]:
    values: List[float] = []
    per_seed_values: Dict[int, float] = {}
    for seed, run in per_seed.items():
        value = _coerce_float(run.val_metrics.get(metric))
        if value is None:
            continue
        values.append(float(value))
        per_seed_values[int(seed)] = float(value)
    return values, per_seed_values


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.array(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return mean, std


def _build_cluster_set(run: FewShotRun) -> ClusterSet:
    return build_cluster_set(
        run.frames.values(),
        is_positive=lambda record: record.label == 1,
        record_id=lambda record: record.frame_id,
        positive_key=lambda record: None,
        negative_key=lambda record: record.center_id
        or record.sequence_id
        or record.case_id,
    )


def _get_policy_metrics(run: FewShotRun, policy: str) -> Mapping[str, float]:
    if policy == "primary":
        return run.primary_metrics
    if policy == "sensitivity":
        return run.sensitivity_metrics
    if policy == "zero_shot" and run.zero_shot is not None:
        return run.zero_shot.metrics
    return {}


def _get_policy_counts(run: FewShotRun, policy: str) -> Mapping[str, int]:
    if policy == "primary":
        return run.primary_counts
    if policy == "sensitivity":
        return run.sensitivity_counts
    if policy == "zero_shot" and run.zero_shot is not None:
        return run.zero_shot.counts
    return {}


def _get_policy_tau(run: FewShotRun, policy: str) -> Optional[float]:
    if policy == "primary":
        return run.tau
    if policy == "sensitivity":
        return run.sensitivity_tau if run.sensitivity_tau is not None else run.tau
    if policy == "zero_shot" and run.zero_shot is not None:
        if run.zero_shot.tau is not None:
            return run.zero_shot.tau
        return run.tau
    return None


def _get_policy_frames(run: FewShotRun, policy: str) -> Mapping[str, EvalFrame]:
    if policy in ("primary", "sensitivity"):
        return run.frames
    if policy == "zero_shot" and run.zero_shot is not None:
        return run.zero_shot.frames
    return {}


def _metrics_for_sample(
    run: FewShotRun,
    frame_ids: Sequence[str],
    metric: str,
    *,
    policy: str,
) -> Optional[float]:
    frames = _get_policy_frames(run, policy)
    records = [frames[fid] for fid in frame_ids if fid in frames]
    if not records:
        return None
    probs = np.array([rec.prob for rec in records], dtype=float)
    labels = np.array([rec.label for rec in records], dtype=int)
    tau = _get_policy_tau(run, policy)
    threshold = run.tau if tau is None else float(tau)
    metrics = compute_binary_metrics(probs, labels, threshold, metric_keys=AGG_METRICS)
    value = metrics.get(metric)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _bootstrap_delta_across_seeds(
    colon_runs: Sequence[FewShotRun],
    baseline_runs: Sequence[FewShotRun],
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
    policy: str,
) -> List[float]:
    if bootstrap <= 0:
        return []
    baseline_by_seed = {run.seed: run for run in baseline_runs}
    rng = np.random.default_rng(rng_seed)
    cluster_cache: Dict[int, ClusterSet] = {}
    replicates: List[float] = []
    for _ in range(bootstrap):
        deltas: List[float] = []
        for colon_run in colon_runs:
            baseline_run = baseline_by_seed.get(colon_run.seed)
            if baseline_run is None:
                continue
            if cluster_cache.get(colon_run.seed) is None:
                cluster_cache[colon_run.seed] = _build_cluster_set(colon_run)
            clusters = cluster_cache[colon_run.seed]
            frame_ids = sample_cluster_ids(clusters, rng)
            colon_value = _metrics_for_sample(colon_run, frame_ids, metric, policy=policy)
            baseline_value = _metrics_for_sample(baseline_run, frame_ids, metric, policy=policy)
            if colon_value is None or baseline_value is None:
                continue
            deltas.append(colon_value - baseline_value)
        if deltas:
            replicates.append(float(np.mean(deltas)))
    return replicates


def _compute_ci(samples: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    if not samples:
        return {"lower": float("nan"), "upper": float("nan")}
    lower_pct = 50.0 * (1.0 - confidence)
    upper_pct = 100.0 - lower_pct
    array = np.array(samples, dtype=float)
    return {
        "lower": float(np.percentile(array, lower_pct)),
        "upper": float(np.percentile(array, upper_pct)),
    }


def _collect_shared_seeds(
    colon_map: Mapping[int, FewShotRun],
    baseline_map: Mapping[int, FewShotRun],
) -> List[int]:
    return sorted(set(colon_map.keys()) & set(baseline_map.keys()))


def _clean_optional_text(value: object) -> Optional[str]:
    text = _clean_text(value)
    return text if text else None


def _extract_dataset_block(dataset: Mapping[str, Any], keys: Sequence[str]) -> Optional[Mapping[str, Any]]:
    for key in keys:
        block = dataset.get(key)
        if isinstance(block, Mapping):
            return block
    return None


def _extract_pack_identifier(run: FewShotRun) -> Optional[str]:
    dataset = run.dataset if isinstance(run.dataset, Mapping) else {}
    test_block = _extract_dataset_block(
        dataset,
        ("test_primary", "test", "test_split", "test_set"),
    )
    if isinstance(test_block, Mapping):
        for key in ("pack", "pack_name", "name", "identifier", "pack_spec"):
            value = _clean_optional_text(test_block.get(key))
            if value:
                return value
    provenance = run.provenance if isinstance(run.provenance, Mapping) else {}
    for key in (
        "test_pack",
        "eval_pack",
        "pack",
        "pack_spec",
        "train_pack",
    ):
        value = _clean_optional_text(provenance.get(key))
        if value:
            return value
    return None


def _extract_csv_sha256_from_block(block: Mapping[str, Any]) -> Optional[str]:
    for key in ("csv_sha256", "sha256", "hash"):
        value = _clean_optional_text(block.get(key))
        if value:
            return value
    return None


def _extract_test_csv_sha256(run: FewShotRun) -> Optional[str]:
    dataset = run.dataset if isinstance(run.dataset, Mapping) else {}
    test_block = _extract_dataset_block(
        dataset,
        ("test_primary", "test", "test_split", "test_set"),
    )
    if isinstance(test_block, Mapping):
        digest = _extract_csv_sha256_from_block(test_block)
        if digest:
            return digest
    provenance = run.provenance if isinstance(run.provenance, Mapping) else {}
    for key in (
        "test_csv_sha256",
        "test_primary_csv_sha256",
        "eval_csv_sha256",
    ):
        value = _clean_optional_text(provenance.get(key))
        if value:
            return value
    for nested in provenance.values():
        if isinstance(nested, Mapping):
            digest = _extract_csv_sha256_from_block(nested)
            if digest:
                return digest
    return None


def _compute_aulc_from_series(budgets: Sequence[int], values: Sequence[float]) -> float:
    if len(budgets) < 2 or len(values) != len(budgets):
        return float("nan")
    x = np.array(list(budgets), dtype=float)
    y = np.array(list(values), dtype=float)
    width = float(x[-1] - x[0])
    if width <= 0:
        return float("nan")
    area = float(np.trapz(y, x))
    return area / width


def _bootstrap_aulc_delta(
    colon_runs_per_budget: Mapping[int, Mapping[int, FewShotRun]],
    baseline_runs_per_budget: Mapping[int, Mapping[int, FewShotRun]],
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
    policy: str,
) -> List[float]:
    if bootstrap <= 0:
        return []
    shared_budgets = sorted(set(colon_runs_per_budget.keys()) & set(baseline_runs_per_budget.keys()))
    if not shared_budgets:
        return []
    ensure_expected_seeds(
        {
            **{f"colon@b{budget}": colon_runs_per_budget.get(budget, {}) for budget in shared_budgets},
            **{f"baseline@b{budget}": baseline_runs_per_budget.get(budget, {}) for budget in shared_budgets},
        },
        expected_seeds=EXPECTED_SEEDS,
        context="Experiment 5C bootstrap ΔAULC",
    )
    shared_seeds = list(EXPECTED_SEEDS)
    rng = np.random.default_rng(rng_seed)
    cluster_cache: Dict[int, ClusterSet] = {}
    replicates: List[float] = []
    for _ in range(bootstrap):
        deltas: List[float] = []
        for seed in shared_seeds:
            colon_series: List[float] = []
            baseline_series: List[float] = []
            clusters = cluster_cache.get(seed)
            if clusters is None:
                first_budget = shared_budgets[0]
                colon_reference = colon_runs_per_budget[first_budget][seed]
                clusters = _build_cluster_set(colon_reference)
                cluster_cache[seed] = clusters
            frame_ids = sample_cluster_ids(clusters, rng)
            valid = True
            for budget in shared_budgets:
                colon_run = colon_runs_per_budget.get(budget, {}).get(seed)
                baseline_run = baseline_runs_per_budget.get(budget, {}).get(seed)
                if colon_run is None or baseline_run is None:
                    valid = False
                    break
                colon_value = _metrics_for_sample(colon_run, frame_ids, metric, policy=policy)
                baseline_value = _metrics_for_sample(baseline_run, frame_ids, metric, policy=policy)
                if colon_value is None or baseline_value is None:
                    valid = False
                    break
                colon_series.append(colon_value)
                baseline_series.append(baseline_value)
            if not valid:
                continue
            colon_aulc = _compute_aulc_from_series(shared_budgets, colon_series)
            baseline_aulc = _compute_aulc_from_series(shared_budgets, baseline_series)
            if not math.isfinite(colon_aulc) or not math.isfinite(baseline_aulc):
                continue
            deltas.append(colon_aulc - baseline_aulc)
        if deltas:
            replicates.append(float(np.mean(deltas)))
    return replicates


def _compute_pairwise_deltas(
    runs_by_model: Mapping[str, Mapping[int, Mapping[int, FewShotRun]]],
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
    expected_seeds: Sequence[int],
    policy: str,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    colon_runs = runs_by_model.get(TARGET_MODEL)
    if not colon_runs:
        return {}
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for baseline in BASELINE_MODELS:
        baseline_runs = runs_by_model.get(baseline)
        if not baseline_runs:
            continue
        shared_budgets = sorted(set(colon_runs.keys()) & set(baseline_runs.keys()))
        baseline_result: Dict[int, Dict[str, Any]] = {}
        for budget in shared_budgets:
            colon_seed_map = colon_runs.get(budget, {})
            baseline_seed_map = baseline_runs.get(budget, {})
            ensure_expected_seeds(
                {
                    f"{TARGET_MODEL}@b{budget}": colon_seed_map,
                    f"{baseline}@b{budget}": baseline_seed_map,
                },
                expected_seeds=expected_seeds,
                context=f"Experiment 5C pairwise ({baseline}) budget {budget}",
            )
            seeds = list(expected_seeds)
            deltas: List[float] = []
            colon_sequence = [colon_seed_map[seed] for seed in seeds]
            baseline_sequence = [baseline_seed_map[seed] for seed in seeds]
            for seed in seeds:
                colon_metrics = _get_policy_metrics(colon_seed_map[seed], policy)
                baseline_metrics = _get_policy_metrics(baseline_seed_map[seed], policy)
                colon_value = _coerce_float(colon_metrics.get(metric)) if isinstance(colon_metrics, Mapping) else None
                baseline_value = _coerce_float(baseline_metrics.get(metric)) if isinstance(baseline_metrics, Mapping) else None
                if colon_value is None or baseline_value is None:
                    continue
                deltas.append(colon_value - baseline_value)
            if not deltas:
                continue
            replicates = _bootstrap_delta_across_seeds(
                colon_sequence,
                baseline_sequence,
                metric=metric,
                bootstrap=bootstrap,
                rng_seed=rng_seed,
                policy=policy,
            )
            baseline_result[budget] = {
                "delta": float(np.mean(deltas)),
                "ci": _compute_ci(replicates) if replicates else {"lower": float("nan"), "upper": float("nan")},
                "replicates": replicates,
                "seeds": len(deltas),
            }
        if baseline_result:
            results[baseline] = baseline_result
    return results


def _compute_aulc_summary(
    runs_by_model: Mapping[str, Mapping[int, Mapping[int, FewShotRun]]],
    metric: str,
    *,
    policy: str,
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for model, per_budget in runs_by_model.items():
        budgets = sorted(per_budget.keys())
        seed_sets: DefaultDict[int, Dict[int, FewShotRun]] = defaultdict(dict)
        for budget in budgets:
            for seed, run in per_budget[budget].items():
                seed_sets[seed][budget] = run
        aulc_values: List[float] = []
        per_seed_values: Dict[int, float] = {}
        for seed, budget_map in seed_sets.items():
            available_budgets = sorted(budget_map.keys())
            if len(available_budgets) < 2:
                continue
            metrics_series: List[float] = []
            for budget in available_budgets:
                metrics = _get_policy_metrics(budget_map[budget], policy)
                value = _coerce_float(metrics.get(metric)) if isinstance(metrics, Mapping) else None
                if value is None:
                    metrics_series = []
                    break
                metrics_series.append(value)
            if not metrics_series:
                continue
            aulc_value = _compute_aulc_from_series(available_budgets, metrics_series)
            if not math.isfinite(aulc_value):
                continue
            aulc_values.append(aulc_value)
            per_seed_values[int(seed)] = aulc_value
        mean, std = _mean_std(aulc_values)
        summary[model] = {
            "mean": mean,
            "std": std,
            "n": len(aulc_values),
            "per_seed": per_seed_values,
        }
    return summary


def _compute_aulc_deltas(
    runs_by_model: Mapping[str, Mapping[int, Mapping[int, FewShotRun]]],
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
    expected_seeds: Sequence[int],
    policy: str,
) -> Dict[str, Dict[str, Any]]:
    colon_runs = runs_by_model.get(TARGET_MODEL)
    if not colon_runs:
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    colon_aulc = _compute_aulc_summary({TARGET_MODEL: colon_runs}, metric, policy=policy)[TARGET_MODEL]
    for baseline in BASELINE_MODELS:
        baseline_runs = runs_by_model.get(baseline)
        if not baseline_runs:
            continue
        shared_budgets = sorted(set(colon_runs.keys()) & set(baseline_runs.keys()))
        if not shared_budgets:
            continue
        per_seed_deltas: List[float] = []
        ensure_expected_seeds(
            {
                **{
                    f"{TARGET_MODEL}@b{budget}": colon_runs.get(budget, {})
                    for budget in shared_budgets
                },
                **{
                    f"{baseline}@b{budget}": baseline_runs.get(budget, {})
                    for budget in shared_budgets
                },
            },
            expected_seeds=expected_seeds,
            context=f"Experiment 5C ΔAULC ({baseline})",
        )
        for seed in expected_seeds:
            colon_series: List[float] = []
            baseline_series: List[float] = []
            budgets = []
            for budget in shared_budgets:
                colon_metrics = _get_policy_metrics(colon_runs[budget][seed], policy)
                baseline_metrics = _get_policy_metrics(baseline_runs[budget][seed], policy)
                colon_value = _coerce_float(colon_metrics.get(metric)) if isinstance(colon_metrics, Mapping) else None
                baseline_value = _coerce_float(baseline_metrics.get(metric)) if isinstance(baseline_metrics, Mapping) else None
                if colon_value is None or baseline_value is None:
                    colon_series = []
                    break
                colon_series.append(colon_value)
                baseline_series.append(baseline_value)
                budgets.append(budget)
            if not colon_series:
                continue
            colon_aulc_value = _compute_aulc_from_series(budgets, colon_series)
            baseline_aulc_value = _compute_aulc_from_series(budgets, baseline_series)
            if not math.isfinite(colon_aulc_value) or not math.isfinite(baseline_aulc_value):
                continue
            per_seed_deltas.append(colon_aulc_value - baseline_aulc_value)
        replicates = _bootstrap_aulc_delta(
            colon_runs,
            baseline_runs,
            metric=metric,
            bootstrap=bootstrap,
            rng_seed=rng_seed,
            policy=policy,
            expected_seeds=expected_seeds,
        )
        results[baseline] = {
            "delta": float(np.mean(per_seed_deltas)) if per_seed_deltas else float("nan"),
            "ci": _compute_ci(replicates) if replicates else {"lower": float("nan"), "upper": float("nan")},
            "replicates": replicates,
            "seeds": len(per_seed_deltas),
            "colon_aulc": colon_aulc,
        }
    return results


def _ensure_zero_shot_complete(
    seed_map: Mapping[int, FewShotRun],
    *,
    expected_seeds: Sequence[int],
    context: str,
) -> None:
    missing: List[int] = []
    for seed in expected_seeds:
        run = seed_map.get(seed)
        if run is None or run.zero_shot is None or not _get_policy_metrics(run, "zero_shot"):
            missing.append(int(seed))
        elif not _get_policy_frames(run, "zero_shot"):
            raise GuardrailViolation(
                f"Missing zero-shot outputs for seed {seed} in {context}"
            )
    if missing:
        raise GuardrailViolation(
            f"Missing zero-shot metrics for seeds {', '.join(str(s) for s in missing)} in {context}"
        )


def _compute_test_composition(
    runs_by_model: Mapping[str, Mapping[int, Mapping[int, FewShotRun]]],
    *,
    expected_seeds: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    composition: Dict[int, Dict[str, Any]] = {}
    for model, per_budget in runs_by_model.items():
        for budget, seed_map in per_budget.items():
            ensure_expected_seeds(
                {f"{model}@b{budget}": seed_map},
                expected_seeds=expected_seeds,
                context=f"Experiment 5C composition ({model}) budget {budget}",
            )
            for seed, run in seed_map.items():
                counts = _get_policy_counts(run, "primary")
                n_pos = counts.get("n_pos")
                n_neg = counts.get("n_neg")
                if n_pos is None or n_neg is None:
                    continue
                total = n_pos + n_neg
                prevalence = float(n_pos / total) if total > 0 else float("nan")
                digest = _extract_test_csv_sha256(run)
                if not digest:
                    raise GuardrailViolation(
                        f"Missing test CSV SHA256 for {model} seed {seed} budget {budget}"
                    )
                pack = _extract_pack_identifier(run)
                entry = composition.get(budget)
                if entry is None:
                    entry = {
                        "budget": budget,
                        "n_pos": int(n_pos),
                        "n_neg": int(n_neg),
                        "prevalence": prevalence,
                        "test_sha256": digest,
                        "pack": pack,
                        "models": {model},
                        "seeds": {seed},
                    }
                    composition[budget] = entry
                else:
                    if entry["n_pos"] != int(n_pos) or entry["n_neg"] != int(n_neg):
                        raise GuardrailViolation(
                            f"Test composition mismatch for budget {budget}: "
                            f"counts differ between runs"
                        )
                    if entry["test_sha256"] != digest:
                        raise GuardrailViolation(
                            f"Test CSV SHA mismatch for budget {budget}: "
                            f"expected {entry['test_sha256']}, found {digest}"
                        )
                    if entry.get("pack") and pack and entry["pack"] != pack:
                        raise GuardrailViolation(
                            f"Pack identifier mismatch for budget {budget}: "
                            f"{entry['pack']} vs {pack}"
                        )
                    if not math.isfinite(entry["prevalence"]) and not math.isfinite(prevalence):
                        pass
                    else:
                        if not math.isclose(
                            entry["prevalence"],
                            prevalence,
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        ):
                            raise GuardrailViolation(
                                f"Prevalence mismatch for budget {budget}"
                            )
                    if entry.get("pack") is None and pack:
                        entry["pack"] = pack
                    entry["models"].add(model)
                    entry["seeds"].add(seed)
    for entry in composition.values():
        entry["models"] = sorted(entry.get("models", ()))
        entry["seeds"] = sorted(entry.get("seeds", ()))
    return composition


def _bootstrap_gain_across_seeds(
    runs: Sequence[FewShotRun],
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
) -> List[float]:
    if bootstrap <= 0:
        return []
    rng = np.random.default_rng(rng_seed)
    cluster_cache: Dict[int, ClusterSet] = {}
    replicates: List[float] = []
    for _ in range(bootstrap):
        gains: List[float] = []
        for run in runs:
            if run.zero_shot is None:
                continue
            if cluster_cache.get(run.seed) is None:
                cluster_cache[run.seed] = _build_cluster_set(run)
            clusters = cluster_cache[run.seed]
            frame_ids = sample_cluster_ids(clusters, rng)
            adapted = _metrics_for_sample(run, frame_ids, metric, policy="primary")
            baseline = _metrics_for_sample(run, frame_ids, metric, policy="zero_shot")
            if adapted is None or baseline is None:
                continue
            gains.append(adapted - baseline)
        if gains:
            replicates.append(float(np.mean(gains)))
    return replicates


def _compute_gain_summary(
    runs_by_model: Mapping[str, Mapping[int, Mapping[int, FewShotRun]]],
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: int,
    expected_seeds: Sequence[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for model, per_budget in runs_by_model.items():
        model_entry: Dict[int, Dict[str, Any]] = {}
        for budget, seed_map in per_budget.items():
            _ensure_zero_shot_complete(
                seed_map,
                expected_seeds=expected_seeds,
                context=f"Experiment 5C gain ({model}) budget {budget}",
            )
            metric_block: Dict[str, Any] = {}
            runs_sequence = [seed_map[seed] for seed in expected_seeds if seed in seed_map]
            for metric in metrics:
                diffs: List[float] = []
                per_seed_values: Dict[int, float] = {}
                valid = True
                for seed in expected_seeds:
                    run = seed_map.get(seed)
                    if run is None or run.zero_shot is None:
                        valid = False
                        break
                    adapted_metrics = _get_policy_metrics(run, "primary")
                    baseline_metrics = _get_policy_metrics(run, "zero_shot")
                    adapted_value = (
                        _coerce_float(adapted_metrics.get(metric))
                        if isinstance(adapted_metrics, Mapping)
                        else None
                    )
                    baseline_value = (
                        _coerce_float(baseline_metrics.get(metric))
                        if isinstance(baseline_metrics, Mapping)
                        else None
                    )
                    if adapted_value is None or baseline_value is None:
                        valid = False
                        break
                    delta = float(adapted_value - baseline_value)
                    diffs.append(delta)
                    per_seed_values[int(seed)] = delta
                if not valid or not diffs:
                    continue
                mean, std = _mean_std(diffs)
                replicates = _bootstrap_gain_across_seeds(
                    runs_sequence,
                    metric=metric,
                    bootstrap=bootstrap,
                    rng_seed=rng_seed,
                )
                metric_block[metric] = {
                    "mean": mean,
                    "std": std,
                    "n": len(diffs),
                    "per_seed": per_seed_values,
                    "ci": _compute_ci(replicates)
                    if replicates
                    else {"lower": float("nan"), "upper": float("nan")},
                    "replicates": replicates,
                }
            if metric_block:
                model_entry[budget] = metric_block
        if model_entry:
            results[model] = model_entry
    return results


def _summarize_policy(
    runs_by_model: Mapping[str, Mapping[int, Mapping[int, FewShotRun]]],
    *,
    policy: str,
    bootstrap: int,
    rng_seed: int,
    seed_validation: SeedValidationResult,
) -> Dict[str, Any]:
    performance: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for model, model_budgets in runs_by_model.items():
        model_entry: Dict[int, Dict[str, Any]] = {}
        for budget, seed_map in model_budgets.items():
            metrics_block: Dict[str, Any] = {}
            for metric in AGG_METRICS:
                values, per_seed_values = _collect_values(seed_map, metric, policy=policy)
                if not values:
                    continue
                mean, std = _mean_std(values)
                metrics_block[metric] = {
                    "mean": mean,
                    "std": std,
                    "n": len(values),
                    "per_seed": per_seed_values,
                }
            val_block: Dict[str, Any] = {}
            for metric in VAL_METRICS:
                values, per_seed_values = _collect_val_values(seed_map, metric)
                if not values:
                    continue
                mean, std = _mean_std(values)
                val_block[metric] = {
                    "mean": mean,
                    "std": std,
                    "n": len(values),
                    "per_seed": per_seed_values,
                }
            model_entry[budget] = {
                "metrics": metrics_block,
                "val_metrics": val_block,
                "seeds": len(seed_map),
            }
        performance[model] = dict(sorted(model_entry.items()))
    learning_curves: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for metric in PAIRWISE_METRICS:
        metric_entry: Dict[str, Dict[int, Dict[str, float]]] = {}
        for model, per_budget in performance.items():
            if not isinstance(per_budget, dict):
                continue
            series: Dict[int, Dict[str, float]] = {}
            for budget, stats in per_budget.items():
                if not isinstance(stats, dict):
                    continue
                metrics_block = stats.get("metrics")
                if not isinstance(metrics_block, Mapping):
                    continue
                metric_stats = metrics_block.get(metric)
                if not isinstance(metric_stats, Mapping):
                    continue
                mean_value = metric_stats.get("mean")
                if mean_value is None:
                    continue
                std_value = metric_stats.get("std")
                n_value = metric_stats.get("n")
                series[budget] = {
                    "mean": float(mean_value),
                    "std": float(std_value) if isinstance(std_value, (int, float)) else 0.0,
                    "n": int(n_value) if isinstance(n_value, (int, float)) else 0,
                }
            if series:
                metric_entry[model] = dict(sorted(series.items()))
        learning_curves[metric] = metric_entry
    pairwise: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for metric in PAIRWISE_METRICS:
        pairwise[metric] = _compute_pairwise_deltas(
            runs_by_model,
            metric=metric,
            bootstrap=bootstrap,
            rng_seed=rng_seed,
            expected_seeds=seed_validation.expected_seeds,
            policy=policy,
        )
    aulc_summary: Dict[str, Dict[str, Any]] = {}
    for metric in AULC_METRICS:
        aulc_summary[metric] = _compute_aulc_summary(
            runs_by_model,
            metric,
            policy=policy,
        )
    aulc_deltas: Dict[str, Dict[str, Any]] = {}
    for metric in AULC_METRICS:
        aulc_deltas[metric] = _compute_aulc_deltas(
            runs_by_model,
            metric=metric,
            bootstrap=bootstrap,
            rng_seed=rng_seed,
            expected_seeds=seed_validation.expected_seeds,
            policy=policy,
        )
    return {
        "performance": performance,
        "learning_curves": learning_curves,
        "pairwise": pairwise,
        "aulc": aulc_summary,
        "aulc_deltas": aulc_deltas,
    }


def summarize_runs(
    runs_by_model: Mapping[str, Mapping[int, Mapping[int, FewShotRun]]],
    *,
    bootstrap: int = 2000,
    rng_seed: int = 12345,
    target_model: str = "ssl_imnet",
    target_budget: int = 500,
) -> Dict[str, Any]:
    if runs_by_model:
        seed_groups = {
            f"{model}@b{budget}": seed_map
            for model, per_budget in runs_by_model.items()
            for budget, seed_map in per_budget.items()
        }
        seed_validation = ensure_expected_seeds(
            seed_groups,
            expected_seeds=EXPECTED_SEEDS,
            context="Experiment 5C",
        )
    else:
        seed_validation = SeedValidationResult((), MappingProxyType({}))
    policy_summaries = {
        policy: _summarize_policy(
            runs_by_model,
            policy=policy,
            bootstrap=max(0, bootstrap),
            rng_seed=rng_seed,
            seed_validation=seed_validation,
        )
        for policy in ("primary", "sensitivity")
    }
    test_composition = _compute_test_composition(
        runs_by_model,
        expected_seeds=seed_validation.expected_seeds,
    )
    gain_summary = _compute_gain_summary(
        runs_by_model,
        metrics=AGG_METRICS,
        bootstrap=max(0, bootstrap),
        rng_seed=rng_seed,
        expected_seeds=seed_validation.expected_seeds,
    )
    budgets = sorted({budget for per_model in runs_by_model.values() for budget in per_model.keys()})
    runs_catalog: List[Dict[str, Any]] = []
    for model, budget_map in runs_by_model.items():
        for budget, seed_map in budget_map.items():
            for seed, run in seed_map.items():
                record = {
                    "model": model,
                    "budget": budget,
                    "seed": seed,
                    "tau": run.tau,
                    "sensitivity_tau": run.sensitivity_tau,
                    "path": str(run.path),
                }
                if run.zero_shot is not None:
                    record["zero_shot_tau"] = run.zero_shot.tau
                    if run.zero_shot.path is not None:
                        record["zero_shot_path"] = str(run.zero_shot.path)
                runs_catalog.append(record)
    return {
        "metadata": {
            "bootstrap": int(bootstrap),
            "rng_seed": int(rng_seed),
            "models": sorted(runs_by_model.keys()),
            "budgets": budgets,
            "target_model": target_model,
            "target_budget": target_budget,
        },
        "runs": runs_catalog,
        "policies": policy_summaries,
        "test_composition": test_composition,
        "gains": gain_summary,
        "validated_seeds": list(seed_validation.expected_seeds),
        "seed_groups": {
            key: list(values) for key, values in seed_validation.observed_seeds.items()
        },
    }


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _get_policy_block(summary: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    policies = summary.get("policies")
    if not isinstance(policies, Mapping):
        raise ValueError("Summary payload does not contain policy aggregates")
    block = policies.get(policy)
    if not isinstance(block, Mapping):
        raise ValueError(f"Summary does not include '{policy}' policy aggregates")
    return block


def write_test_composition_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    composition = summary.get("test_composition")
    if not isinstance(composition, Mapping):
        raise ValueError("Summary payload does not contain test composition data")
    rows: List[Dict[str, Any]] = []
    for budget, payload in sorted(composition.items()):
        if not isinstance(payload, Mapping):
            continue
        models = payload.get("models") if isinstance(payload.get("models"), Sequence) else ()
        seeds = payload.get("seeds") if isinstance(payload.get("seeds"), Sequence) else ()
        rows.append(
            {
                "budget": budget,
                "n_pos": payload.get("n_pos"),
                "n_neg": payload.get("n_neg"),
                "prevalence": payload.get("prevalence"),
                "test_sha256": payload.get("test_sha256"),
                "pack": payload.get("pack"),
                "models": "|".join(str(m) for m in models),
                "seeds": "|".join(str(s) for s in seeds),
            }
        )
    if not rows:
        raise ValueError("No test composition rows available for CSV export")
    fieldnames = [
        "budget",
        "n_pos",
        "n_neg",
        "prevalence",
        "test_sha256",
        "pack",
        "models",
        "seeds",
    ]
    _write_rows(output_path, fieldnames, rows)


def write_performance_csv(
    summary: Mapping[str, Any],
    output_path: Path,
    *,
    policy: str = "primary",
) -> None:
    policy_block = _get_policy_block(summary, policy)
    performance = policy_block.get("performance")
    if not isinstance(performance, Mapping):
        raise ValueError(f"Summary payload does not contain performance data for policy '{policy}'")
    rows: List[Dict[str, Any]] = []
    for model, per_budget in performance.items():
        if not isinstance(per_budget, Mapping):
            continue
        for budget, stats in per_budget.items():
            metrics_block = stats.get("metrics") if isinstance(stats, Mapping) else None
            if not isinstance(metrics_block, Mapping):
                continue
            for metric, metric_stats in metrics_block.items():
                if not isinstance(metric_stats, Mapping):
                    continue
                rows.append(
                    {
                        "policy": policy,
                        "model": model,
                        "budget": budget,
                        "metric": metric,
                        "mean": metric_stats.get("mean"),
                        "std": metric_stats.get("std"),
                        "n": metric_stats.get("n"),
                    }
                )
    if not rows:
        raise ValueError("No performance rows available for CSV export")
    fieldnames = ["policy", "model", "budget", "metric", "mean", "std", "n"]
    _write_rows(output_path, fieldnames, rows)


def write_gain_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    gains = summary.get("gains")
    if not isinstance(gains, Mapping):
        raise ValueError("Summary payload does not contain gain data")
    rows: List[Dict[str, Any]] = []
    for model, per_budget in gains.items():
        if not isinstance(per_budget, Mapping):
            continue
        for budget, metric_block in per_budget.items():
            if not isinstance(metric_block, Mapping):
                continue
            for metric, stats in metric_block.items():
                if not isinstance(stats, Mapping):
                    continue
                ci_value = stats.get("ci") if isinstance(stats.get("ci"), Mapping) else {}
                rows.append(
                    {
                        "model": model,
                        "budget": budget,
                        "metric": metric,
                        "mean": stats.get("mean"),
                        "std": stats.get("std"),
                        "n": stats.get("n"),
                        "ci_lower": ci_value.get("lower"),
                        "ci_upper": ci_value.get("upper"),
                    }
                )
    if not rows:
        raise ValueError("No gain rows available for CSV export")
    fieldnames = ["model", "budget", "metric", "mean", "std", "n", "ci_lower", "ci_upper"]
    _write_rows(output_path, fieldnames, rows)


def write_pairwise_csv(
    summary: Mapping[str, Any],
    output_path: Path,
    *,
    policy: str = "primary",
) -> None:
    policy_block = _get_policy_block(summary, policy)
    pairwise = policy_block.get("pairwise")
    if not isinstance(pairwise, Mapping):
        raise ValueError(f"Summary payload does not contain pairwise data for policy '{policy}'")
    rows: List[Dict[str, Any]] = []
    for metric, baseline_block in pairwise.items():
        if not isinstance(baseline_block, Mapping):
            continue
        for baseline, per_budget in baseline_block.items():
            if not isinstance(per_budget, Mapping):
                continue
            for budget, stats in per_budget.items():
                if not isinstance(stats, Mapping):
                    continue
                ci_value = stats.get("ci") if isinstance(stats.get("ci"), Mapping) else {}
                rows.append(
                    {
                        "policy": policy,
                        "metric": metric,
                        "baseline": baseline,
                        "budget": budget,
                        "delta": stats.get("delta"),
                        "ci_lower": ci_value.get("lower"),
                        "ci_upper": ci_value.get("upper"),
                        "seeds": stats.get("seeds"),
                    }
                )
    if not rows:
        raise ValueError("No pairwise delta rows available for CSV export")
    fieldnames = [
        "policy",
        "metric",
        "baseline",
        "budget",
        "delta",
        "ci_lower",
        "ci_upper",
        "seeds",
    ]
    _write_rows(output_path, fieldnames, rows)


def write_learning_curves_csv(
    summary: Mapping[str, Any],
    output_path: Path,
    *,
    policy: str = "primary",
) -> None:
    policy_block = _get_policy_block(summary, policy)
    learning_curves = policy_block.get("learning_curves")
    if not isinstance(learning_curves, Mapping):
        raise ValueError(
            f"Summary payload does not contain learning curve data for policy '{policy}'"
        )
    rows: List[Dict[str, Any]] = []
    for metric, model_block in learning_curves.items():
        if not isinstance(model_block, Mapping):
            continue
        for model, per_budget in model_block.items():
            if not isinstance(per_budget, Mapping):
                continue
            for budget, stats in per_budget.items():
                if not isinstance(stats, Mapping):
                    continue
                rows.append(
                    {
                        "policy": policy,
                        "metric": metric,
                        "model": model,
                        "budget": budget,
                        "mean": stats.get("mean"),
                        "std": stats.get("std"),
                        "n": stats.get("n"),
                    }
                )
    if not rows:
        raise ValueError("No learning curve rows available for CSV export")
    fieldnames = ["policy", "metric", "model", "budget", "mean", "std", "n"]
    _write_rows(output_path, fieldnames, rows)


def write_aulc_csv(
    summary: Mapping[str, Any],
    output_path: Path,
    *,
    policy: str = "primary",
) -> None:
    policy_block = _get_policy_block(summary, policy)
    aulc = policy_block.get("aulc")
    aulc_deltas = policy_block.get("aulc_deltas")
    if not isinstance(aulc, Mapping):
        raise ValueError(f"Summary payload does not contain AULC data for policy '{policy}'")
    rows: List[Dict[str, Any]] = []
    for metric, model_block in aulc.items():
        if not isinstance(model_block, Mapping):
            continue
        for model, stats in model_block.items():
            if not isinstance(stats, Mapping):
                continue
            rows.append(
                {
                    "policy": policy,
                    "category": "model",
                    "metric": metric,
                    "model": model,
                    "baseline": None,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "n": stats.get("n"),
                    "delta": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "seeds": stats.get("n"),
                }
            )
    if isinstance(aulc_deltas, Mapping):
        for metric, baseline_block in aulc_deltas.items():
            if not isinstance(baseline_block, Mapping):
                continue
            for baseline, stats in baseline_block.items():
                if not isinstance(stats, Mapping):
                    continue
                ci_value = stats.get("ci") if isinstance(stats.get("ci"), Mapping) else {}
                rows.append(
                    {
                        "policy": policy,
                        "category": "delta",
                        "metric": metric,
                        "model": TARGET_MODEL,
                        "baseline": baseline,
                        "mean": stats.get("colon_aulc"),
                        "std": None,
                        "n": stats.get("seeds"),
                        "delta": stats.get("delta"),
                        "ci_lower": ci_value.get("lower"),
                        "ci_upper": ci_value.get("upper"),
                        "seeds": stats.get("seeds"),
                    }
                )
    if not rows:
        raise ValueError("No AULC rows available for CSV export")
    fieldnames = [
        "policy",
        "category",
        "metric",
        "model",
        "baseline",
        "mean",
        "std",
        "n",
        "delta",
        "ci_lower",
        "ci_upper",
        "seeds",
    ]
    _write_rows(output_path, fieldnames, rows)


__all__ = [
    "FewShotRun",
    "discover_runs",
    "load_run",
    "summarize_runs",
    "write_test_composition_csv",
    "write_performance_csv",
    "write_gain_csv",
    "write_pairwise_csv",
    "write_learning_curves_csv",
    "write_aulc_csv",
    "EXPECTED_SEEDS",
]
