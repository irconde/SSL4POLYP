"""Aggregation utilities for Experiment 5C few-shot adaptation results."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .common_loader import CommonFrame, get_default_loader, load_common_run
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

TARGET_MODEL = "ssl_colon"
BASELINE_MODELS: Tuple[str, ...] = ("ssl_imnet", "sup_imnet")
AGG_METRICS: Tuple[str, ...] = ("auprc", "f1", "auroc", "balanced_accuracy", "mcc")
VAL_METRICS: Tuple[str, ...] = ("auprc", "auroc", "loss")
PAIRWISE_METRICS: Tuple[str, ...] = ("auprc", "f1")
AULC_METRICS: Tuple[str, ...] = ("auprc", "f1")


@dataclass(frozen=True)
class EvalFrame:
    frame_id: str
    prob: float
    label: int
    case_id: Optional[str]
    sequence_id: Optional[str]
    center_id: Optional[str]


@dataclass
class FewShotRun:
    model: str
    seed: int
    budget: int
    tau: float
    test_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    provenance: Dict[str, Any]
    dataset: Dict[str, Any]
    frames: Dict[str, EvalFrame]
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
    active_loader = loader or get_default_loader()
    base_run = load_common_run(metrics_path, loader=active_loader)
    payload = base_run.payload
    provenance = dict(base_run.provenance)
    test_metrics = dict(base_run.primary_metrics)
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
    return FewShotRun(
        model=base_run.model,
        seed=base_run.seed,
        budget=int(budget),
        tau=base_run.tau,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        provenance=provenance,
        dataset=dataset_summary,
        frames=frames,
        path=metrics_path,
    )


def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    loader: Optional[ResultLoader] = None,
) -> Dict[str, Dict[int, Dict[int, FewShotRun]]]:
    model_filter = {str(m) for m in models} if models else None
    runs: DefaultDict[str, DefaultDict[int, Dict[int, FewShotRun]]] = defaultdict(lambda: defaultdict(dict))
    active_loader = loader or get_default_loader()
    for metrics_path in sorted(root.rglob("*_last.metrics.json")):
        try:
            run = load_run(metrics_path, loader=active_loader)
        except (OSError, ValueError, GuardrailViolation):
            continue
        if model_filter and run.model not in model_filter:
            continue
        runs[run.model][run.budget][run.seed] = run
    return {
        model: {budget: dict(seed_map) for budget, seed_map in sorted(per_model.items())}
        for model, per_model in runs.items()
    }


def _collect_values(per_seed: Mapping[int, FewShotRun], metric: str) -> Tuple[List[float], Dict[int, float]]:
    values: List[float] = []
    per_seed_values: Dict[int, float] = {}
    for seed, run in per_seed.items():
        value = _coerce_float(run.test_metrics.get(metric))
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
        positive_key=lambda record: record.center_id or record.case_id,
        negative_key=lambda record: record.sequence_id or record.case_id,
    )


def _metrics_for_sample(run: FewShotRun, frame_ids: Sequence[str], metric: str) -> Optional[float]:
    records = [run.frames[fid] for fid in frame_ids if fid in run.frames]
    if not records:
        return None
    probs = np.array([rec.prob for rec in records], dtype=float)
    labels = np.array([rec.label for rec in records], dtype=int)
    metrics = compute_binary_metrics(probs, labels, run.tau, metric_keys=AGG_METRICS)
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
            colon_value = _metrics_for_sample(colon_run, frame_ids, metric)
            baseline_value = _metrics_for_sample(baseline_run, frame_ids, metric)
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
) -> List[float]:
    if bootstrap <= 0:
        return []
    shared_budgets = sorted(set(colon_runs_per_budget.keys()) & set(baseline_runs_per_budget.keys()))
    if not shared_budgets:
        return []
    shared_seeds: Optional[List[int]] = None
    for budget in shared_budgets:
        colon_map = colon_runs_per_budget.get(budget, {})
        baseline_map = baseline_runs_per_budget.get(budget, {})
        seeds = set(colon_map.keys()) & set(baseline_map.keys())
        if shared_seeds is None:
            shared_seeds = sorted(seeds)
        else:
            shared_seeds = sorted(set(shared_seeds) & seeds)
    if not shared_seeds:
        return []
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
                colon_value = _metrics_for_sample(colon_run, frame_ids, metric)
                baseline_value = _metrics_for_sample(baseline_run, frame_ids, metric)
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
            seeds = _collect_shared_seeds(colon_seed_map, baseline_seed_map)
            if not seeds:
                continue
            deltas: List[float] = []
            colon_sequence = [colon_seed_map[seed] for seed in seeds]
            baseline_sequence = [baseline_seed_map[seed] for seed in seeds]
            for seed in seeds:
                colon_value = _coerce_float(colon_seed_map[seed].test_metrics.get(metric))
                baseline_value = _coerce_float(baseline_seed_map[seed].test_metrics.get(metric))
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
                value = _coerce_float(budget_map[budget].test_metrics.get(metric))
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
) -> Dict[str, Dict[str, Any]]:
    colon_runs = runs_by_model.get(TARGET_MODEL)
    if not colon_runs:
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    colon_aulc = _compute_aulc_summary({TARGET_MODEL: colon_runs}, metric)[TARGET_MODEL]
    for baseline in BASELINE_MODELS:
        baseline_runs = runs_by_model.get(baseline)
        if not baseline_runs:
            continue
        shared_budgets = sorted(set(colon_runs.keys()) & set(baseline_runs.keys()))
        if not shared_budgets:
            continue
        per_seed_deltas: List[float] = []
        shared_seeds: Optional[List[int]] = None
        for budget in shared_budgets:
            colon_seed_map = colon_runs.get(budget, {})
            baseline_seed_map = baseline_runs.get(budget, {})
            seeds = set(colon_seed_map.keys()) & set(baseline_seed_map.keys())
            if shared_seeds is None:
                shared_seeds = sorted(seeds)
            else:
                shared_seeds = sorted(set(shared_seeds) & seeds)
        if not shared_seeds:
            continue
        for seed in shared_seeds:
            colon_series: List[float] = []
            baseline_series: List[float] = []
            budgets = []
            for budget in shared_budgets:
                colon_value = _coerce_float(colon_runs[budget][seed].test_metrics.get(metric))
                baseline_value = _coerce_float(baseline_runs[budget][seed].test_metrics.get(metric))
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
        )
        results[baseline] = {
            "delta": float(np.mean(per_seed_deltas)) if per_seed_deltas else float("nan"),
            "ci": _compute_ci(replicates) if replicates else {"lower": float("nan"), "upper": float("nan")},
            "replicates": replicates,
            "seeds": len(per_seed_deltas),
            "colon_aulc": colon_aulc,
        }
    return results


def summarize_runs(
    runs_by_model: Mapping[str, Mapping[int, Mapping[int, FewShotRun]]],
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
    target_model: str = "ssl_imnet",
    target_budget: int = 500,
) -> Dict[str, Any]:
    performance: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for model, model_budgets in runs_by_model.items():
        model_entry: Dict[int, Dict[str, Any]] = {}
        for budget, seed_map in model_budgets.items():
            metrics_block: Dict[str, Any] = {}
            for metric in AGG_METRICS:
                values, per_seed_values = _collect_values(seed_map, metric)
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
    budgets = sorted({budget for per_model in runs_by_model.values() for budget in per_model.keys()})
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
                if not isinstance(metrics_block, dict):
                    continue
                metric_stats = metrics_block.get(metric)
                if not isinstance(metric_stats, dict):
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
            bootstrap=max(0, bootstrap),
            rng_seed=rng_seed,
        )
    aulc_summary: Dict[str, Dict[str, Any]] = {}
    for metric in AULC_METRICS:
        aulc_summary[metric] = _compute_aulc_summary(runs_by_model, metric)
    aulc_deltas: Dict[str, Dict[str, Any]] = {}
    for metric in AULC_METRICS:
        aulc_deltas[metric] = _compute_aulc_deltas(
            runs_by_model,
            metric=metric,
            bootstrap=max(0, bootstrap),
            rng_seed=rng_seed,
        )
    sample_efficiency: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for metric in PAIRWISE_METRICS:
        target_entry = performance.get(target_model, {})
        target_stats_entry = None
        if isinstance(target_entry, dict):
            target_budget_block = target_entry.get(target_budget)
            if isinstance(target_budget_block, dict):
                target_metrics_block = target_budget_block.get("metrics")
                if isinstance(target_metrics_block, dict):
                    target_stats_entry = target_metrics_block.get(metric)
        target_stats = target_stats_entry if isinstance(target_stats_entry, dict) else None
        target_value = target_stats.get("mean") if isinstance(target_stats, Mapping) else None
        metric_result: Dict[str, Dict[str, Optional[float]]] = {}
        for model, per_budget in performance.items():
            if not isinstance(per_budget, dict):
                continue
            selected_budget: Optional[int] = None
            for budget in sorted(per_budget.keys()):
                stats_obj = per_budget.get(budget)
                if not isinstance(stats_obj, dict):
                    continue
                metrics_block_obj = stats_obj.get("metrics")
                if not isinstance(metrics_block_obj, dict):
                    continue
                metric_stats = metrics_block_obj.get(metric)
                if not isinstance(metric_stats, dict):
                    continue
                mean_value = metric_stats.get("mean")
                if mean_value is None:
                    continue
                if target_value is None or float(mean_value) >= float(target_value):
                    selected_budget = budget
                    break
            metric_result[model] = {"target": float(target_value) if target_value is not None else None, "budget": selected_budget}
        sample_efficiency[metric] = metric_result
    runs_catalog: List[Dict[str, Any]] = []
    for model, budget_map in runs_by_model.items():
        for budget, seed_map in budget_map.items():
            for seed, run in seed_map.items():
                runs_catalog.append(
                    {
                        "model": model,
                        "budget": budget,
                        "seed": seed,
                        "tau": run.tau,
                        "path": str(run.path),
                    }
                )
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
        "performance": performance,
        "learning_curves": learning_curves,
        "pairwise": pairwise,
        "aulc": aulc_summary,
        "aulc_deltas": aulc_deltas,
        "sample_efficiency": sample_efficiency,
    }


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_performance_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    performance = summary.get("performance")
    if not isinstance(performance, Mapping):
        raise ValueError("Summary payload does not contain performance data")
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
    fieldnames = ["model", "budget", "metric", "mean", "std", "n"]
    _write_rows(output_path, fieldnames, rows)


def write_learning_curve_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    curves = summary.get("learning_curves")
    if not isinstance(curves, Mapping):
        raise ValueError("Summary payload does not contain learning curve data")
    rows: List[Dict[str, Any]] = []
    for metric, model_block in curves.items():
        if not isinstance(model_block, Mapping):
            continue
        for model, series in model_block.items():
            if not isinstance(series, Mapping):
                continue
            for budget, stats in series.items():
                rows.append(
                    {
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
    fieldnames = ["metric", "model", "budget", "mean", "std", "n"]
    _write_rows(output_path, fieldnames, rows)


def write_pairwise_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    pairwise = summary.get("pairwise")
    if not isinstance(pairwise, Mapping):
        raise ValueError("Summary payload does not contain pairwise comparison data")
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
                ci_value = stats.get("ci")
                ci = ci_value if isinstance(ci_value, Mapping) else {}
                rows.append(
                    {
                        "metric": metric,
                        "baseline": baseline,
                        "budget": budget,
                        "delta": stats.get("delta"),
                        "ci_lower": ci.get("lower"),
                        "ci_upper": ci.get("upper"),
                        "seeds": stats.get("seeds"),
                    }
                )
    if not rows:
        raise ValueError("No pairwise delta rows available for CSV export")
    fieldnames = ["metric", "baseline", "budget", "delta", "ci_lower", "ci_upper", "seeds"]
    _write_rows(output_path, fieldnames, rows)


def write_aulc_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    aulc = summary.get("aulc")
    if not isinstance(aulc, Mapping):
        raise ValueError("Summary payload does not contain AULC data")
    rows: List[Dict[str, Any]] = []
    for metric, model_block in aulc.items():
        if not isinstance(model_block, Mapping):
            continue
        for model, stats in model_block.items():
            if not isinstance(stats, Mapping):
                continue
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "n": stats.get("n"),
                }
            )
    if not rows:
        raise ValueError("No AULC rows available for CSV export")
    fieldnames = ["metric", "model", "mean", "std", "n"]
    _write_rows(output_path, fieldnames, rows)


def write_sample_efficiency_csv(summary: Mapping[str, Any], output_path: Path) -> None:
    sample_efficiency = summary.get("sample_efficiency")
    if not isinstance(sample_efficiency, Mapping):
        raise ValueError("Summary payload does not contain sample efficiency data")
    rows: List[Dict[str, Any]] = []
    for metric, model_block in sample_efficiency.items():
        if not isinstance(model_block, Mapping):
            continue
        for model, stats in model_block.items():
            if not isinstance(stats, Mapping):
                continue
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "target": stats.get("target"),
                    "budget": stats.get("budget"),
                }
            )
    if not rows:
        raise ValueError("No sample efficiency rows available for CSV export")
    fieldnames = ["metric", "model", "target", "budget"]
    _write_rows(output_path, fieldnames, rows)


__all__ = [
    "FewShotRun",
    "discover_runs",
    "load_run",
    "summarize_runs",
    "write_performance_csv",
    "write_learning_curve_csv",
    "write_pairwise_csv",
    "write_aulc_csv",
    "write_sample_efficiency_csv",
]
