from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, overload
from typing import Literal

import numpy as np

from .exp3_report import FrameRecord, compute_strata_metrics
from .result_loader import ResultLoader, GuardrailViolation

PRIMARY_METRICS: Tuple[str, ...] = ("auprc", "f1")
MODEL_LABELS: Dict[str, str] = {
    "sup_imnet": "SUP-ImNet",
    "ssl_imnet": "SSL-ImNet",
    "ssl_colon": "SSL-Colon",
}
METRIC_LABELS: Dict[str, str] = {"auprc": "AUPRC", "f1": "F1"}
TARGET_MODEL = "ssl_colon"
BASELINE_MODELS: Tuple[str, ...] = ("sup_imnet", "ssl_imnet")
PREFERRED_MODELS: Tuple[str, ...] = ("sup_imnet", "ssl_imnet", "ssl_colon")


@dataclass
class RunResult:
    model: str
    percent: float
    seed: int
    tau: float
    metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    sensitivity: Dict[str, float]
    morphology: Dict[str, Dict[str, float]]
    frames: List[FrameRecord]
    cases: Dict[str, List[FrameRecord]]
    provenance: Dict[str, object]
    metrics_path: Path


def _get_loader(*, strict: bool = True) -> ResultLoader:
    return ResultLoader(
        expected_primary_policy="f1_opt_on_val",
        expected_sensitivity_policy="youden_on_val",
        require_sensitivity=True,
        required_curve_keys=("test",),
        strict=strict,
    )


def _normalise_case_id(raw: Optional[str], fallback_index: int) -> str:
    if raw is None:
        return f"case_{fallback_index}"
    text = str(raw).strip()
    return text or f"case_{fallback_index}"


def _normalise_morphology(raw: Optional[str]) -> str:
    if raw is None:
        return "unknown"
    text = str(raw).strip()
    return text.lower() if text else "unknown"


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


def _resolve_outputs_path(metrics_path: Path) -> Path:
    stem = metrics_path.stem
    base = stem[:-5] if stem.endswith("_last") else stem
    return metrics_path.with_name(f"{base}_test_outputs.csv")


def _infer_percent_from_name(metrics_path: Path) -> float:
    match = re.search(r"p(\d+)", metrics_path.stem)
    if match is not None:
        return float(match.group(1))
    return 100.0


def _infer_seed_from_name(metrics_path: Path) -> Optional[int]:
    match = re.search(r"_s(\d+)", metrics_path.stem)
    if match is not None:
        return int(match.group(1))
    return None


def _load_outputs(outputs_path: Path, tau: float) -> Tuple[List[FrameRecord], Dict[str, List[FrameRecord]]]:
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing test outputs CSV: {outputs_path}")
    frames: List[FrameRecord] = []
    cases: DefaultDict[str, List[FrameRecord]] = defaultdict(list)
    with outputs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            prob = _coerce_float(row.get("prob"))
            if prob is None:
                continue
            label = _coerce_int(row.get("label"))
            if label is None:
                label = 0
            pred_value = _coerce_int(row.get("pred"))
            if pred_value is None:
                pred_value = 1 if prob >= tau else 0
            case_id = _normalise_case_id(row.get("case_id") or row.get("sequence_id"), index)
            morph = _normalise_morphology(row.get("morphology"))
            record = FrameRecord(prob=float(prob), label=int(label), pred=int(pred_value), case_id=case_id, morphology=morph)
            frames.append(record)
            cases[case_id].append(record)
    return frames, dict(cases)


def load_run(metrics_path: Path, *, loader: Optional[ResultLoader] = None) -> RunResult:
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    active_loader = loader or _get_loader()
    try:
        active_loader.validate(metrics_path, payload)
    except GuardrailViolation as exc:
        raise GuardrailViolation(f"{exc} (from {metrics_path})") from exc
    provenance = dict(payload.get("provenance") or {})
    model_name = provenance.get("model")
    if not model_name:
        stem = metrics_path.stem
        base = stem[:-5] if stem.endswith("_last") else stem
        model_name = base.split("__", 1)[0]
    percent_value = _coerce_float(provenance.get("subset_percent"))
    if percent_value is None:
        percent_value = _infer_percent_from_name(metrics_path)
    seed_value = _coerce_int(payload.get("seed"))
    if seed_value is None:
        seed_value = _coerce_int(provenance.get("train_seed"))
    if seed_value is None:
        seed_value = _infer_seed_from_name(metrics_path)
    if seed_value is None:
        raise ValueError(f"Unable to determine seed for metrics file '{metrics_path}'")
    test_primary = payload.get("test_primary") or {}
    tau_value = _coerce_float(test_primary.get("tau"))
    if tau_value is None:
        raise ValueError(f"Metrics file '{metrics_path}' does not contain test_primary.tau")
    metrics: Dict[str, float] = {}
    for key, value in test_primary.items():
        numeric = _coerce_float(value)
        if numeric is None:
            continue
        metrics[key] = float(numeric)
    val_metrics_raw = payload.get("val") or {}
    val_metrics = {
        key: float(num)
        for key, num in ((k, _coerce_float(v)) for k, v in val_metrics_raw.items())
        if num is not None
    }
    sensitivity_raw = payload.get("test_sensitivity") or {}
    sensitivity_metrics: Dict[str, float] = {
        key: float(num)
        for key, num in ((k, _coerce_float(v)) for k, v in sensitivity_raw.items())
        if num is not None
    }
    morphology_raw = payload.get("test_morphology") or {}
    morphology: Dict[str, Dict[str, float]] = {}
    if isinstance(morphology_raw, Mapping):
        for stratum, stats in morphology_raw.items():
            if not isinstance(stats, Mapping):
                continue
            filtered: Dict[str, float] = {}
            for key, value in stats.items():
                numeric = _coerce_float(value)
                if numeric is not None:
                    filtered[key] = float(numeric)
            if filtered:
                morphology[stratum] = filtered
    outputs_path = _resolve_outputs_path(metrics_path)
    frames, cases = _load_outputs(outputs_path, float(tau_value))
    return RunResult(
        model=str(model_name),
        percent=float(percent_value),
        seed=int(seed_value),
        tau=float(tau_value),
        metrics=metrics,
        val_metrics=val_metrics,
    sensitivity=sensitivity_metrics,
    morphology=morphology,
        frames=frames,
        cases=cases,
        provenance=provenance,
        metrics_path=metrics_path,
    )


@overload
def discover_runs(
    root: Path,
    *,
    strict: bool = True,
    return_loader: Literal[False] = False,
) -> Dict[str, Dict[float, Dict[int, RunResult]]]:
    ...


@overload
def discover_runs(
    root: Path,
    *,
    strict: bool = True,
    return_loader: Literal[True],
) -> Tuple[Dict[str, Dict[float, Dict[int, RunResult]]], ResultLoader]:
    ...


def discover_runs(
    root: Path,
    *,
    strict: bool = True,
    return_loader: bool = False,
) -> Union[Dict[str, Dict[float, Dict[int, RunResult]]], Tuple[Dict[str, Dict[float, Dict[int, RunResult]]], ResultLoader]]:
    runs: DefaultDict[str, DefaultDict[float, Dict[int, RunResult]]] = defaultdict(lambda: defaultdict(dict))
    loader = _get_loader(strict=strict)
    for metrics_path in sorted(root.rglob("*_last.metrics.json")):
        try:
            run = load_run(metrics_path, loader=loader)
        except (ValueError, FileNotFoundError, GuardrailViolation) as exc:
            raise RuntimeError(f"Failed to load metrics from {metrics_path}") from exc
        runs[run.model][run.percent][run.seed] = run
    result = {
        model: {percent: dict(seed_map) for percent, seed_map in per_model.items()}
        for model, per_model in runs.items()
    }
    if return_loader:
        return result, loader
    return result


def compute_learning_curves(
    runs_by_model: Mapping[str, Mapping[float, Mapping[int, RunResult]]],
    *,
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> Dict[str, Dict[str, Dict[float, Dict[str, float]]]]:
    curves: Dict[str, DefaultDict[str, Dict[float, Dict[str, float]]]] = {
        metric: defaultdict(dict) for metric in metrics
    }
    for model, per_percent in runs_by_model.items():
        for percent, seeds in per_percent.items():
            for metric in metrics:
                values: List[float] = []
                for run in seeds.values():
                    value = run.metrics.get(metric)
                    if value is None or math.isnan(float(value)):
                        continue
                    values.append(float(value))
                if not values:
                    continue
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                curves[metric][model][percent] = {
                    "mean": mean,
                    "std": std,
                    "n": len(values),
                }
    result: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}
    for metric, model_map in curves.items():
        result[metric] = {
            model: dict(sorted(percent_map.items())) for model, percent_map in model_map.items()
        }
    return result


def compute_slopes(
    curves: Mapping[str, Mapping[str, Dict[float, Dict[str, float]]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    slopes: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric, model_map in curves.items():
        metric_slopes: Dict[str, Dict[str, float]] = {}
        for model, points in model_map.items():
            percents = sorted(points.keys())
            if len(percents) < 2:
                continue
            interval_slopes: Dict[str, float] = {}
            for start, end in zip(percents, percents[1:]):
                delta_metric = points[end]["mean"] - points[start]["mean"]
                delta_percent = end - start
                slope_value = delta_metric / delta_percent if delta_percent else float("nan")
                label = _format_interval_label(start, end)
                interval_slopes[label] = slope_value
            if interval_slopes:
                metric_slopes[model] = interval_slopes
        slopes[metric] = metric_slopes
    return slopes


def compute_aulc_from_series(percents: Sequence[float], values: Sequence[float]) -> float:
    if len(percents) < 2:
        return float("nan")
    x = np.array(list(percents), dtype=float)
    y = np.array(list(values), dtype=float)
    width = float(x[-1] - x[0])
    if width <= 0:
        return float("nan")
    area = float(np.trapz(y, x))
    return area / width


def compute_aulc(
    curves: Mapping[str, Mapping[str, Dict[float, Dict[str, float]]]]
) -> Dict[str, Dict[str, float]]:
    aulc: Dict[str, Dict[str, float]] = {}
    for metric, model_map in curves.items():
        metric_aulc: Dict[str, float] = {}
        for model, points in model_map.items():
            percents = sorted(points.keys())
            values = [points[p]["mean"] for p in percents]
            metric_aulc[model] = compute_aulc_from_series(percents, values)
        aulc[metric] = metric_aulc
    return aulc


def compute_pairwise_deltas(
    runs_by_model: Mapping[str, Mapping[float, Mapping[int, RunResult]]],
    metric: str,
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
) -> Dict[str, Dict[float, Dict[str, object]]]:
    colon_runs = runs_by_model.get(TARGET_MODEL)
    if not colon_runs:
        return {}
    results: Dict[str, Dict[float, Dict[str, object]]] = {}
    for baseline in BASELINE_MODELS:
        baseline_runs = runs_by_model.get(baseline)
        if not baseline_runs:
            continue
        shared_percents = sorted(set(colon_runs.keys()) & set(baseline_runs.keys()))
        if not shared_percents:
            continue
        per_percent: Dict[float, Dict[str, object]] = {}
        for percent in shared_percents:
            colon_by_seed = colon_runs.get(percent, {})
            baseline_by_seed = baseline_runs.get(percent, {})
            seeds = sorted(set(colon_by_seed.keys()) & set(baseline_by_seed.keys()))
            if not seeds:
                continue
            deltas: List[float] = []
            for seed in seeds:
                colon_value = colon_by_seed[seed].metrics.get(metric)
                baseline_value = baseline_by_seed[seed].metrics.get(metric)
                if colon_value is None or baseline_value is None:
                    continue
                deltas.append(float(colon_value) - float(baseline_value))
            if not deltas:
                continue
            mean_delta = float(np.mean(deltas))
            replicates = bootstrap_metric_delta(
                {seed: colon_by_seed[seed] for seed in seeds},
                {seed: baseline_by_seed[seed] for seed in seeds},
                metric=metric,
                bootstrap=bootstrap,
                rng_seed=rng_seed,
            )
            ci = compute_ci_bounds(replicates)
            per_percent[percent] = {
                "delta": mean_delta,
                "ci": ci,
                "replicates": replicates,
                "seeds": len(seeds),
            }
        if per_percent:
            results[baseline] = dict(sorted(per_percent.items()))
    return results


def compute_aulc_deltas(
    runs_by_model: Mapping[str, Mapping[float, Mapping[int, RunResult]]],
    metric: str,
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
) -> Dict[str, Dict[str, object]]:
    colon_runs = runs_by_model.get(TARGET_MODEL)
    if not colon_runs:
        return {}
    results: Dict[str, Dict[str, object]] = {}
    for baseline in BASELINE_MODELS:
        baseline_runs = runs_by_model.get(baseline)
        if not baseline_runs:
            continue
        shared_percents = sorted(set(colon_runs.keys()) & set(baseline_runs.keys()))
        if not shared_percents:
            continue
        seeds: Optional[set[int]] = None
        for percent in shared_percents:
            colon_seed_set = set(colon_runs.get(percent, {}).keys())
            baseline_seed_set = set(baseline_runs.get(percent, {}).keys())
            overlap = colon_seed_set & baseline_seed_set
            if seeds is None:
                seeds = set(overlap)
            else:
                seeds &= overlap
        if not seeds:
            continue
        sorted_seeds = sorted(seeds)
        deltas: List[float] = []
        for seed in sorted_seeds:
            colon_values = [colon_runs[p][seed].metrics.get(metric) for p in shared_percents]
            baseline_values = [baseline_runs[p][seed].metrics.get(metric) for p in shared_percents]
            if any(value is None for value in colon_values + baseline_values):
                continue
            colon_series = [float(v) for v in colon_values if v is not None]
            baseline_series = [float(v) for v in baseline_values if v is not None]
            if len(colon_series) != len(shared_percents) or len(baseline_series) != len(shared_percents):
                continue
            colon_aulc = compute_aulc_from_series(shared_percents, colon_series)
            baseline_aulc = compute_aulc_from_series(shared_percents, baseline_series)
            if math.isnan(colon_aulc) or math.isnan(baseline_aulc):
                continue
            deltas.append(colon_aulc - baseline_aulc)
        if not deltas:
            continue
        mean_delta = float(np.mean(deltas))
        replicates = bootstrap_aulc_delta(
            {percent: {seed: colon_runs[percent][seed] for seed in sorted_seeds} for percent in shared_percents},
            {percent: {seed: baseline_runs[percent][seed] for seed in sorted_seeds} for percent in shared_percents},
            percents=shared_percents,
            metric=metric,
            bootstrap=bootstrap,
            rng_seed=rng_seed,
        )
        ci = compute_ci_bounds(replicates)
        results[baseline] = {
            "delta": mean_delta,
            "ci": ci,
            "replicates": replicates,
            "seeds": len(sorted_seeds),
            "percents": shared_percents,
        }
    return results


def bootstrap_metric_delta(
    colon_runs: Mapping[int, RunResult],
    baseline_runs: Mapping[int, RunResult],
    *,
    metric: str,
    bootstrap: int,
    rng_seed: int,
) -> List[float]:
    if bootstrap <= 0:
        return []
    seeds = sorted(set(colon_runs.keys()) & set(baseline_runs.keys()))
    if not seeds:
        return []
    rng = np.random.default_rng(rng_seed)
    replicates: List[float] = []
    for _ in range(bootstrap):
        deltas_seed: List[float] = []
        valid = True
        for seed in seeds:
            colon_run = colon_runs[seed]
            baseline_run = baseline_runs[seed]
            case_ids = sorted(set(colon_run.cases.keys()) & set(baseline_run.cases.keys()))
            if not case_ids:
                valid = False
                break
            sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
            colon_frames: List[FrameRecord] = []
            baseline_frames: List[FrameRecord] = []
            for cid in sampled_ids:
                colon_frames.extend(colon_run.cases[cid])
                baseline_frames.extend(baseline_run.cases[cid])
            colon_metrics = compute_strata_metrics(colon_frames, colon_run.tau)
            baseline_metrics = compute_strata_metrics(baseline_frames, baseline_run.tau)
            colon_value = colon_metrics.get("overall", {}).get(metric)
            baseline_value = baseline_metrics.get("overall", {}).get(metric)
            if colon_value is None or baseline_value is None:
                valid = False
                break
            numeric_colon = float(colon_value)
            numeric_baseline = float(baseline_value)
            if math.isnan(numeric_colon) or math.isnan(numeric_baseline):
                valid = False
                break
            deltas_seed.append(numeric_colon - numeric_baseline)
        if not valid or not deltas_seed:
            continue
        replicates.append(float(np.mean(deltas_seed)))
    return replicates


def bootstrap_aulc_delta(
    colon_runs: Mapping[float, Mapping[int, RunResult]],
    baseline_runs: Mapping[float, Mapping[int, RunResult]],
    *,
    percents: Sequence[float],
    metric: str,
    bootstrap: int,
    rng_seed: int,
) -> List[float]:
    if bootstrap <= 0:
        return []
    seeds: Optional[set[int]] = None
    for percent in percents:
        colon_seed_set = set(colon_runs.get(percent, {}).keys())
        baseline_seed_set = set(baseline_runs.get(percent, {}).keys())
        overlap = colon_seed_set & baseline_seed_set
        if seeds is None:
            seeds = set(overlap)
        else:
            seeds &= overlap
    if not seeds:
        return []
    sorted_seeds = sorted(seeds)
    rng = np.random.default_rng(rng_seed)
    replicates: List[float] = []
    for _ in range(bootstrap):
        deltas_seed: List[float] = []
        valid = True
        for seed in sorted_seeds:
            colon_values: List[float] = []
            baseline_values: List[float] = []
            for percent in percents:
                colon_run = colon_runs[percent][seed]
                baseline_run = baseline_runs[percent][seed]
                case_ids = sorted(set(colon_run.cases.keys()) & set(baseline_run.cases.keys()))
                if not case_ids:
                    valid = False
                    break
                sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
                colon_frames: List[FrameRecord] = []
                baseline_frames: List[FrameRecord] = []
                for cid in sampled_ids:
                    colon_frames.extend(colon_run.cases[cid])
                    baseline_frames.extend(baseline_run.cases[cid])
                colon_metrics = compute_strata_metrics(colon_frames, colon_run.tau)
                baseline_metrics = compute_strata_metrics(baseline_frames, baseline_run.tau)
                colon_value = colon_metrics.get("overall", {}).get(metric)
                baseline_value = baseline_metrics.get("overall", {}).get(metric)
                if colon_value is None or baseline_value is None:
                    valid = False
                    break
                numeric_colon = float(colon_value)
                numeric_baseline = float(baseline_value)
                if math.isnan(numeric_colon) or math.isnan(numeric_baseline):
                    valid = False
                    break
                colon_values.append(numeric_colon)
                baseline_values.append(numeric_baseline)
            if not valid:
                break
            colon_aulc = compute_aulc_from_series(percents, colon_values)
            baseline_aulc = compute_aulc_from_series(percents, baseline_values)
            if math.isnan(colon_aulc) or math.isnan(baseline_aulc):
                valid = False
                break
            deltas_seed.append(colon_aulc - baseline_aulc)
        if not valid or not deltas_seed:
            continue
        replicates.append(float(np.mean(deltas_seed)))
    return replicates


def compute_ci_bounds(samples: Sequence[float]) -> Tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    array = np.array(samples, dtype=float)
    lower = float(np.quantile(array, 0.025))
    upper = float(np.quantile(array, 0.975))
    return lower, upper


def determine_targets(
    curves: Mapping[str, Mapping[str, Dict[float, Dict[str, float]]]],
    *,
    target_percent: float = 100.0,
) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    for metric, model_map in curves.items():
        candidates: List[float] = []
        for baseline in BASELINE_MODELS:
            stats = model_map.get(baseline)
            if not stats:
                continue
            key = _resolve_percent_key(stats, target_percent)
            if key is None:
                continue
            mean = stats[key].get("mean")
            if mean is None or math.isnan(float(mean)):
                continue
            candidates.append(float(mean))
        targets[metric] = max(candidates) if candidates else float("nan")
    return targets


def compute_s_at_target(
    curves: Mapping[str, Mapping[str, Dict[float, Dict[str, float]]]],
    targets: Mapping[str, float],
) -> Dict[str, Dict[str, Optional[float]]]:
    results: Dict[str, Dict[str, Optional[float]]] = {}
    for metric, target in targets.items():
        metric_results: Dict[str, Optional[float]] = {}
        if math.isnan(target):
            results[metric] = metric_results
            continue
        model_map = curves.get(metric, {})
        for model, stats in model_map.items():
            percent_hit: Optional[float] = None
            for percent in sorted(stats.keys()):
                mean = stats[percent].get("mean")
                if mean is None or math.isnan(float(mean)):
                    continue
                if float(mean) >= target - 1e-9:
                    percent_hit = percent
                    break
            metric_results[model] = percent_hit
        results[metric] = metric_results
    return results


def summarize_runs(
    runs_by_model: Mapping[str, Mapping[float, Mapping[int, RunResult]]],
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
) -> Dict[str, object]:
    curves = compute_learning_curves(runs_by_model)
    slopes = compute_slopes(curves)
    aulc = compute_aulc(curves)
    pairwise = {
        metric: compute_pairwise_deltas(runs_by_model, metric, bootstrap=max(0, bootstrap), rng_seed=rng_seed)
        for metric in PRIMARY_METRICS
    }
    aulc_delta = {
        metric: compute_aulc_deltas(runs_by_model, metric, bootstrap=max(0, bootstrap), rng_seed=rng_seed)
        for metric in PRIMARY_METRICS
    }
    targets = determine_targets(curves)
    s_at_target = compute_s_at_target(curves, targets)
    return {
        "runs": runs_by_model,
        "curves": curves,
        "slopes": slopes,
        "aulc": aulc,
        "pairwise": pairwise,
        "aulc_delta": aulc_delta,
        "targets": targets,
        "s_at_target": s_at_target,
    }


def collect_summary(
    runs_root: Path,
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
    strict: bool = True,
) -> Tuple[Dict[str, Dict[float, Dict[int, RunResult]]], Dict[str, object], ResultLoader]:
    runs, loader = discover_runs(runs_root, strict=strict, return_loader=True)
    if not runs:
        return runs, {}, loader
    summary = summarize_runs(runs, bootstrap=max(0, bootstrap), rng_seed=rng_seed)
    return runs, summary, loader


def generate_report(
    runs_root: Path,
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
    strict: bool = True,
) -> str:
    runs, summary, _ = collect_summary(
        runs_root,
        bootstrap=bootstrap,
        rng_seed=rng_seed,
        strict=strict,
    )
    if not runs:
        return "No Experiment 4 runs found.\n"
    return render_report(summary)


def render_report(summary: Mapping[str, object]) -> str:
    runs = summary.get("runs")
    if not runs:
        return "No Experiment 4 runs found.\n"
    lines: List[str] = ["# Experiment 4 learning curve summary", ""]
    curves_obj = summary.get("curves")
    curves = curves_obj if isinstance(curves_obj, Mapping) else {}
    for metric in PRIMARY_METRICS:
        metric_curves_obj = curves.get(metric, {})
        metric_curves = metric_curves_obj if isinstance(metric_curves_obj, Mapping) else {}
        lines.append(f"## Learning curves – {METRIC_LABELS.get(metric, metric.upper())}")
        lines.extend(_render_learning_table(metric_curves))
        lines.append("")
    slopes_obj = summary.get("slopes")
    slopes = slopes_obj if isinstance(slopes_obj, Mapping) else {}
    for metric in PRIMARY_METRICS:
        metric_slopes_obj = slopes.get(metric, {})
        metric_slopes = metric_slopes_obj if isinstance(metric_slopes_obj, Mapping) else {}
        lines.append(f"## Marginal gains – {METRIC_LABELS.get(metric, metric.upper())}")
        lines.extend(_render_slope_table(metric_slopes))
        lines.append("")
    aulc_obj = summary.get("aulc")
    aulc = aulc_obj if isinstance(aulc_obj, Mapping) else {}
    lines.append("## Area under the learning curve")
    lines.extend(_render_aulc_table(aulc))
    lines.append("")
    pairwise_obj = summary.get("pairwise")
    pairwise = pairwise_obj if isinstance(pairwise_obj, Mapping) else {}
    for metric in PRIMARY_METRICS:
        metric_pairwise_obj = pairwise.get(metric, {})
        metric_pairwise = metric_pairwise_obj if isinstance(metric_pairwise_obj, Mapping) else {}
        for baseline, entries in metric_pairwise.items():
            lines.append(
                f"## Pairwise Δ vs {MODEL_LABELS.get(baseline, baseline)} – {METRIC_LABELS.get(metric, metric.upper())}"
            )
            lines.extend(_render_pairwise_table(entries if isinstance(entries, Mapping) else {}))
            lines.append("")
    aulc_delta_obj = summary.get("aulc_delta")
    aulc_delta = aulc_delta_obj if isinstance(aulc_delta_obj, Mapping) else {}
    lines.append("## ΔAULC vs baselines")
    lines.extend(_render_aulc_delta_table(aulc_delta))
    lines.append("")
    targets_obj = summary.get("targets")
    targets = targets_obj if isinstance(targets_obj, Mapping) else {}
    s_at_target_obj = summary.get("s_at_target")
    s_at_target = s_at_target_obj if isinstance(s_at_target_obj, Mapping) else {}
    lines.append("## S@target")
    lines.extend(_render_s_at_target_table(targets, s_at_target))
    lines.append("")
    return "\n".join(line.rstrip() for line in lines).strip() + "\n"


def _render_learning_table(curves: Mapping[str, Dict[float, Mapping[str, float]]]) -> List[str]:
    models = list(curves.keys())
    if not models:
        return ["No runs available."]
    percents = sorted({percent for stats in curves.values() for percent in stats.keys()})
    ordered_models = _ordered_models(models)
    header = "| Percent | " + " | ".join(MODEL_LABELS.get(model, model) for model in ordered_models) + " |"
    separator = "|" + " --- |" * (len(ordered_models) + 1)
    rows = [header, separator]
    for percent in percents:
        row_values = [_format_percent(percent)]
        for model in ordered_models:
            stats = curves.get(model, {}).get(percent)
            if not stats:
                row_values.append("—")
            else:
                row_values.append(_format_mean_std(stats.get("mean"), stats.get("std")))
        rows.append("| " + " | ".join(row_values) + " |")
    return rows


def _render_slope_table(slopes: Mapping[str, Dict[str, float]]) -> List[str]:
    if not slopes:
        return ["No slope data available."]
    intervals = sorted({interval for stats in slopes.values() for interval in stats.keys()}, key=_interval_sort_key)
    ordered_models = _ordered_models(slopes.keys())
    header = "| Model | " + " | ".join(intervals) + " |"
    separator = "|" + " --- |" * (len(intervals) + 1)
    rows = [header, separator]
    for model in ordered_models:
        stats = slopes.get(model, {})
        row = [MODEL_LABELS.get(model, model)]
        for interval in intervals:
            value = stats.get(interval, float("nan"))
            row.append(_format_signed(value))
        rows.append("| " + " | ".join(row) + " |")
    return rows


def _render_aulc_table(aulc: Mapping[str, Mapping[str, float]]) -> List[str]:
    models = {
        model for metric_stats in aulc.values() for model in metric_stats.keys()
    }
    if not models:
        return ["No AULC data available."]
    ordered_models = _ordered_models(models)
    header = "| Model | " + " | ".join(METRIC_LABELS.get(metric, metric.upper()) for metric in PRIMARY_METRICS) + " |"
    separator = "|" + " --- |" * (len(PRIMARY_METRICS) + 1)
    rows = [header, separator]
    for model in ordered_models:
        row = [MODEL_LABELS.get(model, model)]
        for metric in PRIMARY_METRICS:
            value = aulc.get(metric, {}).get(model, float("nan"))
            row.append(_format_scalar(value))
        rows.append("| " + " | ".join(row) + " |")
    return rows


def _render_pairwise_table(entries: Mapping[float, Mapping[str, object]]) -> List[str]:
    if not entries:
        return ["No overlapping runs for comparison."]
    header = "| Percent | Δ | 95% CI | Seeds | Bootstraps |"
    separator = "| --- | --- | --- | --- | --- |"
    rows = [header, separator]
    for percent in sorted(entries.keys()):
        entry_raw = entries[percent]
        entry = dict(entry_raw) if isinstance(entry_raw, Mapping) else {}
        delta_value = _coerce_float(entry.get("delta"))
        ci_obj = entry.get("ci")
        if isinstance(ci_obj, Sequence) and not isinstance(ci_obj, (str, bytes)) and len(ci_obj) >= 2:
            ci_low = _coerce_float(ci_obj[0]) or float("nan")
            ci_high = _coerce_float(ci_obj[1]) or float("nan")
        else:
            ci_low = ci_high = float("nan")
        seeds_value = _coerce_int(entry.get("seeds")) or 0
        reps_obj = entry.get("replicates")
        replicate_count = (
            len(reps_obj)
            if isinstance(reps_obj, Sequence) and not isinstance(reps_obj, (str, bytes))
            else 0
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    _format_percent(percent),
                    _format_signed(delta_value if delta_value is not None else float("nan")),
                    _format_ci(ci_low, ci_high),
                    str(seeds_value),
                    str(replicate_count),
                ]
            )
            + " |"
        )
    return rows


def _render_aulc_delta_table(aulc_delta: Mapping[str, Mapping[str, object]]) -> List[str]:
    header = "| Baseline | Metric | ΔAULC | 95% CI | Seeds | Bootstraps |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    rows = [header, separator]
    any_rows = False
    for metric in PRIMARY_METRICS:
        per_metric = aulc_delta.get(metric, {}) if isinstance(aulc_delta, Mapping) else {}
        if not isinstance(per_metric, Mapping):
            continue
        for baseline, entry in per_metric.items():
            entry_map = dict(entry) if isinstance(entry, Mapping) else {}
            delta_value = _coerce_float(entry_map.get("delta"))
            ci_obj = entry_map.get("ci")
            if isinstance(ci_obj, Sequence) and not isinstance(ci_obj, (str, bytes)) and len(ci_obj) >= 2:
                ci_low = _coerce_float(ci_obj[0]) or float("nan")
                ci_high = _coerce_float(ci_obj[1]) or float("nan")
            else:
                ci_low = ci_high = float("nan")
            seeds_value = _coerce_int(entry_map.get("seeds")) or 0
            reps_obj = entry_map.get("replicates")
            replicate_count = (
                len(reps_obj)
                if isinstance(reps_obj, Sequence) and not isinstance(reps_obj, (str, bytes))
                else 0
            )
            any_rows = True
            rows.append(
                "| "
                + " | ".join(
                    [
                        MODEL_LABELS.get(baseline, baseline),
                        METRIC_LABELS.get(metric, metric.upper()),
                        _format_signed(delta_value if delta_value is not None else float("nan")),
                        _format_ci(ci_low, ci_high),
                        str(seeds_value),
                        str(replicate_count),
                    ]
                )
                + " |"
            )
    if not any_rows:
        return ["No overlapping runs for AULC comparison."]
    return rows


def _render_s_at_target_table(
    targets: Mapping[str, float],
    s_at_target: Mapping[str, Mapping[str, Optional[float]]],
) -> List[str]:
    models = {
        model for metric_map in s_at_target.values() for model in metric_map.keys()
    }
    if not models:
        return ["No models available for S@target computation."]
    ordered_models = _ordered_models(models)
    header = "| Metric | Target | " + " | ".join(MODEL_LABELS.get(model, model) for model in ordered_models) + " |"
    separator = "|" + " --- |" * (len(ordered_models) + 2)
    rows = [header, separator]
    for metric in PRIMARY_METRICS:
        target_value = targets.get(metric, float("nan"))
        row = [METRIC_LABELS.get(metric, metric.upper()), _format_scalar(target_value)]
        per_metric = s_at_target.get(metric, {}) if isinstance(s_at_target, Mapping) else {}
        for model in ordered_models:
            percent = per_metric.get(model)
            row.append(_format_percent(percent))
        rows.append("| " + " | ".join(row) + " |")
    return rows


def _ordered_models(models: Iterable[str]) -> List[str]:
    model_list = list(models)
    ordered = [model for model in PREFERRED_MODELS if model in model_list]
    extras = sorted(model for model in model_list if model not in ordered)
    return ordered + extras


def _format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or math.isnan(float(mean)):
        return "—"
    if std is None or math.isnan(float(std)) or float(std) == 0.0:
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} ± {float(std):.3f}"


def _format_signed(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "—"
    return f"{float(value):+.3f}"


def _format_scalar(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "—"
    return f"{float(value):.3f}"


def _format_ci(lower: float, upper: float) -> str:
    if lower is None or upper is None or math.isnan(float(lower)) or math.isnan(float(upper)):
        return "—"
    return f"{float(lower):.3f} – {float(upper):.3f}"


def _format_percent(value: Optional[float]) -> str:
    if value is None or math.isnan(float(value)):
        return "—"
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-6:
        return f"{int(rounded)}%"
    return f"{numeric:.1f}%"


def _format_percent_numeric(value: float) -> str:
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-6:
        return str(int(rounded))
    return f"{numeric:.1f}"


def _format_interval_label(start: float, end: float) -> str:
    return f"{_format_percent_numeric(start)}→{_format_percent_numeric(end)}"


def _interval_sort_key(interval: str) -> Tuple[float, float]:
    match = re.match(r"(\d+(?:\.\d+)?)→(\d+(?:\.\d+)?)", interval)
    if match:
        return float(match.group(1)), float(match.group(2))
    return math.inf, math.inf


def _resolve_percent_key(stats: Mapping[float, Mapping[str, float]], target: float) -> Optional[float]:
    for percent in stats.keys():
        if math.isclose(percent, target, rel_tol=1e-6, abs_tol=1e-6):
            return percent
    return None


__all__ = [
    "RunResult",
    "discover_runs",
    "summarize_runs",
    "generate_report",
    "render_report",
]
