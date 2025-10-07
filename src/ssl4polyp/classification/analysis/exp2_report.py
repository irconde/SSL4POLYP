from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (  # type: ignore[import]
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRICS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "mcc",
)
DEFAULT_PAIRED_MODELS: Tuple[str, str] = ("ssl_colon", "ssl_imnet")
CI_LEVEL = 0.95


__all__ = [
    "PRIMARY_METRICS",
    "DEFAULT_PAIRED_MODELS",
    "Exp2Run",
    "MetricAggregate",
    "DeltaSummary",
    "Exp2Summary",
    "load_run",
    "discover_runs",
    "summarize_runs",
]


@dataclass(frozen=True)
class EvalFrame:
    frame_id: str
    case_id: str
    prob: float
    label: int


@dataclass
class Exp2Run:
    model: str
    seed: int
    tau: float
    metrics: Dict[str, float]
    frames: List[EvalFrame]
    cases: Dict[str, Tuple[EvalFrame, ...]]
    provenance: Dict[str, Any]
    metrics_path: Path


@dataclass
class MetricAggregate:
    mean: float
    std: float
    n: int
    values: Tuple[float, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "n": int(self.n),
            "values": list(self.values),
        }


@dataclass
class DeltaSummary:
    per_seed: Dict[int, float]
    mean: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    samples: Tuple[float, ...]

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "per_seed": {int(seed): float(value) for seed, value in self.per_seed.items()},
            "mean": float(self.mean),
            "samples": list(self.samples),
        }
        if self.ci_lower is not None and self.ci_upper is not None:
            data["ci_lower"] = float(self.ci_lower)
            data["ci_upper"] = float(self.ci_upper)
        else:
            data["ci_lower"] = None
            data["ci_upper"] = None
        return data


@dataclass
class Exp2Summary:
    model_metrics: Dict[str, Dict[str, MetricAggregate]]
    paired_deltas: Dict[str, DeltaSummary]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_metrics": {
                model: {metric: aggregate.as_dict() for metric, aggregate in metrics.items()}
                for model, metrics in self.model_metrics.items()
            },
            "paired_deltas": {
                metric: summary.as_dict() for metric, summary in self.paired_deltas.items()
            },
        }


def _clean_text(value: object) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, np.floating, int, np.integer)):
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


def _read_outputs(outputs_path: Path) -> Tuple[List[EvalFrame], Dict[str, Tuple[EvalFrame, ...]]]:
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing test outputs CSV: {outputs_path}")
    frames: List[EvalFrame] = []
    cases: DefaultDict[str, List[EvalFrame]] = defaultdict(list)
    with outputs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            prob = _coerce_float(row.get("prob"))
            label = _coerce_int(row.get("label"))
            if prob is None or label is None:
                continue
            frame_id = _clean_text(row.get("frame_id")) or f"frame_{index}"
            case_id = (
                _clean_text(row.get("case_id"))
                or _clean_text(row.get("sequence_id"))
                or frame_id
            )
            frame = EvalFrame(frame_id=frame_id, case_id=case_id, prob=float(prob), label=int(label))
            frames.append(frame)
            cases[case_id].append(frame)
    if not frames:
        raise ValueError(f"No evaluation frames parsed from {outputs_path}")
    return frames, {case: tuple(items) for case, items in cases.items()}


def _compute_binary_metrics(probs: np.ndarray, labels: np.ndarray, tau: float) -> Dict[str, float]:
    if probs.size == 0:
        return {metric: float("nan") for metric in PRIMARY_METRICS}
    preds = (probs >= float(tau)).astype(int)
    total = int(labels.size)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    prevalence = float(n_pos) / float(total) if total else float("nan")
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    try:
        auprc = float(average_precision_score(labels, probs))
    except ValueError:
        auprc = float("nan")
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = float("nan")
    recall_val = float(recall_score(labels, preds, zero_division=0))
    precision_val = float(precision_score(labels, preds, zero_division=0))
    f1_val = float(f1_score(labels, preds, zero_division=0))
    try:
        balanced_acc = float(balanced_accuracy_score(labels, preds))
    except ValueError:
        balanced_acc = float("nan")
    try:
        mcc_val = float(matthews_corrcoef(labels, preds))
    except ValueError:
        mcc_val = float("nan")
    return {
        "auprc": auprc,
        "auroc": auroc,
        "recall": recall_val,
        "precision": precision_val,
        "f1": f1_val,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc_val,
        "prevalence": prevalence,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "count": total,
    }


def _metrics_from_frames(frames: Sequence[EvalFrame], tau: float) -> Dict[str, float]:
    if not frames:
        return {metric: float("nan") for metric in PRIMARY_METRICS}
    probs = np.array([frame.prob for frame in frames], dtype=float)
    labels = np.array([frame.label for frame in frames], dtype=int)
    return _compute_binary_metrics(probs, labels, tau)


def _ci_bounds(values: Sequence[float], *, level: float = CI_LEVEL) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    array = np.array(values, dtype=float)
    if array.size == 0:
        return None
    lower_pct = (1.0 - level) / 2.0 * 100.0
    upper_pct = (1.0 + level) / 2.0 * 100.0
    lower = float(np.percentile(array, lower_pct))
    upper = float(np.percentile(array, upper_pct))
    return lower, upper


def _coerce_metric_block(block: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not isinstance(block, Mapping):
        return metrics
    for key in PRIMARY_METRICS:
        numeric = _coerce_float(block.get(key))
        if numeric is None:
            continue
        metrics[key] = float(numeric)
    return metrics


def load_run(metrics_path: Path) -> Exp2Run:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance_raw = payload.get("provenance")
    provenance = dict(provenance_raw) if isinstance(provenance_raw, Mapping) else {}
    model_name = _clean_text(provenance.get("model"))
    if not model_name:
        model_name = metrics_path.stem.split("__", 1)[0]
    seed_value = _coerce_int(payload.get("seed"))
    if seed_value is None:
        stem_seed = None
        stem = metrics_path.stem[:-5] if metrics_path.stem.endswith("_last") else metrics_path.stem
        if stem.endswith(".metrics"):
            stem = stem[:-8]
        if "_s" in stem:
            try:
                stem_seed = int(stem.rsplit("_s", 1)[-1])
            except ValueError:
                stem_seed = None
        if stem_seed is not None:
            seed_value = stem_seed
    if seed_value is None:
        raise ValueError(f"Metrics file '{metrics_path}' does not specify a seed")
    test_block = payload.get("test_primary") or payload.get("test")
    if not isinstance(test_block, Mapping):
        raise ValueError(f"Metrics file '{metrics_path}' is missing test_primary block")
    tau_value = _coerce_float(test_block.get("tau"))
    if tau_value is None:
        raise ValueError(f"Metrics file '{metrics_path}' is missing test_primary.tau")
    metrics = _coerce_metric_block(test_block)
    outputs_path = _resolve_outputs_path(metrics_path)
    frames, cases = _read_outputs(outputs_path)
    return Exp2Run(
        model=str(model_name),
        seed=int(seed_value),
        tau=float(tau_value),
        metrics=metrics,
        frames=frames,
        cases=cases,
        provenance=provenance,
        metrics_path=metrics_path,
    )


def discover_runs(root: Path, *, models: Optional[Sequence[str]] = None) -> Dict[str, Dict[int, Exp2Run]]:
    root = root.expanduser()
    metrics_paths = sorted(root.rglob("*.metrics.json"))
    runs: Dict[str, Dict[int, Exp2Run]] = {}
    model_filter = {name.lower() for name in models} if models else None
    processed: Dict[str, Path] = {}
    for metrics_path in metrics_paths:
        if metrics_path.name.endswith("_best.metrics.json"):
            continue
        stem = metrics_path.stem
        base = stem[:-5] if stem.endswith("_last") else stem
        if base in processed and not stem.endswith("_last"):
            continue
        if base not in processed or not processed[base].exists():
            processed[base] = metrics_path
        if stem.endswith("_last") and base in processed and not processed[base].stem.endswith("_last"):
            continue
        try:
            run = load_run(metrics_path)
        except (ValueError, FileNotFoundError) as exc:
            raise RuntimeError(f"Failed to load metrics from {metrics_path}") from exc
        if model_filter and run.model.lower() not in model_filter:
            continue
        runs.setdefault(run.model, {})[run.seed] = run
    return runs


def _aggregate(values: Iterable[float]) -> Optional[MetricAggregate]:
    collection = [float(v) for v in values if isinstance(v, (float, int, np.floating, np.integer))]
    clean = [v for v in collection if math.isfinite(v)]
    if not clean:
        return None
    array = np.array(clean, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return MetricAggregate(mean=mean, std=std, n=int(array.size), values=tuple(clean))


def _paired_bootstrap(
    model_a_runs: Mapping[int, Exp2Run],
    model_b_runs: Mapping[int, Exp2Run],
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: Optional[int],
) -> Dict[str, List[float]]:
    seeds = sorted(set(model_a_runs.keys()) & set(model_b_runs.keys()))
    if not seeds:
        return {metric: [] for metric in metrics}
    rng = np.random.default_rng(rng_seed)
    replicates: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for _ in range(max(0, bootstrap)):
        per_seed: Dict[str, List[float]] = {metric: [] for metric in metrics}
        valid = True
        for seed in seeds:
            run_a = model_a_runs[seed]
            run_b = model_b_runs[seed]
            case_ids = sorted(set(run_a.cases.keys()) & set(run_b.cases.keys()))
            if not case_ids:
                valid = False
                break
            sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
            frames_a: List[EvalFrame] = []
            frames_b: List[EvalFrame] = []
            for cid in sampled_ids:
                frames_a.extend(run_a.cases[cid])
                frames_b.extend(run_b.cases[cid])
            metrics_a = _metrics_from_frames(frames_a, run_a.tau)
            metrics_b = _metrics_from_frames(frames_b, run_b.tau)
            for metric in metrics:
                value_a = metrics_a.get(metric)
                value_b = metrics_b.get(metric)
                if value_a is None or value_b is None or not math.isfinite(value_a) or not math.isfinite(value_b):
                    valid = False
                    break
                per_seed[metric].append(float(value_a - value_b))
            if not valid:
                break
        if not valid:
            continue
        for metric in metrics:
            sample_values = per_seed[metric]
            if sample_values:
                replicates[metric].append(float(np.mean(sample_values)))
    return replicates


def summarize_runs(
    runs_by_model: Mapping[str, Mapping[int, Exp2Run]],
    *,
    metrics: Sequence[str] = PRIMARY_METRICS,
    paired_models: Optional[Tuple[str, str]] = DEFAULT_PAIRED_MODELS,
    bootstrap: int = 2000,
    rng_seed: Optional[int] = 20240521,
) -> Exp2Summary:
    model_metrics: Dict[str, Dict[str, MetricAggregate]] = {}
    for model, runs in runs_by_model.items():
        per_metric: Dict[str, MetricAggregate] = {}
        for metric in metrics:
            raw_values = [run.metrics.get(metric) for run in runs.values()]
            values = [float(value) for value in raw_values if isinstance(value, (int, float, np.integer, np.floating))]
            aggregate = _aggregate(values) if values else None
            if aggregate:
                per_metric[metric] = aggregate
        if per_metric:
            model_metrics[model] = per_metric
    paired_deltas: Dict[str, DeltaSummary] = {}
    if paired_models:
        model_a, model_b = paired_models
        runs_a = runs_by_model.get(model_a, {})
        runs_b = runs_by_model.get(model_b, {})
        seeds = sorted(set(runs_a.keys()) & set(runs_b.keys()))
        if seeds:
            per_seed_delta: Dict[str, Dict[int, float]] = {metric: {} for metric in metrics}
            for seed in seeds:
                run_a = runs_a[seed]
                run_b = runs_b[seed]
                for metric in metrics:
                    value_a = run_a.metrics.get(metric)
                    value_b = run_b.metrics.get(metric)
                    if value_a is None or value_b is None:
                        continue
                    if not (math.isfinite(value_a) and math.isfinite(value_b)):
                        continue
                    per_seed_delta[metric][seed] = float(value_a - value_b)
            replicates = _paired_bootstrap(
                runs_a,
                runs_b,
                metrics=metrics,
                bootstrap=bootstrap,
                rng_seed=rng_seed,
            )
            for metric in metrics:
                per_seed_map = per_seed_delta[metric]
                if not per_seed_map:
                    continue
                mean_delta = float(np.mean(list(per_seed_map.values())))
                samples = replicates.get(metric, [])
                ci = _ci_bounds(samples, level=CI_LEVEL) if samples else None
                summary = DeltaSummary(
                    per_seed=dict(sorted(per_seed_map.items())),
                    mean=mean_delta,
                    ci_lower=ci[0] if ci else None,
                    ci_upper=ci[1] if ci else None,
                    samples=tuple(samples),
                )
                paired_deltas[metric] = summary
    return Exp2Summary(model_metrics=model_metrics, paired_deltas=paired_deltas)
