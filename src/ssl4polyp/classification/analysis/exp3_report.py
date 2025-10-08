from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

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

from .common_loader import get_default_loader, load_common_run
from .result_loader import ResultLoader


_STRATA = ("overall", "flat_plus_negs", "polypoid_plus_negs")
_MODEL_LABELS = {
    "sup_imnet": "SUP-ImNet",
    "ssl_imnet": "SSL-ImNet",
    "ssl_colon": "SSL-Colon",
}
_STRATUM_LABELS = {
    "overall": "Overall",
    "flat_plus_negs": "Flat + Negs",
    "polypoid_plus_negs": "Polypoid + Negs",
}
_PRIMARY_METRICS = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "mcc",
)
_METRIC_LABELS = {
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "balanced_accuracy": "Balanced Acc",
    "mcc": "MCC",
}

__all__ = [
    "FrameRecord",
    "RunDataset",
    "discover_runs",
    "compute_strata_metrics",
    "summarise_composition",
    "bootstrap_deltas",
    "generate_report",
]


@dataclass(frozen=True)
class FrameRecord:
    prob: float
    label: int
    pred: int
    case_id: str
    morphology: str


@dataclass
class RunDataset:
    model: str
    seed: int
    tau: float
    frames: List[FrameRecord]
    cases: Dict[str, List[FrameRecord]]


def _normalise_morphology(raw: Optional[object]) -> str:
    if raw is None:
        return "unknown"
    text = str(raw).strip()
    return text.lower() if text else "unknown"


def _binary_metrics(probabilities: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    total = labels.size
    if total == 0:
        return {
            "count": 0,
            "n_pos": 0,
            "n_neg": 0,
            "prevalence": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "auprc": float("nan"),
            "auroc": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "mcc": float("nan"),
            "loss": float("nan"),
        }
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    prevalence = float(n_pos) / float(total) if total > 0 else float("nan")
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    try:
        auprc = float(average_precision_score(labels, probabilities))
    except ValueError:
        auprc = float("nan")
    try:
        auroc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auroc = float("nan")
    recall = float(recall_score(labels, preds, zero_division=0))
    precision = float(precision_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    try:
        balanced_acc = float(balanced_accuracy_score(labels, preds))
    except ValueError:
        balanced_acc = float("nan")
    try:
        mcc = float(matthews_corrcoef(labels, preds))
    except ValueError:
        mcc = float("nan")
    eps = 1e-12
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    loss = float(np.mean(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))))
    return {
        "count": total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": prevalence,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "auprc": auprc,
        "auroc": auroc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "loss": loss,
    }


def compute_strata_metrics(frames: Sequence[FrameRecord], tau: float) -> Dict[str, Dict[str, float]]:
    if not frames:
        return {}
    probs = np.array([record.prob for record in frames], dtype=float)
    labels = np.array([record.label for record in frames], dtype=int)
    preds = (probs >= float(tau)).astype(int)
    morph = np.array([record.morphology for record in frames])
    metrics: Dict[str, Dict[str, float]] = {}
    metrics["overall"] = _binary_metrics(probs, preds, labels)
    pos_mask = labels == 1
    neg_mask = labels == 0
    flat_mask = neg_mask | (pos_mask & (morph == "flat"))
    if np.any(pos_mask & (morph == "flat")):
        metrics["flat_plus_negs"] = _binary_metrics(
            probs[flat_mask], preds[flat_mask], labels[flat_mask]
        )
    polypoid_mask = neg_mask | (pos_mask & (morph == "polypoid"))
    if np.any(pos_mask & (morph == "polypoid")):
        metrics["polypoid_plus_negs"] = _binary_metrics(
            probs[polypoid_mask], preds[polypoid_mask], labels[polypoid_mask]
        )
    return metrics


def load_run(
    metrics_path: Path,
    *,
    loader: Optional[ResultLoader] = None,
) -> RunDataset:
    base_run = load_common_run(metrics_path, loader=loader or get_default_loader())
    frames: List[FrameRecord] = []
    cases: DefaultDict[str, List[FrameRecord]] = defaultdict(list)
    for frame in base_run.frames:
        morphology = _normalise_morphology(frame.row.get("morphology"))
        record = FrameRecord(
            prob=frame.prob,
            label=frame.label,
            pred=frame.pred,
            case_id=frame.case_id,
            morphology=morphology,
        )
        frames.append(record)
        cases[frame.case_id].append(record)
    return RunDataset(
        model=base_run.model,
        seed=base_run.seed,
        tau=base_run.tau,
        frames=frames,
        cases=dict(cases),
    )


def discover_runs(
    root: Path,
    *,
    loader: Optional[ResultLoader] = None,
) -> Dict[str, Dict[int, RunDataset]]:
    runs: DefaultDict[str, Dict[int, RunDataset]] = defaultdict(dict)
    active_loader = loader or get_default_loader()
    for metrics_path in sorted(root.rglob("*_last.metrics.json")):
        run = load_run(metrics_path, loader=active_loader)
        runs[run.model][run.seed] = run
    return runs


def aggregate_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.array(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return mean, std


def bootstrap_deltas(
    colon_runs: Mapping[int, RunDataset],
    baseline_runs: Mapping[int, RunDataset],
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    seeds = sorted(set(colon_runs.keys()) & set(baseline_runs.keys()))
    if not seeds:
        raise ValueError("No overlapping seeds between models for delta computation")
    point_estimates: Dict[str, float] = {}
    replicates: Dict[str, List[float]] = {key: [] for key in _STRATA}
    for stratum in _STRATA:
        deltas = []
        for seed in seeds:
            colon_metrics = compute_strata_metrics(colon_runs[seed].frames, colon_runs[seed].tau)
            baseline_metrics = compute_strata_metrics(baseline_runs[seed].frames, baseline_runs[seed].tau)
            if stratum not in colon_metrics or stratum not in baseline_metrics:
                continue
            deltas.append(colon_metrics[stratum]["auprc"] - baseline_metrics[stratum]["auprc"])
        point_estimates[stratum] = float(np.mean(deltas)) if deltas else float("nan")
    rng = np.random.default_rng(rng_seed)
    max_attempts = 200
    for _ in range(max(0, bootstrap)):
        stratum_values: Dict[str, List[float]] = {key: [] for key in _STRATA}
        valid = True
        for seed in seeds:
            colon_run = colon_runs[seed]
            baseline_run = baseline_runs[seed]
            case_ids = list(colon_run.cases.keys())
            if not case_ids:
                valid = False
                break
            attempts = 0
            while attempts < max_attempts:
                sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
                colon_sample: List[FrameRecord] = []
                baseline_sample: List[FrameRecord] = []
                for cid in sampled_ids:
                    colon_sample.extend(colon_run.cases[cid])
                    baseline_sample.extend(baseline_run.cases[cid])
                colon_metrics = compute_strata_metrics(colon_sample, colon_run.tau)
                baseline_metrics = compute_strata_metrics(baseline_sample, baseline_run.tau)
                if all(
                    stratum in colon_metrics
                    and stratum in baseline_metrics
                    and colon_metrics[stratum]["n_pos"] > 0
                    and baseline_metrics[stratum]["n_pos"] > 0
                    for stratum in _STRATA
                ):
                    for stratum in _STRATA:
                        delta = (
                            colon_metrics[stratum]["auprc"] - baseline_metrics[stratum]["auprc"]
                        )
                        stratum_values[stratum].append(delta)
                    break
                attempts += 1
            else:
                valid = False
                break
        if not valid:
            continue
        for stratum in _STRATA:
            values = stratum_values[stratum]
            if values:
                replicates[stratum].append(float(np.mean(values)))
    return point_estimates, replicates


def summarise_composition(metrics: Mapping[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for stratum, stats in metrics.items():
        summary[stratum] = {
            "n_pos": float(stats.get("n_pos", float("nan"))),
            "n_neg": float(stats.get("n_neg", float("nan"))),
            "prevalence": float(stats.get("prevalence", float("nan"))),
        }
    return summary


def format_mean_std(values: Sequence[float]) -> str:
    mean, std = aggregate_mean_std(values)
    if math.isnan(mean):
        return "—"
    return f"{mean:.3f} ± {std:.3f}"


def format_ci(point: float, samples: Sequence[float]) -> str:
    if not samples or math.isnan(point):
        return "—"
    arr = np.array(samples, dtype=float)
    lower = float(np.quantile(arr, 0.025))
    upper = float(np.quantile(arr, 0.975))
    return f"{point:.3f} (95% CI: {lower:.3f}–{upper:.3f})"


def model_label(model: str) -> str:
    return _MODEL_LABELS.get(model, model)


# Reporting helpers ---------------------------------------------------------

def _ordered_models(metrics_by_model: Mapping[str, Mapping[str, Mapping[str, List[float]]]]) -> List[str]:
    preferred = ["sup_imnet", "ssl_imnet", "ssl_colon"]
    ordered = [model for model in preferred if model in metrics_by_model]
    extras = sorted(set(metrics_by_model.keys()) - set(ordered))
    ordered.extend(extras)
    return ordered


def _collect_metrics(
    runs_by_model: Mapping[str, Mapping[int, RunDataset]]
) -> Tuple[Dict[str, Dict[str, Dict[str, List[float]]]], Optional[Dict[str, Dict[str, float]]]]:
    metrics_by_model: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    composition_reference: Optional[Dict[str, Dict[str, float]]] = None
    for model, runs in runs_by_model.items():
        per_stratum: Dict[str, Dict[str, List[float]]] = {}
        for run in runs.values():
            stratum_metrics = compute_strata_metrics(run.frames, run.tau)
            if model == "ssl_colon" and composition_reference is None:
                composition_reference = summarise_composition(stratum_metrics)
            for stratum, stats in stratum_metrics.items():
                container = per_stratum.setdefault(
                    stratum, {metric: [] for metric in _PRIMARY_METRICS}
                )
                for metric_key in _PRIMARY_METRICS:
                    value = stats.get(metric_key)
                    if value is None:
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(numeric):
                        continue
                    container[metric_key].append(numeric)
        if per_stratum:
            metrics_by_model[model] = per_stratum
    return metrics_by_model, composition_reference


def _render_metric_table(
    stratum: str,
    metrics_by_model: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    model_order: Sequence[str],
) -> List[str]:
    header = ["Model"] + [_METRIC_LABELS[key] for key in _PRIMARY_METRICS]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| --- | " + " | ".join(["---:"] * len(_PRIMARY_METRICS)) + " |")
    for model in model_order:
        stratum_metrics = metrics_by_model.get(model, {}).get(stratum)
        if not stratum_metrics:
            continue
        row = [model_label(model)]
        for metric_key in _PRIMARY_METRICS:
            values = stratum_metrics.get(metric_key, [])
            row.append(format_mean_std(values))
        lines.append("| " + " | ".join(row) + " |")
    if len(lines) == 2:
        lines.append("| _n/a_ | " + " | ".join(["—"] * len(_PRIMARY_METRICS)) + " |")
    return lines


def generate_report(
    runs_root: Path,
    *,
    bootstrap: int = 1000,
    rng_seed: int = 12345,
) -> str:
    runs_by_model = discover_runs(runs_root)
    if not runs_by_model:
        raise FileNotFoundError(f"No metrics files found under {runs_root}")
    metrics_by_model, composition_reference = _collect_metrics(runs_by_model)
    if not metrics_by_model:
        raise RuntimeError("Unable to collect metrics for any model")
    model_order = _ordered_models(metrics_by_model)
    if composition_reference is None:
        sample_model = model_order[0]
        sample_run = next(iter(runs_by_model[sample_model].values()))
        composition_reference = summarise_composition(
            compute_strata_metrics(sample_run.frames, sample_run.tau)
        )

    lines: List[str] = []
    lines.append("# Experiment 3 morphology-balanced report")
    lines.append("")

    if composition_reference:
        lines.append("## Test composition")
        lines.append("")
        lines.append("| Stratum | n_pos | n_neg | Prevalence |")
        lines.append("| --- | ---: | ---: | ---: |")
        for stratum in _STRATA:
            stats = composition_reference.get(stratum)
            if not stats:
                continue
            n_pos = int(stats.get("n_pos", 0))
            n_neg = int(stats.get("n_neg", 0))
            prevalence = stats.get("prevalence", float("nan"))
            prevalence_text = "—" if math.isnan(prevalence) else f"{prevalence:.3f}"
            lines.append(
                f"| {_STRATUM_LABELS.get(stratum, stratum)} | {n_pos} | {n_neg} | {prevalence_text} |"
            )
        lines.append("")

    for stratum in _STRATA:
        if not any(stratum in metrics for metrics in metrics_by_model.values()):
            continue
        lines.append(f"## Metrics at τ_F1(val-morph) — {_STRATUM_LABELS.get(stratum, stratum)}")
        lines.append("")
        lines.extend(_render_metric_table(stratum, metrics_by_model, model_order))
        lines.append("")

    colon_runs = runs_by_model.get("ssl_colon")
    delta_results: Dict[str, Tuple[Dict[str, float], Dict[str, List[float]]]] = {}
    baseline_order: List[str] = []
    if colon_runs:
        for baseline in ("sup_imnet", "ssl_imnet"):
            baseline_runs = runs_by_model.get(baseline)
            if not baseline_runs:
                continue
            point, samples = bootstrap_deltas(
                colon_runs,
                baseline_runs,
                bootstrap=bootstrap,
                rng_seed=rng_seed,
            )
            delta_results[baseline] = (point, samples)
            baseline_order.append(baseline)
    if delta_results:
        lines.append("## Representation deltas (Δ = SSL-Colon − Baseline)")
        lines.append("")
        header = ["Stratum"] + [
            f"{model_label('ssl_colon')} − {model_label(baseline)}" for baseline in baseline_order
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * len(baseline_order)) + " |")
        for stratum in _STRATA:
            row = [_STRATUM_LABELS.get(stratum, stratum)]
            any_values = False
            for baseline in baseline_order:
                point, samples = delta_results[baseline]
                value = format_ci(point.get(stratum, float("nan")), samples.get(stratum, []))
                if value != "—":
                    any_values = True
                row.append(value)
            if any_values:
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("## Interaction effect $I = Δ_{\\text{flat}} - Δ_{\\text{polypoid}}$")
        lines.append("")
        lines.append("| Comparison | I |")
        lines.append("| --- | ---: |")
        for baseline in baseline_order:
            point, samples = delta_results[baseline]
            flat_point = point.get("flat_plus_negs", float("nan"))
            polyp_point = point.get("polypoid_plus_negs", float("nan"))
            interaction_point = flat_point - polyp_point
            flat_samples = samples.get("flat_plus_negs", [])
            polyp_samples = samples.get("polypoid_plus_negs", [])
            count = min(len(flat_samples), len(polyp_samples))
            interaction_samples = [flat_samples[i] - polyp_samples[i] for i in range(count)]
            label = f"{model_label('ssl_colon')} − {model_label(baseline)}"
            lines.append(f"| {label} | {format_ci(interaction_point, interaction_samples)} |")
        lines.append("")

    return "\n".join(lines)


# End of module