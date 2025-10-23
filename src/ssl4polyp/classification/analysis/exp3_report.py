from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning  # type: ignore[import]
from .common_loader import get_default_loader, load_common_run
from .result_loader import ResultLoader
from .common_metrics import compute_binary_metrics
from .seed_checks import ensure_expected_seeds
from .display import (
    PLACEHOLDER,
    format_interval,
    format_mean_std as _format_mean_std_value,
    format_scalar,
)


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
EXPECTED_SEEDS: Tuple[int, ...] = (13, 29, 47)

_PRIMARY_METRICS = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "mcc",
    "loss",
)

_DELTA_METRICS = _PRIMARY_METRICS
_METRIC_LABELS = {
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "balanced_accuracy": "Balanced Acc",
    "mcc": "MCC",
    "loss": "Loss",
}

_POLICY_LABELS = {
    "f1_opt_on_val": "τ_F1(val-morph)",
    "youden_on_val": "τ_Youden(val-morph)",
    "sun_val_frozen": "τ_SUN(val-frozen)",
}

_KNOWN_THRESHOLD_POLICIES = tuple(
    sorted(
        {
            "f1_opt_on_val",
            "youden_on_val",
            "sun_val_frozen",
            "val_opt_youden",
        },
        key=len,
        reverse=True,
    )
)

_APPENDIX_TITLES = {
    "youden_on_val": "Sensitivity operating point",
    "sun_val_frozen": "SUN-frozen operating point",
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
    primary_policy: str
    thresholds: Dict[str, float]
    data_splits: Dict[str, str]
    frames: List[FrameRecord]
    cases: Dict[str, List[FrameRecord]]

    def available_policies(self) -> Sequence[str]:
        return tuple(self.thresholds.keys())

    @property
    def primary_tau(self) -> float:
        return self.tau_for_policy(self.primary_policy)

    def tau_for_policy(self, policy: str) -> float:
        key = _canonical_policy(policy)
        if key is None:
            raise KeyError("Policy name cannot be empty")
        if key not in self.thresholds:
            raise KeyError(f"Policy '{key}' not available for run {self.model!r} seed {self.seed}")
        return float(self.thresholds[key])


def _normalise_morphology(raw: Optional[object]) -> str:
    if raw is None:
        return "unknown"
    text = str(raw).strip()
    return text.lower() if text else "unknown"


def _canonical_policy(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _policy_from_source_key(key: Optional[object]) -> Optional[str]:
    if key is None:
        return None
    text = str(key).strip().lower()
    if not text:
        return None
    for candidate in _KNOWN_THRESHOLD_POLICIES:
        if text.endswith(candidate):
            return candidate
    return None


def _coerce_float(value: Optional[object]) -> Optional[float]:
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


def _binary_metrics(probabilities: np.ndarray, labels: np.ndarray, tau: float) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        metrics = compute_binary_metrics(
            probabilities, labels, tau, metric_keys=_PRIMARY_METRICS
        )
    total = int(labels.size)
    if total == 0:
        metrics.setdefault("count", 0.0)
        metrics.setdefault("n_pos", 0.0)
        metrics.setdefault("n_neg", 0.0)
        metrics.setdefault("prevalence", float("nan"))
        metrics.setdefault("tp", 0.0)
        metrics.setdefault("fp", 0.0)
        metrics.setdefault("tn", 0.0)
        metrics.setdefault("fn", 0.0)
    return metrics


def compute_strata_metrics(frames: Sequence[FrameRecord], tau: float) -> Dict[str, Dict[str, float]]:
    if not frames:
        return {}
    probs = np.array([record.prob for record in frames], dtype=float)
    labels = np.array([record.label for record in frames], dtype=int)
    morph = np.array([record.morphology for record in frames])
    metrics: Dict[str, Dict[str, float]] = {}
    metrics["overall"] = _binary_metrics(probs, labels, tau)
    pos_mask = labels == 1
    neg_mask = labels == 0
    flat_mask = neg_mask | (pos_mask & (morph == "flat"))
    metrics["flat_plus_negs"] = _binary_metrics(probs[flat_mask], labels[flat_mask], tau)
    polypoid_mask = neg_mask | (pos_mask & (morph == "polypoid"))
    metrics["polypoid_plus_negs"] = _binary_metrics(
        probs[polypoid_mask], labels[polypoid_mask], tau
    )
    return metrics


def _get_loader(*, strict: bool = True) -> ResultLoader:
    return get_default_loader(exp_id="exp3b", strict=strict)


def load_run(
    metrics_path: Path,
    *,
    loader: Optional[ResultLoader] = None,
) -> RunDataset:
    base_run = load_common_run(metrics_path, loader=loader or _get_loader())
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
    unexpected_morphologies = sorted(
        {
            record.morphology
            for record in frames
            if record.label == 1 and record.morphology not in {"flat", "polypoid"}
        }
    )
    if unexpected_morphologies:
        details = ", ".join(unexpected_morphologies)
        raise RuntimeError(
            "Experiment 3 requires positive cases to declare 'flat' or 'polypoid' morphology; "
            f"found unexpected labels {{{details}}} in run {base_run.model!r} seed {base_run.seed}"
        )
    payload = base_run.payload
    data_block = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    data_splits: Dict[str, str] = {}
    if isinstance(data_block, Mapping):
        for split in ("train", "val", "test"):
            entry = data_block.get(split)
            if not isinstance(entry, Mapping):
                continue
            raw_path = entry.get("path")
            if isinstance(raw_path, str):
                path_text = raw_path.strip()
                if path_text:
                    data_splits[split] = path_text

    thresholds_payload = payload.get("thresholds") if isinstance(payload.get("thresholds"), Mapping) else {}
    primary_record = (
        thresholds_payload.get("primary")
        if isinstance(thresholds_payload, Mapping) and isinstance(thresholds_payload.get("primary"), Mapping)
        else None
    )
    if not isinstance(primary_record, Mapping):
        raise RuntimeError(
            f"Metrics file '{base_run.metrics_path}' does not define thresholds.primary block"
        )
    expected_primary_policy = _canonical_policy("f1_opt_on_val")
    primary_policy_recorded = _canonical_policy(primary_record.get("policy"))
    if primary_policy_recorded != expected_primary_policy:
        raise RuntimeError(
            f"Metrics file '{base_run.metrics_path}' declares thresholds.primary.policy="
            f"{primary_policy_recorded!r} instead of 'f1_opt_on_val'"
        )
    primary_tau_recorded = _coerce_float(primary_record.get("tau"))
    if primary_tau_recorded is None:
        raise RuntimeError(
            f"Metrics file '{base_run.metrics_path}' is missing thresholds.primary.tau"
        )
    if not math.isclose(primary_tau_recorded, float(base_run.tau), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Metrics file '{base_run.metrics_path}' reports test_primary.tau={base_run.tau} "
            f"but thresholds.primary.tau={primary_tau_recorded}"
        )
    primary_policy = "f1_opt_on_val"
    thresholds: Dict[str, float] = {primary_policy: float(base_run.tau)}

    def _register_policy(
        policy: Optional[str],
        tau_value: Optional[float],
        *,
        source_key: Optional[object] = None,
    ) -> None:
        canonical = _canonical_policy(policy)
        if canonical is None:
            derived = _policy_from_source_key(source_key)
            canonical = _canonical_policy(derived)
        numeric = _coerce_float(tau_value)
        if canonical and numeric is not None:
            thresholds[canonical] = float(numeric)

    sensitivity_record = (
        thresholds_payload.get("sensitivity")
        if isinstance(thresholds_payload, Mapping) and isinstance(thresholds_payload.get("sensitivity"), Mapping)
        else None
    )
    sensitivity_policy = None
    sensitivity_tau = None
    if isinstance(sensitivity_record, Mapping):
        sensitivity_policy = _canonical_policy(sensitivity_record.get("policy"))
        sensitivity_tau = _coerce_float(sensitivity_record.get("tau"))
    if sensitivity_tau is None:
        sensitivity_block = payload.get("test_sensitivity")
        if isinstance(sensitivity_block, Mapping):
            sensitivity_tau = _coerce_float(sensitivity_block.get("tau"))
    _register_policy(sensitivity_policy, sensitivity_tau, source_key="sensitivity")

    if isinstance(thresholds_payload, Mapping):
        for key, value in thresholds_payload.items():
            if key in {"primary", "sensitivity"}:
                continue
            if key == "values" and isinstance(value, Mapping):
                for inner_key, inner_value in value.items():
                    _register_policy(
                        _canonical_policy(inner_key),
                        inner_value,
                        source_key=inner_key,
                    )
                continue
            if not isinstance(value, Mapping):
                continue
            policy_name = _canonical_policy(value.get("policy")) or _canonical_policy(key)
            tau_candidate = _coerce_float(value.get("tau"))
            _register_policy(policy_name, tau_candidate, source_key=key)

    return RunDataset(
        model=base_run.model,
        seed=base_run.seed,
        primary_policy=primary_policy,
        thresholds=thresholds,
        data_splits=data_splits,
        frames=frames,
        cases=dict(cases),
    )


def discover_runs(
    root: Path,
    *,
    loader: Optional[ResultLoader] = None,
) -> Dict[str, Dict[int, RunDataset]]:
    runs: DefaultDict[str, Dict[int, RunDataset]] = defaultdict(dict)
    active_loader = loader or _get_loader()
    for metrics_path in sorted(root.rglob("*_last.metrics.json")):
        try:
            run = load_run(metrics_path, loader=active_loader)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load metrics from {metrics_path} (missing per-frame outputs)"
            ) from exc
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
    policy: str,
    metrics: Sequence[str] = _PRIMARY_METRICS,
    bootstrap: int = 2000,
    rng_seed: int = 12345,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, List[float]]],
]:
    ensure_expected_seeds(
        {"treatment": colon_runs, "baseline": baseline_runs},
        expected_seeds=expected_seeds,
        context="Experiment 3 pairwise delta",
    )
    seeds = list(expected_seeds)
    metric_keys = tuple(dict.fromkeys(metrics))
    if not metric_keys:
        raise ValueError("metrics must not be empty")
    point_estimates: Dict[str, Dict[str, float]] = {
        stratum: {metric: float("nan") for metric in metric_keys} for stratum in _STRATA
    }
    replicates: Dict[str, Dict[str, List[float]]] = {
        stratum: {metric: [] for metric in metric_keys} for stratum in _STRATA
    }
    for stratum in _STRATA:
        metric_deltas: Dict[str, List[float]] = {
            metric: [] for metric in metric_keys
        }
        for seed in seeds:
            colon_tau = colon_runs[seed].tau_for_policy(policy)
            baseline_tau = baseline_runs[seed].tau_for_policy(policy)
            colon_metrics = compute_strata_metrics(colon_runs[seed].frames, colon_tau)
            baseline_metrics = compute_strata_metrics(baseline_runs[seed].frames, baseline_tau)
            if stratum not in colon_metrics or stratum not in baseline_metrics:
                continue
            for metric in metric_keys:
                colon_value = colon_metrics[stratum].get(metric)
                baseline_value = baseline_metrics[stratum].get(metric)
                if colon_value is None or baseline_value is None:
                    continue
                try:
                    colon_float = float(colon_value)
                    baseline_float = float(baseline_value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(colon_float) or math.isnan(baseline_float):
                    continue
                metric_deltas[metric].append(colon_float - baseline_float)
        for metric in metric_keys:
            deltas = metric_deltas[metric]
            point_estimates[stratum][metric] = (
                float(np.mean(deltas)) if deltas else float("nan")
            )
    rng = np.random.default_rng(rng_seed)
    max_attempts = 200
    for _ in range(max(0, bootstrap)):
        stratum_values: Dict[str, Dict[str, List[float]]] = {
            stratum: {metric: [] for metric in metric_keys} for stratum in _STRATA
        }
        valid = True
        for seed in seeds:
            colon_run = colon_runs[seed]
            baseline_run = baseline_runs[seed]
            case_ids = sorted(set(colon_run.cases.keys()) & set(baseline_run.cases.keys()))
            if not case_ids:
                valid = False
                break
            attempts = 0
            seed_success = False
            while attempts < max_attempts:
                sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
                colon_sample: List[FrameRecord] = []
                baseline_sample: List[FrameRecord] = []
                for cid in sampled_ids:
                    colon_sample.extend(colon_run.cases[cid])
                    baseline_sample.extend(baseline_run.cases[cid])
                colon_tau = colon_run.tau_for_policy(policy)
                baseline_tau = baseline_run.tau_for_policy(policy)
                colon_metrics = compute_strata_metrics(colon_sample, colon_tau)
                baseline_metrics = compute_strata_metrics(baseline_sample, baseline_tau)
                attempt_valid = True
                attempt_deltas: Dict[str, Dict[str, float]] = {}
                for stratum in _STRATA:
                    if (
                        stratum not in colon_metrics
                        or stratum not in baseline_metrics
                        or colon_metrics[stratum].get("n_pos", 0) <= 0
                        or baseline_metrics[stratum].get("n_pos", 0) <= 0
                    ):
                        attempt_valid = False
                        break
                    seed_deltas: Dict[str, float] = {}
                    for metric in metric_keys:
                        colon_value = colon_metrics[stratum].get(metric)
                        baseline_value = baseline_metrics[stratum].get(metric)
                        if colon_value is None or baseline_value is None:
                            attempt_valid = False
                            break
                        try:
                            colon_float = float(colon_value)
                            baseline_float = float(baseline_value)
                        except (TypeError, ValueError):
                            attempt_valid = False
                            break
                        if math.isnan(colon_float) or math.isnan(baseline_float):
                            attempt_valid = False
                            break
                        seed_deltas[metric] = colon_float - baseline_float
                    if not attempt_valid:
                        break
                    attempt_deltas[stratum] = seed_deltas
                if attempt_valid:
                    for stratum, seed_deltas in attempt_deltas.items():
                        for metric, delta in seed_deltas.items():
                            stratum_values[stratum][metric].append(delta)
                    seed_success = True
                    break
                attempts += 1
            if not seed_success:
                valid = False
                break
        if not valid:
            continue
        for stratum in _STRATA:
            for metric in metric_keys:
                values = stratum_values[stratum][metric]
                if values:
                    replicates[stratum][metric].append(float(np.mean(values)))
    return point_estimates, replicates


def summarise_composition(metrics: Mapping[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    def _extract_count(stats: Mapping[str, float], key: str) -> float:
        value = stats.get(key, 0.0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(numeric):
            return 0.0
        return numeric

    def _extract_prevalence(stats: Mapping[str, float]) -> float:
        value = stats.get("prevalence", float("nan"))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return numeric

    summary: Dict[str, Dict[str, float]] = {}
    overall_stats = metrics.get("overall", {})
    overall_entry = {
        "n_pos": _extract_count(overall_stats, "n_pos"),
        "n_neg": _extract_count(overall_stats, "n_neg"),
        "prevalence": _extract_prevalence(overall_stats),
    }
    summary["overall"] = overall_entry
    overall_neg = overall_entry["n_neg"]
    for stratum in _STRATA:
        if stratum == "overall":
            continue
        stats = metrics.get(stratum)
        if stats:
            summary[stratum] = {
                "n_pos": _extract_count(stats, "n_pos"),
                "n_neg": _extract_count(stats, "n_neg"),
                "prevalence": _extract_prevalence(stats),
            }
            continue
        prevalence = 0.0 if overall_neg > 0 else float("nan")
        summary[stratum] = {
            "n_pos": 0.0,
            "n_neg": overall_neg,
            "prevalence": prevalence,
        }
    return summary


def format_mean_std(values: Sequence[float]) -> str:
    mean, std = aggregate_mean_std(values)
    if math.isnan(mean):
        return PLACEHOLDER
    return _format_mean_std_value(mean, std)


def format_ci(point: float, samples: Sequence[float]) -> str:
    try:
        point_value = float(point)
    except (TypeError, ValueError):
        return PLACEHOLDER
    if math.isnan(point_value):
        return PLACEHOLDER
    if not samples:
        return PLACEHOLDER
    arr = np.array(samples, dtype=float)
    if arr.size == 0:
        return PLACEHOLDER
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return PLACEHOLDER
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    lower = float(np.quantile(arr, 0.025))
    upper = float(np.quantile(arr, 0.975))
    interval = format_interval(lower, upper)
    if interval == PLACEHOLDER:
        return PLACEHOLDER
    mean_text = _format_mean_std_value(point_value, std)
    return f"{mean_text} (95% CI: {interval})"


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
    runs_by_model: Mapping[str, Mapping[int, RunDataset]],
    *,
    policy: str,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    metrics_by_model: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for model, runs in runs_by_model.items():
        per_stratum: Dict[str, Dict[str, List[float]]] = {}
        for run in runs.values():
            try:
                tau = run.tau_for_policy(policy)
            except KeyError:
                continue
            stratum_metrics = compute_strata_metrics(run.frames, tau)
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
    return metrics_by_model


def _resolve_composition_reference(
    runs_by_model: Mapping[str, Mapping[int, RunDataset]]
) -> Optional[Dict[str, Dict[str, float]]]:
    for runs in runs_by_model.values():
        for run in runs.values():
            try:
                metrics = compute_strata_metrics(run.frames, run.primary_tau)
            except KeyError:
                continue
            if metrics:
                return summarise_composition(metrics)
    return None


def _runs_have_policy(
    runs_by_model: Mapping[str, Mapping[int, RunDataset]], policy: str
) -> bool:
    canonical = _canonical_policy(policy)
    if canonical is None:
        return False
    for runs in runs_by_model.values():
        for run in runs.values():
            if canonical not in run.thresholds:
                return False
    return True


def _ensure_sun_morphology_test(
    runs_by_model: Mapping[str, Mapping[int, RunDataset]]
) -> None:
    expected_suffixes = ("sun_morphology/test.csv", "sun_morphology/test")
    for model, runs in runs_by_model.items():
        for run in runs.values():
            test_path = run.data_splits.get("test")
            if not test_path:
                raise RuntimeError(
                    f"Run {model!r} seed {run.seed} is missing data.test.path"
                )
            normalised = str(test_path).strip().replace("\\", "/")
            if not normalised:
                raise RuntimeError(
                    f"Run {model!r} seed {run.seed} has empty data.test.path"
                )
            if not any(normalised.endswith(suffix) for suffix in expected_suffixes):
                raise RuntimeError(
                    f"Run {model!r} seed {run.seed} uses unexpected test split {test_path!r}; "
                    "expected sun_morphology/test"
                )


def _collect_deltas(
    runs_by_model: Mapping[str, Mapping[int, RunDataset]],
    *,
    policy: str,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: int,
) -> Tuple[
    Dict[
        str,
        Tuple[
            Dict[str, Dict[str, float]],
            Dict[str, Dict[str, List[float]]],
        ],
    ],
    List[str],
]:
    colon_runs = runs_by_model.get("ssl_colon")
    if not colon_runs:
        raise RuntimeError(
            "Experiment 3 delta computation requires SSL-Colon runs."
        )
    missing_baselines = [
        baseline for baseline in ("sup_imnet", "ssl_imnet") if not runs_by_model.get(baseline)
    ]
    if missing_baselines:
        missing_labels = ", ".join(model_label(baseline) for baseline in missing_baselines)
        raise RuntimeError(
            "Experiment 3 delta computation requires baseline runs for "
            f"{missing_labels}."
        )
    delta_results: Dict[
        str,
        Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, List[float]]]],
    ] = {}
    baseline_order: List[str] = []
    for baseline in ("sup_imnet", "ssl_imnet"):
        baseline_runs = runs_by_model.get(baseline)
        assert baseline_runs is not None
        point, samples = bootstrap_deltas(
            colon_runs,
            baseline_runs,
            policy=policy,
            metrics=metrics,
            bootstrap=bootstrap,
            rng_seed=rng_seed,
        )
        delta_results[baseline] = (point, samples)
        baseline_order.append(baseline)
    return delta_results, baseline_order


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


def _render_metrics_block(
    lines: List[str],
    *,
    policy: str,
    metrics_by_model: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    model_order: Sequence[str],
    heading_level: int,
) -> None:
    label = _POLICY_LABELS.get(policy, policy)
    heading = "#" * heading_level
    for stratum in _STRATA:
        if not any(stratum in metrics for metrics in metrics_by_model.values()):
            continue
        lines.append(
            f"{heading} Metrics at {label} — {_STRATUM_LABELS.get(stratum, stratum)}"
        )
        lines.append("")
        lines.extend(_render_metric_table(stratum, metrics_by_model, model_order))
        lines.append("")


def _render_deltas_block(
    lines: List[str],
    *,
    policy: str,
    delta_results: Mapping[
        str,
        Tuple[
            Dict[str, Dict[str, float]],
            Dict[str, Dict[str, List[float]]],
        ],
    ],
    baseline_order: Sequence[str],
    metrics: Sequence[str],
    heading_level: int,
) -> None:
    if not delta_results or not baseline_order:
        return
    label = _POLICY_LABELS.get(policy, policy)
    heading = "#" * heading_level
    metric_keys = tuple(dict.fromkeys(metrics))
    if not metric_keys:
        return
    sections: List[str] = []
    for metric in metric_keys:
        metric_label = _METRIC_LABELS.get(metric, metric)
        header = ["Stratum"] + [
            f"{model_label('ssl_colon')} − {model_label(baseline)}"
            for baseline in baseline_order
        ]
        rows: List[str] = []
        for stratum in _STRATA:
            row = [_STRATUM_LABELS.get(stratum, stratum)]
            any_values = False
            for baseline in baseline_order:
                point, samples = delta_results[baseline]
                stratum_points = point.get(stratum, {})
                stratum_samples = samples.get(stratum, {})
                value = format_ci(
                    stratum_points.get(metric, float("nan")),
                    stratum_samples.get(metric, []),
                )
                if value != PLACEHOLDER:
                    any_values = True
                row.append(value)
            if any_values:
                rows.append("| " + " | ".join(row) + " |")
        if not rows:
            continue
        sections.append(f"**{metric_label}**")
        sections.append("")
        sections.append("| " + " | ".join(header) + " |")
        sections.append("| --- | " + " | ".join(["---:"] * len(baseline_order)) + " |")
        sections.extend(rows)
        sections.append("")
    if not sections:
        return
    lines.append(
        f"{heading} Representation deltas at {label} (Δ = {model_label('ssl_colon')} − Baseline)"
    )
    lines.append("")
    lines.extend(sections)


def _render_interaction_block(
    lines: List[str],
    *,
    policy: str,
    delta_results: Mapping[
        str,
        Tuple[
            Dict[str, Dict[str, float]],
            Dict[str, Dict[str, List[float]]],
        ],
    ],
    baseline_order: Sequence[str],
    metrics: Sequence[str],
    heading_level: int,
) -> None:
    if not delta_results or not baseline_order:
        return
    label = _POLICY_LABELS.get(policy, policy)
    heading = "#" * heading_level
    metric_keys = tuple(dict.fromkeys(metrics))
    if not metric_keys:
        return
    sections: List[str] = []
    for metric in metric_keys:
        metric_label = _METRIC_LABELS.get(metric, metric)
        rows: List[str] = []
        for baseline in baseline_order:
            point, samples = delta_results[baseline]
            flat_point = point.get("flat_plus_negs", {}).get(metric, float("nan"))
            polyp_point = point.get("polypoid_plus_negs", {}).get(metric, float("nan"))
            interaction_point = flat_point - polyp_point
            flat_samples = samples.get("flat_plus_negs", {}).get(metric, [])
            polyp_samples = samples.get("polypoid_plus_negs", {}).get(metric, [])
            count = min(len(flat_samples), len(polyp_samples))
            interaction_samples = [
                flat_samples[i] - polyp_samples[i] for i in range(count)
            ]
            label_text = f"{model_label('ssl_colon')} − {model_label(baseline)}"
            rows.append(
                f"| {label_text} | {format_ci(interaction_point, interaction_samples)} |"
            )
        if not rows:
            continue
        sections.append(f"**{metric_label}**")
        sections.append("")
        sections.append("| Comparison | I |")
        sections.append("| --- | ---: |")
        sections.extend(rows)
        sections.append("")
    if not sections:
        return
    lines.append(
        f"{heading} Interaction effect at {label} $I = Δ_{{\\text{{flat}}}} - Δ_{{\\text{{polypoid}}}}$"
    )
    lines.append("")
    lines.extend(sections)


def generate_report(
    runs_root: Path,
    *,
    bootstrap: int = 2000,
    rng_seed: int = 12345,
) -> str:
    loader = _get_loader()
    runs_by_model = discover_runs(runs_root, loader=loader)
    if not runs_by_model:
        raise FileNotFoundError(f"No metrics files found under {runs_root}")
    ensure_expected_seeds(
        runs_by_model,
        expected_seeds=EXPECTED_SEEDS,
        context="Experiment 3",
    )
    _ensure_sun_morphology_test(runs_by_model)
    primary_policy: Optional[str] = None
    for runs in runs_by_model.values():
        if runs:
            primary_policy = next(iter(runs.values())).primary_policy
            break
    if not primary_policy:
        raise RuntimeError("Unable to determine primary threshold policy")
    if not _runs_have_policy(runs_by_model, primary_policy):
        raise RuntimeError(
            f"Missing primary threshold policy '{primary_policy}' across runs"
        )

    metrics_by_model = _collect_metrics(runs_by_model, policy=primary_policy)
    if not metrics_by_model:
        raise RuntimeError("Unable to collect metrics for any model")
    model_order = _ordered_models(metrics_by_model)
    composition_reference = _resolve_composition_reference(runs_by_model)
    if composition_reference is None:
        sample_model = model_order[0]
        sample_run = next(iter(runs_by_model[sample_model].values()))
        composition_reference = summarise_composition(
            compute_strata_metrics(
                sample_run.frames, sample_run.tau_for_policy(primary_policy)
            )
        )

    available_policies = {
        policy for runs in runs_by_model.values() for run in runs.values() for policy in run.thresholds
    }
    appendix_policies: List[str] = []
    sensitivity_policy = "youden_on_val"
    if _canonical_policy(primary_policy) != _canonical_policy(sensitivity_policy):
        if _runs_have_policy(runs_by_model, sensitivity_policy):
            appendix_policies.append(sensitivity_policy)
        else:
            raise RuntimeError(
                "Sensitivity policy 'youden_on_val' is required but missing across runs"
            )
    for candidate in sorted(available_policies):
        if candidate in {primary_policy, sensitivity_policy}:
            continue
        if candidate in appendix_policies:
            continue
        if _runs_have_policy(runs_by_model, candidate):
            appendix_policies.append(candidate)

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
            prevalence_text = format_scalar(prevalence)
            lines.append(
                f"| {_STRATUM_LABELS.get(stratum, stratum)} | {n_pos} | {n_neg} | {prevalence_text} |"
            )
        lines.append("")

    _render_metrics_block(
        lines,
        policy=primary_policy,
        metrics_by_model=metrics_by_model,
        model_order=model_order,
        heading_level=2,
    )
    delta_results, baseline_order = _collect_deltas(
        runs_by_model,
        policy=primary_policy,
        metrics=_DELTA_METRICS,
        bootstrap=bootstrap,
        rng_seed=rng_seed,
    )
    _render_deltas_block(
        lines,
        policy=primary_policy,
        delta_results=delta_results,
        baseline_order=baseline_order,
        metrics=_DELTA_METRICS,
        heading_level=2,
    )
    _render_interaction_block(
        lines,
        policy=primary_policy,
        delta_results=delta_results,
        baseline_order=baseline_order,
        metrics=_DELTA_METRICS,
        heading_level=2,
    )

    for policy in appendix_policies:
        appendix_metrics = _collect_metrics(runs_by_model, policy=policy)
        if not appendix_metrics:
            continue
        appendix_label = _POLICY_LABELS.get(policy, policy)
        appendix_title = _APPENDIX_TITLES.get(policy, f"Results at {appendix_label}")
        lines.append(f"## Appendix: {appendix_title} ({appendix_label})")
        lines.append("")
        _render_metrics_block(
            lines,
            policy=policy,
            metrics_by_model=appendix_metrics,
            model_order=model_order,
            heading_level=3,
        )
        appendix_deltas, appendix_baselines = _collect_deltas(
            runs_by_model,
            policy=policy,
            metrics=_DELTA_METRICS,
            bootstrap=bootstrap,
            rng_seed=rng_seed,
        )
        _render_deltas_block(
            lines,
            policy=policy,
            delta_results=appendix_deltas,
            baseline_order=appendix_baselines,
            metrics=_DELTA_METRICS,
            heading_level=3,
        )
        _render_interaction_block(
            lines,
            policy=policy,
            delta_results=appendix_deltas,
            baseline_order=appendix_baselines,
            metrics=_DELTA_METRICS,
            heading_level=3,
        )

    return "\n".join(lines)


# End of module