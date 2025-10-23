from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
    overload,
)

import numpy as np

from .common_loader import candidate_outputs_csv_paths
from .common_metrics import (
    DEFAULT_BINARY_METRIC_KEYS,
    _clean_text,
    _coerce_float,
    _coerce_int,
    compute_binary_metrics,
)
from .display import (
    PLACEHOLDER,
    format_ci,
    format_mean_std,
    format_scalar,
    format_signed,
)
from .result_loader import GuardrailViolation, ResultLoader, build_report_manifest
from .seed_checks import SeedValidationResult, ensure_expected_seeds

PRIMARY_METRICS: Tuple[str, ...] = DEFAULT_BINARY_METRIC_KEYS
ALL_METRICS: Tuple[str, ...] = PRIMARY_METRICS
EXPECTED_MODELS: Tuple[str, ...] = ("ssl_imnet", "ssl_colon")
DEFAULT_PAIRED_MODELS: Tuple[str, str] = ("ssl_colon", "ssl_imnet")
EXPECTED_SEEDS: Tuple[int, ...] = (13, 29, 47)
CI_LEVEL = 0.95
DEFAULT_BOOTSTRAP = 2000
DEFAULT_RNG_SEED = 20240521
_PAIRED_T_CRITICAL_DF2 = 4.302652729911275
MODEL_LABELS: Dict[str, str] = {
    "ssl_imnet": "SSL-ImNet",
    "ssl_colon": "SSL-Colon",
}
METRIC_LABELS: Dict[str, str] = {
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "balanced_accuracy": "Balanced Acc",
    "mcc": "MCC",
    "loss": "Loss",
}

__all__ = [
    "PRIMARY_METRICS",
    "ALL_METRICS",
    "EXPECTED_MODELS",
    "DEFAULT_PAIRED_MODELS",
    "EXPECTED_SEEDS",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_RNG_SEED",
    "EvalFrame",
    "Exp2Run",
    "CompositionSummary",
    "MetricAggregate",
    "DeltaSummary",
    "Exp2Summary",
    "load_run",
    "discover_runs",
    "summarize_runs",
    "render_markdown",
    "write_csv_tables",
    "collect_summary",
    "build_manifest",
    "_metrics_from_frames",
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
    tau_primary: float
    tau_sensitivity: Optional[float]
    primary_metrics: Dict[str, float]
    sensitivity_metrics: Dict[str, float]
    frames: Tuple[EvalFrame, ...]
    cases: Dict[str, Tuple[EvalFrame, ...]]
    provenance: Mapping[str, Any]
    metrics_path: Path


@dataclass(frozen=True)
class CompositionSummary:
    n_pos: int
    n_neg: int

    @property
    def total(self) -> int:
        return self.n_pos + self.n_neg

    @property
    def prevalence(self) -> float:
        total = self.total
        return (self.n_pos / total) if total else float("nan")

    def as_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "n_pos": int(self.n_pos),
            "n_neg": int(self.n_neg),
            "total": int(self.total),
            "prevalence": float(self.prevalence),
        }


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
    std: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    samples: Tuple[float, ...]

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "per_seed": {int(seed): float(value) for seed, value in self.per_seed.items()},
            "mean": float(self.mean),
            "std": float(self.std) if self.std is not None else None,
            "samples": list(self.samples),
        }
        payload["ci_lower"] = float(self.ci_lower) if self.ci_lower is not None else None
        payload["ci_upper"] = float(self.ci_upper) if self.ci_upper is not None else None
        return payload


@dataclass
class Exp2Summary:
    composition: CompositionSummary
    primary_metrics: Dict[str, Dict[str, MetricAggregate]]
    primary_deltas: Dict[str, DeltaSummary]
    sensitivity_metrics: Dict[str, Dict[str, MetricAggregate]]
    sensitivity_deltas: Dict[str, DeltaSummary]
    seed_validation: SeedValidationResult

    def as_dict(self) -> Dict[str, Any]:
        return {
            "composition": self.composition.as_dict(),
            "primary": {
                "model_metrics": {
                    model: {metric: aggregate.as_dict() for metric, aggregate in metrics.items()}
                    for model, metrics in self.primary_metrics.items()
                },
                "paired_deltas": {
                    metric: summary.as_dict() for metric, summary in self.primary_deltas.items()
                },
            },
            "sensitivity": {
                "model_metrics": {
                    model: {metric: aggregate.as_dict() for metric, aggregate in metrics.items()}
                    for model, metrics in self.sensitivity_metrics.items()
                },
                "paired_deltas": {
                    metric: summary.as_dict() for metric, summary in self.sensitivity_deltas.items()
                },
            },
            "seed_validation": {
                "expected": [int(seed) for seed in self.seed_validation.expected_seeds],
                "observed": {
                    str(model): [int(seed) for seed in seeds]
                    for model, seeds in self.seed_validation.observed_seeds.items()
                },
            },
        }


def _get_loader(*, strict: bool = True) -> ResultLoader:
    return ResultLoader(exp_id="exp2", strict=strict)


def _normalise_case_id(raw: Optional[object], fallback_index: int) -> str:
    text = _clean_text(raw)
    if text:
        return text
    return f"case_{fallback_index}"


def _normalise_frame_id(raw: Optional[object], fallback_index: int) -> str:
    text = _clean_text(raw)
    if text:
        return text
    return f"frame_{fallback_index}"


def _output_candidates(metrics_path: Path) -> List[Path]:
    return list(candidate_outputs_csv_paths(metrics_path))


def _read_outputs(metrics_path: Path) -> Tuple[Tuple[EvalFrame, ...], Dict[str, Tuple[EvalFrame, ...]]]:
    last_error: Optional[Exception] = None
    for candidate in _output_candidates(metrics_path):
        if not candidate.exists():
            continue
        frames: List[EvalFrame] = []
        cases: DefaultDict[str, List[EvalFrame]] = defaultdict(list)
        try:
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader):
                    prob = _coerce_float(row.get("prob"))
                    label = _coerce_int(row.get("label"))
                    if prob is None or label is None:
                        continue
                    case_id = _normalise_case_id(row.get("case_id") or row.get("sequence_id"), index)
                    frame_id = _normalise_frame_id(row.get("frame_id"), index)
                    frame = EvalFrame(
                        frame_id=frame_id,
                        case_id=case_id,
                        prob=float(prob),
                        label=int(label),
                    )
                    frames.append(frame)
                    cases[case_id].append(frame)
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
            continue
        if not frames:
            last_error = ValueError(f"No evaluation frames parsed from {candidate}")
            continue
        return tuple(frames), {case: tuple(items) for case, items in cases.items()}
    if last_error:
        raise last_error
    raise FileNotFoundError(
        f"No test outputs CSV found for metrics file '{metrics_path}'. "
        f"Tried: {', '.join(str(path) for path in _output_candidates(metrics_path))}"
    )


def _metrics_from_frames(frames: Sequence[EvalFrame], tau: float) -> Dict[str, float]:
    if not frames:
        return {metric: float("nan") for metric in ALL_METRICS}
    probs = np.array([frame.prob for frame in frames], dtype=float)
    labels = np.array([frame.label for frame in frames], dtype=int)
    return compute_binary_metrics(probs, labels, tau, metric_keys=ALL_METRICS)


def _ensure_loss(metrics: MutableMapping[str, float], frames: Sequence[EvalFrame], tau: float) -> None:
    if "loss" in metrics and math.isfinite(_coerce_float(metrics.get("loss")) or float("nan")):
        return
    computed = _metrics_from_frames(frames, tau)
    metrics["loss"] = computed.get("loss", float("nan"))
    for key in ("n_pos", "n_neg", "prevalence", "tp", "fp", "tn", "fn"):
        if key not in metrics and key in computed:
            metrics[key] = computed[key]


def load_run(metrics_path: Path, *, loader: Optional[ResultLoader] = None) -> Exp2Run:
    active_loader = loader or _get_loader()
    payload_raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        loaded = active_loader.extract(metrics_path, payload_raw)
    except GuardrailViolation as exc:
        raise GuardrailViolation(f"{exc} (from {metrics_path})") from exc
    payload = loaded.payload
    provenance_block = payload.get("provenance")
    provenance = dict(provenance_block) if isinstance(provenance_block, Mapping) else {}
    model_name = _clean_text(provenance.get("model"))
    if not model_name:
        stem = metrics_path.stem
        model_name = stem.split("_", 1)[0]
    model = str(model_name).lower()

    seed_value = _coerce_int(payload.get("seed"))
    if seed_value is None:
        stem = metrics_path.stem
        if stem.endswith("_last"):
            stem = stem[:-5]
        if stem.endswith(".metrics"):
            stem = stem[:-8]
        if "_s" in stem:
            try:
                seed_value = int(stem.rsplit("_s", 1)[-1])
            except ValueError:
                seed_value = None
    if seed_value is None:
        raise ValueError(f"Metrics file '{metrics_path}' does not specify a seed")

    frames, cases = _read_outputs(metrics_path)

    primary_metrics = dict(loaded.primary_metrics)
    tau_primary = _coerce_float(primary_metrics.get("tau"))
    if tau_primary is None:
        raise ValueError(f"Metrics file '{metrics_path}' is missing test_primary.tau")
    _ensure_loss(primary_metrics, frames, tau_primary)

    sensitivity_metrics = dict(loaded.sensitivity_metrics)
    tau_sensitivity = _coerce_float(sensitivity_metrics.get("tau"))
    if sensitivity_metrics:
        if tau_sensitivity is None:
            raise ValueError(
                f"Metrics file '{metrics_path}' is missing test_sensitivity.tau despite sensitivity block"
            )
        _ensure_loss(sensitivity_metrics, frames, tau_sensitivity)
    else:
        tau_sensitivity = None

    return Exp2Run(
        model=model,
        seed=int(seed_value),
        tau_primary=float(tau_primary),
        tau_sensitivity=float(tau_sensitivity) if tau_sensitivity is not None else None,
        primary_metrics=primary_metrics,
        sensitivity_metrics=sensitivity_metrics,
        frames=frames,
        cases=cases,
        provenance=provenance,
        metrics_path=metrics_path,
    )


@overload
def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    strict: bool = True,
    return_loader: Literal[False] = False,
) -> Dict[str, Dict[int, Exp2Run]]:
    ...


@overload
def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    strict: bool = True,
    return_loader: Literal[True],
) -> Tuple[Dict[str, Dict[int, Exp2Run]], ResultLoader]:
    ...


def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    strict: bool = True,
    return_loader: bool = False,
) -> Union[Dict[str, Dict[int, Exp2Run]], Tuple[Dict[str, Dict[int, Exp2Run]], ResultLoader]]:
    root = root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Runs root '{root}' does not exist")
    loader = _get_loader(strict=strict)
    runs: Dict[str, Dict[int, Exp2Run]] = {}
    if models is None:
        model_filter = {name.lower() for name in EXPECTED_MODELS}
    else:
        model_filter = {name.lower() for name in models}
    metrics_paths = sorted(root.rglob("*.metrics.json"))
    chosen: Dict[str, Path] = {}
    for metrics_path in metrics_paths:
        if metrics_path.name.endswith("_best.metrics.json"):
            continue
        stem = metrics_path.stem
        base = stem[:-5] if stem.endswith("_last") else stem
        previous = chosen.get(base)
        if previous and previous.exists() and not previous.name.endswith("_last.metrics.json"):
            continue
        chosen[base] = metrics_path
    for metrics_path in chosen.values():
        try:
            run = load_run(metrics_path, loader=loader)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load metrics from {metrics_path} (missing per-frame outputs)"
            ) from exc
        except (ValueError, GuardrailViolation) as exc:
            raise RuntimeError(f"Failed to load metrics from {metrics_path}") from exc
        if model_filter and run.model not in model_filter:
            continue
        runs.setdefault(run.model, {})[run.seed] = run
    if return_loader:
        return runs, loader
    return runs


def _validate_composition(runs_by_model: Mapping[str, Mapping[int, Exp2Run]]) -> CompositionSummary:
    reference: Optional[Tuple[int, int]] = None
    for model_runs in runs_by_model.values():
        for run in model_runs.values():
            n_pos = _coerce_int(run.primary_metrics.get("n_pos")) or 0
            n_neg = _coerce_int(run.primary_metrics.get("n_neg")) or 0
            current = (n_pos, n_neg)
            if reference is None:
                reference = current
            elif reference != current:
                raise ValueError(
                    "Mismatch in SUN-test composition across runs: "
                    f"expected {reference}, found {current} (run={run.metrics_path})."
                )
    if reference is None:
        raise ValueError("No runs available to determine SUN-test composition")
    return CompositionSummary(n_pos=reference[0], n_neg=reference[1])


def _aggregate(values: Iterable[float]) -> Optional[MetricAggregate]:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    clean = [value for value in numeric if math.isfinite(value)]
    if not clean:
        return None
    array = np.array(clean, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return MetricAggregate(mean=mean, std=std, n=int(array.size), values=tuple(clean))


def _collect_model_metrics(
    runs_by_model: Mapping[str, Mapping[int, Exp2Run]],
    *,
    block: str,
    metrics: Sequence[str],
) -> Dict[str, Dict[str, MetricAggregate]]:
    aggregates: Dict[str, Dict[str, MetricAggregate]] = {}
    for model, runs in runs_by_model.items():
        per_metric: Dict[str, MetricAggregate] = {}
        for metric in metrics:
            raw_values = [
                (
                    run.primary_metrics
                    if block == "primary"
                    else run.sensitivity_metrics
                ).get(metric)
                for run in runs.values()
            ]
            numeric_values = [
                float(value)
                for value in raw_values
                if isinstance(value, (int, float, np.integer, np.floating))
            ]
            aggregate = _aggregate(numeric_values)
            if aggregate:
                per_metric[metric] = aggregate
        if per_metric:
            aggregates[model] = per_metric
    return aggregates


def _paired_bootstrap(
    treatment_runs: Mapping[int, Exp2Run],
    baseline_runs: Mapping[int, Exp2Run],
    *,
    metrics: Sequence[str],
    block: str,
    bootstrap: int,
    rng_seed: Optional[int],
) -> Dict[str, List[float]]:
    seeds = sorted(set(treatment_runs.keys()) & set(baseline_runs.keys()))
    if not seeds:
        return {metric: [] for metric in metrics}
    rng = np.random.default_rng(rng_seed)
    replicates: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for _ in range(max(0, bootstrap)):
        per_seed_samples: Dict[str, List[float]] = {metric: [] for metric in metrics}
        valid = True
        for seed in seeds:
            treatment = treatment_runs[seed]
            baseline = baseline_runs[seed]
            tau_treatment = treatment.tau_primary if block == "primary" else treatment.tau_sensitivity
            tau_baseline = baseline.tau_primary if block == "primary" else baseline.tau_sensitivity
            if tau_treatment is None or tau_baseline is None:
                valid = False
                break
            case_ids = sorted(set(treatment.cases.keys()) & set(baseline.cases.keys()))
            if not case_ids:
                valid = False
                break
            sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
            treatment_frames: List[EvalFrame] = []
            baseline_frames: List[EvalFrame] = []
            for cid in sampled_ids:
                treatment_frames.extend(treatment.cases[cid])
                baseline_frames.extend(baseline.cases[cid])
            metrics_treatment = _metrics_from_frames(treatment_frames, tau_treatment)
            metrics_baseline = _metrics_from_frames(baseline_frames, tau_baseline)
            for metric in metrics:
                value_t = metrics_treatment.get(metric)
                value_b = metrics_baseline.get(metric)
                if value_t is None or value_b is None:
                    valid = False
                    break
                if not (math.isfinite(value_t) and math.isfinite(value_b)):
                    valid = False
                    break
                per_seed_samples[metric].append(float(value_t - value_b))
            if not valid:
                break
        if not valid:
            continue
        for metric in metrics:
            sample_values = per_seed_samples[metric]
            if sample_values:
                replicates[metric].append(float(np.mean(sample_values)))
    return replicates


def _compute_delta_summaries(
    treatment_runs: Mapping[int, Exp2Run],
    baseline_runs: Mapping[int, Exp2Run],
    *,
    metrics: Sequence[str],
    block: str,
    bootstrap: int,
    rng_seed: Optional[int],
) -> Dict[str, DeltaSummary]:
    seeds = sorted(set(treatment_runs.keys()) & set(baseline_runs.keys()))
    if not seeds:
        return {}
    per_seed_delta: Dict[str, Dict[int, float]] = {metric: {} for metric in metrics}
    for seed in seeds:
        treatment = treatment_runs[seed]
        baseline = baseline_runs[seed]
        source_treatment = treatment.primary_metrics if block == "primary" else treatment.sensitivity_metrics
        source_baseline = baseline.primary_metrics if block == "primary" else baseline.sensitivity_metrics
        for metric in metrics:
            value_t = source_treatment.get(metric)
            value_b = source_baseline.get(metric)
            if value_t is None or value_b is None:
                continue
            if not (math.isfinite(value_t) and math.isfinite(value_b)):
                continue
            per_seed_delta[metric][seed] = float(value_t - value_b)
    replicates = _paired_bootstrap(
        treatment_runs,
        baseline_runs,
        metrics=metrics,
        block=block,
        bootstrap=bootstrap,
        rng_seed=rng_seed,
    )
    summaries: Dict[str, DeltaSummary] = {}
    for metric in metrics:
        per_seed_map = per_seed_delta[metric]
        if not per_seed_map:
            continue
        values = [value for value in per_seed_map.values() if math.isfinite(value)]
        if not values:
            continue
        array = np.array(values, dtype=float)
        n = int(array.size)
        mean_delta = float(np.mean(array))
        std_delta = float(np.std(array, ddof=1)) if n > 1 else 0.0
        samples = replicates.get(metric, [])
        ci: Optional[Tuple[float, float]] = None
        if n > 1 and math.isfinite(std_delta):
            if std_delta <= 0.0:
                point = float(mean_delta)
                ci = (point, point)
            else:
                std_error = std_delta / math.sqrt(float(n))
                margin = _PAIRED_T_CRITICAL_DF2 * std_error
                lower = float(mean_delta) - margin
                upper = float(mean_delta) + margin
                ci = (lower, upper)
        summaries[metric] = DeltaSummary(
            per_seed=dict(sorted(per_seed_map.items())),
            mean=mean_delta,
            std=std_delta,
            ci_lower=ci[0] if ci else None,
            ci_upper=ci[1] if ci else None,
            samples=tuple(samples),
        )
    return summaries


def summarize_runs(
    runs_by_model: Mapping[str, Mapping[int, Exp2Run]],
    *,
    metrics: Sequence[str] = ALL_METRICS,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    rng_seed: Optional[int] = DEFAULT_RNG_SEED,
    paired_models: Tuple[str, str] = DEFAULT_PAIRED_MODELS,
) -> Exp2Summary:
    if not runs_by_model:
        raise ValueError("No runs discovered for Experiment 2")
    target_groups = {model: runs_by_model.get(model, {}) for model in EXPECTED_MODELS}
    seed_validation = ensure_expected_seeds(
        target_groups,
        expected_seeds=EXPECTED_SEEDS,
        context="Experiment 2",
    )
    treatment_label, baseline_label = paired_models
    treatment_runs = runs_by_model.get(
        treatment_label, target_groups.get(treatment_label, {})
    )
    baseline_runs = runs_by_model.get(
        baseline_label, target_groups.get(baseline_label, {})
    )
    missing_labels = [
        label
        for label, runs in (
            (treatment_label, treatment_runs),
            (baseline_label, baseline_runs),
        )
        if not runs
    ]
    if missing_labels:
        formatted = ", ".join(sorted(missing_labels))
        raise ValueError(
            "Experiment 2 paired comparison requires runs for each model; "
            f"missing runs for: {formatted}"
        )
    composition = _validate_composition(target_groups)
    primary_metrics = _collect_model_metrics(target_groups, block="primary", metrics=metrics)
    sensitivity_metrics = _collect_model_metrics(target_groups, block="sensitivity", metrics=metrics)

    primary_deltas = (
        _compute_delta_summaries(
            treatment_runs,
            baseline_runs,
            metrics=metrics,
            block="primary",
            bootstrap=bootstrap,
            rng_seed=rng_seed,
        )
        if treatment_runs and baseline_runs
        else {}
    )
    sensitivity_deltas = (
        _compute_delta_summaries(
            treatment_runs,
            baseline_runs,
            metrics=metrics,
            block="sensitivity",
            bootstrap=bootstrap,
            rng_seed=rng_seed,
        )
        if treatment_runs and baseline_runs
        else {}
    )

    return Exp2Summary(
        composition=composition,
        primary_metrics=primary_metrics,
        primary_deltas=primary_deltas,
        sensitivity_metrics=sensitivity_metrics,
        sensitivity_deltas=sensitivity_deltas,
        seed_validation=seed_validation,
    )


def _ordered_models(models: Iterable[str]) -> List[str]:
    ordered: List[str] = [model for model in EXPECTED_MODELS if model in models]
    extras = sorted(set(models) - set(ordered))
    ordered.extend(extras)
    return ordered


def _render_metric_table(
    title: str,
    metrics_by_model: Mapping[str, Mapping[str, MetricAggregate]],
    *,
    metrics: Sequence[str],
) -> List[str]:
    if not metrics_by_model:
        return [title, "", "No metrics available.", ""]
    lines = [title, ""]
    header = ["Model"] + [METRIC_LABELS.get(metric, metric.upper()) for metric in metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * len(metrics)) + " |")
    for model in _ordered_models(metrics_by_model.keys()):
        aggregates = metrics_by_model.get(model)
        if not aggregates:
            continue
        row = [MODEL_LABELS.get(model, model)]
        for metric in metrics:
            aggregate = aggregates.get(metric)
            if not aggregate:
                row.append(PLACEHOLDER)
                continue
            row.append(format_mean_std(aggregate.mean, aggregate.std))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _render_delta_table(
    title: str,
    deltas: Mapping[str, DeltaSummary],
    *,
    metrics: Sequence[str],
    comparison_label: str,
) -> List[str]:
    if not deltas:
        return [title, "", "No paired delta results available.", ""]
    lines = [title, ""]
    header = ["Metric", f"Mean Δ ({comparison_label})", "±SD", "95% CI", "Per-seed"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| Metric | ---: | ---: | ---: | --- |")
    for metric in metrics:
        summary = deltas.get(metric)
        if not summary:
            continue
        row_metric = METRIC_LABELS.get(metric, metric.upper())
        mean_text = format_signed(summary.mean)
        std_text = format_scalar(summary.std) if summary.std is not None else PLACEHOLDER
        if summary.ci_lower is not None and summary.ci_upper is not None:
            ci_text = format_ci(summary.ci_lower, summary.ci_upper)
        else:
            ci_text = PLACEHOLDER
        per_seed_text = ", ".join(
            f"s{seed}={format_signed(value)}" for seed, value in sorted(summary.per_seed.items())
        )
        if not per_seed_text:
            per_seed_text = PLACEHOLDER
        lines.append(
            "| "
            + " | ".join([row_metric, mean_text, std_text, ci_text, per_seed_text])
            + " |"
        )
    lines.append("")
    return lines


def render_markdown(summary: Exp2Summary) -> str:
    lines: List[str] = []
    lines.append("# Experiment 2 — SSL-ImNet vs SSL-Colon (SUN)")
    lines.append("")
    composition = summary.composition
    lines.append("## T1 — SUN-test composition")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| n_pos | {composition.n_pos} |")
    lines.append(f"| n_neg | {composition.n_neg} |")
    lines.append(f"| Total | {composition.total} |")
    lines.append(f"| Prevalence | {format_scalar(composition.prevalence)} |")
    lines.append("")
    lines.extend(
        _render_metric_table(
            "## T2 — Model performance at τ_F1(val) (test_primary)",
            summary.primary_metrics,
            metrics=ALL_METRICS,
        )
    )
    lines.extend(
        _render_delta_table(
            "## T3 — Paired deltas at τ_F1(val) (Δ = SSL-Colon − SSL-ImNet)",
            summary.primary_deltas,
            metrics=ALL_METRICS,
            comparison_label="SSL-Colon − SSL-ImNet",
        )
    )
    lines.append("## Appendix — Sensitivity threshold (τ_Youden(val))")
    lines.append("")
    lines.extend(
        _render_metric_table(
            "### A.1 — Model performance at τ_Youden(val) (test_sensitivity)",
            summary.sensitivity_metrics,
            metrics=ALL_METRICS,
        )
    )
    lines.extend(
        _render_delta_table(
            "### A.2 — Paired deltas at τ_Youden(val)",
            summary.sensitivity_deltas,
            metrics=ALL_METRICS,
            comparison_label="SSL-Colon − SSL-ImNet",
        )
    )
    return "\n".join(lines)


def write_csv_tables(summary: Exp2Summary, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, Path] = {}

    def _write_table(filename: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        return path

    comp_path = _write_table(
        "t1_composition.csv",
        ("metric", "value"),
        [
            ("n_pos", summary.composition.n_pos),
            ("n_neg", summary.composition.n_neg),
            ("total", summary.composition.total),
            ("prevalence", summary.composition.prevalence),
        ],
    )
    created["t1_composition"] = comp_path

    primary_rows: List[List[Any]] = []
    for model, metrics in summary.primary_metrics.items():
        row: List[Any] = [model]
        for metric in ALL_METRICS:
            aggregate = metrics.get(metric)
            if aggregate:
                row.extend([aggregate.mean, aggregate.std])
            else:
                row.extend([float("nan"), float("nan")])
        primary_rows.append(row)
    primary_path = _write_table(
        "t2_primary_metrics.csv",
        ["model"]
        + [item for metric in ALL_METRICS for item in (f"{metric}_mean", f"{metric}_std")],
        primary_rows,
    )
    created["t2_primary_metrics"] = primary_path

    sensitivity_rows: List[List[Any]] = []
    for model, metrics in summary.sensitivity_metrics.items():
        row = [model]
        for metric in ALL_METRICS:
            aggregate = metrics.get(metric)
            if aggregate:
                row.extend([aggregate.mean, aggregate.std])
            else:
                row.extend([float("nan"), float("nan")])
        sensitivity_rows.append(row)
    sensitivity_path = _write_table(
        "appendix_sensitivity_metrics.csv",
        ["model"]
        + [item for metric in ALL_METRICS for item in (f"{metric}_mean", f"{metric}_std")],
        sensitivity_rows,
    )
    created["appendix_sensitivity_metrics"] = sensitivity_path

    def _delta_rows(deltas: Mapping[str, DeltaSummary]) -> List[Sequence[Any]]:
        rows: List[Sequence[Any]] = []
        for metric in ALL_METRICS:
            summary_delta = deltas.get(metric)
            if not summary_delta:
                continue
            row = [
                metric,
                summary_delta.mean,
                summary_delta.std if summary_delta.std is not None else float("nan"),
                summary_delta.ci_lower if summary_delta.ci_lower is not None else float("nan"),
                summary_delta.ci_upper if summary_delta.ci_upper is not None else float("nan"),
            ]
            for seed in EXPECTED_SEEDS:
                row.append(summary_delta.per_seed.get(seed, float("nan")))
            rows.append(row)
        return rows

    primary_delta_path = _write_table(
        "t3_primary_deltas.csv",
        ["metric", "mean", "std", "ci_lower", "ci_upper"] + [f"s{seed}" for seed in EXPECTED_SEEDS],
        _delta_rows(summary.primary_deltas),
    )
    created["t3_primary_deltas"] = primary_delta_path

    sensitivity_delta_path = _write_table(
        "appendix_sensitivity_deltas.csv",
        ["metric", "mean", "std", "ci_lower", "ci_upper"] + [f"s{seed}" for seed in EXPECTED_SEEDS],
        _delta_rows(summary.sensitivity_deltas),
    )
    created["appendix_sensitivity_deltas"] = sensitivity_delta_path

    return created


def collect_summary(
    runs_root: Path,
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    rng_seed: Optional[int] = DEFAULT_RNG_SEED,
    strict: bool = True,
) -> Tuple[Dict[str, Dict[int, Exp2Run]], Optional[Exp2Summary], ResultLoader]:
    runs, loader = discover_runs(runs_root, strict=strict, return_loader=True)
    if not runs:
        return {}, None, loader
    summary = summarize_runs(runs, bootstrap=bootstrap, rng_seed=rng_seed)
    return runs, summary, loader


def build_manifest(
    summary: Optional[Exp2Summary],
    *,
    loader: ResultLoader,
    manifest_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    extra_outputs: Optional[Iterable[Path]] = None,
    rng_seed: Optional[int] = None,
    bootstrap: Optional[int] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "experiment": "exp2",
    }
    if summary is not None:
        metadata["summary"] = summary.as_dict()
    manifest = build_report_manifest(
        output_path=output_path,
        loader=loader,
        runs=loader.loaded_runs,
        extra_outputs=extra_outputs,
        rng_seed=rng_seed,
        bootstrap=bootstrap,
        metadata=metadata,
        validated_seeds=summary.seed_validation.expected_seeds if summary else None,
        seed_groups=summary.seed_validation.observed_seeds if summary else None,
    )
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
