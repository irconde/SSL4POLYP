from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
    Literal,
)

import numpy as np

from .common_metrics import (
    DEFAULT_BINARY_METRIC_KEYS,
    _clean_text,
    _coerce_float,
    _coerce_int,
    compute_binary_metrics,
)
from .display import PLACEHOLDER, format_ci, format_mean_std, format_scalar, format_signed
from .exp2_report import DeltaSummary, MetricAggregate
from .result_loader import (
    CurveMetadata,
    GuardrailViolation,
    ResultLoader,
    build_report_manifest,
    compute_file_sha256,
)
from .seed_checks import SeedValidationResult, ensure_expected_seeds

PRIMARY_METRICS: Tuple[str, ...] = DEFAULT_BINARY_METRIC_KEYS
EXPECTED_MODELS: Tuple[str, ...] = ("sup_imnet", "ssl_imnet")
EXPECTED_SEEDS: Tuple[int, ...] = (13, 29, 47)
MODEL_LABELS: Dict[str, str] = {
    "sup_imnet": "SUP-ImNet",
    "ssl_imnet": "SSL-ImNet",
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
CI_LEVEL = 0.95
DEFAULT_BOOTSTRAP = 2000
DEFAULT_RNG_SEED = 20240521


@dataclass(frozen=True)
class EvalFrame:
    frame_id: str
    case_id: str
    prob: float
    label: int


@dataclass
class Exp1Run:
    model: str
    seed: int
    primary_metrics: Dict[str, float]
    sensitivity_metrics: Dict[str, float]
    tau_primary: float
    tau_sensitivity: Optional[float]
    frames: Tuple[EvalFrame, ...]
    cases: Dict[str, Tuple[EvalFrame, ...]]
    metrics_path: Path
    curves: Dict[str, CurveMetadata]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class CompositionSummary:
    n_pos: int
    n_neg: int

    def as_dict(self) -> Dict[str, Union[int, float]]:
        total = self.n_pos + self.n_neg
        prevalence = float(self.n_pos) / float(total) if total else float("nan")
        return {
            "n_pos": int(self.n_pos),
            "n_neg": int(self.n_neg),
            "total": int(total),
            "prevalence": prevalence,
        }


@dataclass
class Exp1Summary:
    composition: CompositionSummary
    primary_metrics: Dict[str, Dict[str, MetricAggregate]]
    primary_deltas: Dict[str, DeltaSummary]
    sensitivity_metrics: Dict[str, Dict[str, MetricAggregate]]
    sensitivity_deltas: Dict[str, DeltaSummary]
    seed_validation: SeedValidationResult
    curve_assets: Dict[str, Dict[int, Dict[str, CurveMetadata]]]

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
            "curve_assets": {
                model: {
                    int(seed): {key: metadata.as_dict() for key, metadata in curves.items()}
                    for seed, curves in per_model.items()
                }
                for model, per_model in self.curve_assets.items()
            },
        }


def _get_loader(*, strict: bool = True) -> ResultLoader:
    return ResultLoader(exp_id="exp1", required_curve_keys=(), strict=strict)

def _resolve_outputs_path(metrics_path: Path) -> Tuple[Path, ...]:
    """Return candidate CSV paths for per-frame outputs.

    Legacy metrics files in this project often use compound suffixes such as
    ``*_last.metrics.json``.  ``Path.stem`` only strips the final suffix, so naively
    appending ``_test_outputs.csv`` to the stem can yield names like
    ``*_last.metrics_test_outputs.csv`` which do not exist on disk.  We therefore
    normalise the stem by progressively removing both the ``_last`` sentinel and
    the ``.metrics`` fragment before constructing candidate paths.
    """

    name = metrics_path.name
    # Strip the known metrics suffix first to recover the logical base name.
    if name.endswith(".metrics.json"):
        base = name[: -len(".metrics.json")]
    else:
        base = metrics_path.stem

    candidates: List[Path] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(metrics_path.with_name(f"{candidate}_test_outputs.csv"))

    _push(base)
    if base.endswith("_last"):
        _push(base[: -len("_last")])
    if base.endswith(".metrics"):
        _push(base[: -len(".metrics")])

    return tuple(candidates)


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


def _read_outputs(outputs_path: Path) -> Tuple[Tuple[EvalFrame, ...], Dict[str, Tuple[EvalFrame, ...]]]:
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing test outputs CSV: {outputs_path}")
    frames: List[EvalFrame] = []
    cases: DefaultDict[str, List[EvalFrame]] = DefaultDict(list)
    with outputs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            prob = _coerce_float(row.get("prob"))
            label = _coerce_int(row.get("label"))
            if prob is None or label is None:
                continue
            case_id = _normalise_case_id(row.get("case_id") or row.get("sequence_id"), index)
            frame_id = _normalise_frame_id(row.get("frame_id"), index)
            frame = EvalFrame(frame_id=frame_id, case_id=case_id, prob=float(prob), label=int(label))
            frames.append(frame)
            cases.setdefault(case_id, []).append(frame)
    if not frames:
        raise ValueError(f"No evaluation frames parsed from {outputs_path}")
    return tuple(frames), {case: tuple(items) for case, items in cases.items()}


def _metrics_from_frames(frames: Sequence[EvalFrame], tau: float) -> Dict[str, float]:
    if not frames:
        return {metric: float("nan") for metric in PRIMARY_METRICS}
    probs = np.array([frame.prob for frame in frames], dtype=float)
    labels = np.array([frame.label for frame in frames], dtype=int)
    return compute_binary_metrics(probs, labels, tau, metric_keys=PRIMARY_METRICS)


def _merge_metric_block(
    target: MutableMapping[str, float], computed: Mapping[str, float]
) -> None:
    for key, value in computed.items():
        existing = _coerce_float(target.get(key))
        if existing is None or not math.isfinite(existing):
            target[key] = float(value)


def _aggregate(values: Iterable[float]) -> Optional[MetricAggregate]:
    numeric = [float(value) for value in values if isinstance(value, (int, float, np.integer, np.floating))]
    clean = [value for value in numeric if math.isfinite(value)]
    if not clean:
        return None
    array = np.array(clean, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return MetricAggregate(mean=mean, std=std, n=int(array.size), values=tuple(clean))


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


def _collect_curves(payload: Mapping[str, Any], metrics_path: Path) -> Dict[str, CurveMetadata]:
    exports = payload.get("curve_exports")
    curves: Dict[str, CurveMetadata] = {}
    if not isinstance(exports, Mapping):
        return curves
    for key, entry in exports.items():
        if not isinstance(entry, Mapping):
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        rel_path = Path(raw_path)
        resolved_path = rel_path if rel_path.is_absolute() else (metrics_path.parent / rel_path)
        if not resolved_path.exists():
            continue
        sha_field = entry.get("sha256")
        from_digest = None
        if isinstance(sha_field, str) and sha_field.strip():
            from_digest = sha_field.strip().lower()
        digest = from_digest or compute_file_sha256(resolved_path)
        metadata = {
            str(meta_key): entry[meta_key]
            for meta_key in entry
            if meta_key not in {"path", "sha256"}
        }
        curves[str(key)] = CurveMetadata(
            key=str(key),
            path=resolved_path.resolve(),
            sha256=digest,
            metadata=metadata,
        )
    return curves


def load_run(metrics_path: Path, *, loader: Optional[ResultLoader] = None) -> Exp1Run:
    active_loader = loader or _get_loader()
    payload_raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    loaded = active_loader.extract(metrics_path, payload_raw)
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
        if "_s" in stem:
            try:
                seed_value = int(stem.rsplit("_s", 1)[-1])
            except ValueError:
                seed_value = None
    if seed_value is None:
        raise ValueError(f"Metrics file '{metrics_path}' does not specify a seed")
    primary_metrics = dict(loaded.primary_metrics)
    sensitivity_metrics = dict(loaded.sensitivity_metrics)
    tau_primary_raw = primary_metrics.get("tau")
    tau_primary = _coerce_float(tau_primary_raw)
    if tau_primary is None:
        raise ValueError(f"Metrics file '{metrics_path}' is missing test_primary.tau")
    tau_sensitivity_raw = sensitivity_metrics.get("tau") if sensitivity_metrics else None
    tau_sensitivity = _coerce_float(tau_sensitivity_raw) if tau_sensitivity_raw is not None else None
    frames: Optional[Tuple[EvalFrame, ...]] = None
    cases: Optional[Dict[str, Tuple[EvalFrame, ...]]] = None
    outputs_candidates = _resolve_outputs_path(metrics_path)
    last_error: Optional[Exception] = None
    for candidate in outputs_candidates:
        try:
            frames, cases = _read_outputs(candidate)
            break
        except FileNotFoundError as exc:
            last_error = exc
    else:
        tried = ", ".join(str(path) for path in outputs_candidates)
        message = (
            f"No test outputs CSV found for metrics file '{metrics_path}'. Tried: {tried}"
        )
        raise FileNotFoundError(message) from last_error
    assert frames is not None and cases is not None
    computed_primary = _metrics_from_frames(frames, float(tau_primary))
    _merge_metric_block(primary_metrics, computed_primary)
    if tau_sensitivity is not None:
        computed_sensitivity = _metrics_from_frames(frames, float(tau_sensitivity))
        _merge_metric_block(sensitivity_metrics, computed_sensitivity)
    curves = dict(loaded.curves)
    return Exp1Run(
        model=model,
        seed=int(seed_value),
        primary_metrics=primary_metrics,
        sensitivity_metrics=sensitivity_metrics,
        tau_primary=float(tau_primary),
        tau_sensitivity=float(tau_sensitivity) if tau_sensitivity is not None else None,
        frames=frames,
        cases=cases,
        metrics_path=metrics_path,
        curves=curves,
        provenance=provenance,
    )


@overload
def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    strict: bool = True,
    return_loader: Literal[False] = False,
) -> Dict[str, Dict[int, Exp1Run]]:
    ...


@overload
def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    strict: bool = True,
    return_loader: Literal[True],
) -> Tuple[Dict[str, Dict[int, Exp1Run]], ResultLoader]:
    ...


def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
    strict: bool = True,
    return_loader: bool = False,
) -> Union[Dict[str, Dict[int, Exp1Run]], Tuple[Dict[str, Dict[int, Exp1Run]], ResultLoader]]:
    root = root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Runs root '{root}' does not exist")
    loader = _get_loader(strict=strict)
    runs: Dict[str, Dict[int, Exp1Run]] = {}
    if models is None:
        model_filter = {name.lower() for name in EXPECTED_MODELS}
    else:
        model_filter = {name.lower() for name in models}
    metrics_paths = sorted(root.rglob("*.metrics.json"))
    for metrics_path in metrics_paths:
        if metrics_path.name.endswith("_best.metrics.json"):
            continue
        try:
            run = load_run(metrics_path, loader=loader)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load metrics from {metrics_path} (missing per-frame outputs). {exc}"
            ) from exc
        except (ValueError, GuardrailViolation) as exc:
            raise RuntimeError(f"Failed to load metrics from {metrics_path}") from exc
        if model_filter and run.model not in model_filter:
            continue
        runs.setdefault(run.model, {})[run.seed] = run
    if return_loader:
        return runs, loader
    return runs


def _validate_composition(runs_by_model: Mapping[str, Mapping[int, Exp1Run]]) -> CompositionSummary:
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
                    "Mismatch in SUN test composition across runs: "
                    f"expected {reference}, found {current} (run={run.metrics_path})."
                )
    if reference is None:
        raise ValueError("No runs available to determine SUN test composition")
    return CompositionSummary(n_pos=reference[0], n_neg=reference[1])


def _aggregate_model_metrics(
    runs_by_model: Mapping[str, Mapping[int, Exp1Run]],
    *,
    metrics: Sequence[str],
    block: str,
) -> Dict[str, Dict[str, MetricAggregate]]:
    aggregates: Dict[str, Dict[str, MetricAggregate]] = {}
    for model, model_runs in runs_by_model.items():
        per_metric: Dict[str, MetricAggregate] = {}
        for metric in metrics:
            values: List[float] = []
            for run in model_runs.values():
                source = run.primary_metrics if block == "primary" else run.sensitivity_metrics
                value = source.get(metric)
                if value is None or not math.isfinite(float(value)):
                    continue
                values.append(float(value))
            aggregate = _aggregate(values)
            if aggregate:
                per_metric[metric] = aggregate
        if per_metric:
            aggregates[model] = per_metric
    return aggregates


def _paired_bootstrap(
    treatment_runs: Mapping[int, Exp1Run],
    baseline_runs: Mapping[int, Exp1Run],
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: Optional[int],
    block: str,
) -> Dict[str, List[float]]:
    seeds = sorted(set(treatment_runs.keys()) & set(baseline_runs.keys()))
    if not seeds or bootstrap <= 0:
        return {metric: [] for metric in metrics}
    rng = np.random.default_rng(rng_seed)
    replicates: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for _ in range(max(0, bootstrap)):
        per_seed_samples: Dict[str, List[float]] = {metric: [] for metric in metrics}
        valid = True
        for seed in seeds:
            treatment = treatment_runs[seed]
            baseline = baseline_runs[seed]
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
            tau_treatment = treatment.tau_primary if block == "primary" else (treatment.tau_sensitivity or treatment.tau_primary)
            tau_baseline = baseline.tau_primary if block == "primary" else (baseline.tau_sensitivity or baseline.tau_primary)
            metrics_treatment = _metrics_from_frames(treatment_frames, tau_treatment)
            metrics_baseline = _metrics_from_frames(baseline_frames, tau_baseline)
            for metric in metrics:
                value_a = metrics_treatment.get(metric)
                value_b = metrics_baseline.get(metric)
                if value_a is None or value_b is None:
                    continue
                if not (math.isfinite(value_a) and math.isfinite(value_b)):
                    continue
                per_seed_samples[metric].append(float(value_a - value_b))
        if not valid:
            continue
        for metric in metrics:
            samples = per_seed_samples[metric]
            if samples:
                replicates[metric].append(float(np.mean(samples)))
    return replicates


def _compute_delta_summaries(
    treatment_runs: Mapping[int, Exp1Run],
    baseline_runs: Mapping[int, Exp1Run],
    *,
    metrics: Sequence[str],
    bootstrap: int,
    rng_seed: Optional[int],
    block: str,
) -> Dict[str, DeltaSummary]:
    seeds = sorted(set(treatment_runs.keys()) & set(baseline_runs.keys()))
    if not seeds:
        return {}
    per_seed_delta: Dict[str, Dict[int, float]] = {metric: {} for metric in metrics}
    for seed in seeds:
        treatment = treatment_runs[seed]
        baseline = baseline_runs[seed]
        for metric in metrics:
            source_treatment = treatment.primary_metrics if block == "primary" else treatment.sensitivity_metrics
            source_baseline = baseline.primary_metrics if block == "primary" else baseline.sensitivity_metrics
            value_a = source_treatment.get(metric)
            value_b = source_baseline.get(metric)
            if value_a is None or value_b is None:
                continue
            if not (math.isfinite(float(value_a)) and math.isfinite(float(value_b))):
                continue
            per_seed_delta[metric][seed] = float(value_a - value_b)
    replicates = _paired_bootstrap(
        treatment_runs,
        baseline_runs,
        metrics=metrics,
        bootstrap=bootstrap,
        rng_seed=rng_seed,
        block=block,
    )
    summaries: Dict[str, DeltaSummary] = {}
    for metric in metrics:
        per_seed_map = per_seed_delta[metric]
        if not per_seed_map:
            continue
        seed_values = [value for value in per_seed_map.values() if math.isfinite(value)]
        if not seed_values:
            continue
        array = np.array(seed_values, dtype=float)
        mean_delta = float(np.mean(array))
        std_delta = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        samples = replicates.get(metric, [])
        ci = _ci_bounds(samples, level=CI_LEVEL) if samples else None
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
    runs_by_model: Mapping[str, Mapping[int, Exp1Run]],
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    rng_seed: Optional[int] = DEFAULT_RNG_SEED,
) -> Exp1Summary:
    if not runs_by_model:
        raise ValueError("No runs discovered for Experiment 1")
    target_groups = {
        model: runs_by_model.get(model, {})
        for model in EXPECTED_MODELS
        if model in runs_by_model
    }
    seed_validation = ensure_expected_seeds(
        target_groups,
        expected_seeds=EXPECTED_SEEDS,
        context="Experiment 1",
    )
    composition = _validate_composition(runs_by_model)
    primary_metrics = _aggregate_model_metrics(runs_by_model, metrics=PRIMARY_METRICS, block="primary")
    sensitivity_metrics = _aggregate_model_metrics(
        runs_by_model,
        metrics=PRIMARY_METRICS,
        block="sensitivity",
    )
    treatment_runs = runs_by_model.get("ssl_imnet", {})
    baseline_runs = runs_by_model.get("sup_imnet", {})
    if not treatment_runs or not baseline_runs:
        raise ValueError(
            "Experiment 1 requires both 'ssl_imnet' and 'sup_imnet' runs to compute paired deltas"
        )
    ensure_expected_seeds(
        {
            "ssl_imnet": treatment_runs,
            "sup_imnet": baseline_runs,
        },
        expected_seeds=seed_validation.expected_seeds,
        context="Experiment 1 pairwise (SSL-ImNet vs SUP-ImNet)",
    )
    primary_deltas = _compute_delta_summaries(
        treatment_runs,
        baseline_runs,
        metrics=PRIMARY_METRICS,
        bootstrap=bootstrap,
        rng_seed=rng_seed,
        block="primary",
    )
    sensitivity_deltas = _compute_delta_summaries(
        treatment_runs,
        baseline_runs,
        metrics=PRIMARY_METRICS,
        bootstrap=bootstrap,
        rng_seed=rng_seed,
        block="sensitivity",
    )
    curve_assets: Dict[str, Dict[int, Dict[str, CurveMetadata]]] = {}
    for model, model_runs in runs_by_model.items():
        per_seed: Dict[int, Dict[str, CurveMetadata]] = {}
        for seed, run in model_runs.items():
            if run.curves:
                per_seed[int(seed)] = dict(run.curves)
        if per_seed:
            curve_assets[model] = per_seed
    return Exp1Summary(
        composition=composition,
        primary_metrics=primary_metrics,
        primary_deltas=primary_deltas,
        sensitivity_metrics=sensitivity_metrics,
        sensitivity_deltas=sensitivity_deltas,
        seed_validation=seed_validation,
        curve_assets=curve_assets,
    )


def _ordered_models(models: Iterable[str]) -> List[str]:
    preferred = [model for model in EXPECTED_MODELS if model in models]
    extras = sorted(set(models) - set(preferred))
    return preferred + extras


def _metric_rows_for_model(model: str, aggregates: Mapping[str, MetricAggregate]) -> List[str]:
    row = [MODEL_LABELS.get(model, model)]
    for metric in PRIMARY_METRICS:
        aggregate = aggregates.get(metric)
        if not aggregate:
            row.append(PLACEHOLDER)
            continue
        row.append(format_mean_std(aggregate.mean, aggregate.std))
    return row


def _render_metric_table(title: str, metrics_by_model: Mapping[str, Dict[str, MetricAggregate]]) -> List[str]:
    if not metrics_by_model:
        return [title, "", "No metrics available.", ""]
    lines: List[str] = [title, ""]
    header = ["Model"] + [METRIC_LABELS.get(metric, metric.upper()) for metric in PRIMARY_METRICS]
    ordered = _ordered_models(metrics_by_model.keys())
    rows = ["| " + " | ".join(header) + " |"]
    rows.append("| " + " | ".join(["---"] + ["---:"] * len(PRIMARY_METRICS)) + " |")
    for model in ordered:
        aggregates = metrics_by_model.get(model, {})
        rows.append("| " + " | ".join(_metric_rows_for_model(model, aggregates)) + " |")
    lines.extend(rows)
    lines.append("")
    return lines


def _delta_std(per_seed: Mapping[int, float]) -> Optional[float]:
    values = [float(value) for value in per_seed.values() if math.isfinite(value)]
    if len(values) < 2:
        return 0.0 if values else None
    array = np.array(values, dtype=float)
    return float(np.std(array, ddof=1))


def _render_delta_table(title: str, deltas: Mapping[str, DeltaSummary]) -> List[str]:
    if not deltas:
        return [title, "", "No paired delta results available.", ""]
    lines = [title, ""]
    header = ["Metric", "Mean Δ (SSL − SUP)", "±SD", "95% CI", "Per-seed"]
    rows = ["| " + " | ".join(header) + " |"]
    rows.append("| Metric | ---: | ---: | ---: | --- |")
    for metric in PRIMARY_METRICS:
        summary = deltas.get(metric)
        if not summary:
            continue
        mean_text = format_signed(summary.mean)
        std_value = _delta_std(summary.per_seed)
        std_text = format_mean_std(summary.mean, std_value) if std_value is not None else PLACEHOLDER
        ci_text = format_ci(summary.ci_lower, summary.ci_upper) if summary.ci_lower is not None and summary.ci_upper is not None else PLACEHOLDER
        per_seed_text = ", ".join(
            f"s{seed}={format_signed(value)}" for seed, value in sorted(summary.per_seed.items())
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    METRIC_LABELS.get(metric, metric.upper()),
                    mean_text,
                    std_text.split("±")[-1].strip() if "±" in std_text else PLACEHOLDER,
                    ci_text,
                    per_seed_text or PLACEHOLDER,
                ]
            )
            + " |"
        )
    lines.extend(rows)
    lines.append("")
    return lines


def render_markdown(summary: Exp1Summary) -> str:
    lines: List[str] = []
    lines.append("# Experiment 1 report — SUP-ImNet vs SSL-ImNet (SUN)")
    lines.append("")
    composition = summary.composition.as_dict()
    lines.append("## T1 – SUN test composition")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    total = composition["total"]
    prevalence = composition["prevalence"]
    lines.append(f"| n_pos | {composition['n_pos']} |")
    lines.append(f"| n_neg | {composition['n_neg']} |")
    lines.append(f"| Total | {total} |")
    lines.append(f"| Prevalence | {format_scalar(prevalence)} |")
    lines.append("")
    lines.extend(
        _render_metric_table(
            "## T2 – Primary threshold (τ from F1-opt on val)", summary.primary_metrics
        )
    )
    lines.extend(
        _render_delta_table(
            "## T3 – Paired deltas at primary threshold (SSL − SUP)", summary.primary_deltas
        )
    )
    lines.extend(
        _render_metric_table(
            "## Appendix – Sensitivity threshold (τ from Youden)", summary.sensitivity_metrics
        )
    )
    lines.extend(
        _render_delta_table(
            "## Appendix – Paired deltas at sensitivity threshold", summary.sensitivity_deltas
        )
    )
    if summary.curve_assets:
        lines.append("## Curve assets")
        lines.append("")
        for model in _ordered_models(summary.curve_assets.keys()):
            lines.append(f"- {MODEL_LABELS.get(model, model)}:")
            per_seed = summary.curve_assets[model]
            for seed in sorted(per_seed.keys()):
                curves = per_seed[seed]
                for key, metadata in sorted(curves.items()):
                    lines.append(
                        f"  - seed {seed}, {key}: {metadata.path} (sha256={metadata.sha256})"
                    )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_csv_tables(summary: Exp1Summary, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    composition_path = output_dir / "exp1_t1_composition.csv"
    composition = summary.composition.as_dict()
    with composition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n_pos", "n_neg", "total", "prevalence"])
        writer.writerow([
            composition["n_pos"],
            composition["n_neg"],
            composition["total"],
            composition["prevalence"],
        ])
    created.append(composition_path)

    def _write_model_metrics(path: Path, metrics_by_model: Mapping[str, Dict[str, MetricAggregate]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "metric", "mean", "std", "n"])
            for model in _ordered_models(metrics_by_model.keys()):
                aggregates = metrics_by_model.get(model, {})
                for metric, aggregate in aggregates.items():
                    writer.writerow(
                        [model, metric, aggregate.mean, aggregate.std, aggregate.n]
                    )

    primary_metrics_path = output_dir / "exp1_t2_primary.csv"
    _write_model_metrics(primary_metrics_path, summary.primary_metrics)
    created.append(primary_metrics_path)

    sensitivity_metrics_path = output_dir / "exp1_t2_sensitivity.csv"
    _write_model_metrics(sensitivity_metrics_path, summary.sensitivity_metrics)
    created.append(sensitivity_metrics_path)

    def _write_delta(path: Path, deltas: Mapping[str, DeltaSummary]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            seed_headers = [f"seed_{seed}" for seed in summary.seed_validation.expected_seeds]
            writer.writerow([
                "metric",
                "mean_delta",
                "ci_lower",
                "ci_upper",
                "samples",
                "std",
                *seed_headers,
            ])
            for metric in PRIMARY_METRICS:
                summary_delta = deltas.get(metric)
                if not summary_delta:
                    continue
                std_value = _delta_std(summary_delta.per_seed)
                row = [
                    metric,
                    summary_delta.mean,
                    summary_delta.ci_lower,
                    summary_delta.ci_upper,
                    len(summary_delta.samples),
                    std_value,
                ]
                for seed in summary.seed_validation.expected_seeds:
                    row.append(summary_delta.per_seed.get(seed))
                writer.writerow(row)

    primary_delta_path = output_dir / "exp1_t3_primary.csv"
    _write_delta(primary_delta_path, summary.primary_deltas)
    created.append(primary_delta_path)

    sensitivity_delta_path = output_dir / "exp1_t3_sensitivity.csv"
    _write_delta(sensitivity_delta_path, summary.sensitivity_deltas)
    created.append(sensitivity_delta_path)
    return created


def build_manifest(
    summary: Exp1Summary,
    *,
    loader: ResultLoader,
    manifest_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    extra_outputs: Optional[Iterable[Path]] = None,
    rng_seed: Optional[int] = None,
    bootstrap: Optional[int] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "experiment": "exp1",
        "summary": summary.as_dict(),
    }
    manifest = build_report_manifest(
        output_path=output_path,
        loader=loader,
        runs=loader.loaded_runs,
        rng_seed=rng_seed,
        bootstrap=bootstrap,
        extra_outputs=extra_outputs,
        metadata=metadata,
        validated_seeds=summary.seed_validation.expected_seeds,
        seed_groups=summary.seed_validation.observed_seeds,
    )
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "PRIMARY_METRICS",
    "EXPECTED_MODELS",
    "Exp1Run",
    "Exp1Summary",
    "discover_runs",
    "load_run",
    "summarize_runs",
    "render_markdown",
    "write_csv_tables",
    "build_manifest",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_RNG_SEED",
]
