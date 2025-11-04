from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import numpy as np

from .common_loader import resolve_outputs_csv
# Local bootstrap utilities for paired deltas.
from .bootstrapper import BootstrapDeltaResult, Bootstrapper

# Metrics used for retention/drop calculations and AUSC summaries.
RETENTION_METRICS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "loss",
)


# Metrics used when reporting retention deltas.
PRIMARY_RETENTION_METRICS: Tuple[str, ...] = (
    "f1",
    "recall",
)


EXPECTED_PRIMARY_TAU_POLICY = "sun_val_frozen"


REQUIRED_SEEDS: Tuple[int, ...] = (13, 29, 47)


@dataclass(frozen=True)
class PerturbationDefinition:
    """Canonical definition of a perturbation severity for Experiment 5B."""

    canonical_tag: str
    raw_severity: float
    display: str


EXPERIMENT5B_REQUIRED_PERTURBATIONS: Dict[str, Tuple[PerturbationDefinition, ...]] = {
    "blur": (
        PerturbationDefinition("blur_sigma_0p5", 0.5, "σ=0.5"),
        PerturbationDefinition("blur_sigma_1p0", 1.0, "σ=1.0"),
        PerturbationDefinition("blur_sigma_1p5", 1.5, "σ=1.5"),
        PerturbationDefinition("blur_sigma_2p0", 2.0, "σ=2.0"),
    ),
    "jpeg": (
        PerturbationDefinition("jpeg_q_95", 5.0, "q=95"),
        PerturbationDefinition("jpeg_q_75", 25.0, "q=75"),
        PerturbationDefinition("jpeg_q_50", 50.0, "q=50"),
        PerturbationDefinition("jpeg_q_30", 70.0, "q=30"),
    ),
    "brightness_contrast": (
        PerturbationDefinition("bc_b0p8_c0p8", 0.2, "brightness=0.8, contrast=0.8"),
        PerturbationDefinition("bc_b0p6_c0p6", 0.4, "brightness=0.6, contrast=0.6"),
    ),
    "occlusion": (
        PerturbationDefinition("occ_a5p", 0.05, "area=5%"),
        PerturbationDefinition("occ_a10p", 0.10, "area=10%"),
        PerturbationDefinition("occ_a20p", 0.20, "area=20%"),
    ),
}


PERTURBATION_SEVERITY_TOLERANCE = 1e-6


INTEGER_METRIC_KEYS: Tuple[str, ...] = (
    "tp",
    "tn",
    "fp",
    "fn",
    "n_pos",
    "n_neg",
    "n_total",
    "count",
)


class TauPolicyError(RuntimeError):
    """Raised when Experiment 5B metrics report an unexpected τ policy."""


@dataclass(frozen=True)
class TagInfo:
    """Metadata describing how a perturbation tag maps to a severity family."""

    family: str
    raw_severity: float
    normalized_severity: float = 0.0


@dataclass
class RunPerturbationResult:
    """Structured record representing a single run's perturbation metrics."""

    model: str
    seed: int
    tau: Optional[float]
    tau_policy: Optional[str]
    metrics: Dict[str, float]
    perturbations: Dict[str, Dict[str, float]]
    case_metrics: Dict[str, Dict[str, Dict[str, Any]]]
    provenance: Dict[str, object]
    path: Path


@dataclass
class FamilyBootstrapSeedData:
    """Prepared per-seed case data for corruption family bootstrapping."""

    seed: int
    case_ids: Tuple[str, ...]
    cluster_ids: Tuple[str, ...]
    severity_tags: Tuple[str, ...]
    severity_levels: Tuple[float, ...]
    target_arrays: Dict[str, np.ndarray]
    baseline_arrays: Dict[str, np.ndarray]


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


def _coerce_int_like(value: object) -> Optional[int]:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    rounded = int(round(float(numeric)))
    if not math.isfinite(rounded):
        return None
    return rounded


def _coerce_string(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    text = str(value).strip()
    return text or None


def _parse_fixed_point(value: str) -> float:
    text = value.strip().lower().replace("p", ".")
    try:
        numeric = float(text)
    except ValueError:
        return float("nan")
    return numeric


def _parse_tag(tag: str) -> TagInfo:
    label = str(tag).strip()
    if not label:
        return TagInfo(family="unknown", raw_severity=float("nan"))
    if label == "clean":
        return TagInfo(family="baseline", raw_severity=0.0, normalized_severity=0.0)
    if label.lower() == "all-perturbed":
        return TagInfo(family="aggregate", raw_severity=float("nan"), normalized_severity=float("nan"))
    if label.startswith("blur_sigma_"):
        suffix = label.split("blur_sigma_", 1)[1]
        level = _parse_fixed_point(suffix)
        return TagInfo(family="blur", raw_severity=level)
    if label.startswith("jpeg_q_"):
        suffix = label.split("jpeg_q_", 1)[1]
        quality = _coerce_float(suffix)
        if quality is None:
            return TagInfo(family="jpeg", raw_severity=float("nan"))
        # Higher severity for lower quality; invert around 100.
        return TagInfo(family="jpeg", raw_severity=100.0 - quality)
    if label.startswith("brightness_"):
        factor = _parse_fixed_point(label.split("brightness_", 1)[1])
        if math.isnan(factor):
            severity = float("nan")
        else:
            severity = max(0.0, 1.0 - factor)
        return TagInfo(family="brightness", raw_severity=severity)
    if label.startswith("contrast_"):
        factor = _parse_fixed_point(label.split("contrast_", 1)[1])
        if math.isnan(factor):
            severity = float("nan")
        else:
            severity = max(0.0, 1.0 - factor)
        return TagInfo(family="contrast", raw_severity=severity)
    if label.startswith("bc_b") and "_c" in label:
        # brightness/contrast e.g. bc_b0p8_c0p6
        try:
            brightness_token, contrast_token = label.split("_c", 1)
            brightness = _parse_fixed_point(brightness_token.split("bc_b", 1)[1])
            contrast = _parse_fixed_point(contrast_token)
            if math.isnan(brightness) or math.isnan(contrast):
                level = float("nan")
            else:
                # Treat severity as the deficit from unity of the weaker factor.
                deficit = 1.0 - min(brightness, contrast)
                level = max(0.0, deficit)
        except (IndexError, ValueError):
            level = float("nan")
        return TagInfo(family="brightness_contrast", raw_severity=level)
    if label.startswith("occ_a"):
        suffix = label.split("occ_a", 1)[1]
        area = _parse_fixed_point(suffix)
        if math.isfinite(area) and area > 1.0:
            area = area / 100.0
        return TagInfo(family="occlusion", raw_severity=area)
    return TagInfo(family=label, raw_severity=float("nan"), normalized_severity=float("nan"))


def _convert_per_seed_map(
    data: Mapping[str, Mapping[int, float]]
) -> Dict[str, Dict[int, float]]:
    output: Dict[str, Dict[int, float]] = {}
    for metric, per_seed in data.items():
        if not isinstance(per_seed, Mapping):
            continue
        metric_map: Dict[int, float] = {}
        for seed, value in per_seed.items():
            if value is None or not math.isfinite(float(value)):
                continue
            metric_map[int(seed)] = float(value)
        if metric_map:
            output[str(metric)] = metric_map
    return output


def _convert_case_per_seed_map(
    data: Mapping[str, Mapping[str, Mapping[int, Mapping[str, float]]]]
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    output: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for tag, metric_map in data.items():
        if not isinstance(metric_map, Mapping):
            continue
        metric_output: Dict[str, Dict[int, Dict[str, float]]] = {}
        for metric, per_seed in metric_map.items():
            if not isinstance(per_seed, Mapping):
                continue
            seed_output: Dict[int, Dict[str, float]] = {}
            for seed, case_map in per_seed.items():
                if not isinstance(case_map, Mapping):
                    continue
                sanitized: Dict[str, float] = {}
                for case_id, value in case_map.items():
                    numeric = _coerce_float(value)
                    if numeric is None or not math.isfinite(numeric):
                        continue
                    sanitized[str(case_id)] = float(numeric)
                if sanitized:
                    seed_output[int(seed)] = sanitized
            if seed_output:
                metric_output[str(metric)] = seed_output
        if metric_output:
            output[str(tag)] = metric_output
    return output


def _convert_case_cluster_map(
    data: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> Dict[str, Dict[int, Dict[str, str]]]:
    output: Dict[str, Dict[int, Dict[str, str]]] = {}
    for tag, per_seed in data.items():
        if not isinstance(per_seed, Mapping):
            continue
        seed_output: Dict[int, Dict[str, str]] = {}
        for seed, case_map in per_seed.items():
            if not isinstance(case_map, Mapping):
                continue
            sanitized: Dict[str, str] = {}
            for case_id, cluster in case_map.items():
                if cluster is None:
                    continue
                sanitized[str(case_id)] = str(cluster)
            if sanitized:
                seed_output[int(seed)] = sanitized
        if seed_output:
            output[str(tag)] = seed_output
    return output


def _sanitize_metric_mapping(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        numeric = _coerce_float(value)
        if numeric is not None and math.isfinite(numeric):
            sanitized[str(key)] = float(numeric)
        else:
            sanitized[str(key)] = value
    return sanitized


def _parse_case_metrics_block(
    block: Optional[object],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not isinstance(block, Mapping):
        return {}
    output: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tag, tag_block in block.items():
        cases: Dict[str, Dict[str, Any]] = {}
        if isinstance(tag_block, Mapping):
            for case_id, metrics in tag_block.items():
                if not isinstance(metrics, Mapping):
                    continue
                sanitized = _sanitize_metric_mapping(metrics)
                if sanitized:
                    cases[str(case_id)] = sanitized
        elif isinstance(tag_block, Sequence) and not isinstance(tag_block, (str, bytes)):
            for entry in tag_block:
                if not isinstance(entry, Mapping):
                    continue
                case_id = entry.get("case_id") or entry.get("case") or entry.get("id")
                metrics_source = entry.get("metrics")
                extra_fields: Dict[str, Any] = {}
                if not isinstance(metrics_source, Mapping):
                    metrics_source = entry
                else:
                    extra_fields = {str(key): value for key, value in entry.items() if key != "metrics"}
                sanitized = _sanitize_metric_mapping(metrics_source)
                if extra_fields:
                    sanitized.update(extra_fields)
                if case_id and sanitized:
                    cases[str(case_id)] = sanitized
        if cases:
            output[str(tag)] = cases
    return output


def _build_tag_catalog(runs: Mapping[str, Mapping[int, RunPerturbationResult]]) -> Dict[str, TagInfo]:
    catalog: Dict[str, TagInfo] = {}
    families: DefaultDict[str, List[float]] = defaultdict(list)
    for run_map in runs.values():
        for run in run_map.values():
            for tag in run.perturbations:
                if tag not in catalog:
                    catalog[tag] = _parse_tag(tag)
                info = catalog[tag]
                if info.family not in {"baseline", "aggregate", "unknown"} and not math.isnan(info.raw_severity):
                    families[info.family].append(info.raw_severity)
    # Ensure clean is available even if absent from perturbations map.
    if "clean" not in catalog:
        catalog["clean"] = TagInfo(family="baseline", raw_severity=0.0, normalized_severity=0.0)
    for tag, info in list(catalog.items()):
        if info.family in {"baseline", "aggregate", "unknown"}:
            continue
        levels = families.get(info.family)
        if not levels:
            catalog[tag] = TagInfo(family=info.family, raw_severity=info.raw_severity, normalized_severity=float("nan"))
            continue
        min_level = min(levels)
        max_level = max(levels)
        if math.isnan(info.raw_severity):
            normalized = float("nan")
        elif max_level > min_level:
            normalized = (info.raw_severity - min_level) / (max_level - min_level)
        else:
            normalized = 1.0
        catalog[tag] = TagInfo(
            family=info.family,
            raw_severity=info.raw_severity,
            normalized_severity=max(0.0, normalized),
        )
    return catalog


def _validate_required_perturbations(tag_catalog: Mapping[str, TagInfo]) -> None:
    missing_by_family: Dict[str, List[PerturbationDefinition]] = {}
    observed_tags: DefaultDict[str, List[str]] = defaultdict(list)
    observed_levels: DefaultDict[str, List[float]] = defaultdict(list)

    for tag, info in tag_catalog.items():
        family = info.family
        if family in {"baseline", "aggregate", "unknown"}:
            continue
        observed_tags[family].append(tag)
        if info.raw_severity is not None and math.isfinite(float(info.raw_severity)):
            observed_levels[family].append(float(info.raw_severity))

    for family, requirements in EXPERIMENT5B_REQUIRED_PERTURBATIONS.items():
        present_levels = observed_levels.get(family, [])
        for definition in requirements:
            if any(abs(level - definition.raw_severity) <= PERTURBATION_SEVERITY_TOLERANCE for level in present_levels):
                continue
            missing_by_family.setdefault(family, []).append(definition)

    if not missing_by_family:
        return

    issues: List[str] = []
    for family in sorted(missing_by_family.keys()):
        missing_descriptions = ", ".join(
            f"{definition.display} (expected tag '{definition.canonical_tag}')"
            for definition in missing_by_family[family]
        )
        observed = ", ".join(sorted(observed_tags.get(family, []))) or "none"
        issues.append(
            f"{family}: missing {missing_descriptions}; observed tags: {observed}"
        )

    raise ValueError(
        "Experiment 5B metrics are missing required perturbation severities: " + "; ".join(issues)
    )


def _resolve_outputs_path(metrics_path: Path) -> Path:
    return resolve_outputs_csv(metrics_path)


def load_run(metrics_path: Path) -> RunPerturbationResult:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance_raw = payload.get("provenance") or {}
    provenance: Dict[str, object] = dict(provenance_raw) if isinstance(provenance_raw, Mapping) else {}
    model_name = str(provenance.get("model") or metrics_path.stem.split("_", 1)[0])
    tau_policy = _coerce_string(provenance.get("tau_policy"))
    if not tau_policy:
        thresholds_block = payload.get("thresholds")
        if isinstance(thresholds_block, Mapping):
            primary_block = thresholds_block.get("primary")
            if isinstance(primary_block, Mapping):
                tau_policy = _coerce_string(primary_block.get("policy"))
    expected_tau_policy = _coerce_string(EXPECTED_PRIMARY_TAU_POLICY)
    if expected_tau_policy:
        if not tau_policy:
            raise TauPolicyError(
                f"Metrics file '{metrics_path}' is missing tau_policy; expected '{expected_tau_policy}'."
            )
        if tau_policy != expected_tau_policy:
            raise TauPolicyError(
                "Metrics file '{}' reports tau_policy '{}' but expected '{}'.".format(
                    metrics_path, tau_policy, expected_tau_policy
                )
            )
    seed_value = _coerce_int(payload.get("seed"))
    if seed_value is None:
        seed_value = _coerce_int(provenance.get("train_seed"))
    if seed_value is None:
        raise ValueError(f"Metrics file '{metrics_path}' does not specify a seed")
    test_primary_raw = payload.get("test_primary") or {}
    metrics: Dict[str, float] = {}
    for key, value in test_primary_raw.items():
        numeric = _coerce_float(value)
        if numeric is not None:
            metrics[key] = numeric
    tau_value = metrics.get("tau")
    pert_block = payload.get("test_perturbations") or {}
    per_tag_raw = pert_block.get("per_tag") if isinstance(pert_block, Mapping) else None
    if not isinstance(per_tag_raw, Mapping) or not per_tag_raw:
        raise ValueError(f"Metrics file '{metrics_path}' does not contain test_perturbations.per_tag")
    perturbations: Dict[str, Dict[str, float]] = {}
    case_metrics = _parse_case_metrics_block(pert_block.get("per_case"))
    if not case_metrics:
        case_metrics = _parse_case_metrics_block(pert_block.get("per_case_metrics"))
    for tag, stats in per_tag_raw.items():
        if not isinstance(stats, Mapping):
            continue
        sanitized: Dict[str, float] = {}
        for key, value in stats.items():
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            sanitized[str(key)] = numeric
        if sanitized:
            perturbations[str(tag)] = sanitized
    if not perturbations:
        raise ValueError(f"Metrics file '{metrics_path}' does not provide numeric perturbation metrics")
    outputs_path = _resolve_outputs_path(metrics_path)
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing test outputs CSV: {outputs_path}")
    return RunPerturbationResult(
        model=model_name,
        seed=int(seed_value),
        tau=tau_value,
        tau_policy=tau_policy,
        metrics=metrics,
        perturbations=perturbations,
        case_metrics=case_metrics,
        provenance=provenance,
        path=metrics_path,
    )


def discover_runs(
    root: Path,
    *,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[int, RunPerturbationResult]]:
    model_filter = {str(m) for m in models} if models else None
    runs: DefaultDict[str, Dict[int, RunPerturbationResult]] = defaultdict(dict)
    skip_tokens = {"_best", "_intermediate", "_ema"}
    for metrics_path in sorted(root.rglob("*.metrics.json")):
        stem = metrics_path.stem
        if any(token in stem for token in skip_tokens):
            continue
        try:
            run = load_run(metrics_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load metrics from {metrics_path} (missing per-frame outputs)"
            ) from exc
        except TauPolicyError:
            raise
        except (OSError, ValueError):
            continue
        if model_filter and run.model not in model_filter:
            continue
        runs[run.model][run.seed] = run
    return {model: dict(seed_map) for model, seed_map in runs.items()}


def _compute_stats(values: Sequence[float]) -> Dict[str, float]:
    numeric = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not numeric:
        return {}
    mean = float(np.mean(numeric))
    std = float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(numeric)}


def _deduplicate_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda item: (item[0], item[1]))
    deduplicated: List[Tuple[float, float]] = []
    last_severity: Optional[float] = None
    for severity, value in ordered:
        if last_severity is not None and abs(severity - last_severity) < 1e-12:
            deduplicated[-1] = (severity, value)
        else:
            deduplicated.append((severity, value))
            last_severity = severity
    return deduplicated


def _normalised_trapz(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    width = float(xs[-1] - xs[0])
    if width <= 0.0:
        return None
    area = float(np.trapz(ys, xs))
    return area / width


def _safe_numeric(value: Any, default: float = float("inf")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def _flatten_stats(row: MutableMapping[str, object], prefix: str, stats_map: Mapping[str, Any]) -> None:
    for metric, stats in stats_map.items():
        if not isinstance(stats, Mapping):
            continue
        mean = stats.get("mean")
        std = stats.get("std")
        count = stats.get("n")
        row[f"{prefix}{metric}_mean"] = mean
        row[f"{prefix}{metric}_std"] = std
        row[f"{prefix}{metric}_n"] = int(count) if count is not None else None


def _get_retention_per_seed_map(
    model_entry: Mapping[str, Any], family: str, metric: str
) -> Dict[int, float]:
    if family == "macro":
        macro_per_seed = model_entry.get("ausc_macro_retention_per_seed")
        if isinstance(macro_per_seed, Mapping):
            metric_map = macro_per_seed.get(metric)
            if isinstance(metric_map, Mapping):
                return {int(seed): float(value) for seed, value in metric_map.items()}
        return {}
    retention_per_seed = model_entry.get("ausc_retention_per_seed")
    if not isinstance(retention_per_seed, Mapping):
        return {}
    family_block = retention_per_seed.get(family)
    if not isinstance(family_block, Mapping):
        return {}
    metric_block = family_block.get(metric)
    if not isinstance(metric_block, Mapping):
        return {}
    return {int(seed): float(value) for seed, value in metric_block.items()}


def _get_retention_per_seed_for_tag(
    model_entry: Mapping[str, Any], tag: str, metric: str
) -> Dict[int, float]:
    per_tag = model_entry.get("per_tag")
    if not isinstance(per_tag, Mapping):
        return {}
    tag_entry = per_tag.get(tag)
    if not isinstance(tag_entry, Mapping):
        return {}
    retention_block = tag_entry.get("retention_per_seed")
    if not isinstance(retention_block, Mapping):
        return {}
    metric_block = retention_block.get(metric)
    if not isinstance(metric_block, Mapping):
        return {}
    return {int(seed): float(value) for seed, value in metric_block.items()}


def _normalise_cluster_identifier(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return None
    numeric = _coerce_float(value)
    if numeric is not None and math.isfinite(numeric):
        if float(numeric).is_integer():
            return str(int(round(float(numeric))))
        return str(float(numeric))
    return str(value)


def _extract_cluster_value(case_stats: Mapping[str, Any], cluster_key: str) -> Optional[str]:
    if not cluster_key:
        return None
    if not isinstance(case_stats, Mapping):
        return None
    direct_value = _normalise_cluster_identifier(case_stats.get(cluster_key))
    if direct_value is not None:
        return direct_value
    metadata = case_stats.get("metadata")
    if isinstance(metadata, Mapping):
        meta_value = _normalise_cluster_identifier(metadata.get(cluster_key))
        if meta_value is not None:
            return meta_value
    clusters = case_stats.get("clusters")
    if isinstance(clusters, Mapping):
        cluster_value = _normalise_cluster_identifier(clusters.get(cluster_key))
        if cluster_value is not None:
            return cluster_value
    return None


def _get_case_retention_per_seed(
    model_entry: Mapping[str, Any], tag: str, metric: str
) -> Dict[int, Dict[str, float]]:
    per_case = model_entry.get("per_case_retention_per_seed")
    if not isinstance(per_case, Mapping):
        return {}
    tag_block = per_case.get(tag)
    if not isinstance(tag_block, Mapping):
        return {}
    metric_block = tag_block.get(metric)
    if not isinstance(metric_block, Mapping):
        return {}
    output: Dict[int, Dict[str, float]] = {}
    for seed, case_map in metric_block.items():
        if not isinstance(case_map, Mapping):
            continue
        sanitized: Dict[str, float] = {}
        for case_id, value in case_map.items():
            numeric = _coerce_float(value)
            if numeric is None or not math.isfinite(numeric):
                continue
            sanitized[str(case_id)] = float(numeric)
        if sanitized:
            output[int(seed)] = sanitized
    return output


def _get_case_clusters_per_seed(
    model_entry: Mapping[str, Any], tag: str
) -> Dict[int, Dict[str, str]]:
    per_case = model_entry.get("per_case_clusters_per_seed")
    if not isinstance(per_case, Mapping):
        return {}
    tag_block = per_case.get(tag)
    if not isinstance(tag_block, Mapping):
        return {}
    output: Dict[int, Dict[str, str]] = {}
    for seed, case_map in tag_block.items():
        if not isinstance(case_map, Mapping):
            continue
        sanitized: Dict[str, str] = {}
        for case_id, cluster in case_map.items():
            if cluster is None:
                continue
            sanitized[str(case_id)] = str(cluster)
        if sanitized:
            output[int(seed)] = sanitized
    return output


def _derive_rng(rng_seed: int, *components: object) -> np.random.Generator:
    hasher = hashlib.sha1()
    hasher.update(str(int(rng_seed)).encode("utf-8", errors="ignore"))
    for component in components:
        hasher.update(str(component).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    seed_bytes = hasher.digest()[:8]
    seed_int = int.from_bytes(seed_bytes, "little", signed=False)
    if seed_int == 0:
        seed_int = 1
    return np.random.default_rng(seed_int)


def _bootstrap_tag_delta(
    target_entry: Mapping[str, Any],
    baseline_entry: Mapping[str, Any],
    tag: str,
    metric: str,
    *,
    bootstrap: int,
    rng_seed: int,
    cluster_key: str,
) -> Optional[BootstrapDeltaResult]:
    if bootstrap <= 0:
        return None
    target_cases = _get_case_retention_per_seed(target_entry, tag, metric)
    baseline_cases = _get_case_retention_per_seed(baseline_entry, tag, metric)
    target_clusters = _get_case_clusters_per_seed(target_entry, tag)
    baseline_clusters = _get_case_clusters_per_seed(baseline_entry, tag)
    if not target_cases or not baseline_cases:
        return None
    seeds = sorted(set(target_cases.keys()) & set(baseline_cases.keys()))
    if not seeds:
        return None
    metrics_payload: Dict[str, Dict[int, Tuple[float, ...]]] = {"target": {}, "baseline": {}}
    clusters_payload: Dict[str, Dict[int, Tuple[str, ...]]] = {"target": {}, "baseline": {}}
    for seed in seeds:
        target_map = target_cases.get(seed)
        baseline_map = baseline_cases.get(seed)
        if not target_map or not baseline_map:
            continue
        shared_cases = [case for case in target_map.keys() if case in baseline_map]
        if not shared_cases:
            continue
        target_values = [float(target_map[case]) for case in shared_cases]
        baseline_values = [float(baseline_map[case]) for case in shared_cases]
        if not target_values or not baseline_values:
            continue
        metrics_payload["target"][seed] = tuple(target_values)
        metrics_payload["baseline"][seed] = tuple(baseline_values)
        if cluster_key:
            target_seed_clusters = target_clusters.get(seed, {})
            baseline_seed_clusters = baseline_clusters.get(seed, {})
            if not isinstance(target_seed_clusters, Mapping):
                target_seed_clusters = {}
            if not isinstance(baseline_seed_clusters, Mapping):
                baseline_seed_clusters = {}
            cluster_sequence: List[str] = []
            for case in shared_cases:
                cluster_value: Optional[str] = None
                if case in target_seed_clusters:
                    cluster_value = target_seed_clusters[case]
                elif case in baseline_seed_clusters:
                    cluster_value = baseline_seed_clusters[case]
                if cluster_value is None:
                    cluster_value = case
                cluster_sequence.append(str(cluster_value))
            if cluster_sequence:
                clusters_payload["target"][seed] = tuple(cluster_sequence)
                clusters_payload["baseline"][seed] = tuple(cluster_sequence)
    if not metrics_payload["target"] or not metrics_payload["baseline"]:
        return None
    rng = _derive_rng(rng_seed, "tag", tag, metric)
    metrics_mapping = cast(
        Mapping[Hashable, Mapping[int, Sequence[float]]],
        metrics_payload,
    )
    clusters_mapping = cast(
        Mapping[Hashable, Mapping[int, Sequence[Hashable]]],
        clusters_payload,
    )
    bootstrapper = Bootstrapper(metrics_mapping, clusters=clusters_mapping, rng=rng)
    return bootstrapper.paired_delta("target", "baseline", bootstrap=bootstrap, ci=0.95)


def _collect_family_seed_data(
    target_entry: Mapping[str, Any],
    baseline_entry: Mapping[str, Any],
    family: str,
    metric: str,
    tag_catalog: Mapping[str, TagInfo],
    *,
    cluster_key: str,
) -> List[FamilyBootstrapSeedData]:
    tag_data: Dict[
        str,
        Tuple[
            Dict[int, Dict[str, float]],
            Dict[int, Dict[str, float]],
            Dict[int, Dict[str, str]],
            Dict[int, Dict[str, str]],
            float,
        ],
    ] = {}
    for tag, info in tag_catalog.items():
        if info.family != family or tag == "clean":
            continue
        severity = info.normalized_severity
        if severity is None or not math.isfinite(severity):
            continue
        target_cases = _get_case_retention_per_seed(target_entry, tag, metric)
        baseline_cases = _get_case_retention_per_seed(baseline_entry, tag, metric)
        if not target_cases or not baseline_cases:
            continue
        target_clusters = _get_case_clusters_per_seed(target_entry, tag)
        baseline_clusters = _get_case_clusters_per_seed(baseline_entry, tag)
        shared_seeds = set(target_cases.keys()) & set(baseline_cases.keys())
        if not shared_seeds:
            continue
        tag_data[tag] = (
            target_cases,
            baseline_cases,
            target_clusters,
            baseline_clusters,
            float(severity),
        )
    if not tag_data:
        return []
    shared_seed_ids: Optional[Set[int]] = None
    for target_cases, baseline_cases, _, _, _ in tag_data.values():
        available = set(target_cases.keys()) & set(baseline_cases.keys())
        if shared_seed_ids is None:
            shared_seed_ids = set(int(seed) for seed in available)
        else:
            shared_seed_ids &= {int(seed) for seed in available}
    if not shared_seed_ids:
        return []
    ordered_tags = sorted(tag_data.items(), key=lambda item: item[1][2])
    seed_data: List[FamilyBootstrapSeedData] = []
    for seed in sorted(shared_seed_ids):
        case_candidates: Optional[Set[str]] = None
        for tag, (target_cases, baseline_cases, _, _, _) in ordered_tags:
            target_map_raw = target_cases.get(seed)
            baseline_map_raw = baseline_cases.get(seed)
            if not target_map_raw or not baseline_map_raw:
                case_candidates = set()
                break
            target_map = cast(Dict[str, float], target_map_raw)
            baseline_map = cast(Dict[str, float], baseline_map_raw)
            shared_case_ids: Set[str] = set(target_map.keys()) & set(baseline_map.keys())
            if not shared_case_ids:
                case_candidates = set()
                break
            if case_candidates is None:
                case_candidates = set(shared_case_ids)
            else:
                case_candidates &= shared_case_ids
        if not case_candidates:
            continue
        case_ids = tuple(sorted(case_candidates))
        if not case_ids:
            continue
        target_arrays: Dict[str, np.ndarray] = {}
        baseline_arrays: Dict[str, np.ndarray] = {}
        severity_tags: List[str] = []
        severity_levels: List[float] = []
        cluster_assignments: Dict[str, str] = {}
        skip_seed = False
        for tag, (
            target_cases,
            baseline_cases,
            target_clusters,
            baseline_clusters,
            severity,
        ) in ordered_tags:
            target_map = target_cases.get(seed)
            baseline_map = baseline_cases.get(seed)
            if not target_map or not baseline_map:
                skip_seed = True
                break
            target_cluster_map = target_clusters.get(seed, {}) if cluster_key else {}
            baseline_cluster_map = baseline_clusters.get(seed, {}) if cluster_key else {}
            target_values = np.array([float(target_map[cid]) for cid in case_ids], dtype=float)
            baseline_values = np.array([float(baseline_map[cid]) for cid in case_ids], dtype=float)
            if target_values.size == 0 or baseline_values.size == 0:
                skip_seed = True
                break
            if cluster_key:
                for case_id in case_ids:
                    if case_id in cluster_assignments:
                        continue
                    cluster_value: Optional[str] = None
                    if case_id in target_cluster_map:
                        cluster_value = target_cluster_map[case_id]
                    elif case_id in baseline_cluster_map:
                        cluster_value = baseline_cluster_map[case_id]
                    if cluster_value is None and cluster_key == "case_id":
                        cluster_value = case_id
                    if cluster_value is None:
                        cluster_value = case_id
                    cluster_assignments[case_id] = str(cluster_value)
            target_arrays[tag] = target_values
            baseline_arrays[tag] = baseline_values
            severity_tags.append(tag)
            severity_levels.append(float(severity))
        if skip_seed or not severity_tags:
            continue
        if not cluster_assignments and cluster_key:
            cluster_assignments = {case_id: case_id for case_id in case_ids}
        cluster_ids: Tuple[str, ...]
        if cluster_key:
            cluster_ids = tuple(cluster_assignments.get(case_id, case_id) for case_id in case_ids)
        else:
            cluster_ids = case_ids
        seed_data.append(
            FamilyBootstrapSeedData(
                seed=int(seed),
                case_ids=case_ids,
                cluster_ids=cluster_ids,
                severity_tags=tuple(severity_tags),
                severity_levels=tuple(severity_levels),
                target_arrays=target_arrays,
                baseline_arrays=baseline_arrays,
            )
        )
    return seed_data


def _bootstrap_family_delta(
    target_entry: Mapping[str, Any],
    baseline_entry: Mapping[str, Any],
    family: str,
    metric: str,
    *,
    tag_catalog: Mapping[str, TagInfo],
    bootstrap: int,
    rng_seed: int,
    severity_lockstep: bool,
    cluster_key: str,
) -> Optional[BootstrapDeltaResult]:
    seed_data = _collect_family_seed_data(
        target_entry,
        baseline_entry,
        family,
        metric,
        tag_catalog,
        cluster_key=cluster_key,
    )
    if not seed_data:
        return None
    per_seed_delta: Dict[int, float] = {}
    for record in seed_data:
        aggregated_target: List[Tuple[float, float]] = []
        aggregated_baseline: List[Tuple[float, float]] = []
        for tag, severity in zip(record.severity_tags, record.severity_levels):
            if not math.isfinite(severity):
                continue
            target_mean = float(np.mean(record.target_arrays[tag]))
            baseline_mean = float(np.mean(record.baseline_arrays[tag]))
            aggregated_target.append((severity, target_mean))
            aggregated_baseline.append((severity, baseline_mean))
        if not aggregated_target or not aggregated_baseline:
            continue
        aggregated_target.sort(key=lambda item: item[0])
        aggregated_baseline.sort(key=lambda item: item[0])
        target_series = _deduplicate_points([(0.0, 1.0)] + aggregated_target)
        baseline_series = _deduplicate_points([(0.0, 1.0)] + aggregated_baseline)
        target_ausc = _normalised_trapz(target_series)
        baseline_ausc = _normalised_trapz(baseline_series)
        if (
            target_ausc is None
            or baseline_ausc is None
            or not math.isfinite(target_ausc)
            or not math.isfinite(baseline_ausc)
        ):
            continue
        per_seed_delta[int(record.seed)] = float(target_ausc - baseline_ausc)
    if not per_seed_delta:
        return None
    mean_delta = float(np.mean(list(per_seed_delta.values())))
    samples: List[float] = []
    if bootstrap > 0:
        rng = _derive_rng(rng_seed, "family", family, metric)
        for _ in range(bootstrap):
            seed_deltas: List[float] = []
            for record in seed_data:
                n_cases = len(record.case_ids)
                if n_cases == 0:
                    continue
                clusters = tuple(dict.fromkeys(record.cluster_ids))
                cluster_to_indices = {
                    cluster: np.array(
                        [idx for idx, value in enumerate(record.cluster_ids) if value == cluster],
                        dtype=int,
                    )
                    for cluster in clusters
                }
                if clusters:
                    sampled_cluster_indices = rng.choice(
                        len(clusters), size=len(clusters), replace=True
                    )
                    base_indices = np.concatenate(
                        [cluster_to_indices[clusters[index]] for index in sampled_cluster_indices]
                    )
                else:
                    base_indices = np.arange(n_cases, dtype=int)
                resampled_target: List[Tuple[float, float]] = []
                resampled_baseline: List[Tuple[float, float]] = []
                for tag, severity in zip(record.severity_tags, record.severity_levels):
                    if not math.isfinite(severity):
                        continue
                    if severity_lockstep:
                        sample_indices = base_indices
                    else:
                        if clusters:
                            sampled_cluster_indices = rng.choice(
                                len(clusters), size=len(clusters), replace=True
                            )
                            sample_indices = np.concatenate(
                                [
                                    cluster_to_indices[clusters[index]]
                                    for index in sampled_cluster_indices
                                ]
                            )
                        else:
                            sample_indices = rng.choice(n_cases, size=n_cases, replace=True)
                    if sample_indices.size == 0:
                        continue
                    target_values = record.target_arrays[tag][sample_indices]
                    baseline_values = record.baseline_arrays[tag][sample_indices]
                    resampled_target.append((severity, float(np.mean(target_values))))
                    resampled_baseline.append((severity, float(np.mean(baseline_values))))
                if not resampled_target or not resampled_baseline:
                    continue
                resampled_target.sort(key=lambda item: item[0])
                resampled_baseline.sort(key=lambda item: item[0])
                target_series = _deduplicate_points([(0.0, 1.0)] + resampled_target)
                baseline_series = _deduplicate_points([(0.0, 1.0)] + resampled_baseline)
                target_ausc = _normalised_trapz(target_series)
                baseline_ausc = _normalised_trapz(baseline_series)
                if (
                    target_ausc is None
                    or baseline_ausc is None
                    or not math.isfinite(target_ausc)
                    or not math.isfinite(baseline_ausc)
                ):
                    continue
                seed_deltas.append(float(target_ausc - baseline_ausc))
            if seed_deltas:
                samples.append(float(np.mean(seed_deltas)))
    if samples:
        lower_pct = 2.5
        upper_pct = 97.5
        ci_lower = float(np.percentile(samples, lower_pct))
        ci_upper = float(np.percentile(samples, upper_pct))
    else:
        ci_lower = None
        ci_upper = None
    return BootstrapDeltaResult(
        mean=mean_delta,
        per_seed=per_seed_delta,
        samples=tuple(samples),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def _build_t1_table(models_summary: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, entry in sorted(models_summary.items()):
        per_tag = entry.get("per_tag") if isinstance(entry, Mapping) else None
        clean_entry = per_tag.get("clean") if isinstance(per_tag, Mapping) else None
        if not isinstance(clean_entry, Mapping):
            continue
        row: Dict[str, object] = {"model": model, "family": "baseline", "tag": "clean"}
        _flatten_stats(row, "", clean_entry.get("metrics", {}))
        rows.append(row)
    return rows


def _build_t2_tables(
    severity_rows: Sequence[Mapping[str, object]]
) -> Dict[str, List[Dict[str, object]]]:
    family_tables: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in severity_rows:
        family = row.get("family")
        tag = row.get("tag")
        if family in {None, "baseline", "aggregate", "unknown"}:
            continue
        if tag == "clean":
            continue
        record = {key: value for key, value in row.items() if not isinstance(value, Mapping)}
        family_tables[str(family)].append(record)
    for family, rows in family_tables.items():
        rows.sort(
            key=lambda entry: (
                str(entry.get("model")),
                _safe_numeric(entry.get("severity_normalized"), default=float("inf")),
            )
        )
    return dict(family_tables)


def _build_t3_table(models_summary: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, entry in sorted(models_summary.items()):
        ausc_map = entry.get("ausc") if isinstance(entry, Mapping) else None
        if not isinstance(ausc_map, Mapping):
            continue
        for family, payload in sorted(ausc_map.items()):
            if not isinstance(payload, Mapping):
                continue
            row: Dict[str, object] = {"model": model, "family": family}
            metrics_block = payload.get("metrics")
            if isinstance(metrics_block, Mapping):
                _flatten_stats(row, "ausc_", metrics_block)
            retention_block = payload.get("retention")
            if isinstance(retention_block, Mapping):
                _flatten_stats(row, "retention_", retention_block)
            rows.append(row)
        macro_block = entry.get("ausc_macro_retention") if isinstance(entry, Mapping) else None
        if isinstance(macro_block, Mapping) and macro_block:
            macro_row: Dict[str, object] = {"model": model, "family": "macro"}
            _flatten_stats(macro_row, "retention_", macro_block)
            rows.append(macro_row)
    rows.sort(key=lambda entry: (str(entry.get("model")), str(entry.get("family"))))
    return rows


def _build_t4_table(
    models_summary: Mapping[str, Mapping[str, Any]],
    families: Sequence[str],
    tag_catalog: Mapping[str, TagInfo],
    *,
    bootstrap: int,
    rng_seed: int,
    severity_lockstep: bool,
    cluster_key: str,
) -> List[Dict[str, object]]:
    target_key = "ssl_colon"
    baselines = ("ssl_imnet", "sup_imnet")
    target_entry = models_summary.get(target_key)
    if not isinstance(target_entry, Mapping):
        return []
    rows: List[Dict[str, object]] = []
    for baseline in baselines:
        baseline_entry = models_summary.get(baseline)
        if not isinstance(baseline_entry, Mapping):
            continue
        for family in list(families) + ["macro"]:
            for metric in PRIMARY_RETENTION_METRICS:
                target_map = _get_retention_per_seed_map(target_entry, family, metric)
                baseline_map = _get_retention_per_seed_map(baseline_entry, family, metric)
                if not target_map or not baseline_map:
                    continue
                seeds = sorted(set(target_map.keys()) & set(baseline_map.keys()))
                if not seeds:
                    continue
                per_seed_delta: Dict[int, float] = {
                    int(seed): float(target_map[seed] - baseline_map[seed]) for seed in seeds
                }
                stats = _compute_stats(list(per_seed_delta.values()))
                if not stats:
                    continue
                bootstrap_result = _bootstrap_family_delta(
                    target_entry,
                    baseline_entry,
                    family,
                    metric,
                    tag_catalog=tag_catalog,
                    bootstrap=bootstrap,
                    rng_seed=rng_seed,
                    severity_lockstep=severity_lockstep,
                    cluster_key=cluster_key,
                )
                if bootstrap_result:
                    per_seed_delta = {
                        int(seed): float(value)
                        for seed, value in bootstrap_result.per_seed.items()
                    }
                    stats = _compute_stats(list(per_seed_delta.values())) or {}
                    delta_mean = float(bootstrap_result.mean)
                    ci_lower = bootstrap_result.ci_lower
                    ci_upper = bootstrap_result.ci_upper
                else:
                    mean_value = stats.get("mean") if stats else None
                    delta_mean = float(mean_value) if mean_value is not None else None
                    ci_lower = None
                    ci_upper = None
                std_value = stats.get("std") if stats else None
                delta_std = float(std_value) if std_value is not None else None
                count_value = stats.get("n") if stats else None
                delta_n = int(count_value) if count_value is not None else None
                row: Dict[str, object] = {
                    "comparison": f"ssl_colon - {baseline}",
                    "baseline": baseline,
                    "family": family,
                    "metric": metric,
                    "delta_mean": delta_mean,
                    "delta_std": delta_std,
                    "delta_n": delta_n,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
                for seed, value in sorted(per_seed_delta.items()):
                    row[f"seed_{seed}"] = value
                rows.append(row)
    rows.sort(
        key=lambda entry: (
            str(entry.get("baseline")),
            str(entry.get("family")),
            str(entry.get("metric")),
        )
    )
    return rows


def _build_t5_table(
    models_summary: Mapping[str, Mapping[str, Any]],
    tag_catalog: Mapping[str, TagInfo],
    *,
    bootstrap: int,
    rng_seed: int,
    cluster_key: str,
) -> List[Dict[str, object]]:
    target_key = "ssl_colon"
    baselines = ("ssl_imnet", "sup_imnet")
    target_entry = models_summary.get(target_key)
    if not isinstance(target_entry, Mapping):
        return []
    rows: List[Dict[str, object]] = []
    for baseline in baselines:
        baseline_entry = models_summary.get(baseline)
        if not isinstance(baseline_entry, Mapping):
            continue
        for tag, info in sorted(tag_catalog.items()):
            if info.family in {"baseline", "aggregate", "unknown"}:
                continue
            if tag == "clean":
                continue
            for metric in PRIMARY_RETENTION_METRICS:
                target_map = _get_retention_per_seed_for_tag(target_entry, tag, metric)
                baseline_map = _get_retention_per_seed_for_tag(baseline_entry, tag, metric)
                if not target_map or not baseline_map:
                    continue
                seeds = sorted(set(target_map.keys()) & set(baseline_map.keys()))
                if not seeds:
                    continue
                per_seed_delta: Dict[int, float] = {
                    int(seed): float(target_map[seed] - baseline_map[seed]) for seed in seeds
                }
                stats = _compute_stats(list(per_seed_delta.values()))
                if not stats:
                    continue
                bootstrap_result = _bootstrap_tag_delta(
                    target_entry,
                    baseline_entry,
                    tag,
                    metric,
                    bootstrap=bootstrap,
                    rng_seed=rng_seed,
                    cluster_key=cluster_key,
                )
                if bootstrap_result:
                    per_seed_delta = {
                        int(seed): float(value)
                        for seed, value in bootstrap_result.per_seed.items()
                    }
                    stats = _compute_stats(list(per_seed_delta.values())) or {}
                    delta_mean = float(bootstrap_result.mean)
                    ci_lower = bootstrap_result.ci_lower
                    ci_upper = bootstrap_result.ci_upper
                else:
                    mean_value = stats.get("mean") if stats else None
                    delta_mean = float(mean_value) if mean_value is not None else None
                    ci_lower = None
                    ci_upper = None
                std_value = stats.get("std") if stats else None
                delta_std = float(std_value) if std_value is not None else None
                count_value = stats.get("n") if stats else None
                delta_n = int(count_value) if count_value is not None else None
                row: Dict[str, object] = {
                    "comparison": f"ssl_colon - {baseline}",
                    "baseline": baseline,
                    "family": info.family,
                    "tag": tag,
                    "severity": info.raw_severity,
                    "severity_normalized": info.normalized_severity,
                    "metric": metric,
                    "delta_mean": delta_mean,
                    "delta_std": delta_std,
                    "delta_n": delta_n,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
                for seed, value in sorted(per_seed_delta.items()):
                    row[f"seed_{seed}"] = value
                rows.append(row)
    rows.sort(
        key=lambda entry: (
            str(entry.get("baseline")),
            str(entry.get("family")),
            _safe_numeric(entry.get("severity_normalized"), default=float("inf")),
            str(entry.get("metric")),
        )
    )
    return rows


def _build_tables(
    models_summary: Mapping[str, Mapping[str, Any]],
    tag_catalog: Mapping[str, TagInfo],
    severity_rows: Sequence[Mapping[str, object]],
    *,
    bootstrap: int,
    rng_seed: int,
    severity_lockstep: bool,
    cluster_key: str,
) -> Dict[str, object]:
    families = sorted(
        {
            info.family
            for info in tag_catalog.values()
            if info.family not in {"baseline", "aggregate", "unknown"}
        }
    )
    return {
        "t1_clean": _build_t1_table(models_summary),
        "t2": _build_t2_tables(severity_rows),
        "t3_ausc": _build_t3_table(models_summary),
        "t4_delta_ausc": _build_t4_table(
            models_summary,
            families,
            tag_catalog,
            bootstrap=bootstrap,
            rng_seed=rng_seed,
            severity_lockstep=severity_lockstep,
            cluster_key=cluster_key,
        ),
        "t5_delta_retention_by_severity": _build_t5_table(
            models_summary,
            tag_catalog,
            bootstrap=bootstrap,
            rng_seed=rng_seed,
            cluster_key=cluster_key,
        ),
    }


def _build_provenance(
    runs: Mapping[str, Mapping[int, RunPerturbationResult]],
    *,
    bootstrap: int,
    rng_seed: int,
    cluster_key: str,
    frozen_tau: bool,
    severity_lockstep: bool,
) -> Dict[str, object]:
    provenance: Dict[str, object] = {
        "models": {},
        "seeds": sorted({int(seed) for seed_map in runs.values() for seed in seed_map}),
    }
    models_block: Dict[str, object] = {}
    for model, seed_map in sorted(runs.items()):
        seed_entries: Dict[int, Dict[str, Any]] = {}
        for seed, run in sorted(seed_map.items()):
            seed_entries[int(seed)] = {
                "metrics_path": str(run.path),
                "tau": run.tau,
                "tau_policy": run.tau_policy,
                "provenance": run.provenance,
            }
        models_block[model] = {
            "seeds": sorted(seed_map.keys()),
            "runs": seed_entries,
        }
    provenance["models"] = models_block
    provenance["bootstrap"] = {
        "replicates": int(bootstrap),
        "rng_seed": int(rng_seed),
        "cluster_key": str(cluster_key),
        "cluster_key_description": f"{cluster_key} (SUN perturbations)",
        "severity_lockstep": bool(severity_lockstep),
        "frozen_tau": bool(frozen_tau),
    }
    provenance["expected_tau_policy"] = EXPECTED_PRIMARY_TAU_POLICY
    return provenance


def _summarize_model(
    model: str,
    runs: Mapping[int, RunPerturbationResult],
    tag_catalog: Mapping[str, TagInfo],
    *,
    metrics: Sequence[str],
    cluster_key: str,
) -> Dict[str, object]:
    per_tag_values: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_tag_retention: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_tag_delta: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_tag_drop: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ausc_acc: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ausc_retention_acc: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_seed_values: DefaultDict[str, DefaultDict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    per_seed_retention: DefaultDict[str, DefaultDict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    per_seed_drop: DefaultDict[str, DefaultDict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    per_seed_delta: DefaultDict[str, DefaultDict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    ausc_per_seed: DefaultDict[str, DefaultDict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    ausc_retention_per_seed: DefaultDict[str, DefaultDict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    per_case_retention: DefaultDict[
        str, DefaultDict[str, DefaultDict[int, Dict[str, float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    per_case_delta: DefaultDict[
        str, DefaultDict[str, DefaultDict[int, Dict[str, float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    per_case_clusters: DefaultDict[str, DefaultDict[int, Dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for run in runs.values():
        clean_metrics = run.perturbations.get("clean", {})
        expected_counts: Optional[Tuple[int, int, int]] = None
        for tag, stats in run.perturbations.items():
            for key, value in stats.items():
                numeric = _coerce_float(value)
                if numeric is not None:
                    per_tag_values[tag][key].append(numeric)
                    per_seed_values[tag][key][run.seed] = numeric
            tp = _coerce_int_like(stats.get("tp"))
            fp = _coerce_int_like(stats.get("fp"))
            tn = _coerce_int_like(stats.get("tn"))
            fn = _coerce_int_like(stats.get("fn"))
            n_pos = _coerce_int_like(stats.get("n_pos"))
            n_neg = _coerce_int_like(stats.get("n_neg"))
            total = None
            if tp is not None and fp is not None and tn is not None and fn is not None:
                total = tp + fp + tn + fn
            if n_pos is not None and n_neg is not None:
                pos_neg_total = n_pos + n_neg
                if total is not None and total != pos_neg_total:
                    raise ValueError(
                        f"Confusion counts mismatch for model '{model}' seed {run.seed} tag '{tag}'"
                    )
                total = pos_neg_total
            if total is not None and n_pos is not None and n_neg is not None:
                if expected_counts is None:
                    expected_counts = (n_pos, n_neg, total)
                elif expected_counts != (n_pos, n_neg, total):
                    raise ValueError(
                        "Test membership drift detected across severities for model "
                        f"'{model}' seed {run.seed} (tag '{tag}')"
                    )
        retention_seed_tracker: Dict[str, bool] = {metric: False for metric in metrics}
        case_metrics_map = run.case_metrics if isinstance(run.case_metrics, Mapping) else {}
        clean_case_metrics = case_metrics_map.get("clean") if isinstance(case_metrics_map, Mapping) else None
        baseline_case_metrics: Dict[str, Dict[str, float]] = {}
        if isinstance(clean_case_metrics, Mapping):
            for case_id, case_stats in clean_case_metrics.items():
                if not isinstance(case_stats, Mapping):
                    continue
                canonical_id = str(case_id)
                cluster_value = _extract_cluster_value(case_stats, cluster_key)
                if cluster_value is None and cluster_key == "case_id":
                    cluster_value = canonical_id
                sanitized: Dict[str, float] = {}
                for metric in metrics:
                    numeric = _coerce_float(case_stats.get(metric))
                    if numeric is None or not math.isfinite(numeric):
                        continue
                    sanitized[str(metric)] = float(numeric)
                if sanitized:
                    baseline_case_metrics[canonical_id] = sanitized
                    if cluster_value is not None:
                        per_case_clusters["clean"][run.seed][canonical_id] = str(cluster_value)
                    for metric_key, value in sanitized.items():
                        per_case_retention["clean"][metric_key][run.seed][canonical_id] = 1.0
                        per_case_delta["clean"][metric_key][run.seed][canonical_id] = 0.0
        for metric in metrics:
            baseline_value = _coerce_float(clean_metrics.get(metric))
            if baseline_value is None:
                continue
            per_tag_retention["clean"][metric].append(1.0)
            per_tag_drop["clean"][metric].append(0.0)
            per_tag_delta["clean"][metric].append(0.0)
            per_seed_retention["clean"][metric][run.seed] = 1.0
            per_seed_drop["clean"][metric][run.seed] = 0.0
            per_seed_delta["clean"][metric][run.seed] = 0.0
            retention_seed_tracker[metric] = True
        for tag, stats in run.perturbations.items():
            if tag == "clean":
                continue
            base = clean_metrics
            for metric in metrics:
                metric_value = _coerce_float(stats.get(metric))
                baseline_value = _coerce_float(base.get(metric))
                if metric_value is None or baseline_value is None or not math.isfinite(baseline_value) or baseline_value == 0.0:
                    continue
                retention_value = metric_value / baseline_value
                per_tag_retention[tag][metric].append(retention_value)
                per_tag_drop[tag][metric].append(1.0 - retention_value)
                per_tag_delta[tag][metric].append(metric_value - baseline_value)
                per_seed_retention[tag][metric][run.seed] = retention_value
                per_seed_drop[tag][metric][run.seed] = 1.0 - retention_value
                per_seed_delta[tag][metric][run.seed] = metric_value - baseline_value
        if isinstance(case_metrics_map, Mapping):
            for tag, case_block in case_metrics_map.items():
                if not isinstance(case_block, Mapping):
                    continue
                canonical_tag = str(tag)
                if canonical_tag == "clean":
                    continue
                for case_id, case_stats in case_block.items():
                    if not isinstance(case_stats, Mapping):
                        continue
                    canonical_case = str(case_id)
                    baseline_stats = baseline_case_metrics.get(canonical_case)
                    if not baseline_stats:
                        continue
                    cluster_value = _extract_cluster_value(case_stats, cluster_key)
                    if cluster_value is None:
                        cluster_value = per_case_clusters["clean"].get(run.seed, {}).get(
                            canonical_case
                        )
                    if cluster_value is None and cluster_key == "case_id":
                        cluster_value = canonical_case
                    if cluster_value is not None:
                        per_case_clusters[canonical_tag][run.seed][canonical_case] = str(
                            cluster_value
                        )
                    for metric in metrics:
                        metric_value = _coerce_float(case_stats.get(metric))
                        if metric_value is None or not math.isfinite(metric_value):
                            continue
                        baseline_value = baseline_stats.get(str(metric))
                        if baseline_value is None or not math.isfinite(baseline_value):
                            continue
                        if baseline_value != 0.0:
                            retention_value = metric_value / baseline_value
                            if math.isfinite(retention_value):
                                per_case_retention[canonical_tag][str(metric)][run.seed][
                                    canonical_case
                                ] = float(retention_value)
                        delta_value = metric_value - baseline_value
                        if math.isfinite(delta_value):
                            per_case_delta[canonical_tag][str(metric)][run.seed][
                                canonical_case
                            ] = float(delta_value)
        # Compute AUSC per family for this run.
        family_points: DefaultDict[str, DefaultDict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for tag, stats in run.perturbations.items():
            info = tag_catalog.get(tag)
            if info is None or info.family in {"baseline", "aggregate", "unknown"}:
                continue
            severity = info.normalized_severity
            if severity is None or math.isnan(severity):
                continue
            for metric in metrics:
                numeric = _coerce_float(stats.get(metric))
                if numeric is None:
                    continue
                family_points[info.family][metric].append((severity, numeric))
        clean_cache: Dict[str, float] = {
            metric: val
            for metric in metrics
            if (val := _coerce_float(clean_metrics.get(metric))) is not None and math.isfinite(val)
        }
        for family, family_metric_points in family_points.items():
            for metric, points in family_metric_points.items():
                clean_value = clean_cache.get(metric)
                if clean_value is None:
                    continue
                series = _deduplicate_points([(0.0, clean_value)] + points)
                ausc_value = _normalised_trapz(series)
                if ausc_value is not None and math.isfinite(ausc_value):
                    ausc_acc[family][metric].append(ausc_value)
                    ausc_per_seed[family][metric][run.seed] = ausc_value
                if clean_value == 0.0:
                    continue
                retention_series = _deduplicate_points([(0.0, 1.0)] + [(sev, val / clean_value) for sev, val in points])
                retention_ausc = _normalised_trapz(retention_series)
                if retention_ausc is not None and math.isfinite(retention_ausc):
                    ausc_retention_acc[family][metric].append(retention_ausc)
                    ausc_retention_per_seed[family][metric][run.seed] = retention_ausc

    per_tag_summary: Dict[str, Dict[str, Any]] = {}
    for tag in sorted(per_tag_values.keys() | per_tag_retention.keys() | per_tag_delta.keys()):
        metrics_block = {
            key: stats
            for key in sorted(per_tag_values[tag].keys())
            if (stats := _compute_stats(per_tag_values[tag][key]))
        }
        retention_block = {
            key: stats
            for key in sorted(per_tag_retention[tag].keys())
            if (stats := _compute_stats(per_tag_retention[tag][key]))
        }
        drop_block = {
            key: stats
            for key in sorted(per_tag_drop[tag].keys())
            if (stats := _compute_stats(per_tag_drop[tag][key]))
        }
        delta_block = {
            key: stats
            for key in sorted(per_tag_delta[tag].keys())
            if (stats := _compute_stats(per_tag_delta[tag][key]))
        }
        per_tag_summary[tag] = {
            "metrics": metrics_block,
            "retention": retention_block,
            "drop": drop_block,
            "delta": delta_block,
            "metrics_per_seed": _convert_per_seed_map(per_seed_values.get(tag, {})),
            "retention_per_seed": _convert_per_seed_map(per_seed_retention.get(tag, {})),
            "drop_per_seed": _convert_per_seed_map(per_seed_drop.get(tag, {})),
            "delta_per_seed": _convert_per_seed_map(per_seed_delta.get(tag, {})),
        }

    severity_rows: List[Dict[str, object]] = []
    for tag, summary in per_tag_summary.items():
        info = tag_catalog.get(tag, TagInfo(family="unknown", raw_severity=float("nan"), normalized_severity=float("nan")))
        row: Dict[str, object] = {
            "model": model,
            "tag": tag,
            "family": info.family,
            "severity": info.raw_severity,
            "severity_normalized": info.normalized_severity,
        }
        _flatten_stats(row, "", summary.get("metrics", {}))
        _flatten_stats(row, "retention_", summary.get("retention", {}))
        _flatten_stats(row, "drop_", summary.get("drop", {}))
        _flatten_stats(row, "delta_", summary.get("delta", {}))
        severity_rows.append(row)

    families_summary: Dict[str, Dict[str, object]] = {}
    clean_entry = per_tag_summary.get("clean")
    for family in sorted({info.family for info in tag_catalog.values() if info.family not in {"baseline", "aggregate", "unknown"}}):
        tag_entries: List[Dict[str, object]] = []
        if clean_entry:
            tag_entries.append(
                {
                    "tag": "clean",
                    "severity": 0.0,
                    "severity_normalized": 0.0,
                    "metrics": clean_entry.get("metrics", {}),
                    "retention": clean_entry.get("retention", {}),
                    "drop": clean_entry.get("drop", {}),
                    "delta": clean_entry.get("delta", {}),
                }
            )
        for tag, summary in per_tag_summary.items():
            info = tag_catalog.get(tag)
            if info is None or info.family != family:
                continue
            tag_entries.append(
                {
                    "tag": tag,
                    "severity": info.raw_severity,
                    "severity_normalized": info.normalized_severity,
                    "metrics": summary.get("metrics", {}),
                    "retention": summary.get("retention", {}),
                    "drop": summary.get("drop", {}),
                    "delta": summary.get("delta", {}),
                }
            )
        if tag_entries:
            tag_entries.sort(
                key=lambda entry: (
                    _safe_numeric(entry.get("severity_normalized")),
                    str(entry.get("tag")),
                )
            )
            families_summary[family] = {"tags": tag_entries}

    ausc_summary: Dict[str, Dict[str, Any]] = {}
    for family, metric_series_defaultdict in ausc_acc.items():
        metrics_stats: Dict[str, Dict[str, float]] = {}
        metric_series_map: Dict[str, List[float]] = dict(metric_series_defaultdict)
        for metric, values in metric_series_map.items():
            stats = _compute_stats(values)
            if stats:
                metrics_stats[metric] = stats
        retention_stats: Dict[str, Dict[str, float]] = {}
        retention_map: Dict[str, List[float]] = dict(ausc_retention_acc.get(family, {}))
        for metric, values in retention_map.items():
            stats = _compute_stats(values)
            if stats:
                retention_stats[metric] = stats
        if metrics_stats or retention_stats:
            ausc_summary[family] = {
                "metrics": metrics_stats,
                "retention": retention_stats,
            }

    return {
        "per_tag": per_tag_summary,
        "families": families_summary,
        "severity_rows": severity_rows,
        "ausc": ausc_summary,
        "ausc_per_seed": {family: _convert_per_seed_map(metrics) for family, metrics in ausc_per_seed.items()},
        "ausc_retention_per_seed": {
            family: _convert_per_seed_map(metrics) for family, metrics in ausc_retention_per_seed.items()
        },
        "per_case_retention_per_seed": _convert_case_per_seed_map(per_case_retention),
        "per_case_delta_per_seed": _convert_case_per_seed_map(per_case_delta),
        "per_case_clusters_per_seed": _convert_case_cluster_map(per_case_clusters),
    }


def summarize_runs(
    runs: Mapping[str, Mapping[int, RunPerturbationResult]],
    *,
    metrics: Sequence[str] = RETENTION_METRICS,
    bootstrap: int = 2000,
    rng_seed: int = 12345,
    cluster_key: str = "case_id",
    frozen_tau: bool = True,
    severity_lockstep: bool = True,
) -> Dict[str, object]:
    tag_catalog = _build_tag_catalog(runs)
    _validate_required_perturbations(tag_catalog)
    models_summary: Dict[str, Dict[str, object]] = {}
    severity_rows: List[Dict[str, object]] = []
    for model, seed_map in runs.items():
        missing = sorted(set(REQUIRED_SEEDS) - {int(seed) for seed in seed_map.keys()})
        if missing:
            missing_str = ', '.join(str(seed) for seed in missing)
            raise ValueError(f"Model '{model}' is missing required seeds: {missing_str}")
    for model, model_runs in sorted(runs.items()):
        summary = _summarize_model(
            model,
            model_runs,
            tag_catalog,
            metrics=metrics,
            cluster_key=cluster_key,
        )
        model_entry: Dict[str, object] = {
            "per_tag": summary["per_tag"],
            "families": summary["families"],
            "ausc": summary["ausc"],
            "ausc_per_seed": summary.get("ausc_per_seed", {}),
            "ausc_retention_per_seed": summary.get("ausc_retention_per_seed", {}),
            "per_case_retention_per_seed": summary.get("per_case_retention_per_seed", {}),
            "per_case_delta_per_seed": summary.get("per_case_delta_per_seed", {}),
            "per_case_clusters_per_seed": summary.get("per_case_clusters_per_seed", {}),
        }
        models_summary[model] = model_entry
        summary_rows = summary.get("severity_rows", [])
        if isinstance(summary_rows, list):
            severity_rows.extend([row for row in summary_rows if isinstance(row, dict)])

    families = sorted(
        {
            info.family
            for info in tag_catalog.values()
            if info.family not in {"baseline", "aggregate", "unknown"}
        }
    )

    for model, model_entry in models_summary.items():
        retention_per_seed = model_entry.get("ausc_retention_per_seed")
        if not isinstance(retention_per_seed, Mapping):
            continue
        macro_retention_per_seed: Dict[str, Dict[int, float]] = {}
        macro_retention_stats: Dict[str, Dict[str, float]] = {}
        for metric in PRIMARY_RETENTION_METRICS:
            seed_values: Dict[int, float] = {}
            for seed in REQUIRED_SEEDS:
                family_values: List[float] = []
                for family in families:
                    family_block = retention_per_seed.get(family)
                    if not isinstance(family_block, Mapping):
                        family_values = []
                        break
                    metric_block = family_block.get(metric)
                    if not isinstance(metric_block, Mapping):
                        family_values = []
                        break
                    value = metric_block.get(int(seed))
                    if value is None or not math.isfinite(float(value)):
                        family_values = []
                        break
                    family_values.append(float(value))
                if len(family_values) == len(families) and families:
                    seed_values[int(seed)] = float(np.mean(family_values))
            if seed_values:
                macro_retention_per_seed[metric] = seed_values
                macro_stats = _compute_stats(list(seed_values.values()))
                if macro_stats:
                    macro_retention_stats[metric] = macro_stats
        if macro_retention_stats:
            model_entry["ausc_macro_retention"] = macro_retention_stats
            model_entry["ausc_macro_retention_per_seed"] = macro_retention_per_seed

    tables = _build_tables(
        models_summary,
        tag_catalog,
        severity_rows,
        bootstrap=max(0, int(bootstrap)),
        rng_seed=int(rng_seed),
        severity_lockstep=bool(severity_lockstep),
        cluster_key=str(cluster_key),
    )
    provenance = _build_provenance(
        runs,
        bootstrap=int(bootstrap),
        rng_seed=int(rng_seed),
        cluster_key=str(cluster_key),
        frozen_tau=bool(frozen_tau),
        severity_lockstep=bool(severity_lockstep),
    )

    catalog_serialized = {
        tag: {
            "family": info.family,
            "severity": info.raw_severity,
            "severity_normalized": info.normalized_severity,
        }
        for tag, info in sorted(tag_catalog.items())
    }
    return {
        "tag_catalog": catalog_serialized,
        "models": models_summary,
        "severity_rows": severity_rows,
        "tables": tables,
        "provenance": provenance,
    }

def write_severity_csv(summary: Mapping[str, object], output_path: Path) -> None:
    rows = summary.get("severity_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Summary does not contain severity_rows data")
    fieldnames = sorted({key for row in rows if isinstance(row, Mapping) for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            writer.writerow(row)


def write_tables(summary: Mapping[str, object], output_dir: Path) -> Dict[str, Path]:
    tables = summary.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError("Summary does not include tabular outputs")
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> Optional[Path]:
        filtered_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
        if not filtered_rows:
            return None
        fieldnames = sorted({key for row in filtered_rows for key in row.keys()})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in filtered_rows:
                writer.writerow(row)
        return path

    created: Dict[str, Path] = {}

    t1_rows = tables.get("t1_clean")
    if isinstance(t1_rows, Sequence):
        path = _write(output_dir / "T1_clean.csv", t1_rows)
        if path:
            created["t1_clean"] = path

    t2_tables = tables.get("t2")
    if isinstance(t2_tables, Mapping):
        family_filenames = {
            "blur": "T2_blur.csv",
            "gaussian_blur": "T2_blur.csv",
            "jpeg": "T2_jpeg.csv",
            "brightness": "T2_brightness.csv",
            "contrast": "T2_contrast.csv",
            "brightness_contrast": "T2_brightness_contrast.csv",
            "occlusion": "T2_occlusion.csv",
        }
        for family, rows in t2_tables.items():
            if not isinstance(rows, Sequence):
                continue
            filename = family_filenames.get(family, f"T2_{family}.csv")
            path = _write(output_dir / filename, rows)
            if path:
                created[f"t2_{family}"] = path

    t3_rows = tables.get("t3_ausc")
    if isinstance(t3_rows, Sequence):
        path = _write(output_dir / "T3_ausc.csv", t3_rows)
        if path:
            created["t3_ausc"] = path

    t4_rows = tables.get("t4_delta_ausc")
    if isinstance(t4_rows, Sequence):
        path = _write(output_dir / "T4_delta_ausc.csv", t4_rows)
        if path:
            created["t4_delta_ausc"] = path

    t5_rows = tables.get("t5_delta_retention_by_severity")
    if isinstance(t5_rows, Sequence):
        path = _write(output_dir / "T5_delta_retention_by_severity.csv", t5_rows)
        if path:
            created["t5_delta_retention_by_severity"] = path

    return created


__all__ = [
    "PRIMARY_RETENTION_METRICS",
    "EXPECTED_PRIMARY_TAU_POLICY",
    "RunPerturbationResult",
    "TauPolicyError",
    "discover_runs",
    "load_run",
    "summarize_runs",
    "write_severity_csv",
    "write_tables",
]
