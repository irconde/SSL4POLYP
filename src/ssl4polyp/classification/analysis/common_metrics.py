from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

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

__all__ = [
    "DEFAULT_BINARY_METRIC_KEYS",
    "_clean_text",
    "_coerce_float",
    "_coerce_int",
    "compute_binary_metrics",
    "ClusterSet",
    "build_cluster_set",
    "sample_cluster_ids",
]

_DEFAULT_BINARY_METRICS: Tuple[str, ...] = (
    "auprc",
    "auroc",
    "recall",
    "precision",
    "f1",
    "balanced_accuracy",
    "mcc",
    "loss",
)


# Public alias for the default binary metric set so report modules can share
# a single definition when they need to stay in sync with the computation
# defaults.  The tuple is intentionally immutable to guard against accidental
# mutation across imports.
DEFAULT_BINARY_METRIC_KEYS: Tuple[str, ...] = _DEFAULT_BINARY_METRICS


def _clean_text(value: Optional[object]) -> Optional[str]:
    """Normalise arbitrary inputs to stripped text or ``None``."""

    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: object) -> Optional[float]:
    """Parse floats from heterogeneous inputs, filtering non-finite values."""

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
    """Parse integers from heterogeneous inputs."""

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


def compute_binary_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    tau: float,
    *,
    metric_keys: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Compute binary classification metrics for probabilities and labels."""

    metrics = tuple(metric_keys) if metric_keys is not None else _DEFAULT_BINARY_METRICS
    metric_set = set(metrics)
    total = int(labels.size)
    if probs.size == 0 or total == 0:
        result: Dict[str, float] = {
            "count": 0.0,
            "n_pos": 0.0,
            "n_neg": 0.0,
            "prevalence": float("nan"),
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
        }
        for key in metrics:
            result[key] = float("nan")
        return result
    preds = (probs >= float(tau)).astype(int)
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
    eps = 1e-12
    clipped = np.clip(probs.astype(float), eps, 1.0 - eps)
    loss_val = float(
        np.mean(
            -(labels.astype(float) * np.log(clipped)
            + (1 - labels.astype(float)) * np.log(1 - clipped))
        )
    )
    full_metrics: Dict[str, float] = {
        "count": float(total),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "prevalence": prevalence,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "auprc": auprc,
        "auroc": auroc,
        "recall": recall_val,
        "precision": precision_val,
        "f1": f1_val,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc_val,
        "loss": loss_val,
    }
    return {
        key: full_metrics[key]
        for key in full_metrics
        if key in metric_set or key not in _DEFAULT_BINARY_METRICS
    }


T = TypeVar("T")


@dataclass(frozen=True)
class ClusterSet:
    positives: Tuple[Tuple[str, ...], ...]
    negatives: Tuple[Tuple[str, ...], ...]


def build_cluster_set(
    records: Iterable[T],
    *,
    is_positive: Callable[[T], bool],
    record_id: Callable[[T], str],
    positive_key: Callable[[T], Optional[str]],
    negative_key: Callable[[T], Optional[str]],
) -> ClusterSet:
    """Construct bootstrap clusters for positive and negative frames."""

    pos_clusters: DefaultDict[str, List[str]] = defaultdict(list)
    neg_clusters: DefaultDict[str, List[str]] = defaultdict(list)
    for record in records:
        identifier = record_id(record)
        if is_positive(record):
            key = positive_key(record) or f"pos_frame::{identifier}"
            pos_clusters[key].append(identifier)
        else:
            key = negative_key(record) or f"neg_frame::{identifier}"
            neg_clusters[key].append(identifier)
    positives = tuple(tuple(cluster) for cluster in pos_clusters.values())
    negatives = tuple(tuple(cluster) for cluster in neg_clusters.values())
    return ClusterSet(positives=positives, negatives=negatives)


def sample_cluster_ids(clusters: ClusterSet, rng: np.random.Generator) -> List[str]:
    """Sample frame identifiers from clustered positives and negatives."""

    sampled: List[str] = []
    if clusters.positives:
        indices = rng.integers(0, len(clusters.positives), size=len(clusters.positives))
        for idx in indices:
            sampled.extend(clusters.positives[idx])
    if clusters.negatives:
        indices = rng.integers(0, len(clusters.negatives), size=len(clusters.negatives))
        for idx in indices:
            sampled.extend(clusters.negatives[idx])
    return sampled
