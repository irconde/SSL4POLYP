"""Single binary-metrics module for training JSONs and reports (promoted from ``analysis_v2/metrics_v2.py``).

Definitions (positive class = polyp = label 1, prediction = ``prob >= tau``):
  auroc, auprc : sklearn ``roc_auc_score`` / ``average_precision_score`` on ``prob``
  precision    : tp / (tp + fp)            (0.0 if undefined, as sklearn zero_division=0)
  recall       : tp / (tp + fn)            (sensitivity of the positive class)
  specificity  : tn / (tn + fp)
  f1           : 2tp / (2tp + fp + fn)     (0.0 if undefined)
  balanced_accuracy : (recall + specificity) / 2
Undefined AUROC/AUPRC/BA (one class only) -> NaN.

Calibration and per-polyp metrics (SAP section 2):
  brier        : mean((prob - label)^2)
  ece          : expected calibration error with 15 equal-mass bins
  per-polyp sensitivity : share of positive cases with >= k frames above tau (k = 1, 5)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "METRICS",
    "DEGENERATE_THRESHOLD",
    "ECE_BINS",
    "binary_metrics",
    "brier_score",
    "equal_mass_bins",
    "expected_calibration_error",
    "is_degenerate",
    "metrics_at",
    "per_polyp_sensitivity",
    "reliability_table",
]

METRICS = ("auroc", "auprc", "f1", "precision", "recall", "specificity", "balanced_accuracy")
DEGENERATE_THRESHOLD = 0.10
ECE_BINS = 15


def _as_arrays(probs: Sequence[float], labels: Sequence[int]):
    probs_np = np.asarray(probs, dtype=np.float64).ravel()
    labels_np = np.asarray(labels, dtype=np.int64).ravel()
    if probs_np.size != labels_np.size:
        raise ValueError("probs and labels must have the same length")
    return probs_np, labels_np


def binary_metrics(
    probs: Sequence[float],
    labels: Sequence[int],
    preds: Sequence[int],
) -> Dict[str, float]:
    """Return positive-class metrics for explicit ``preds`` (AUROC/AUPRC from ``probs``)."""

    probs_np, labels_np = _as_arrays(probs, labels)
    pred = np.asarray(preds).ravel().astype(bool)
    out: Dict[str, float] = {k: float("nan") for k in METRICS}
    out.update(tp=0.0, fp=0.0, tn=0.0, fn=0.0, n_pos=0.0, n_neg=0.0)
    if labels_np.size == 0:
        return out
    pos = labels_np == 1
    tp = float(np.sum(pred & pos))
    fp = float(np.sum(pred & ~pos))
    fn = float(np.sum(~pred & pos))
    tn = float(np.sum(~pred & ~pos))
    out.update(tp=tp, fp=fp, tn=tn, fn=fn, n_pos=tp + fn, n_neg=tn + fp)
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    out["recall"], out["specificity"] = recall, spec
    out["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    out["f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    if not (math.isnan(recall) or math.isnan(spec)):
        out["balanced_accuracy"] = 0.5 * (recall + spec)
    if tp + fn > 0 and tn + fp > 0:
        out["auroc"] = float(roc_auc_score(labels_np, probs_np))
        out["auprc"] = float(average_precision_score(labels_np, probs_np))
    return out


def metrics_at(
    probs: Sequence[float],
    labels: Sequence[int],
    tau: float,
    rows: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return :func:`binary_metrics` with ``pred = prob >= tau`` (optionally on ``rows``)."""

    probs_np, labels_np = _as_arrays(probs, labels)
    if rows is not None:
        probs_np, labels_np = probs_np[rows], labels_np[rows]
    return binary_metrics(probs_np, labels_np, probs_np >= float(tau))


def is_degenerate(metrics: Dict[str, float], threshold: float = DEGENERATE_THRESHOLD) -> bool:
    """SAP: specificity or sensitivity at the operating threshold below ``threshold``."""

    values = [metrics.get("specificity"), metrics.get("recall")]
    return any(v is not None and not math.isnan(v) and v < threshold for v in values)


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    probs_np, labels_np = _as_arrays(probs, labels)
    if probs_np.size == 0:
        return float("nan")
    return float(np.mean((probs_np - labels_np) ** 2))


def equal_mass_bins(probs: Sequence[float], n_bins: int = ECE_BINS) -> List[np.ndarray]:
    """Split frame indices sorted by ``prob`` into ``n_bins`` bins of (near) equal size.

    Ties are broken by the original index (stable sort) so the binning is deterministic.
    """

    probs_np = np.asarray(probs, dtype=np.float64).ravel()
    order = np.argsort(probs_np, kind="mergesort")
    return [chunk for chunk in np.array_split(order, n_bins) if chunk.size > 0]


def reliability_table(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = ECE_BINS
) -> List[Dict[str, float]]:
    """Per-bin mean confidence and observed positive frequency (equal-mass bins)."""

    probs_np, labels_np = _as_arrays(probs, labels)
    rows: List[Dict[str, float]] = []
    for idx, members in enumerate(equal_mass_bins(probs_np, n_bins)):
        rows.append(
            {
                "bin": float(idx),
                "n": float(members.size),
                "prob_min": float(probs_np[members].min()),
                "prob_max": float(probs_np[members].max()),
                "mean_prob": float(probs_np[members].mean()),
                "frac_pos": float(labels_np[members].mean()),
            }
        )
    return rows


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = ECE_BINS
) -> float:
    """ECE on the positive-class probability with ``n_bins`` equal-mass bins."""

    probs_np, labels_np = _as_arrays(probs, labels)
    total = probs_np.size
    if total == 0:
        return float("nan")
    ece = 0.0
    for row in reliability_table(probs_np, labels_np, n_bins):
        ece += (row["n"] / total) * abs(row["frac_pos"] - row["mean_prob"])
    return float(ece)


def per_polyp_sensitivity(
    probs: Sequence[float],
    labels: Sequence[int],
    case_ids: Sequence[str],
    tau: float,
    min_frames: Sequence[int] = (1, 5),
) -> Dict[str, float]:
    """Share of positive cases with at least ``k`` frames at or above ``tau`` (SUN only)."""

    probs_np, labels_np = _as_arrays(probs, labels)
    cases = np.asarray([str(c) for c in case_ids], dtype=object)
    if cases.size != probs_np.size:
        raise ValueError("case_ids must align with probs")
    positive_cases = sorted({c for c, y in zip(cases.tolist(), labels_np.tolist()) if y == 1})
    out: Dict[str, float] = {"n_pos_cases": float(len(positive_cases))}
    hits = {c: int(np.sum((cases == c) & (labels_np == 1) & (probs_np >= float(tau)))) for c in positive_cases}
    for k in min_frames:
        key = f"per_polyp_sensitivity_ge{int(k)}"
        if not positive_cases:
            out[key] = float("nan")
        else:
            out[key] = float(sum(1 for c in positive_cases if hits[c] >= int(k)) / len(positive_cases))
    return out
