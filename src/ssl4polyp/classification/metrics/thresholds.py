"""Threshold utilities for binary classification models.

This module centralises helper functions used to derive decision thresholds
from validation logits and to persist/retrieve them alongside checkpoints.
It provides deterministic candidate generation and multiple optimisation
policies with well-defined tie-breaking behaviour.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_curve


ThresholdMap = Dict[str, float]

_EPS = 1e-12
_MAX_THRESHOLD_CANDIDATES = 200


@dataclass(frozen=True)
class ThresholdPolicyResult:
    """Structured result describing a policy-driven threshold search."""

    tau: float
    record: Dict[str, Any]
    metrics: Dict[str, float]
    candidates: Sequence[float]


def _policy_tiebreakers(policy: str) -> Sequence[str]:
    # All validation-derived policies share the same deterministic tie-breaking order.
    policy = policy.lower()
    if policy not in {
        "f1_opt_on_val",
        "youden_on_val",
        "val_opt_youden",
    }:
        return ["higher_recall", "lower_tau"]
    return ["higher_recall", "lower_tau"]


def _prepare_binary_scores(logits: torch.Tensor) -> torch.Tensor:
    """Normalise ``logits`` into positive-class scores for binary problems."""

    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.ndim != 2:
        raise ValueError(
            "Binary threshold computation expects logits with shape (N,) or (N, 2)"
        )
    if logits.size(1) == 1:
        return torch.sigmoid(logits.squeeze(1))
    if logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    raise ValueError(
        "Binary threshold computation received logits with more than two classes"
    )


def compute_youden_j_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Return the threshold maximising Youden's J statistic.

    Parameters
    ----------
    logits:
        Logits produced by a binary classifier. Accepted shapes are ``(N,)``,
        ``(N, 1)`` or ``(N, 2)``. The function converts them to probabilities
        for the positive class.
    targets:
        Integer tensor containing the binary labels for the associated logits.

    Returns
    -------
    float
        Threshold on the positive-class probability that maximises
        ``Youden's J = sensitivity + specificity - 1``.
    """

    if logits.numel() == 0:
        raise ValueError("Cannot compute threshold on empty logits tensor")

    scores = _prepare_binary_scores(logits).detach().cpu().numpy().astype(float)
    labels = targets.detach().cpu().numpy().astype(int)

    if scores.shape[0] != labels.shape[0]:
        raise ValueError("Logits and targets must have matching first dimension")
    if np.unique(labels).size < 2:
        raise ValueError("Youden's J threshold requires both positive and negative samples")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    tau = float(thresholds[best_idx])
    if np.isinf(tau):
        # ``roc_curve`` may return ``inf`` when predictions are perfect. In this
        # case pick a value marginally above the maximum score which effectively
        # classifies positives perfectly while keeping negatives untouched.
        tau = float(np.nextafter(scores.max(), 1.0))
    return tau


def format_threshold_key(dataset: str, split: str, policy: str) -> str:
    """Create a canonical key name for persisted thresholds."""

    return f"{dataset.lower()}_{split.lower()}_{policy.lower()}"


def save_thresholds(path: Path, thresholds: Mapping[str, float]) -> None:
    """Persist ``thresholds`` to ``path`` as a small JSON document."""

    serialisable = {key: float(value) for key, value in thresholds.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"thresholds": serialisable}, handle, indent=2)


def load_thresholds(path: Path) -> ThresholdMap:
    """Load a thresholds mapping previously written by :func:`save_thresholds`."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    raw = payload.get("thresholds", payload)
    result: ThresholdMap = {}
    for key, value in raw.items():
        try:
            result[key] = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid threshold value for key {key!r}: {value!r}") from exc
    return result


def resolve_threshold(
    thresholds: Mapping[str, float],
    key: Optional[str],
) -> Optional[float]:
    """Return the threshold value for ``key`` if present."""

    if key is None:
        return None
    if key not in thresholds:
        return None
    return float(thresholds[key])


def compute_threshold_from_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    policy: str = "youden",
) -> float:
    """Compute a threshold following ``policy`` using ``loader`` for logits."""

    policy = policy.lower()
    if policy != "youden":
        raise ValueError(f"Unsupported threshold policy '{policy}'")

    was_training = model.training
    model.eval()
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) != 3:
                raise ValueError(
                    "Threshold computation expects batches with (images, labels, metadata)"
                )
            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).detach().cpu()
            logits_list.append(outputs)
            targets_list.append(labels.detach().cpu())
    if was_training:
        model.train()
    if not logits_list:
        raise ValueError("Cannot compute threshold from an empty dataloader")
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return compute_youden_j_threshold(logits, targets)


def _prepare_candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Return deterministic candidate thresholds capped to a fixed budget."""

    if scores.ndim != 1:
        raise ValueError("Scores array must be one-dimensional")
    base = np.unique(scores)
    base = np.concatenate(([0.0], base, [1.0]))
    base = np.clip(base, 0.0, 1.0)
    base = np.unique(base)
    if base.size <= _MAX_THRESHOLD_CANDIDATES:
        return base.astype(float, copy=False)
    indices = np.linspace(0, base.size - 1, num=_MAX_THRESHOLD_CANDIDATES, dtype=int)
    indices[0] = 0
    indices[-1] = base.size - 1
    return base[indices].astype(float, copy=False)


def _compute_confusion_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds = scores[:, None] >= candidates[None, :]
    positives = labels[:, None] == 1
    negatives = ~positives
    tp = np.logical_and(preds, positives).sum(axis=0, dtype=np.int64)
    fp = np.logical_and(preds, negatives).sum(axis=0, dtype=np.int64)
    fn = np.logical_and(~preds, positives).sum(axis=0, dtype=np.int64)
    tn = np.logical_and(~preds, negatives).sum(axis=0, dtype=np.int64)
    return tp, fp, tn, fn


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    result = np.zeros_like(num, dtype=float)
    mask = denom > 0
    result[mask] = num[mask] / denom[mask]
    return result


def _apply_tiebreak(
    candidate_indices: np.ndarray,
    metric_values: np.ndarray,
    direction: str,
) -> np.ndarray:
    if candidate_indices.size <= 1:
        return candidate_indices
    values = metric_values[candidate_indices]
    if direction == "higher":
        target = values.max()
        mask = values >= (target - _EPS)
    elif direction == "lower":
        target = values.min()
        mask = values <= (target + _EPS)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported tiebreak direction '{direction}'")
    return candidate_indices[mask]


def _compute_metrics_for_tau(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "youden_j": tpr - fpr,
    }


def _build_policy_record(
    *,
    policy: str,
    tau: float,
    split_name: str,
    n_candidates: int,
    tiebreakers: Sequence[str],
    epoch: int,
    degenerate: bool,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "policy": policy,
        "tau": float(tau),
        "split": split_name,
        "n_candidates": int(n_candidates),
        "tiebreakers": list(tiebreakers),
        "epoch": int(epoch),
        "degenerate_val": bool(degenerate),
        "notes": notes or {},
    }
    return record


def compute_policy_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    policy: str,
    split_name: str,
    epoch: int,
    previous_tau: Optional[float] = None,
) -> ThresholdPolicyResult:
    """Compute a decision threshold following the requested ``policy``."""

    policy = policy.strip().lower()
    supported = {"f1_opt_on_val", "youden_on_val", "val_opt_youden"}
    if policy not in supported:
        raise ValueError(f"Unsupported threshold policy '{policy}'")

    scores_np = np.asarray(scores, dtype=float).ravel()
    labels_np = np.asarray(labels, dtype=int).ravel()
    if scores_np.size == 0:
        raise ValueError("Cannot compute threshold with empty scores array")
    if scores_np.size != labels_np.size:
        raise ValueError("Scores and labels must have matching shapes")

    unique_labels = np.unique(labels_np)
    degenerate = unique_labels.size < 2
    notes: Dict[str, Any] = {}

    if degenerate:
        if previous_tau is not None and math.isfinite(previous_tau):
            tau = float(previous_tau)
            notes["carried_forward"] = True
        else:
            tau = 0.5
            notes["default_tau"] = 0.5
        tp, fp, tn, fn = _compute_confusion_arrays(scores_np, labels_np, np.array([tau]))
        metrics = _compute_metrics_for_tau(int(tp[0]), int(fp[0]), int(tn[0]), int(fn[0]))
        record = _build_policy_record(
            policy=policy,
            tau=tau,
            split_name=split_name,
            n_candidates=0,
            tiebreakers=_policy_tiebreakers(policy),
            epoch=epoch,
            degenerate=True,
            notes=notes,
        )
        return ThresholdPolicyResult(tau=tau, record=record, metrics=metrics, candidates=[float(tau)])

    candidates = _prepare_candidate_thresholds(scores_np)
    tp, fp, tn, fn = _compute_confusion_arrays(scores_np, labels_np, candidates)
    recalls = _safe_divide(tp, tp + fn)
    precisions = _safe_divide(tp, tp + fp)
    f1_scores = _safe_divide(2 * tp, (2 * tp) + fp + fn)
    tprs = recalls
    fprs = _safe_divide(fp, fp + tn)

    if policy == "f1_opt_on_val":
        objective = f1_scores
    else:
        objective = tprs - fprs
    tiebreak_spec = [
        (recalls, "higher"),
        (candidates, "lower"),
    ]

    best_value = objective.max()
    candidate_indices = np.where(objective >= (best_value - _EPS))[0]
    for values, direction in tiebreak_spec:
        candidate_indices = _apply_tiebreak(candidate_indices, values, direction)
        if candidate_indices.size == 1:
            break
    best_idx = int(candidate_indices[0])
    tau = float(candidates[best_idx])
    metrics = _compute_metrics_for_tau(int(tp[best_idx]), int(fp[best_idx]), int(tn[best_idx]), int(fn[best_idx]))
    record = _build_policy_record(
        policy=policy,
        tau=tau,
        split_name=split_name,
        n_candidates=int(candidates.size),
        tiebreakers=_policy_tiebreakers(policy),
        epoch=epoch,
        degenerate=False,
        notes=notes,
    )
    record_with_metrics = dict(record)
    record_with_metrics["metrics"] = dict(metrics)
    return ThresholdPolicyResult(
        tau=tau,
        record=record_with_metrics,
        metrics=metrics,
        candidates=candidates.tolist(),
    )


def resolve_frozen_sun_threshold(
    thresholds_block: Mapping[str, Any],
    *,
    source_key: str = "primary",
    expected_split_substring: str = "sun_full/val",
    checkpoint_path: Optional[Path] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Return (tau, record) when reusing a frozen SUN validation threshold."""

    if not isinstance(thresholds_block, Mapping):
        raise ValueError("Thresholds block must be a mapping to reuse frozen thresholds")
    candidate = thresholds_block.get(source_key)
    if candidate is None and source_key != "primary":
        candidate = thresholds_block.get("primary")
    if not isinstance(candidate, Mapping):
        available = ", ".join(sorted(k for k in thresholds_block.keys()))
        raise KeyError(
            f"Threshold entry '{source_key}' not found in thresholds block (available: {available})"
        )
    tau = candidate.get("tau")
    if tau is None or not isinstance(tau, (int, float)) or not math.isfinite(float(tau)):
        raise ValueError(f"Frozen threshold entry '{source_key}' does not provide a valid tau")
    source_policy = candidate.get("policy")
    source_split_raw = candidate.get("split")
    source_split = (
        str(source_split_raw).strip()
        if isinstance(source_split_raw, str) and source_split_raw.strip()
        else None
    )
    candidate_notes = candidate.get("notes") if isinstance(candidate, Mapping) else None
    notes: Dict[str, Any] = {}
    if isinstance(candidate_notes, Mapping):
        notes.update({str(key): candidate_notes[key] for key in candidate_notes.keys()})
    if expected_split_substring and source_split:
        if expected_split_substring not in source_split:
            notes["unexpected_source_split"] = source_split
    elif expected_split_substring and source_split is None:
        notes["unexpected_source_split"] = None
    notes["source_policy"] = source_policy
    notes["source_key"] = source_key
    if checkpoint_path is not None:
        notes["source_checkpoint"] = str(checkpoint_path)

    epoch_value = candidate.get("epoch") if isinstance(candidate, Mapping) else None
    if isinstance(epoch_value, (int, np.integer)):
        epoch = int(epoch_value)
    elif isinstance(epoch_value, (float, np.floating)) and math.isfinite(float(epoch_value)):
        epoch = int(epoch_value)
    else:
        epoch = -1

    degenerate = bool(candidate.get("degenerate_val")) if isinstance(candidate, Mapping) else False
    split_value = source_split or (expected_split_substring or None)
    canonical_source_split = expected_split_substring or source_split or None
    if source_split:
        notes.setdefault("source_split_path", source_split)

    record: Dict[str, Any] = {
        "policy": "sun_val_frozen",
        "tau": float(tau),
        "split": split_value,
        "n_candidates": 0,
        "tiebreakers": [],
        "epoch": epoch,
        "degenerate_val": degenerate,
        "notes": notes,
        "source_policy": source_policy,
        "source_split": canonical_source_split,
        "source_key": source_key,
    }
    return float(tau), record

