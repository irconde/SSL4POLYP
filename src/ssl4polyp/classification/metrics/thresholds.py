"""Threshold utilities for binary classification models.

This module centralises helper functions used to derive decision thresholds
from validation logits and to persist/retrieve them alongside checkpoints.
Currently only the Youden's J statistic is implemented which is the policy
used across the classification experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from sklearn.metrics import roc_curve


ThresholdMap = Dict[str, float]


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

