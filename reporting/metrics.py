"""Reporting-specific metric utilities."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ssl4polyp.classification.analysis.common_metrics import (  # type: ignore[import]
    _coerce_float,
    _coerce_int,
)


def _iter_csv_rows(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _extract_columns(rows: Iterable[dict[str, object]], *, prob_field: str, label_field: str) -> tuple[list[float], list[int]]:
    probabilities: list[float] = []
    labels: list[int] = []
    for row in rows:
        prob = _coerce_float(row.get(prob_field))
        label = _coerce_int(row.get(label_field))
        if prob is None or label is None:
            continue
        probabilities.append(float(prob))
        labels.append(int(label))
    return probabilities, labels


def _binary_cross_entropy(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if not probabilities or not labels:
        return float("nan")
    if len(probabilities) != len(labels):
        raise ValueError("Probability and label sequences must have the same length")
    probs = np.asarray(probabilities, dtype=float)
    labs = np.asarray(labels, dtype=int)
    if probs.size == 0:
        return float("nan")
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0 - eps)
    losses = -(labs * np.log(clipped) + (1 - labs) * np.log(1 - clipped))
    mean_loss = float(np.mean(losses))
    if math.isnan(mean_loss):
        return float("nan")
    return mean_loss


def bce_loss_from_csv(
    csv_path: Path,
    *,
    prob_field: str = "prob",
    label_field: str = "label",
) -> float:
    """Compute binary cross-entropy loss from an outputs CSV file.

    Parameters
    ----------
    csv_path:
        Path to the CSV containing per-frame outputs. The file must include
        probability and label columns.
    prob_field:
        Column name containing the positive-class probability. Defaults to
        ``"prob"`` to match exported evaluation outputs.
    label_field:
        Column name containing the binary ground-truth label. Defaults to
        ``"label"``.

    Returns
    -------
    float
        The mean binary cross-entropy loss computed across all valid rows. If
        the CSV does not contain any valid samples the function returns NaN.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Outputs CSV not found: {path}")
    probabilities, labels = _extract_columns(
        _iter_csv_rows(path), prob_field=prob_field, label_field=label_field
    )
    return _binary_cross_entropy(probabilities, labels)


__all__ = ["bce_loss_from_csv"]
