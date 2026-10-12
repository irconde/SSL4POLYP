"""Shared cluster bootstrap used by the E1-E4 reports.

Scheme (identical for every experiment):

* Frames are resampled by *cluster* (SUN case, PolypGen centre or negative sequence).
* Sampling is stratified by class: positive clusters and negative clusters are drawn
  separately, each with replacement and with as many draws as there are clusters.
* One draw is shared by the two models being compared, by every training seed and by
  every budget.  All of them are evaluated on the same test frames, so the test-set
  sampling noise is common; drawing independently per seed would average that noise
  away and make the interval too narrow.
* ``B = 2000`` replicates, seed ``12345`` and percentile intervals unless overridden.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import]

__all__ = [
    "CI_LEVEL",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_RNG_SEED",
    "ClusterSampler",
    "bootstrap_replicates",
    "bootstrap_scalar",
    "leave_one_out_masks",
    "metric_on_rows",
    "percentile_ci",
    "require_aligned",
]

DEFAULT_BOOTSTRAP = 2000
DEFAULT_RNG_SEED = 12345
CI_LEVEL = 0.95

Statistic = Callable[..., Optional[Mapping[str, float]]]


def percentile_ci(samples: Sequence[float], level: float = CI_LEVEL) -> Tuple[float, float]:
    """Return the equal-tailed percentile interval, or ``(nan, nan)`` without samples."""

    if len(samples) == 0:
        return float("nan"), float("nan")
    array = np.asarray(samples, dtype=float)
    tail = 50.0 * (1.0 - level)
    return float(np.percentile(array, tail)), float(np.percentile(array, 100.0 - tail))


def require_aligned(
    reference: Sequence[str],
    others: Mapping[Hashable, Sequence[str]],
    *,
    context: str,
) -> None:
    """Raise if any run does not list exactly the same frames in the same order."""

    reference_list = list(reference)
    for name, frame_ids in others.items():
        if list(frame_ids) != reference_list:
            raise ValueError(
                f"{context}: per-frame outputs of {name!r} do not match the reference test set "
                "(different frames or order); a shared bootstrap draw would be invalid."
            )


class ClusterSampler:
    """Class-stratified cluster sampler over the rows of one test set.

    ``keys[i]`` is the cluster of row ``i``; rows with an empty key become singleton
    clusters.  Positive and negative rows never share a cluster.
    """

    def __init__(self, labels: Sequence[int], keys: Sequence[Optional[str]]) -> None:
        labels_arr = np.asarray(labels, dtype=int)
        if labels_arr.shape[0] != len(keys):
            raise ValueError("labels and keys must have the same length")
        self.labels = labels_arr
        pos: Dict[str, List[int]] = {}
        neg: Dict[str, List[int]] = {}
        for row, (label, key) in enumerate(zip(labels_arr, keys)):
            text = str(key).strip() if key is not None else ""
            name = text if text else f"row::{row}"
            (pos if label == 1 else neg).setdefault(name, []).append(row)
        self.pos_names = tuple(pos)
        self.neg_names = tuple(neg)
        self._pos = [np.asarray(rows, dtype=int) for rows in pos.values()]
        self._neg = [np.asarray(rows, dtype=int) for rows in neg.values()]

    @property
    def n_pos_clusters(self) -> int:
        return len(self._pos)

    @property
    def n_neg_clusters(self) -> int:
        return len(self._neg)

    def all_rows(self) -> np.ndarray:
        return np.arange(self.labels.shape[0], dtype=int)

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        parts: List[np.ndarray] = []
        for clusters in (self._pos, self._neg):
            if not clusters:
                continue
            picks = rng.integers(0, len(clusters), size=len(clusters))
            parts.extend(clusters[int(i)] for i in picks)
        return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def _draw(samplers, rng: np.random.Generator):
    if isinstance(samplers, Mapping):
        return {name: sampler.draw(rng) for name, sampler in samplers.items()}
    return samplers.draw(rng)


def bootstrap_replicates(
    sampler,
    statistic: Statistic,
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
    max_attempts: int = 1,
) -> Dict[str, List[float]]:
    """Evaluate ``statistic(rows)`` on ``bootstrap`` cluster resamples.

    ``sampler`` is one :class:`ClusterSampler` (``statistic`` receives the sampled row
    indices) or a mapping ``name -> ClusterSampler`` for statistics that span several test
    sets, e.g. PolypGen and SUN (``statistic`` receives ``{name: rows}``).
    ``statistic`` returns a mapping
    ``name -> value`` (or ``None`` when the draw is unusable, in which case the draw is
    repeated up to ``max_attempts`` times).  Non-finite values are skipped per name.
    """

    replicates: Dict[str, List[float]] = {}
    if bootstrap <= 0:
        return replicates
    rng = np.random.default_rng(rng_seed)
    for _ in range(int(bootstrap)):
        result: Optional[Mapping[str, float]] = None
        for _attempt in range(max(1, int(max_attempts))):
            result = statistic(_draw(sampler, rng))
            if result is not None:
                break
        if result is None:
            continue
        for name, value in result.items():
            if value is None or not math.isfinite(float(value)):
                continue
            replicates.setdefault(name, []).append(float(value))
    return replicates


def bootstrap_scalar(
    sampler: ClusterSampler,
    statistic: Callable[[np.ndarray], Optional[float]],
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    rng_seed: int = DEFAULT_RNG_SEED,
    max_attempts: int = 1,
) -> List[float]:
    """Convenience wrapper of :func:`bootstrap_replicates` for a single statistic."""

    def _wrapped(rows: np.ndarray) -> Optional[Mapping[str, float]]:
        value = statistic(rows)
        return None if value is None else {"value": float(value)}

    return bootstrap_replicates(
        sampler, _wrapped, bootstrap=bootstrap, rng_seed=rng_seed, max_attempts=max_attempts
    ).get("value", [])


def leave_one_out_masks(keys: Sequence[Optional[str]]) -> Dict[str, np.ndarray]:
    """Return ``{cluster: keep_mask}`` where ``keep_mask`` drops that cluster's rows."""

    array = np.asarray([str(k).strip() if k is not None else "" for k in keys], dtype=object)
    masks: Dict[str, np.ndarray] = {}
    for name in sorted({str(k) for k in array if str(k)}):
        masks[name] = array != name
    return masks


_EPS = 1e-12


def metric_on_rows(
    probs: np.ndarray,
    labels: np.ndarray,
    tau: float,
    rows: np.ndarray,
    metric: str,
) -> Optional[float]:
    """Positive-class ``metric`` on ``rows`` (repeats allowed) of one run.

    Same definitions as :func:`common_metrics.compute_binary_metrics` (recall,
    precision and F1 of the positive class with ``zero_division=0``; AP for AUPRC),
    but computes only what is requested, which keeps large bootstraps tractable.
    Returns ``None`` when the metric is undefined for the sample.
    """

    if rows.size == 0:
        return None
    p = probs[rows]
    y = labels[rows]
    key = metric.lower()
    if key in ("auprc", "auroc"):
        if y.min() == y.max():
            return None
        if key == "auprc":
            return float(average_precision_score(y, p))
        return float(roc_auc_score(y, p))
    if key == "loss":
        clipped = np.clip(p.astype(float), _EPS, 1.0 - _EPS)
        yf = y.astype(float)
        return float(np.mean(-(yf * np.log(clipped) + (1.0 - yf) * np.log(1.0 - clipped))))
    pred = p >= tau
    pos = y == 1
    tp = float(np.sum(pred & pos))
    fp = float(np.sum(pred & ~pos))
    fn = float(np.sum(~pred & pos))
    tn = float(np.sum(~pred & ~pos))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if key == "recall":
        return recall
    if key == "precision":
        return precision
    if key == "f1":
        denom = 2.0 * tp + fp + fn
        return 2.0 * tp / denom if denom > 0 else 0.0
    if key == "balanced_accuracy":
        if (tp + fn) == 0 or (tn + fp) == 0:
            return None
        return 0.5 * (recall + tn / (tn + fp))
    if key == "mcc":
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return None
