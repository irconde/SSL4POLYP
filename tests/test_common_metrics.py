from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest  # type: ignore[import]

from ssl4polyp.classification.analysis.common_metrics import (  # type: ignore[import]
    ClusterSet,
    build_cluster_set,
    compute_binary_metrics,
    sample_cluster_ids,
    _clean_text,
    _coerce_float,
    _coerce_int,
)


@dataclass(frozen=True)
class _DummyRecord:
    frame_id: str
    label: int
    center: Optional[str]
    sequence: Optional[str]
    case: Optional[str]


def test_clean_and_coerce_helpers_handle_nan_inputs() -> None:
    assert _clean_text("  abc  ") == "abc"
    assert _clean_text("   ") is None
    assert _clean_text(None) is None

    assert _coerce_float("0.5") == pytest.approx(0.5)
    assert _coerce_float(np.float64(1.2)) == pytest.approx(1.2)
    assert _coerce_float("nan") is None
    assert _coerce_float(np.nan) is None

    assert _coerce_int("7") == 7
    assert _coerce_int(True) == 1
    assert _coerce_int("not-an-int") is None


def test_compute_binary_metrics_empty_and_subset() -> None:
    empty = compute_binary_metrics(np.array([], dtype=float), np.array([], dtype=int), tau=0.5)
    assert empty["count"] == pytest.approx(0.0)
    for key in ("auprc", "f1"):
        assert np.isnan(empty[key])

    probs = np.array([0.9, 0.2, 0.85, 0.1], dtype=float)
    labels = np.array([1, 0, 1, 0], dtype=int)
    metrics = compute_binary_metrics(probs, labels, tau=0.5, metric_keys=("auprc", "f1"))
    assert metrics["count"] == pytest.approx(4.0)
    assert metrics["tp"] == pytest.approx(2.0)
    assert "precision" not in metrics
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_build_cluster_set_uses_custom_keys() -> None:
    records = [
        _DummyRecord(frame_id="f1", label=1, center="A", sequence="s1", case="c1"),
        _DummyRecord(frame_id="f2", label=0, center="A", sequence="s1", case="c1"),
        _DummyRecord(frame_id="f3", label=1, center="A", sequence="s2", case="c1"),
        _DummyRecord(frame_id="f4", label=0, center=None, sequence=None, case="c2"),
    ]
    clusters = build_cluster_set(
        records,
        is_positive=lambda record: record.label == 1,
        record_id=lambda record: record.frame_id,
        positive_key=lambda record: record.center,
        negative_key=lambda record: record.sequence,
    )
    assert clusters.positives == (("f1", "f3"),)
    assert clusters.negatives == (("f2",), ("f4",))


def test_sample_cluster_ids_matches_rng_sequence() -> None:
    clusters = ClusterSet(
        positives=(("p1", "p2"), ("p3",)),
        negatives=(("n1",), ("n2", "n3")),
    )
    seed = 17
    expected_rng = np.random.default_rng(seed)
    expected: list[str] = []
    pos_indices = expected_rng.integers(0, len(clusters.positives), size=len(clusters.positives))
    for idx in pos_indices:
        expected.extend(clusters.positives[idx])
    neg_indices = expected_rng.integers(0, len(clusters.negatives), size=len(clusters.negatives))
    for idx in neg_indices:
        expected.extend(clusters.negatives[idx])

    rng = np.random.default_rng(seed)
    sampled = sample_cluster_ids(clusters, rng)
    assert sampled == expected
    assert Counter(sampled).keys() <= {"p1", "p2", "p3", "n1", "n2", "n3"}
