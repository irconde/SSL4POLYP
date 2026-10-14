import sys
from pathlib import Path

import numpy as np
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ssl4polyp.classification.analysis.cluster_bootstrap import aulc, run_bootstrap, unit_rows
from ssl4polyp.classification.analysis.frame_outputs import FrameOutputs, unit_case
from ssl4polyp.classification.metrics import thresholds
from ssl4polyp.classification.metrics.binary import (
    brier_score,
    expected_calibration_error,
    is_degenerate,
    metrics_at,
    per_polyp_sensitivity,
)


def test_metrics_at_positive_class_definitions() -> None:
    probs = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.7])
    labels = np.array([1, 1, 1, 0, 0, 0])
    m = metrics_at(probs, labels, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert not is_degenerate(m)
    assert is_degenerate(metrics_at(probs, labels, 0.0))


def test_calibration_metrics() -> None:
    probs = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    assert brier_score(probs, labels) == 0.0
    assert expected_calibration_error(probs, labels, n_bins=2) == 0.0
    probs = np.full(10, 0.5)
    labels = np.array([1] * 2 + [0] * 8)
    # equal-mass bins of 2 frames: |1-0.5|*0.2 + |0-0.5|*0.8
    assert expected_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.5, abs=1e-12)


def test_per_polyp_sensitivity() -> None:
    probs = np.array([0.9] * 5 + [0.1] * 28 + [0.9] + [0.1] * 32 + [0.9] * 33)
    labels = np.array([1] * 66 + [0] * 33)
    cases = ["A"] * 33 + ["B"] * 33 + ["N"] * 33
    out = per_polyp_sensitivity(probs, labels, cases, tau=0.5)
    assert out["n_pos_cases"] == 2
    assert out["per_polyp_sensitivity_ge1"] == 1.0
    assert out["per_polyp_sensitivity_ge5"] == 0.5


def test_f1_threshold_search_is_exact() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    labels = (rng.random(1000) < scores).astype(int)
    result = thresholds.compute_policy_threshold(
        scores, labels, policy="f1_opt_on_val", split_name="x", epoch=1
    )
    assert len(result.candidates) == np.unique(scores).size
    best = max(metrics_at(scores, labels, t)["f1"] for t in np.unique(scores))
    assert metrics_at(scores, labels, result.tau)["f1"] == pytest.approx(best)


def test_bootstrap_shared_sample_and_point_estimate() -> None:
    labels = np.array([1] * 6 + [0] * 6)
    cases = ["p1"] * 3 + ["p2"] * 3 + ["n1"] * 3 + ["n2"] * 3
    def outputs(probs):
        return FrameOutputs(
            path=Path("x"),
            frame_ids=[str(i) for i in range(12)],
            probs=np.asarray(probs, dtype=float),
            labels=labels,
            preds=np.zeros(12, dtype=int),
            columns={"case_id": cases},
        )
    a = outputs([0.9] * 6 + [0.1] * 6)
    b = outputs([0.6, 0.4, 0.6, 0.6, 0.4, 0.6] + [0.4, 0.6, 0.4, 0.4, 0.4, 0.4])
    runs = {"a": (a, 0.5), "b": (b, 0.5)}
    units = {k: unit_rows(v[0], unit_case) for k, v in runs.items()}
    point, reps, spaces = run_bootstrap(
        runs, units, lambda g: {"d": g["a"]()["auroc"] - g["b"]()["auroc"]}, B=50, seed=1
    )
    assert point["d"] > 0
    assert reps["d"].size == 50
    assert spaces[None].counts() == {"y0": 2, "y1": 2}


def test_aulc_normalised() -> None:
    assert aulc([5, 100], [0.5, 0.5]) == pytest.approx(0.5)
