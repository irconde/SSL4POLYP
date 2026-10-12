from __future__ import annotations

import numpy as np
import pytest  # type: ignore[import]

from ssl4polyp.classification.analysis.bootstrap_common import (  # type: ignore[import]
    ClusterSampler,
    bootstrap_replicates,
    bootstrap_scalar,
    leave_one_out_masks,
    metric_on_rows,
    percentile_ci,
    require_aligned,
)

# 3 positive clusters (sizes 2, 1, 3) and 2 negative clusters (sizes 2, 2).
LABELS = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
KEYS = ["pA", "pA", "pB", "pC", "pC", "pC", "nA", "nA", "nB", "nB"]


def test_draw_is_stratified_by_class() -> None:
    sampler = ClusterSampler(LABELS, KEYS)
    assert sampler.n_pos_clusters == 3 and sampler.n_neg_clusters == 2
    rng = np.random.default_rng(0)
    for _ in range(50):
        rows = sampler.draw(rng)
        labels = np.asarray(LABELS)[rows]
        # Always 3 positive cluster draws and 2 negative ones, whatever the sample.
        assert (labels == 1).sum() >= 3 and (labels == 0).sum() >= 4
        assert (labels == 0).sum() % 2 == 0  # negative clusters have 2 rows each


def test_replicates_are_deterministic_for_a_seed() -> None:
    sampler = ClusterSampler(LABELS, KEYS)
    stat = lambda rows: float(np.asarray(LABELS)[rows].mean())  # noqa: E731
    first = bootstrap_scalar(sampler, stat, bootstrap=40, rng_seed=7)
    again = bootstrap_scalar(sampler, stat, bootstrap=40, rng_seed=7)
    other = bootstrap_scalar(sampler, stat, bootstrap=40, rng_seed=8)
    assert first == again and first != other and len(first) == 40


def test_shared_draw_gives_identical_rows_to_every_consumer() -> None:
    # One statistic evaluating several "seeds" on the same rows sees the same draw, so a
    # seed-wise delta between two identical models is exactly zero in every replicate.
    sampler = ClusterSampler(LABELS, KEYS)
    probs = {13: np.linspace(0.1, 0.9, 10), 29: np.linspace(0.2, 0.8, 10)}

    def statistic(rows: np.ndarray):
        deltas = [float(p[rows].mean() - p[rows].mean()) for p in probs.values()]
        return {"delta": float(np.mean(deltas))}

    out = bootstrap_replicates(sampler, statistic, bootstrap=20, rng_seed=1)
    assert out["delta"] == [0.0] * 20


def test_unusable_draws_are_retried_then_skipped() -> None:
    sampler = ClusterSampler(LABELS, KEYS)
    calls = {"n": 0}

    def statistic(rows: np.ndarray):
        calls["n"] += 1
        return None

    out = bootstrap_replicates(sampler, statistic, bootstrap=5, rng_seed=0, max_attempts=3)
    assert out == {} and calls["n"] == 15


def test_non_finite_values_are_dropped_per_name() -> None:
    sampler = ClusterSampler(LABELS, KEYS)
    out = bootstrap_replicates(
        sampler, lambda rows: {"a": float("nan"), "b": 1.0}, bootstrap=4, rng_seed=0
    )
    assert "a" not in out and out["b"] == [1.0] * 4


def test_empty_keys_become_singleton_clusters() -> None:
    sampler = ClusterSampler([1, 1, 0], [None, "", "n"])
    assert sampler.n_pos_clusters == 2 and sampler.n_neg_clusters == 1


def test_percentile_ci() -> None:
    lower, upper = percentile_ci(list(range(101)))
    assert lower == pytest.approx(2.5) and upper == pytest.approx(97.5)
    low_nan, up_nan = percentile_ci([])
    assert np.isnan(low_nan) and np.isnan(up_nan)


def test_require_aligned_detects_mismatch() -> None:
    require_aligned(["a", "b"], {"run1": ["a", "b"]}, context="test")
    with pytest.raises(ValueError):
        require_aligned(["a", "b"], {"run2": ["b", "a"]}, context="test")


def test_leave_one_out_masks() -> None:
    masks = leave_one_out_masks(["C1", "C2", "C1", ""])
    assert set(masks) == {"C1", "C2"}
    assert masks["C1"].tolist() == [False, True, False, True]


@pytest.mark.parametrize(
    "metric", ["auprc", "auroc", "recall", "precision", "f1", "balanced_accuracy", "mcc", "loss"]
)
def test_metric_on_rows_matches_compute_binary_metrics(metric: str) -> None:
    from ssl4polyp.classification.analysis.common_metrics import compute_binary_metrics

    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=200)
    probs = np.clip(0.5 * labels + rng.normal(0.25, 0.25, size=200), 0.01, 0.99)
    rows = rng.integers(0, 200, size=200)  # with repeats, as a bootstrap draw
    expected = compute_binary_metrics(probs[rows], labels[rows], 0.45)[metric]
    assert metric_on_rows(probs, labels, 0.45, rows, metric) == pytest.approx(expected)


def test_exp5a_leave_one_center_out_flags_influential_center() -> None:
    from types import SimpleNamespace

    from ssl4polyp.classification.analysis.exp5a_report import (  # type: ignore[import]
        EvalFrame,
        compute_leave_one_center_out,
    )

    def make_run(good: bool):
        frames = {}
        index = 0
        for center in ("C1", "C2", "C3"):
            for label in (1, 1, 0, 0):
                # The good model is right everywhere; the bad one is wrong only in C3.
                correct = good or center != "C3"
                prob = (0.9 if label == 1 else 0.1) if correct else (0.1 if label == 1 else 0.9)
                frames[f"f{index}"] = EvalFrame(
                    frame_id=f"f{index}", prob=prob, label=label, case_id=None,
                    center_id=center, sequence_id=None, origin="polypgen_clean",
                )
                index += 1
        return SimpleNamespace(frames=frames, tau=0.5)

    runs = {"ssl_colon": {13: make_run(True)}, "sup_imnet": {13: make_run(False)}}
    loco = compute_leave_one_center_out(runs, centers=["C1", "C2", "C3"], metrics=("f1",))
    payload = loco["pairwise"]["sup_imnet"]["f1"]
    assert payload["full"] > 0
    assert payload["most_influential"] == "C3"
    assert payload["by_center"]["C3"] == pytest.approx(0.0)  # without C3 the models tie
    assert payload["sign_change"] is False


def test_exp3_flat_leave_one_case_out_needs_multiple_flat_cases() -> None:
    from types import SimpleNamespace

    from ssl4polyp.classification.analysis.exp3_report import (  # type: ignore[import]
        FrameRecord,
        compute_flat_leave_one_case_out,
    )

    def make_run(flat_a_correct: bool):
        rows = []
        for case, correct in (("fA", flat_a_correct), ("fB", True)):
            rows.append(FrameRecord(prob=0.9 if correct else 0.1, label=1, pred=int(correct),
                                    case_id=case, morphology="flat", frame_id=f"{case}_p"))
        for case in ("n1", "n2"):
            rows.append(FrameRecord(prob=0.1, label=0, pred=0, case_id=case, morphology="", frame_id=f"{case}_n"))
        return SimpleNamespace(frames=rows, tau_for_policy=lambda policy: 0.5)

    runs = {"ssl_colon": {13: make_run(True)}, "sup_imnet": {13: make_run(False)}}
    loco = compute_flat_leave_one_case_out(runs, policy="f1_opt_on_val", metrics=("recall",))
    payload = loco["sup_imnet"]["recall"]
    assert payload["full"] == pytest.approx(0.5)
    assert payload["most_influential"] == "fA"
    assert payload["by_case"]["fA"] == pytest.approx(0.0)

    single = {"ssl_colon": {13: SimpleNamespace(
        frames=[FrameRecord(prob=0.9, label=1, pred=1, case_id="only", morphology="flat", frame_id="x")],
        tau_for_policy=lambda policy: 0.5)}}
    assert compute_flat_leave_one_case_out(single, policy="f1_opt_on_val") == {}
