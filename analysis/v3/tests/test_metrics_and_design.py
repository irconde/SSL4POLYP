"""Metrics come from binary.py; contrasts, AULC and the bootstrap behave as specified."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.v3.design import Contrast, View, aulc_weights, bootstrap, build_views
from analysis.v3.loader import load_campaign, load_frames
from analysis.v3.metrics import BINARY_PATH, binary
from analysis.v3.sap import E1_PERCENTS


def test_binary_module_is_the_campaign_file():
    assert BINARY_PATH.name == "binary.py" and "classification/metrics" in str(BINARY_PATH)
    assert binary.ECE_BINS == 15 and binary.DEGENERATE_THRESHOLD == 0.10


def test_view_metrics_equal_binary_on_the_csv(sim_full):
    out, truth = sim_full
    c = load_campaign(truth["roots"])
    views = build_views(c)
    run = c.get("e4", 100, "sup_imnet", 2, 29)
    df = pd.read_csv(run.frames["zeroshot"].path)
    for arm in ("A", "B"):
        v = views[("e4", 100, "sup_imnet", 2, 29, arm)]
        expected = binary.metrics_at(df["prob"].to_numpy(), df["label"].to_numpy(), run.tau[arm])
        got = v.metrics()
        for k in ("auroc", "auprc", "f1", "precision", "recall", "specificity", "balanced_accuracy", "tp", "fp", "tn", "fn"):
            assert got[k] == expected[k], k
        assert got["brier"] == binary.brier_score(df["prob"], df["label"])
        assert got["ece"] == binary.expected_calibration_error(df["prob"], df["label"], 15)
    # pred column written at τ_A
    assert np.array_equal(df["pred"].to_numpy(), (df["prob"].to_numpy() >= run.tau["A"]).astype(int))


def test_e2_strata_and_negative_source_views(sim_full):
    out, truth = sim_full
    c = load_campaign(truth["roots"])
    views = build_views(c)
    f = views[("e2", None, "ssl_colon", 3, 47, "flat")]
    y = f.frames.labels[f.rows]
    morph = f.frames.col("morphology")[f.rows]
    assert set(morph[y == 1]) == {"flat"} and (y == 0).sum() == 495
    iso = views[("neg", 50, "ssl_colon", 3, 47, "isolated")]
    seq = views[("neg", 50, "ssl_colon", 3, 47, "sequence")]
    clean = set(c.get("e3", None, "ssl_colon", 3, 47).frames["test"].col("frame_id"))
    for v, src in ((iso, "isolated"), (seq, "sequence")):
        pos = v.rows[v.frames.labels[v.rows] == 1]
        neg = v.rows[v.frames.labels[v.rows] == 0]
        assert set(v.frames.col("frame_id")[pos]) <= clean
        assert set(v.frames.col("negative_source")[neg]) == {src}
    assert np.array_equal(iso.rows[iso.frames.labels[iso.rows] == 1], seq.rows[seq.frames.labels[seq.rows] == 1])


def test_aulc_weights_equal_normalised_trapezoid():
    rng = np.random.default_rng(0)
    y = rng.uniform(size=len(E1_PERCENTS))
    trap = getattr(np, "trapezoid", None) or np.trapz
    assert np.isclose(np.dot(aulc_weights(E1_PERCENTS), y), trap(y, E1_PERCENTS) / (100 - 5))


def _toy_view(key, probs, labels, cases, tau=0.5, group=("sun", 1)):
    n = len(labels)
    text = {c: np.array([""] * n, dtype=object) for c in ("frame_id", "case_id", "sequence_id", "center_id", "origin", "morphology", "negative_source")}
    text["case_id"] = np.array(cases, dtype=object)
    from analysis.v3.loader import Frames

    f = Frames(path=None, probs=np.asarray(probs, float), labels=np.asarray(labels), text=text)
    return View(key, f, tau, np.arange(n), "case", group, per_polyp=True)


def test_per_polyp_counts_bootstrap_copies_as_distinct_cases():
    v = _toy_view("v", [0.9, 0.9, 0.1, 0.2], [1, 1, 0, 0], ["P1", "P1", "N1", "N1"])
    rows = np.array([0, 1, 0, 1, 2, 3])          # P1 drawn twice, N1 once
    copies = np.array([0, 0, 1, 1, 2, 2])
    m = v.metrics(rows, copies)
    assert m["n_pos_cases"] == 2 and m["per_polyp_sensitivity_ge1"] == 1.0


def test_bootstrap_is_reproducible_and_independent_of_workers():
    rng = np.random.default_rng(1)
    views = {}
    contrasts_terms = {}
    for s in (13, 29, 47):
        cases = [f"P{i}" for i in range(6) for _ in range(5)] + [f"N{i}" for i in range(6) for _ in range(5)]
        labels = [1] * 30 + [0] * 30
        for g, shift in (("a", 1.0), ("b", 0.5)):
            probs = 1 / (1 + np.exp(-(np.array(labels) * shift + rng.normal(0, 1, 60))))
            views[(g, s)] = _toy_view((g, s), probs, labels, cases)
        contrasts_terms[(1, s)] = [(("a", s), "auroc", 1.0), (("b", s), "auroc", -1.0)]
    c = Contrast("toy", "x", {}, contrasts_terms)
    r1, _ = bootstrap(views, [c], B=60, seed=12345, workers=1)
    r2, _ = bootstrap(views, [c], B=60, seed=12345, workers=4)
    assert np.array_equal(r1, r2, equal_nan=True)
    r3, _ = bootstrap(views, [c], B=60, seed=1, workers=1)
    assert not np.array_equal(r1, r3)
