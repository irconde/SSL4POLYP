"""End-to-end checks on simulated campaigns: H2 handling (pending / available) and checks a–d by Monte Carlo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.v3 import mixed
from analysis.v3.analysis import run_analysis
from analysis.v3.sap import HYPOTHESES

from .conftest import WORKERS


def test_h2_pending_without_matched_regime(sim_no_matched):
    _, truth = sim_no_matched
    res = run_analysis(truth["roots"], B=40, workers=WORKERS, secondary=False, sensitivity=False, log=lambda *_: None)
    h = res.hypotheses.set_index("hypothesis")
    assert h.loc["H2", "status"] == "pending" and "pending" in res.notes["H2"]
    assert (h.drop("H2")["holm_m"] == 6).all()
    assert h.loc["H5", "note"].startswith("mean over 3 regimes")
    assert np.isfinite(h.drop("H2")["estimate"].astype(float)).all()


def test_h2_and_holm_over_seven_with_matched_regime(sim_full):
    _, truth = sim_full
    res = run_analysis(truth["roots"], B=40, workers=WORKERS, secondary=False, sensitivity=False, log=lambda *_: None)
    h = res.hypotheses.set_index("hypothesis")
    assert (h["holm_m"] == 7).all() and (h["status"] == "final").all()
    assert h.loc["H5", "note"] == ""
    # per-run differences of H2 recomputed by hand from the run metrics
    w = {5: 2.5 / 95, 10: 10 / 95, 25: 20 / 95, 50: 37.5 / 95, 100: 25 / 95}
    k, s = 4, 29
    hand = sum(w[p] * (res.point[("e1", p, "ssl_colon", k, s, "all")]["auprc"] - res.point[("e1", p, "ssl_imnet_matched", k, s, "all")]["auprc"])
               for p in w)
    assert np.isclose(res.per_run["H2"][(k, s)], hand)


def test_a_effects_recovered_within_intervals(monte_carlo):
    df, summary = monte_carlo
    eff = df[df["scenario"] == "effects"]
    assert (eff["clipped"] == 0).all()
    per_h = summary["effects"]["per_hypothesis"]
    # model CI (lme4/fallback) covers the true μ: pooled coverage over hypotheses and repetitions
    assert eff["model_covers"].mean() >= 0.85, eff.groupby("hypothesis")["model_covers"].mean().to_dict()
    for h in HYPOTHESES:
        assert abs(per_h[h]["bias"]) < 0.25 * abs(per_h[h]["effect"]), (h, per_h[h])
        assert per_h[h]["confirmation_rate"] >= 0.8, (h, per_h[h])


def test_b_null_scenario_does_not_confirm_systematically(monte_carlo):
    df, summary = monte_carlo
    null = summary["null"]
    assert null["clipped_targets_total"] == 0
    assert null["confirmations_per_test"] <= 0.05, null
    assert null["family_wise_any_confirmed"] <= 0.25, null
    for h, v in null["per_hypothesis"].items():
        assert v["confirmation_rate"] <= 0.2, (h, v)


def test_c_holm_and_double_rule_applied_in_every_repetition(monte_carlo):
    df, _ = monte_carlo
    for (_, _), g in df.groupby(["scenario", "sim_seed"]):
        adj = mixed.holm(dict(zip(g["hypothesis"], g["p"])))
        assert np.allclose([adj[h] for h in g["hypothesis"]], g["p_holm"])
        assert (g["holm_m"] == 7).all()
        rule = (g["p_holm"] < 0.05) & ((g["boot_lo"] > 0) | (g["boot_hi"] < 0))
        assert (rule == g["confirmed"].astype(bool)).all()


def test_d_inference_without_partition_variance(monte_carlo):
    """SAP v3.3 uses the t-test on the partition means in every fit (the v3.2c fallback switch is tested in
    test_mixed_and_rules); with zero partition variance the effects are still recovered and confirmed."""

    df, summary = monte_carlo
    assert set(df["model_method"]) == {"partition_means_t"}
    nop = df[df["scenario"] == "no_partition_variance"]
    assert nop["model_covers"].mean() >= 0.85 and nop["confirmed"].mean() >= 0.8
