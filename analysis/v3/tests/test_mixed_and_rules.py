"""Mixed model, fallback rule, Holm and the double confirmation rule (checks c and d)."""
from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis.v3 import mixed
from analysis.v3.analysis import confirm
from analysis.v3.sap import HYPOTHESES


def _table(values):
    return {(k, s): v for (k, s), v in values.items()}


def test_lmer_matches_balanced_anova_satterthwaite():
    rng = np.random.default_rng(3)
    u = rng.normal(0, 0.03, 5)
    d = {(k, s): 0.02 + u[k - 1] + rng.normal(0, 0.01) for k in range(1, 6) for s in (13, 29, 47)}
    fit = mixed.fit_many({"x": d}).iloc[0]
    assert fit["method"] == "lmer" and not fit["singular"] and fit["converged"]
    arr = np.array([[d[(k, s)] for s in (13, 29, 47)] for k in range(1, 6)])
    K, n = arr.shape
    msb = n * arr.mean(axis=1).var(ddof=1)
    se = np.sqrt(msb / (K * n))
    assert np.isclose(fit["estimate"], arr.mean(), atol=1e-10)
    assert np.isclose(fit["se"], se, rtol=1e-4)
    assert np.isclose(fit["df"], K - 1, rtol=1e-3)
    assert np.isclose(fit["p"], 2 * stats.t.sf(abs(arr.mean() / se), K - 1), rtol=1e-3)
    q = stats.t.ppf(0.975, fit["df"])
    assert np.isclose(fit["ci_lo"], fit["estimate"] - q * fit["se"]) and np.isclose(fit["ci_hi"], fit["estimate"] + q * fit["se"])
    w = np.sum((arr - arr.mean(axis=1, keepdims=True)) ** 2) / (K * (n - 1))
    assert np.isclose(fit["var_seed"], w, rtol=1e-4)
    assert np.isclose(fit["var_partition"], (msb - w) / n, rtol=1e-3)


def test_fallback_when_partition_variance_is_zero():
    """Equal partition means (and large seed noise) => REML σ²_partition = 0 => singular => t-test on K means."""

    dev = (-0.02, 0.0, 0.02)
    offsets = (0.0005, -0.0005, 0.0003, -0.0003, 0.0)
    d = {(k, s): 0.01 + offsets[k - 1] + dev[j] for k in range(1, 6) for j, s in enumerate((13, 29, 47))}
    fit = mixed.fit_many({"x": d}).iloc[0]
    assert fit["singular"] in (True, "TRUE") or bool(fit["singular"])
    assert fit["method"] == "fallback_t"
    means = [np.mean([d[(k, s)] for s in (13, 29, 47)]) for k in range(1, 6)]
    tt = stats.ttest_1samp(means, 0.0)
    assert np.isclose(fit["estimate"], np.mean(means)) and np.isclose(fit["df"], 4)
    assert np.isclose(fit["p"], tt.pvalue, rtol=1e-8)
    lo, hi = stats.t.interval(0.95, 4, loc=np.mean(means), scale=stats.sem(means))
    assert np.isclose(fit["ci_lo"], lo) and np.isclose(fit["ci_hi"], hi)


def test_single_partition_uses_seed_t_test_and_constant_data_is_flagged():
    d1 = {(1, 13): 0.01, (1, 29): 0.03, (1, 47): 0.02}
    const = {(k, s): 0.0 for k in range(1, 6) for s in (13, 29, 47)}
    fits = mixed.fit_many({"k1": d1, "const": const}).set_index("id")
    assert fits.loc["k1", "method"] == "t_seeds_K1"
    assert np.isclose(fits.loc["k1", "p"], stats.ttest_1samp(list(d1.values()), 0).pvalue)
    assert fits.loc["const", "method"] == "constant" and np.isnan(fits.loc["const", "p"])


def test_holm_matches_r_p_adjust():
    p = {f"H{i}": v for i, v in enumerate([0.001, 0.04, 0.03, 0.2, 0.012, 0.0049, 0.5], start=1)}
    ours = mixed.holm(p)
    expr = "cat(sprintf('%.15g', p.adjust(c(" + ",".join(repr(v) for v in p.values()) + "), method='holm')), sep=' ')"
    r = subprocess.run([mixed.rscript(), "-e", expr], capture_output=True, text=True, check=True).stdout.split()
    assert np.allclose([ours[k] for k in p], [float(x) for x in r], rtol=1e-12)


def _summary(rows):
    base = {"mean": 0.0, "sd_partition": 0.01, "sd_seed": 0.01, "n_runs": 15, "n_partitions": 5, "model_method": "lmer",
            "model_estimate": 0.0, "model_se": 0.01, "model_df": 4.0, "model_ci_lo": -0.01, "model_ci_hi": 0.01,
            "var_partition": 1e-4, "var_seed": 1e-4, "singular": False, "boot_B": 2000}
    return pd.DataFrame([{**base, "id": h, "model_p": p, "boot_lo": lo, "boot_hi": hi} for h, p, lo, hi in rows])


def test_double_rule_and_holm_family_of_seven():
    rows = [
        ("H1", 0.001, 0.01, 0.05),    # Holm 0.007, CI excludes 0  -> confirmed
        ("H2", 0.002, -0.01, 0.04),   # Holm 0.012, CI includes 0  -> not (bootstrap condition)
        ("H3", 0.009, 0.02, 0.06),    # Holm 0.045, CI excludes 0  -> confirmed
        ("H4", 0.011, 0.01, 0.03),    # Holm max(4 × 0.011, 0.045) = 0.045 (monotone step-down) -> confirmed
        ("H5", 0.02, -0.08, -0.02),   # Holm 0.06 -> not (Holm condition) although p < 0.05 and CI excludes 0
        ("H6", 0.3, 0.01, 0.02),      # not
        ("H7", 0.04, 0.01, 0.02),     # Holm 0.08 -> not
    ]
    hyp = {h: object() for h in HYPOTHESES}
    out = confirm(_summary(rows), hyp, {}).set_index("hypothesis")
    assert (out["holm_m"] == 7).all() and (out["status"] == "final").all()
    expected_holm = {"H1": 0.007, "H2": 0.012, "H3": 0.045, "H4": 0.045, "H5": 0.06, "H6": 0.3, "H7": 0.08}
    for h, v in expected_holm.items():
        assert np.isclose(out.loc[h, "p_holm"], v), h
    assert out["confirmed"].to_dict() == {"H1": True, "H2": False, "H3": True, "H4": True, "H5": False, "H6": False, "H7": False}


def test_holm_over_six_while_h2_is_pending():
    rows = [(h, 0.008, 0.01, 0.02) for h in HYPOTHESES if h != "H2"]
    hyp = {h: (None if h == "H2" else object()) for h in HYPOTHESES}
    out = confirm(_summary(rows), hyp, {"H2": "pending: no SSL-imnet-matched results found"}).set_index("hypothesis")
    assert out.loc["H2", "status"] == "pending" and not out.loc["H2", "confirmed"]
    assert (out.drop("H2")["holm_m"] == 6).all()
    assert out.drop("H2")["status"].str.startswith("provisional").all()
    assert np.allclose(out.drop("H2")["p_holm"], 6 * 0.008)
