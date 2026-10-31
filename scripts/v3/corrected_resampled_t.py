#!/usr/bin/env python3
"""H1–H7 with the Nadeau–Bengio corrected resampled t-test, and the overlap between the SUN partitions.

Post hoc sensitivity analysis, not registered: nothing is trained and registered results are read, never written.

The registered test (SAP v3.3) is the one-sample t-test on the J = 5 partition means: SE² = s²/J, with s² the variance
of the partition means. The corrected resampled t-test of Nadeau & Bengio (2003, Machine Learning 52:239–281) replaces
1/J by (1/J + n_test/n_train), to account for the correlation between resamples that share data. The estimate and the
df (J − 1) do not change. Holm is applied over H1–H7, as registered. The per-run differences are the registered ones
(``results_v3/analysis/paired_differences.csv``); the uncorrected test computed here must reproduce the registered
estimate, SE, CI, p and Holm p before the corrected values are written.

Sizes, in SUN cases (``data_packs/v3/k*/``): test split 30 (15 positive cases and 15 negative virtual cases), train
split 140, E1 subsets 8, 14, 36, 70 and 140. See the README for the ratio used per hypothesis.

    PYTHONPATH=. python scripts/v3/corrected_resampled_t.py --out results_v3/verification/corrected_resampled_t
Run from the repository root in ``ssl4polyp-analysis``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.v3.design import aulc_weights
from analysis.v3.sap import E1_PERCENTS, HYPOTHESES, PARTITIONS, REPO

PACKS = REPO / "data_packs/v3"
J = len(PARTITIONS)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def cases(df: pd.DataFrame, label: str) -> set:
    return set(df.loc[df["label"] == label, "case_id"])


def videos(df: pd.DataFrame) -> set:
    """Source video of the negative virtual cases (``NEG_SUN_caseN_vXXX`` → ``NEG_SUN_caseN``)."""
    return set(df.loc[df["label"] == "0", "case_id"].str.replace(r"_v\d+$", "", regex=True))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m, run, out = len(p), 0.0, {}
    for i, h in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[h]))
        out[h] = run
    return pd.Series(out)[p.index]


def ttest(est: float, s2: float, factor: float) -> dict:
    se = float(np.sqrt(s2 * factor))
    q = stats.t.ppf(0.975, J - 1)
    t = est / se
    return {"se": se, "t": t, "ci_lo": est - q * se, "ci_hi": est + q * se, "p": float(2 * stats.t.sf(abs(t), J - 1))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    # sizes in cases per partition and split
    test = {k: read(PACKS / f"k{k}/base/test.csv") for k in PARTITIONS}
    train = {k: read(PACKS / f"k{k}/base/train.csv") for k in PARTITIONS}
    val = {k: read(PACKS / f"k{k}/base/val.csv") for k in PARTITIONS}
    subsets = {p: {k: read(PACKS / f"k{k}/sun_subsets/sun_p{p:02d}/train.csv") for k in PARTITIONS} for p in E1_PERCENTS}
    size_rows = []
    for k in PARTITIONS:
        row = {"k": k}
        for name, d in (("test", test[k]), ("val", val[k]), ("train", train[k])):
            row[f"{name}_pos_cases"], row[f"{name}_neg_cases"] = len(cases(d, "1")), len(cases(d, "0"))
        for p in E1_PERCENTS:
            row[f"e1_p{p}_train_cases"] = len(cases(subsets[p][k], "1")) + len(cases(subsets[p][k], "0"))
        size_rows.append(row)
    sizes = pd.DataFrame(size_rows)
    n_test = sizes["test_pos_cases"] + sizes["test_neg_cases"]
    n_train = sizes["train_pos_cases"] + sizes["train_neg_cases"]
    if n_test.nunique() != 1 or n_train.nunique() != 1:
        raise RuntimeError("partition sizes differ between partitions")
    n_test, n_train = int(n_test.iloc[0]), int(n_train.iloc[0])
    n_e1 = {p: int(sizes[f"e1_p{p}_train_cases"].iloc[0]) for p in E1_PERCENTS}
    w = dict(zip(E1_PERCENTS, aulc_weights(E1_PERCENTS)))
    r_sun = n_test / n_train
    r_aulc = sum(w[p] * n_test / n_e1[p] for p in E1_PERCENTS)
    ratios = {"H1": r_aulc, "H2": r_aulc, "H3": r_sun, "H4": r_sun, "H5": r_sun, "H6": r_sun, "H7": r_sun}
    ratio_basis = {
        "H1": "E1 AULC: Σ_p w_p · 30 / n_train(p), w = registered AULC weights, n_train = 8, 14, 36, 70, 140 cases",
        "H2": "as H1",
        "H3": "SUN partition: 30 test cases / 140 train cases (sun_full model, E2 evaluates the partition test split)",
        "H4": "SUN partition 30 / 140: resampling of the SUN training data; E3 test (polypgen_clean_test) is fixed and shared",
        "H5": "as H4 (unadapted sun_full models on PolypGen)",
        "H6": "as H4 (arms A and B: unadapted sun_full models on PolypGen)",
        "H7": "as H4 (arm C adapts the sun_full model on a PolypGen support)",
    }

    # pairwise overlap of the test splits (and test of one partition in the training split of the other)
    overlap = []
    for a, b in itertools.combinations(PARTITIONS, 2):
        ta, tb = test[a], test[b]
        overlap.append({"k_a": a, "k_b": b,
                        "shared_test_pos_cases": len(cases(ta, "1") & cases(tb, "1")),
                        "shared_test_neg_cases": len(cases(ta, "0") & cases(tb, "0")),
                        "shared_test_neg_videos": len(videos(ta) & videos(tb)),
                        "test_neg_videos_a": len(videos(ta)), "test_neg_videos_b": len(videos(tb)),
                        "test_a_cases_in_train_b": len((cases(ta, "1") | cases(ta, "0")) & (cases(train[b], "1") | cases(train[b], "0"))),
                        "test_b_cases_in_train_a": len((cases(tb, "1") | cases(tb, "0")) & (cases(train[a], "1") | cases(train[a], "0"))),
                        "shared_train_cases": len((cases(train[a], "1") | cases(train[a], "0")) & (cases(train[b], "1") | cases(train[b], "0")))})
    overlap = pd.DataFrame(overlap)

    # registered per-run differences → partition means → registered and corrected tests
    pdiff = pd.read_csv(REPO / "results_v3/analysis/paired_differences.csv", float_precision="round_trip")
    cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
    reg_h = read(REPO / "results_v3/analysis/tables/table_hypotheses.csv").set_index("hypothesis")
    rows = []
    for h in HYPOTHESES:
        d = pdiff[pdiff["id"] == h]
        means = d.groupby("k")["d"].mean()
        if len(d) != 15 or len(means) != J:
            raise RuntimeError(f"{h}: {len(d)} runs, {len(means)} partitions")
        est, s2 = float(d["d"].mean()), float(means.var(ddof=1))
        reg, cor = ttest(est, s2, 1 / J), ttest(est, s2, 1 / J + ratios[h])
        rows.append({"hypothesis": h, "estimate": est, "sd_partition_means": np.sqrt(s2), "n_test_over_n_train": ratios[h],
                     "ratio_basis": ratio_basis[h], "se_factor": np.sqrt((1 / J + ratios[h]) / (1 / J)),
                     **{f"registered_{k}": v for k, v in reg.items()}, **{f"corrected_{k}": v for k, v in cor.items()},
                     "boot_lo": float(cs.loc[h, "boot_lo"]), "boot_hi": float(cs.loc[h, "boot_hi"])})
    res = pd.DataFrame(rows).set_index("hypothesis")
    res["registered_p_holm"] = holm(res["registered_p"])
    res["corrected_p_holm"] = holm(res["corrected_p"])
    boot_excludes_0 = (res["boot_lo"] > 0) | (res["boot_hi"] < 0)
    res["registered_double_rule"] = (res["registered_p_holm"] < 0.05) & boot_excludes_0
    res["corrected_double_rule"] = (res["corrected_p_holm"] < 0.05) & boot_excludes_0
    # alternative for E1: the ratio of the full training split (100 % budget) for H1 and H2; Holm over H1–H7 again
    alt = res[["estimate", "sd_partition_means", "n_test_over_n_train", "boot_lo", "boot_hi"]].copy()
    alt.loc[["H1", "H2"], "n_test_over_n_train"] = r_sun
    for h in alt.index:
        c = ttest(alt.loc[h, "estimate"], alt.loc[h, "sd_partition_means"] ** 2, 1 / J + alt.loc[h, "n_test_over_n_train"])
        for k, v in c.items():
            alt.loc[h, f"corrected_{k}"] = v
    alt["corrected_p_holm"] = holm(alt["corrected_p"])
    alt["corrected_double_rule"] = (alt["corrected_p_holm"] < 0.05) & boot_excludes_0

    # check: the uncorrected test reproduces the registered values
    check = []
    for h in HYPOTHESES:
        for q, rq, src in (("estimate", "mean", cs), ("registered_se", "model_se", cs), ("registered_ci_lo", "model_ci_lo", cs),
                           ("registered_ci_hi", "model_ci_hi", cs), ("registered_p", "model_p", cs), ("registered_p_holm", "p_holm", None)):
            ref = float(cs.loc[h, rq]) if src is not None else float(reg_h.loc[h, rq])
            diff = abs(float(res.loc[h, q]) - ref)
            check.append({"hypothesis": h, "quantity": q, "recomputed": float(res.loc[h, q]), "registered": ref, "abs_diff": diff,
                          "equal": diff <= 1e-9 * max(1.0, abs(ref))})
    check = pd.DataFrame(check)
    for h in HYPOTHESES:
        if bool(res.loc[h, "registered_double_rule"]) != (reg_h.loc[h, "confirmed"] == "True"):
            raise RuntimeError(f"{h}: double rule differs from the registered verdict")
    if not check["equal"].all():
        raise RuntimeError(f"registered values not reproduced: {int(check['equal'].sum())}/{len(check)}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    res.reset_index().to_csv(out / "corrected_t.csv", index=False, float_format="%.10g")
    alt.reset_index().to_csv(out / "corrected_t_alt_e1_full_train.csv", index=False, float_format="%.10g")
    sizes.to_csv(out / "partition_sizes.csv", index=False)
    overlap.to_csv(out / "partition_overlap.csv", index=False)
    check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
    info = {"J": J, "df": J - 1, "n_test_cases": n_test, "n_train_cases": n_train, "e1_train_cases": n_e1,
            "aulc_weights": w, "ratio_sun": r_sun, "ratio_e1_aulc": r_aulc, "ratio_e1_p100": n_test / n_e1[100],
            "registered_check": {"n": len(check), "equal": int(check["equal"].sum()), "max_abs_diff": float(check["abs_diff"].max())},
            "holm_family": list(HYPOTHESES)}
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    print(res[["estimate", "n_test_over_n_train", "registered_p", "registered_p_holm", "corrected_ci_lo", "corrected_ci_hi",
               "corrected_p", "corrected_p_holm", "registered_double_rule", "corrected_double_rule"]].to_string())
    print("alternative (H1, H2 with 30/140):")
    print(alt[["n_test_over_n_train", "corrected_ci_lo", "corrected_ci_hi", "corrected_p", "corrected_p_holm", "corrected_double_rule"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
