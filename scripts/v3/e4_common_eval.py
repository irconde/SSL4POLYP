#!/usr/bin/env python3
"""E4 arm contrasts on the evaluation frames common to the four budgets (50, 100, 200, 500).

Post hoc sensitivity analysis, not registered: nothing is trained and registered results are read, never written.

The E4 evaluation split of each pack (partition k, seed s) depends on the budget: the support frames are removed from
it, and the 500-frame split holds no isolated negatives. Comparisons across budgets therefore mix the support size with
the test composition. Here, for each pack, the common evaluation set is the frames present in the evaluation splits of
the four budgets (and of every regime). Balanced accuracy, sensitivity, specificity and AUROC of the four arms are
recomputed on it from the stored per-frame outputs, with each run's operating thresholds as fixed in the campaign
(``*_last.metrics.json``, ``e4_arms``; nothing is re-estimated).

Contrasts B − A, C − A, D − C and D − A per budget, mean of the four regimes, with the registered inference
(``analysis/v3``): paired estimate (mean of the 15 per-run values), t-test on the 5 partition means (``mixed``, variant
``ttest``) and cluster bootstrap of the E4 units (B = 2000, seed 12345). The contrasts on the original splits use every
E4 view, so the bootstrap spaces and draws are those of the registered e4 family; they must reproduce the registered
values (``results_v3/analysis/contrast_summary.csv``, ``arm|<budget>|pooled|...``; Table S17) before the common-set
ones are reported. The common-set views are subsets of the same units, so they share the draws.

    PYTHONPATH=. python scripts/v3/e4_common_eval.py --out results_v3/verification/e4_common_eval
Run from the repository root in ``ssl4polyp-analysis``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from functools import reduce  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.v3 import mixed  # noqa: E402
from analysis.v3.design import Contrast, View, bootstrap, build_views, ci95, complete_terms  # noqa: E402
from analysis.v3.loader import load_campaign  # noqa: E402
from analysis.v3.sap import B_BOOT, BOOT_SEED, E4_BUDGETS, LABEL, MODEL_VARIANT, REPO, THRESHOLD_FREE  # noqa: E402
from analysis.v3.tables import to_markdown  # noqa: E402

REG_ROOT = REPO / "checkpoints_v3/campaign/classification"
METRICS = ("balanced_accuracy", "recall", "specificity", "auroc")
COMPARISONS = (("B", "A"), ("C", "A"), ("D", "C"), ("D", "A"))
SETS = ("original", "common")


def slot(arm: str, which: str) -> str:
    return arm if which == "original" else f"{arm}@common"


def composition(frames, rows) -> dict:
    lab, src = frames.labels[rows], frames.col("negative_source")[rows]
    return {"n": int(rows.size), "positives": int((lab == 1).sum()),
            "neg_isolated": int(((lab == 0) & (src == "isolated")).sum()),
            "neg_sequence": int(((lab == 0) & (src == "sequence")).sum())}


def arm_contrasts(views, runs, regimes) -> List[Contrast]:
    out, n = [], len(regimes)
    for which in SETS:
        for b in E4_BUDGETS:
            for treat, base in COMPARISONS:
                same_scores = {treat, base} in ({"A", "B"}, {"C", "D"})
                for metric in METRICS:
                    if same_scores and metric in THRESHOLD_FREE:
                        continue  # identical probabilities: the AUROC difference is 0 by construction
                    def make(k, s, b=b, treat=treat, base=base, metric=metric, which=which):
                        return [t for g in regimes for t in [(("e4", b, g, k, s, slot(treat, which)), metric, 1.0 / n),
                                                             (("e4", b, g, k, s, slot(base, which)), metric, -1.0 / n)]]

                    terms = complete_terms(views, runs, make)
                    if len(terms) != len(runs):
                        raise RuntimeError(f"{which} {b} {treat}-{base} {metric}: {len(terms)} complete runs of {len(runs)}")
                    out.append(Contrast(f"{which}|arm|{b}|pooled|{treat}-{base}|{metric}", "e4",
                                        {"set": which, "budget": b, "comparison": f"{treat} - {base}", "metric": metric}, terms))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    t0 = time.time()
    camp = load_campaign([REG_ROOT])
    failed = [x for x in camp.checks if not x["passed"]]
    if failed:
        raise RuntimeError(f"{len(failed)} integrity checks failed, first: {failed[0]}")
    views = build_views(camp)
    regimes, partitions, seeds = camp.regimes(), camp.partitions(), camp.seeds()
    runs = [(k, s) for k in partitions for s in seeds]
    idx = camp.index()

    # 1. common evaluation set per pack: frames in the evaluation split of every budget and regime
    comp_rows, common = [], {}
    for k, s in runs:
        ids = {(b, g): idx[("e4", b, g, k, s)].frames["test"].col("frame_id") for b in E4_BUDGETS for g in regimes}
        common[(k, s)] = reduce(np.intersect1d, ids.values())
        for g in regimes:
            for b in E4_BUDGETS:
                f = idx[("e4", b, g, k, s)].frames["test"]
                for which, rows in (("original", np.arange(len(f))), ("common", np.flatnonzero(np.isin(f.col("frame_id"), common[(k, s)])))):
                    comp_rows.append({"k": k, "seed": s, "regime": g, "budget": b, "set": which, **composition(f, rows)})
    comp = pd.DataFrame(comp_rows)
    # the common set is the same frames for every budget and regime of a pack
    cc = comp[comp["set"] == "common"].groupby(["k", "seed"])[["n", "positives", "neg_isolated", "neg_sequence"]].nunique()
    if not (cc == 1).all().all():
        raise RuntimeError("common set composition differs across budgets or regimes within a pack")
    pack = comp[comp["set"] == "common"].drop_duplicates(["k", "seed"])[["k", "seed", "n", "positives", "neg_isolated", "neg_sequence"]]
    orig = comp[(comp["set"] == "original") & (comp["regime"] == regimes[0])].pivot(index=["k", "seed"], columns="budget", values="n")
    pack = pack.merge(orig.add_prefix("n_split_").reset_index(), on=["k", "seed"])
    for b in E4_BUDGETS:
        pack[f"frac_of_split_{b}"] = pack["n"] / pack[f"n_split_{b}"]

    # 2. common-set views: the registered E4 views restricted to the common frames, same τ per arm and run
    for key in [vk for vk in views if vk[0] == "e4"]:
        v = views[key]
        k, s = key[3], key[4]
        rows = np.flatnonzero(np.isin(v.frames.col("frame_id"), common[(k, s)]))
        ck = key[:5] + (slot(key[5], "common"),)
        views[ck] = View(ck, v.frames, v.tau, rows, v.unit_kind, v.group)
    e4_keys = sorted([vk for vk in views if vk[0] == "e4"], key=repr)
    point = {vk: views[vk].metrics() for vk in e4_keys}
    run_metrics = pd.DataFrame([{"set": "common" if vk[5].endswith("@common") else "original", "budget": vk[1],
                                 "regime": vk[2], "k": vk[3], "seed": vk[4], "arm": vk[5].split("@")[0],
                                 "tau": views[vk].tau, "n": int(views[vk].rows.size),
                                 **{m: point[vk][m] for m in METRICS}} for vk in e4_keys])
    arm_means = run_metrics.groupby(["set", "budget", "arm"])[list(METRICS)].mean().reset_index()

    # 3. contrasts with the registered inference
    mine = arm_contrasts(views, runs, regimes)
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in mine}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    reps, counts = bootstrap(views, mine, B=B_BOOT, seed=BOOT_SEED, workers=args.workers)
    print(f"[bootstrap] e4: {len(mine)} contrasts, units {counts} ({time.time() - t0:.0f}s)")
    rows = []
    for j, c in enumerate(mine):
        f, sp = fits.loc[c.id], mixed.spread(per_run[c.id])
        lo, hi = ci95(reps[:, j])
        rows.append({"id": c.id, **c.meta, "estimate": sp["mean"], "sd_partition": sp["sd_partition"], "sd_seed": sp["sd_seed"],
                     "n_runs": sp["n_runs"], "method": f["method"], "t_ci_lo": f["ci_lo"], "t_ci_hi": f["ci_hi"],
                     "p_unadjusted": f["p"], "boot_lo": lo, "boot_hi": hi, "boot_B": int(np.isfinite(reps[:, j]).sum())})
    long = pd.DataFrame(rows)

    # check: original-split contrasts = registered values (estimate, t-test CI, bootstrap CI, B)
    cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
    check = []
    for _, r in long[long["set"] == "original"].iterrows():
        reg_id = r["id"].split("|", 1)[1]
        ref = cs.loc[reg_id]
        for q, rq in (("estimate", "mean"), ("t_ci_lo", "model_ci_lo"), ("t_ci_hi", "model_ci_hi"),
                      ("boot_lo", "boot_lo"), ("boot_hi", "boot_hi"), ("boot_B", "boot_B")):
            d = abs(float(r[q]) - float(ref[rq]))
            check.append({"contrast": r["id"], "registered_id": reg_id, "quantity": q, "recomputed": r[q],
                          "registered": ref[rq], "abs_diff": d, "exact": d <= 5e-10 * max(1.0, abs(float(ref[rq])))})
    check = pd.DataFrame(check)
    n_exact = int(check["exact"].sum())
    if n_exact != len(check):
        raise RuntimeError(f"original-split contrasts do not reproduce the registered values: {n_exact}/{len(check)} exact")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "composition_by_pack_regime_budget.csv", index=False)
    pack.to_csv(out / "common_set_by_pack.csv", index=False, float_format="%.6g")
    run_metrics.to_csv(out / "run_metrics_original_vs_common.csv", index=False, float_format="%.10g")
    arm_means.to_csv(out / "arm_means.csv", index=False, float_format="%.10g")
    long.to_csv(out / "contrasts_long.csv", index=False, float_format="%.10g")
    check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
    keys = ["budget", "comparison", "metric"]
    wide = long[long["set"] == "original"][keys].reset_index(drop=True)
    for which in SETS:
        sub = long[long["set"] == which].reset_index(drop=True)
        for col in ("estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"):
            wide[f"{which}_{col}"] = sub[col]
    wide["common_minus_original"] = wide["common_estimate"] - wide["original_estimate"]
    wide.to_csv(out / "contrasts_original_vs_common.csv", index=False, float_format="%.10g")
    info = {"root": str(REG_ROOT.relative_to(REPO)), "regimes": regimes, "partitions": partitions, "seeds": seeds,
            "B": B_BOOT, "boot_seed": BOOT_SEED, "model_variant": MODEL_VARIANT, "bootstrap_units": {str(g): v for g, v in counts.items()},
            "n_contrasts": len(mine), "common_set_n": {"min": int(pack["n"].min()), "max": int(pack["n"].max())},
            "common_set_same_across_budgets_and_regimes": True,
            "registered_check": {"n": len(check), "exact": n_exact, "max_abs_diff": float(check["abs_diff"].max())},
            "elapsed_s": round(time.time() - t0, 1)}
    md = ["# E4 arm contrasts: original splits (registered) vs common evaluation set", "",
          "Post hoc sensitivity analysis, not registered. Mean of the four regimes ("
          + ", ".join(LABEL.get(g, g) for g in regimes) + "). Estimate = mean of the paired per-run values (5 partitions × "
          "3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345, the registered draws).",
          "", to_markdown(pack, "Common evaluation set per pack"), to_markdown(arm_means, "Arm means (15 runs × 4 regimes)"),
          to_markdown(wide, "Arm contrasts"),
          f"Original splits recomputed: {n_exact}/{len(check)} quantities equal to results_v3/analysis/contrast_summary.csv "
          f"(max |diff| {info['registered_check']['max_abs_diff']:.3g})."]
    (out / "contrasts_original_vs_common.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
