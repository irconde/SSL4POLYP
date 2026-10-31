#!/usr/bin/env python3
"""E4 arm contrasts without degenerate predictors (a) and with seed medians (b).

Post hoc sensitivity analysis, not registered: nothing is trained and registered results are read, never written.

Contrasts (balanced accuracy): H6 (B − A) and H7 (C − A), mean of the four budgets and the four regimes, and B − A, C − A,
D − C, D − A per budget, mean of the regimes. Each is, per run (partition k, seed s), a mean of paired arm differences
over its pairs (regime, budget) — the registered definition (``analysis/v3/contrasts.py``).

* (a) **exclusion**: a pair is left out when either of its two arms is a degenerate predictor with the registered rule
  (specificity or sensitivity < 0.10 at the operating threshold, ``binary.is_degenerate``; the per-view flag of
  ``results_v3/analysis/run_metrics.csv``). The per-run value is the mean over the remaining pairs; a run with no pair
  left is dropped. The excluded pairs are fixed from the point estimates and kept fixed in the bootstrap replicates.
* (b) **seed median**: nothing excluded; the value of each partition is the median of its three per-run values instead
  of their mean. Estimate = mean of the five partition medians; t-test on the partition medians.

Inference as registered (``analysis/v3``): t-test on the partition values (``mixed``, variant ``ttest``) and cluster
bootstrap of the E4 units (B = 2000, seed 12345). Every contrast is built as 15 single-run contrasts over all the E4
views, so the bootstrap space and draws are those of the registered e4 family, and each replicate is aggregated as in
(registered), (a) or (b); a replicate is undefined (NaN, discarded) when any of the runs it averages is. The registered
aggregation must reproduce ``results_v3/analysis/contrast_summary.csv`` before anything is written.

    PYTHONPATH=. python scripts/v3/degenerate_sensitivity.py --out results_v3/verification/degenerate_sensitivity
Run from the repository root in ``ssl4polyp-analysis``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.v3 import mixed  # noqa: E402
from analysis.v3.design import Contrast, Space, bootstrap, build_views, ci95  # noqa: E402
from analysis.v3.loader import load_campaign  # noqa: E402
from analysis.v3.sap import B_BOOT, BOOT_SEED, E4_BUDGETS, MODEL_VARIANT, REPO  # noqa: E402
from analysis.v3.tables import to_markdown  # noqa: E402

REG_ROOT = REPO / "checkpoints_v3/campaign/classification"
METRIC = "balanced_accuracy"
PER_BUDGET = (("B", "A"), ("C", "A"), ("D", "C"), ("D", "A"))
VARIANTS = ("registered", "a_exclusion", "b_seed_median")


def definitions() -> List[dict]:
    out = [{"id": "H6", "comparison": "B - A", "budget": "mean of budgets", "treat": "B", "base": "A", "budgets": E4_BUDGETS},
           {"id": "H7", "comparison": "C - A", "budget": "mean of budgets", "treat": "C", "base": "A", "budgets": E4_BUDGETS}]
    for b in E4_BUDGETS:
        for t, a in PER_BUDGET:
            out.append({"id": f"arm|{b}|pooled|{t}-{a}|{METRIC}", "comparison": f"{t} - {a}", "budget": b, "treat": t, "base": a,
                        "budgets": (b,)})
    return out


def run_terms(d: dict, regimes, k: int, s: int, excluded: set) -> List[Tuple]:
    pairs = [(g, b) for g in regimes for b in d["budgets"] if (g, b, k, s) not in excluded]
    n = len(pairs)
    return [t for g, b in pairs for t in [(("e4", b, g, k, s, d["treat"]), METRIC, 1.0 / n),
                                          (("e4", b, g, k, s, d["base"]), METRIC, -1.0 / n)]]


def aggregate(values: Dict[Tuple[int, int], np.ndarray], how: str) -> np.ndarray:
    """values: run → array over replicates; mean over runs, or mean over partitions of the seed median."""
    runs = sorted(values)
    mat = np.vstack([values[r] for r in runs])
    if how == "mean":
        return mat.mean(axis=0)
    ks = sorted({k for k, _ in runs})
    return np.vstack([np.median(mat[[i for i, (k, _) in enumerate(runs) if k == kk]], axis=0) for kk in ks]).mean(axis=0)


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
    e4 = sorted([vk for vk in views if vk[0] == "e4"], key=repr)
    point = {vk: views[vk].metrics() for vk in e4}

    # degenerate flags: recomputed per view, must equal the registered run_metrics.csv
    rm = pd.read_csv(REPO / "results_v3/analysis/run_metrics.csv")
    rm = rm[rm["family"] == "e4"].set_index(["variant", "regime", "k", "seed", "slot"])["degenerate"]
    deg = {vk: bool(point[vk]["degenerate"]) for vk in e4}
    mism = [vk for vk in e4 if deg[vk] != bool(rm.loc[(float(vk[1]), vk[2], vk[3], vk[4], vk[5])])]
    if mism:
        raise RuntimeError(f"degenerate flags differ from run_metrics.csv: {mism[:3]}")

    defs = definitions()
    single: List[Contrast] = []
    excl_rows, index = [], {}
    for d in defs:
        excluded = {(g, b, k, s) for g in regimes for b in d["budgets"] for k, s in runs
                    if deg[("e4", b, g, k, s, d["treat"])] or deg[("e4", b, g, k, s, d["base"])]}
        kept_runs = [(k, s) for k, s in runs if any((g, b, k, s) not in excluded for g in regimes for b in d["budgets"])]
        excl_rows.append({"id": d["id"], "comparison": d["comparison"], "budget": d["budget"],
                          "pairs": len(regimes) * len(d["budgets"]) * len(runs), "pairs_excluded": len(excluded),
                          "pairs_excluded_treat_degenerate": sum(deg[("e4", b, g, k, s, d["treat"])] for g, b, k, s in excluded),
                          "pairs_excluded_base_degenerate": sum(deg[("e4", b, g, k, s, d["base"])] for g, b, k, s in excluded),
                          "runs_dropped": len(runs) - len(kept_runs),
                          **{f"excluded_{g}": sum(1 for e in excluded if e[0] == g) for g in regimes}})
        for var, ex, rr in (("all", set(), runs), ("excl", excluded, kept_runs)):
            for k, s in rr:
                cid = f"{d['id']}#{var}#{k}#{s}"
                single.append(Contrast(cid, "e4", {}, {(k, s): run_terms(d, regimes, k, s, ex)}))
                index.setdefault((d["id"], var), []).append(((k, s), len(single) - 1))
    exclusion = pd.DataFrame(excl_rows)

    # the bootstrap space = the registered e4 space (all E4 views are used)
    needed = {vk for c in single for vk in c.views()}
    if Space([views[vk] for vk in needed]).flat != Space([views[vk] for vk in e4]).flat:
        raise RuntimeError("bootstrap space differs from the registered E4 space")
    point_d = np.array([list(c.per_run(lambda vk, m: point[vk][m]).values())[0] for c in single])
    reps, counts = bootstrap(views, single, B=B_BOOT, seed=BOOT_SEED, workers=args.workers)
    print(f"[bootstrap] e4: {len(single)} single-run contrasts, units {counts} ({time.time() - t0:.0f}s)")

    rows, per_run_out, fits_in = [], [], {}
    for d in defs:
        for variant, var, how in (("registered", "all", "mean"), ("a_exclusion", "excl", "mean"), ("b_seed_median", "all", "median")):
            members = index[(d["id"], var)]
            dvals = {r: float(point_d[j]) for r, j in members}
            for r, v in dvals.items():
                per_run_out.append({"id": d["id"], "variant": variant, "k": r[0], "seed": r[1], "d": v})
            if how == "mean":
                fits_in[f"{d['id']}|{variant}"] = dvals
                est = float(np.mean([dvals[r] for r in sorted(dvals)]))
            else:
                med = pd.Series(dvals).groupby(level=0).median()
                fits_in[f"{d['id']}|{variant}"] = {(k, 0): float(v) for k, v in med.items()}
                est = float(med.mean())
            rep = aggregate({r: reps[:, j] for r, j in members}, how)
            lo, hi = ci95(rep)
            rows.append({"id": d["id"], "comparison": d["comparison"], "budget": d["budget"], "variant": variant,
                         "n_runs": len(dvals), "n_partitions": len({k for k, _ in dvals}), "estimate": est,
                         "boot_lo": lo, "boot_hi": hi, "boot_B": int(np.isfinite(rep).sum())})
    res = pd.DataFrame(rows)
    fits = mixed.fit_many(fits_in, MODEL_VARIANT).set_index("id")
    key = res["id"] + "|" + res["variant"]
    for col, src in (("t_ci_lo", "ci_lo"), ("t_ci_hi", "ci_hi"), ("p_unadjusted", "p"), ("method", "method")):
        res[col] = key.map(fits[src])

    # check: the registered aggregation = contrast_summary.csv (estimate, t-test CI, bootstrap CI, B)
    cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
    check = []
    for _, r in res[res["variant"] == "registered"].iterrows():
        ref = cs.loc[r["id"]]
        for q, rq in (("estimate", "mean"), ("t_ci_lo", "model_ci_lo"), ("t_ci_hi", "model_ci_hi"),
                      ("boot_lo", "boot_lo"), ("boot_hi", "boot_hi"), ("boot_B", "boot_B")):
            diff = abs(float(r[q]) - float(ref[rq]))
            check.append({"id": r["id"], "quantity": q, "recomputed": r[q], "registered": ref[rq], "abs_diff": diff,
                          "exact": diff <= 5e-10 * max(1.0, abs(float(ref[rq])))})
    check = pd.DataFrame(check)
    n_exact = int(check["exact"].sum())
    if n_exact != len(check):
        raise RuntimeError(f"registered values not reproduced: {n_exact}/{len(check)} exact")

    wide = res.pivot(index="id", columns="variant", values="estimate")
    h7 = {"registered": float(wide.loc["H7", "registered"]), "a_exclusion": float(wide.loc["H7", "a_exclusion"])}
    h7["share_of_registered_effect_removed_by_a"] = (h7["registered"] - h7["a_exclusion"]) / h7["registered"]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    order = {d["id"]: i for i, d in enumerate(defs)}
    res = res.sort_values(["id", "variant"], key=lambda s: s.map(order) if s.name == "id" else s.map({v: i for i, v in enumerate(VARIANTS)}))
    res.to_csv(out / "contrasts.csv", index=False, float_format="%.10g")
    exclusion.to_csv(out / "excluded_pairs.csv", index=False)
    pd.DataFrame(per_run_out).to_csv(out / "per_run.csv", index=False, float_format="%.10g")
    check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
    pd.DataFrame([{"family": "e4", "variant": vk[1], "regime": vk[2], "k": vk[3], "seed": vk[4], "arm": vk[5],
                   "specificity": point[vk]["specificity"], "sensitivity": point[vk]["recall"], "balanced_accuracy": point[vk][METRIC]}
                  for vk in e4 if deg[vk]]).to_csv(out / "degenerate_views.csv", index=False, float_format="%.10g")
    info = {"root": str(REG_ROOT.relative_to(REPO)), "regimes": regimes, "partitions": partitions, "seeds": seeds, "B": B_BOOT,
            "boot_seed": BOOT_SEED, "model_variant": MODEL_VARIANT, "bootstrap_units": {str(g): v for g, v in counts.items()},
            "bootstrap_space_equals_registered_e4": True, "degenerate_views": int(sum(deg.values())),
            "registered_check": {"n": len(check), "exact": n_exact, "max_abs_diff": float(check["abs_diff"].max())},
            "H7": h7, "elapsed_s": round(time.time() - t0, 1)}
    show = res[["id", "variant", "n_runs", "estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi", "boot_B"]]
    md = ["# E4 arm contrasts: registered vs (a) degenerate pairs excluded vs (b) seed medians", "",
          "Post hoc sensitivity analysis, not registered. Balanced accuracy, mean of the four regimes.", "",
          to_markdown(exclusion, "Pairs excluded in (a)"), to_markdown(show, "Contrasts"),
          f"Registered aggregation recomputed: {n_exact}/{len(check)} quantities equal to contrast_summary.csv."]
    (out / "contrasts.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
