#!/usr/bin/env python3
"""Sensitivity of the E1 AULC to the budget axis: linear (registered) vs logarithmic.

Post hoc sensitivity analysis, not registered: nothing is trained and registered results are read, never written.

The registered AULC (SAP section 2) is the trapezoid over the linear budget axis (5, 10, 25, 50, 100 %) divided by its
width. The log-axis AULC is the trapezoid over ln(budget) divided by ln(100/5). Both are linear in the per-budget
metrics, so each AULC difference is a contrast with per-budget weights, inferred with the registered code
(``analysis/v3``): paired estimate (mean of the 15 per-run values), t-test on the 5 partition means (``mixed``, variant
``ttest``) and cluster bootstrap of the e1 family (B = 2000, seed 12345). The registered e1 contrasts (H1, H2 and the
secondary ones) are bootstrapped alongside, so the resampling spaces and the draws are those of the registered run.

Contrasts: SSL-colon − {SUP-imnet (H1), SSL-imnet, SSL-imnet-matched (H2)}, AULC of AUPRC and of AUROC, on both axes.
The linear ones are built here with the same weights function applied to x and must reproduce the registered values
(``results_v3/analysis/contrast_summary.csv``) before the log ones are reported.

    PYTHONPATH=. python scripts/v3/aulc_log_axis.py --out results_v3/verification/aulc_log_axis
Run from the repository root in ``ssl4polyp-analysis``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.v3 import mixed  # noqa: E402
from analysis.v3.contrasts import COLON, hypothesis_contrasts, secondary_contrasts  # noqa: E402
from analysis.v3.design import Contrast, bootstrap, build_views, ci95, complete_terms  # noqa: E402
from analysis.v3.loader import load_campaign  # noqa: E402
from analysis.v3.sap import B_BOOT, BOOT_SEED, COMPARATORS, E1_PERCENTS, MODEL_VARIANT, REPO  # noqa: E402
from analysis.v3.tables import to_markdown  # noqa: E402

REG_ROOT = REPO / "checkpoints_v3/campaign/classification"
METRICS = ("auprc", "auroc")
AXES = {"linear": lambda p: float(p), "log": lambda p: math.log(p)}
REGISTERED_ID = {("sup_imnet", "auprc"): "H1", ("ssl_imnet_matched", "auprc"): "H2"}
NAMES = {"sup_imnet": "SUP-imnet", "ssl_imnet": "SSL-imnet", "ssl_imnet_matched": "SSL-imnet-matched"}


def trapezoid_weights(xs: Sequence[float]) -> List[float]:
    """Weights w with Σ w_i y_i = trapezoid(y, x) / (x_max − x_min)."""

    w = [0.0] * len(xs)
    for i in range(len(xs) - 1):
        h = (xs[i + 1] - xs[i]) / 2.0
        w[i] += h
        w[i + 1] += h
    width = xs[-1] - xs[0]
    return [v / width for v in w]


def aulc_contrasts(views, runs) -> List[Contrast]:
    out = []
    for axis, f in AXES.items():
        w = dict(zip(E1_PERCENTS, trapezoid_weights([f(p) for p in E1_PERCENTS])))
        for metric in METRICS:
            for base in COMPARATORS:
                def make(k, s, base=base, metric=metric, w=w):
                    terms = []
                    for p in E1_PERCENTS:
                        terms += [(("e1", p, COLON, k, s, "all"), metric, w[p]), (("e1", p, base, k, s, "all"), metric, -w[p])]
                    return terms

                terms = complete_terms(views, runs, make)
                if len(terms) != len(runs):
                    raise RuntimeError(f"{axis} {metric} {base}: {len(terms)} complete runs of {len(runs)}")
                out.append(Contrast(f"{axis}|{COLON}-{base}|{metric}", "e1",
                                    {"axis": axis, "metric": metric, "base": base, "weights": [w[p] for p in E1_PERCENTS]}, terms))
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
    mine = aulc_contrasts(views, runs)
    hyp, _ = hypothesis_contrasts(views, regimes, partitions, seeds)
    registered = [c for c in hyp.values() if c is not None and c.family == "e1"]
    registered += [c for c in secondary_contrasts(views, regimes, partitions, seeds) if c.family == "e1"]

    point = {vk: views[vk].metrics() for vk in {vk for c in mine for vk in c.views()}}
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in mine}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    members = mine + registered
    reps, counts = bootstrap(views, members, B=B_BOOT, seed=BOOT_SEED, workers=args.workers)
    print(f"[bootstrap] e1: {len(members)} contrasts, units {counts} ({time.time() - t0:.0f}s)")
    rows = []
    for j, c in enumerate(mine):
        f, sp = fits.loc[c.id], mixed.spread(per_run[c.id])
        lo, hi = ci95(reps[:, j])
        m, base = c.meta["metric"], c.meta["base"]
        rows.append({"id": c.id, "axis": c.meta["axis"], "metric": f"AULC({m})", "comparison": f"{COLON} - {base}",
                     "registered_as": REGISTERED_ID.get((base, m), f"aulc|{COLON}-{base}|{m}"),
                     "weights_5_10_25_50_100": " ".join(f"{x:.6f}" for x in c.meta["weights"]),
                     "estimate": sp["mean"], "sd_partition": sp["sd_partition"], "sd_seed": sp["sd_seed"],
                     "n_runs": sp["n_runs"], "method": f["method"], "t_ci_lo": f["ci_lo"], "t_ci_hi": f["ci_hi"],
                     "p_unadjusted": f["p"], "boot_lo": lo, "boot_hi": hi, "boot_B": int(np.isfinite(reps[:, j]).sum())})
    long = pd.DataFrame(rows)

    # check: the linear contrasts built here = registered values (estimate, t-test CI, bootstrap CI, B)
    cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
    check = []
    for _, r in long[long["axis"] == "linear"].iterrows():
        for reg_id in {r["registered_as"], f"aulc|{r['comparison'].replace(' - ', '-')}|{r['metric'][5:-1]}"}:
            ref = cs.loc[reg_id]
            for q, rq in (("estimate", "mean"), ("t_ci_lo", "model_ci_lo"), ("t_ci_hi", "model_ci_hi"),
                          ("boot_lo", "boot_lo"), ("boot_hi", "boot_hi"), ("boot_B", "boot_B")):
                d = abs(float(r[q]) - float(ref[rq]))
                check.append({"contrast": r["id"], "registered_id": reg_id, "quantity": q, "recomputed": r[q],
                              "registered": ref[rq], "abs_diff": d, "exact": d <= 5e-10 * max(1.0, abs(float(ref[rq])))})
    check = pd.DataFrame(check)
    n_exact = int(check["exact"].sum())
    if n_exact != len(check):
        raise RuntimeError(f"linear AULC does not reproduce the registered values: {n_exact}/{len(check)} exact")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "aulc_axis_long.csv", index=False, float_format="%.10g")
    check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
    wide = long[long["axis"] == "linear"][["comparison", "metric", "registered_as"]].reset_index(drop=True)
    for axis in AXES:
        sub = long[long["axis"] == axis].reset_index(drop=True)
        for col in ("estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"):
            wide[f"{axis}_{col}"] = sub[col]
    wide["log_minus_linear"] = wide["log_estimate"] - wide["linear_estimate"]
    wide.to_csv(out / "aulc_linear_vs_log.csv", index=False, float_format="%.10g")
    weights = pd.DataFrame({"budget_percent": E1_PERCENTS,
                            **{f"weight_{a}": trapezoid_weights([f(p) for p in E1_PERCENTS]) for a, f in AXES.items()}})
    weights.to_csv(out / "aulc_weights.csv", index=False, float_format="%.10g")
    info = {"root": str(REG_ROOT.relative_to(REPO)), "partitions": partitions, "seeds": seeds, "B": B_BOOT,
            "boot_seed": BOOT_SEED, "model_variant": MODEL_VARIANT, "n_contrasts": len(mine),
            "n_registered_bootstrapped_alongside": len(registered),
            "registered_check": {"n": len(check), "exact": n_exact, "max_abs_diff": float(check["abs_diff"].max())},
            "elapsed_s": round(time.time() - t0, 1)}
    md = ["# E1 AULC on a linear (registered) vs logarithmic budget axis", "",
          "Post hoc sensitivity analysis, not registered. Estimate = mean of the paired per-run values (5 partitions × "
          "3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345, the registered "
          "draws).", "", to_markdown(weights, "Weights per budget"), to_markdown(wide, "AULC differences"),
          f"Linear AULC recomputed: {n_exact}/{len(check)} quantities equal to results_v3/analysis/contrast_summary.csv "
          f"(max |diff| {info['registered_check']['max_abs_diff']:.3g})."]
    (out / "aulc_linear_vs_log.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
