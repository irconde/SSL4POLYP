#!/usr/bin/env python3
"""Sensitivity of the degenerate-predictor count to its threshold (registered: 0.10).

Post hoc sensitivity analysis, not registered: nothing is trained and registered results are read, never written.

SAP v3 section 2 (unchanged up to v3.3): a run is a degenerate predictor when its test specificity or sensitivity at
the operating threshold is below 0.10 (``binary.is_degenerate``, strict inequality). From the registered per-run
metrics (``results_v3/analysis/run_metrics.csv``), runs are counted as degenerate with thresholds 0.05, 0.10 and 0.20
for each condition of Table S18 (E1 per budget, E3, E4 per arm and budget) and each regime. Specificity and
sensitivity are recomputed from the confusion counts (tn / (tn + fp), tp / (tp + fn)) and must equal the stored
values; the 0.10 count must equal the registered ``table_degenerate.csv``. For the runs degenerate at 0.10: minimum,
median and maximum specificity and sensitivity, and which of the two is below the threshold.

    PYTHONPATH=. python scripts/v3/degenerate_threshold.py --out results_v3/verification/degenerate_threshold
Run from the repository root in ``ssl4polyp-analysis``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.v3.sap import LABEL, REPO

RUN_METRICS = REPO / "results_v3/analysis/run_metrics.csv"
REGISTERED = REPO / "results_v3/analysis/tables/table_degenerate.csv"
THRESHOLDS = (0.05, 0.10, 0.20)
FAMILIES = ("e1", "e3", "e4")
KEYS = ["family", "variant", "slot", "regime"]


def condition(r) -> str:
    if r["family"] == "e1":
        return f"E1 {int(r['variant'])} %"
    if r["family"] == "e3":
        return "E3 zero-shot"
    return f"E4 arm {r['slot']}, {int(r['variant'])} frames"


def describe(s: pd.Series, name: str) -> dict:
    if s.empty:
        return {f"{name}_min": np.nan, f"{name}_median": np.nan, f"{name}_max": np.nan}
    return {f"{name}_min": s.min(), f"{name}_median": s.median(), f"{name}_max": s.max()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rm = pd.read_csv(RUN_METRICS, float_precision="round_trip")
    rm = rm[rm["family"].isin(FAMILIES)].copy()
    spec = rm["tn"] / (rm["tn"] + rm["fp"])
    sens = rm["tp"] / (rm["tp"] + rm["fn"])
    recompute = {"specificity": float((spec - rm["specificity"]).abs().max()),
                 "recall": float((sens - rm["recall"]).abs().max())}
    if max(recompute.values()) > 1e-9:
        raise RuntimeError(f"specificity/sensitivity differ from the confusion counts: {recompute}")
    rm["specificity"], rm["sensitivity"] = spec, sens
    for t in THRESHOLDS:
        rm[f"degenerate_{t:.2f}"] = (rm["specificity"] < t) | (rm["sensitivity"] < t)
    if not (rm["degenerate_0.10"] == rm["degenerate"].astype(bool)).all():
        raise RuntimeError("degenerate at 0.10 differs from the registered per-run flag")

    rows = []
    for key, g in rm.groupby(KEYS, sort=True, dropna=False):
        r = dict(zip(KEYS, key))
        d10 = g[g["degenerate_0.10"]]
        rows.append({"condition": condition(r), **r, "regime_label": LABEL.get(r["regime"], r["regime"]), "n_runs": len(g),
                     **{f"n_degenerate_{t:.2f}": int(g[f"degenerate_{t:.2f}"].sum()) for t in THRESHOLDS},
                     "n_low_specificity_0.10": int((d10["specificity"] < 0.10).sum()),
                     "n_low_sensitivity_0.10": int((d10["sensitivity"] < 0.10).sum()),
                     **describe(d10["specificity"], "deg010_specificity"), **describe(d10["sensitivity"], "deg010_sensitivity")})
    table = pd.DataFrame(rows)
    order = {"e1": 0, "e3": 1, "e4": 2}
    table = table.sort_values(["family", "variant", "slot", "regime"], key=lambda s: s.map(order) if s.name == "family" else s)

    reg = pd.read_csv(REGISTERED)
    reg = reg[reg["experiment"].isin(FAMILIES)].rename(columns={"experiment": "family"})
    m = table.merge(reg[KEYS + ["n_runs", "n_degenerate"]], on=KEYS, how="outer", suffixes=("", "_registered"), indicator=True)
    mismatch = m[(m["_merge"] != "both") | (m["n_degenerate_0.10"] != m["n_degenerate"]) | (m["n_runs"] != m["n_runs_registered"])]
    if not mismatch.empty:
        raise RuntimeError(f"0.10 counts differ from table_degenerate.csv in {len(mismatch)} conditions")

    by_exp = []
    for fam, g in rm.groupby("family"):
        for regime, gg in g.groupby("regime"):
            by_exp.append({"family": fam, "regime": regime, "n_runs": len(gg),
                           **{f"n_degenerate_{t:.2f}": int(gg[f"degenerate_{t:.2f}"].sum()) for t in THRESHOLDS}})
    by_exp = pd.DataFrame(by_exp)
    deg_runs = rm[rm["degenerate_0.10"]].copy()
    deg_runs["condition"] = deg_runs.apply(condition, axis=1)
    deg_runs = deg_runs[["condition"] + KEYS + ["k", "seed", "tau", "tp", "fp", "tn", "fn", "specificity", "sensitivity",
                                                 "degenerate_0.05", "degenerate_0.10", "degenerate_0.20"]]
    summary_010 = {name: describe(deg_runs[name], name) for name in ("specificity", "sensitivity")}

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "degenerate_by_condition.csv", index=False, float_format="%.10g")
    by_exp.to_csv(out / "degenerate_by_experiment.csv", index=False)
    deg_runs.to_csv(out / "degenerate_runs_0.10.csv", index=False, float_format="%.10g")
    info = {"source": str(RUN_METRICS.relative_to(REPO)), "thresholds": THRESHOLDS, "inequality": "strict (<)",
            "n_runs": len(rm), "n_conditions": len(table),
            "totals": {f"{t:.2f}": int(rm[f"degenerate_{t:.2f}"].sum()) for t in THRESHOLDS},
            "recomputed_from_counts_max_abs_diff": recompute, "registered_0.10_counts_equal": True,
            "degenerate_0.10_runs": {"n": len(deg_runs), **{k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary_010.items()}}}
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
