#!/usr/bin/env python3
"""Drift report: campaign v3 results vs the previous manuscript and vs ``analysis_v2`` (spec v3.1 section 6).

Run after ``run_analysis``. Inputs: ``<out>/run_metrics.csv``, ``<out>/contrast_summary.csv`` and
``<out>/paired_differences.csv`` (v3); ``analysis_v2/manuscript_values.py`` (values quoted in the previous
manuscript, transcribed from ``manuscript_results_values.md``); ``analysis_v2/paired_differences.csv`` and the
``analysis_v2/table*.csv`` levels (v2 recomputation, one partition, seeds 13/29/47, three regimes).

For every value: v3 mean over all paired runs (K = 5 partitions × 3 seeds) and over partition k = 1 only (the
partition of v2), the difference with the reference, and for paired differences the sign and "significance"
(95 % CI excluding 0: the v2/manuscript bootstrap CI vs the v3 bootstrap CI and the v3 model CI). Changes of sign
or of significance are flagged, as are level differences larger than 0.02. E4 AP levels are not compared (their
evaluation splits differ between packs: SAP section 6, revision c); E4 AP paired differences within a pack are.

    PYTHONPATH=. python -m analysis.v3.drift --out results_v3/analysis
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .sap import REPO

V2 = REPO / "analysis_v2"
LEVEL_FLAG = 0.02
PCT = {"p5": 5, "p10": 10, "p25": 25, "p50": 50, "p100": 100}
BUD = {"s50": 50, "s100": 100, "s200": 200, "s500": 500}


def _manuscript():
    spec = importlib.util.spec_from_file_location("manuscript_values_v2", V2 / "manuscript_values.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _excl(lo, hi) -> Optional[bool]:
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)) or (isinstance(hi, float) and math.isnan(hi)):
        return None
    return bool(lo > 0 or hi < 0)


def _sign(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(np.sign(x))


class V3:
    def __init__(self, out: Path):
        self.rm = pd.read_csv(out / "run_metrics.csv")
        self.cs = pd.read_csv(out / "contrast_summary.csv").set_index("id")
        self.pd = pd.read_csv(out / "paired_differences.csv")

    def level(self, family: str, variant, slot: str, regime: str, metric: str) -> Tuple[float, float, float, int]:
        rm = self.rm[(self.rm["family"] == family) & (self.rm["slot"] == slot) & (self.rm["regime"] == regime)]
        rm = rm[rm["variant"].isna()] if variant is None else rm[rm["variant"] == variant]
        if rm.empty or metric not in rm:
            return (float("nan"),) * 3 + (0,)
        k1 = rm[rm["k"] == 1][metric]
        return float(rm[metric].mean()), float(rm[metric].std(ddof=1)), float(k1.mean()) if len(k1) else float("nan"), len(rm)

    def contrast(self, cid: str) -> Optional[Dict[str, float]]:
        if cid not in self.cs.index:
            return None
        r = self.cs.loc[cid]
        d = self.pd[self.pd["id"] == cid]
        k1 = d[d["k"] == 1]["d"]
        return {"mean": float(r["mean"]), "k1_mean": float(k1.mean()) if len(k1) else float("nan"),
                "model_lo": float(r["model_ci_lo"]), "model_hi": float(r["model_ci_hi"]),
                "boot_lo": float(r["boot_lo"]), "boot_hi": float(r["boot_hi"])}


def cmp_id(fam: str, variant, slot: str, base: str, metric: str) -> str:
    return f"cmp|{fam}|{variant}|{slot}|ssl_colon-{base}|{metric}"


def paired_row(source: str, label: str, ref_est, ref_lo, ref_hi, v3: Optional[Dict[str, float]], cid: str) -> dict:
    row = {"source": source, "quantity": label, "v3_id": cid, "ref_estimate": ref_est, "ref_ci_lo": ref_lo, "ref_ci_hi": ref_hi}
    if v3 is None:
        row["status"] = "no v3 counterpart"
        return row
    row.update(v3_estimate=v3["mean"], v3_k1_estimate=v3["k1_mean"], v3_boot_lo=v3["boot_lo"], v3_boot_hi=v3["boot_hi"],
               v3_model_lo=v3["model_lo"], v3_model_hi=v3["model_hi"], diff=v3["mean"] - ref_est if ref_est is not None else float("nan"))
    s_ref, s_new = _sign(ref_est), _sign(v3["mean"])
    e_ref, e_boot, e_model = _excl(ref_lo, ref_hi), _excl(v3["boot_lo"], v3["boot_hi"]), _excl(v3["model_lo"], v3["model_hi"])
    row["sign_change"] = bool(s_ref is not None and s_new is not None and s_ref != 0 and s_new != 0 and s_ref != s_new)
    row["ref_ci_excludes_0"] = e_ref
    row["v3_boot_ci_excludes_0"] = e_boot
    row["v3_model_ci_excludes_0"] = e_model
    row["significance_change_boot"] = None if e_ref is None or e_boot is None else bool(e_ref != e_boot)
    row["significance_change_model"] = None if e_ref is None or e_model is None else bool(e_ref != e_model)
    row["status"] = "compared"
    return row


def level_row(source: str, label: str, ref_mean, ref_sd, v3: Tuple[float, float, float, int], note: str = "") -> dict:
    mean, sd, k1, n = v3
    diff = mean - ref_mean if ref_mean is not None else float("nan")
    return {"source": source, "quantity": label, "ref_mean": ref_mean, "ref_sd": ref_sd, "v3_mean": mean, "v3_sd": sd,
            "v3_k1_mean": k1, "v3_n_runs": n, "diff": diff, "diff_k1": k1 - ref_mean if ref_mean is not None else float("nan"),
            "flag_abs_diff_gt_0.02": bool(abs(diff) > LEVEL_FLAG) if not math.isnan(diff) else None, "note": note}


# --------------------------------------------------------------------------- manuscript
def vs_manuscript(v3: V3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    M = _manuscript()
    levels: List[dict] = []
    for (cond, reg), vals in M.TABLE3.items():
        for metric, v in zip(M.M3, vals):
            m, s = v if isinstance(v, tuple) else (v, None)
            levels.append(level_row("manuscript Table 3", f"E1 {cond} {reg} {metric}", m, s, v3.level("e1", PCT[cond], "all", reg, metric)))
    for (stratum, reg), vals in M.TABLE5.items():
        for metric, v in zip(M.M3, vals):
            m, s = v if isinstance(v, tuple) else (v, None)
            levels.append(level_row("manuscript Table 5", f"E2 {stratum} {reg} {metric}", m, s, v3.level("e2", None, stratum, reg, metric)))
    for reg, vals in M.TABLE6.items():
        for metric, v in zip(M.M6, vals):
            m, s = v if isinstance(v, tuple) else (v, None)
            levels.append(level_row("manuscript Table 6", f"E3 {reg} {metric}", m, s, v3.level("e3", None, "all", reg, metric)))
    for reg, vals in M.TABLE7.items():
        for src, (m, s) in zip(("isolated", "sequence"), vals):
            levels.append(level_row("manuscript Table 7", f"negative source {src} {reg} auroc", m, s, v3.level("neg", 50, src, reg, "auroc")))
    for (b, reg), vals in M.TABLE8.items():
        for metric, (m, s) in zip(M.M8, vals):
            if metric == "auprc":
                levels.append(level_row("manuscript Table 8", f"E4 {b} adapted {reg} auprc", m, s, (float("nan"),) * 3 + (0,),
                                        "not compared: E4 AP depends on the pack (SAP 6, rev. c)"))
                continue
            levels.append(level_row("manuscript Table 8", f"E4 {b} adapted (arm C) {reg} {metric}", m, s, v3.level("e4", BUD[b], "C", reg, metric)))
    for (b, reg), (az, aa, bz, ba) in M.TABLE9.items():
        for label, val, arm, metric in (("zero-shot auroc", az, "A", "auroc"), ("adapted auroc", aa, "C", "auroc"),
                                        ("zero-shot balanced_accuracy", bz, "A", "balanced_accuracy"),
                                        ("adapted balanced_accuracy", ba, "C", "balanced_accuracy")):
            levels.append(level_row("manuscript Table 9", f"E4 {b} {label} (arm {arm}) {reg}", val, None, v3.level("e4", BUD[b], arm, reg, metric)))
    paired: List[dict] = []
    for metric, by in M.TABLE4.items():
        for base, (ac, ab, delta, lo, hi) in by.items():
            paired.append(paired_row("manuscript Table 4", f"E1 AULC({metric}) ssl_colon - {base}", delta, lo, hi,
                                     v3.contrast(f"aulc|ssl_colon-{base}|{metric}"), f"aulc|ssl_colon-{base}|{metric}"))
            for reg, val in (("ssl_colon", ac), (base, ab)):
                c = v3.contrast(f"aulc_level|{reg}|{metric}")
                levels.append(level_row("manuscript Table 4", f"E1 AULC({metric}) {reg}", val, None,
                                        (c["mean"] if c else float("nan"), float("nan"), c["k1_mean"] if c else float("nan"), 15 if c else 0)))
    for exp, cond, base, metric, est, lo, hi, where in M.PAIRED:
        if exp == "E1":
            cid = cmp_id("e1", PCT[cond], "all", base, metric)
        elif exp == "E2":
            cid = cmp_id("e2", None, cond, base, metric)
        elif exp == "E3":
            cid = cmp_id("e3", None, "all", base, metric)
        elif exp == "Table7":
            cid = cmp_id("neg", 50, cond, base, metric)
        else:
            b = cond.split()[0]
            cid = cmp_id("e4", BUD[b], "C", base, metric)
        paired.append(paired_row(f"manuscript {where}", f"{exp} {cond} ssl_colon - {base} {metric}", est, lo, hi, v3.contrast(cid), cid))
    for reg, gap in M.GAPS.items():
        paired.append(paired_row("manuscript 3.2", f"E2 AUROC gap flat - polypoid {reg}", gap, None, None, v3.contrast(f"gap|{reg}|auroc"), f"gap|{reg}|auroc"))
    return pd.DataFrame(levels), pd.DataFrame(paired)


# --------------------------------------------------------------------------- analysis_v2
def vs_v2(v3: V3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pdv2 = pd.read_csv(V2 / "paired_differences.csv")
    paired: List[dict] = []
    for _, r in pdv2.iterrows():
        exp, cond, comp, metric = r["experiment"], r["condition"], r["comparison"], r["metric"]
        base = comp.split(" - ")[1] if " - " in comp else None
        cid = None
        if exp == "E1" and cond == "AULC":
            cid = f"aulc|ssl_colon-{base}|{metric}" if base else f"aulc_level|{comp}|{metric}"
        elif exp == "E1":
            cid = cmp_id("e1", PCT[cond], "all", base, metric)
        elif exp == "E2" and cond == "gap flat-polypoid":
            cid = f"gapdiff|ssl_colon-{base}|{metric}" if base else f"gap|{comp}|{metric}"
        elif exp == "E2":
            cid = cmp_id("e2", None, cond, base, metric)
        elif exp == "E3":
            cid = cmp_id("e3", None, "all", base, metric)
        elif exp == "Table7-negative-source":
            cid = cmp_id("neg", 50, cond, base, metric)
        elif exp == "E4":
            b, kind = cond.split()[0], " ".join(cond.split()[1:])
            if kind == "adaptation effect":
                cid = f"arm|{BUD[b]}|{comp}|C-A|{metric}"
            else:
                cid = cmp_id("e4", BUD[b], "C" if kind == "adapted" else "A", base, metric)
        label = f"{exp} {cond} {comp} {metric}"
        paired.append(paired_row("analysis_v2 paired_differences", label, float(r["estimate"]), float(r["ci_lo"]), float(r["ci_hi"]),
                                 v3.contrast(cid) if cid else None, cid or ""))
    levels: List[dict] = []
    tables = (("table3_e1_performance.csv", "e1", lambda c, s: (PCT[c], "all")),
              ("table5_e2_morphology.csv", "e2", lambda c, s: (None, s)),
              ("table6_e3_zeroshot.csv", "e3", lambda c, s: (None, "all")),
              ("table7_negative_source.csv", "neg", lambda c, s: (50, s)),
              ("table8_e4_fewshot_adapted.csv", "e4", lambda c, s: (BUD[c], "C")),
              ("table8b_e4_fewshot_zeroshot_matched.csv", "e4", lambda c, s: (BUD[c], "A")))
    for fname, fam, key in tables:
        t = pd.read_csv(V2 / fname)
        for _, r in t.iterrows():
            variant, slot = key(r["condition"], r["stratum"])
            for metric in ("auroc", "auprc", "f1", "recall", "specificity", "balanced_accuracy"):
                if f"{metric}_mean" not in r:
                    continue
                label = f"{fam} {r['condition']} {r['stratum']} {r['model']} {metric}"
                if fam == "e4" and metric == "auprc":
                    levels.append(level_row(f"analysis_v2 {fname}", label, float(r[f'{metric}_mean']), float(r[f'{metric}_sd']),
                                            (float("nan"),) * 3 + (0,), "not compared: E4 AP depends on the pack (SAP 6, rev. c)"))
                    continue
                levels.append(level_row(f"analysis_v2 {fname}", label, float(r[f"{metric}_mean"]), float(r[f"{metric}_sd"]),
                                        v3.level(fam, variant, slot, r["model"], metric)))
    return pd.DataFrame(levels), pd.DataFrame(paired)


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, cols: List[str]) -> str:
    if df.empty:
        return "(none)\n"
    def f(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return f"{v:+.3f}" if abs(v) < 10 else f"{v:.0f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines) + "\n"


def _flags(p: pd.DataFrame) -> pd.DataFrame:
    if p.empty or "status" not in p:
        return p
    c = p[p["status"] == "compared"]
    return c[(c["sign_change"] == True) | (c["significance_change_boot"] == True) | (c["significance_change_model"] == True)]  # noqa: E712


def write(out: Path) -> Path:
    v3 = V3(out)
    lm, pm = vs_manuscript(v3)
    l2, p2 = vs_v2(v3)
    d = out / "drift"
    d.mkdir(parents=True, exist_ok=True)
    lm.to_csv(d / "drift_manuscript_levels.csv", index=False, float_format="%.6g")
    pm.to_csv(d / "drift_manuscript_paired.csv", index=False, float_format="%.6g")
    l2.to_csv(d / "drift_v2_levels.csv", index=False, float_format="%.6g")
    p2.to_csv(d / "drift_v2_paired.csv", index=False, float_format="%.6g")
    pc = ["quantity", "ref_estimate", "ref_ci_lo", "ref_ci_hi", "v3_estimate", "v3_k1_estimate", "v3_boot_lo", "v3_boot_hi",
          "v3_model_lo", "v3_model_hi", "sign_change", "significance_change_boot", "significance_change_model"]
    lc = ["quantity", "ref_mean", "v3_mean", "v3_k1_mean", "diff", "diff_k1"]
    L = ["# Drift report: campaign v3 vs the previous manuscript and analysis_v2", "",
         "Generated by `analysis/v3/drift.py`. v3 = mean over all paired runs (5 partitions × 3 seeds); v3 k1 = partition k = 1 "
         "only (the partition used before). Paired differences: SSL-colon − comparator. Significance = 95 % CI excludes 0 "
         "(reference: bootstrap CI quoted in the manuscript or computed by analysis_v2; v3: cluster-bootstrap CI and model CI). "
         "Level flags: |v3 − reference| > 0.02. E4 AP levels are not compared (SAP 6, rev. c). The manuscript values come from "
         "`analysis_v2/manuscript_values.py` (transcribed from `manuscript_results_values.md`, which is not in the repository).", ""]
    for title, levels, paired in (("Previous manuscript", lm, pm), ("analysis_v2", l2, p2)):
        comp = paired[paired.get("status", pd.Series(dtype=str)) == "compared"] if not paired.empty else paired
        fl = _flags(paired)
        lv = levels[levels["v3_n_runs"] > 0] if not levels.empty else levels
        big = lv[lv["flag_abs_diff_gt_0.02"] == True] if not lv.empty else lv  # noqa: E712
        L += [f"## {title}", "",
              f"- Paired differences compared: {len(comp)} (no v3 counterpart: {int((paired['status'] != 'compared').sum()) if not paired.empty else 0}).",
              f"- Sign changes: {int(comp['sign_change'].sum()) if len(comp) else 0}; significance changes vs v3 bootstrap CI: "
              f"{int((comp['significance_change_boot'] == True).sum()) if len(comp) else 0}; vs v3 model CI: "  # noqa: E712
              f"{int((comp['significance_change_model'] == True).sum()) if len(comp) else 0}.",  # noqa: E712
              f"- Levels compared: {len(lv)}; |difference| > 0.02: {len(big)}; median |difference| {lv['diff'].abs().median():.3f}." if len(lv) else "- Levels compared: 0.",
              "", "### Paired differences with a change of sign or significance", "", _md(fl, pc),
              "### Levels with |difference| > 0.02", "", _md(big.sort_values("diff", key=abs, ascending=False) if len(big) else big, lc), ""]
    path = d / "drift_report.md"
    path.write_text("\n".join(L))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    print(write(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
