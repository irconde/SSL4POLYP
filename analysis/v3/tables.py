"""Tables (CSV + Markdown): equivalents of manuscript Tables 3–9 and the new SAP v3.2c/v3.3 tables."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import mixed
from .analysis import Result, paired_differences
from .metrics import binary
from .sap import CALIBRATION, LABEL, METRICS, PER_POLYP, REGIMES

SLOT_ORDER = ("all", "overall", "flat", "polypoid", "A", "B", "C", "D", "isolated", "sequence")

LEVEL_METRICS = METRICS + CALIBRATION
E4_LEVEL_METRICS = tuple(m for m in LEVEL_METRICS if m != "auprc")  # SAP 6 (rev. c): E4 AP only within a pack
CONTRAST_COLUMNS = ["mean", "sd_partition", "sd_seed", "n_runs", "n_partitions", "model_method", "model_estimate",
                    "model_ci_lo", "model_ci_hi", "model_p", "boot_lo", "boot_hi", "boot_B"]


def levels(rm: pd.DataFrame, family: str, metrics: Sequence[str], by: Sequence[str] = ("variant", "slot", "regime")) -> pd.DataFrame:
    """Mean over runs and the SDs between partitions and between seeds (SAP section 7)."""

    sub = rm[rm["family"] == family]
    rows = []
    for key, g in sub.groupby(list(by), dropna=False, sort=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n_runs"] = len(g)
        for m in metrics:
            sp = mixed.spread({(k, s): v for k, s, v in zip(g["k"], g["seed"], g[m])})
            row[f"{m}_mean"], row[f"{m}_sd_partition"], row[f"{m}_sd_seed"] = sp["mean"], sp["sd_partition"], sp["sd_seed"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"regime": {g: i for i, g in enumerate(REGIMES)}, "slot": {s: i for i, s in enumerate(SLOT_ORDER)}}
    keys = [c for c in by if c in out]
    out = out.sort_values(keys, key=lambda col: col.map(order[col.name]).fillna(99) if col.name in order
                          else pd.to_numeric(col, errors="coerce").fillna(-1)).reset_index(drop=True)
    if "variant" in out:
        out["variant"] = out["variant"].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else int(v))
    if "regime" in out:
        out.insert(list(by).index("regime") + 1, "regime_label", out["regime"].map(lambda g: LABEL.get(g, g)))
    return out


def contrasts(summary: pd.DataFrame, mask: pd.Series, keep: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in keep if c in summary.columns] + [c for c in CONTRAST_COLUMNS if c in summary.columns]
    return summary.loc[mask, cols].reset_index(drop=True)


# --------------------------------------------------------------------------- markdown
def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer() and abs(v) >= 1:
            return str(int(v))
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def to_markdown(df: pd.DataFrame, title: str, note: str = "") -> str:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    if df.empty:
        return "\n".join(lines + ["(no rows)", ""])
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines + [""])


def compact_levels(df: pd.DataFrame, metrics: Sequence[str], keys: Sequence[str]) -> pd.DataFrame:
    """Markdown view: 'mean (SD partition / SD seed)' per metric."""

    out = df[[c for c in keys if c in df.columns]].copy()
    for m in metrics:
        if f"{m}_mean" in df:
            out[m] = [f"{a:.4f} ({b:.4f}/{c:.4f})" if np.isfinite(a) else ""
                      for a, b, c in zip(df[f"{m}_mean"], df[f"{m}_sd_partition"], df[f"{m}_sd_seed"])]
    return out


# --------------------------------------------------------------------------- reliability
def reliability(res: Result) -> pd.DataFrame:
    """Mean equal-mass reliability curve (15 bins) per experiment, budget, arm and regime (mean over runs)."""

    rows = []
    for (fam, variant, regime, k, s, slot), v in res.views.items():
        if fam not in ("sun_full", "e3", "e4"):
            continue
        p, y = v.frames.probs[v.rows], v.frames.labels[v.rows]
        for b in binary.reliability_table(p, y, binary.ECE_BINS):
            rows.append({"family": fam, "variant": variant, "slot": slot, "regime": regime, "k": k, "seed": s, **b})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (df.groupby(["family", "variant", "slot", "regime", "bin"], dropna=False)
             .agg(mean_prob=("mean_prob", "mean"), frac_pos=("frac_pos", "mean"), n_runs=("n", "size")).reset_index())
    return agg


def negsource_counts(res: Result) -> pd.DataFrame:
    rows = []
    for (fam, variant, regime, k, s, slot), v in res.views.items():
        if fam != "neg":
            continue
        y = v.frames.labels[v.rows]
        rows.append({"regime": regime, "k": k, "seed": s, "negative_source": slot, "positives_common": int((y == 1).sum()),
                     "negatives": int((y == 0).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- all tables
def build_tables(res: Result) -> Dict[str, Dict[str, object]]:
    """``{name: {"df": DataFrame, "title": str, "md": DataFrame for Markdown, "note": str}}``."""

    rm, sm = res.run_metrics, res.summary
    a = sm.get("analysis", pd.Series("", index=sm.index)).fillna("")
    T: Dict[str, Dict[str, object]] = {}

    def put(name, df, title, md=None, note=""):
        T[name] = {"df": df, "title": title, "md": df if md is None else md, "note": note}

    hyp = res.hypotheses.copy()
    md_cols = [c for c in ("hypothesis", "definition", "estimate", "sd_partition", "sd_seed", "model_method", "model_ci_lo",
                           "model_ci_hi", "p", "p_holm", "boot_lo", "boot_hi", "confirmed", "status", "note") if c in hyp]
    put("table_hypotheses", hyp, "Primary hypotheses H1–H7 (SAP sections 3–5)", hyp[md_cols],
        "Estimate = mean of the paired per-run differences; model CI and p from the one-sample t-test on the K "
        "partition means (SAP v3.3, df = K − 1); Holm over the hypotheses listed in holm_family; confirmed = Holm p < 0.05 and the "
        "cluster-bootstrap 95 % CI excludes 0.")

    sun = levels(rm, "sun_full", LEVEL_METRICS + PER_POLYP, by=("regime",))
    put("table_sun_full", sun, "sun_full test (baseline and parent of E2–E4)", compact_levels(sun, LEVEL_METRICS + PER_POLYP, ("regime_label",)))

    t3 = levels(rm, "e1", LEVEL_METRICS + PER_POLYP, by=("variant", "regime")).rename(columns={"variant": "percent"})
    put("table3_e1_performance", t3, "Table 3: E1 by training budget", compact_levels(t3, METRICS, ("percent", "regime_label")),
        "mean (SD between partitions / SD between seeds) over the paired runs.")

    t4 = pd.concat([
        contrasts(sm, a == "AULC level", ["regime", "metric"]),
        contrasts(sm, a == "AULC comparison", ["comparison", "metric"]),
    ], ignore_index=True)
    put("table4_e1_aulc", t4, "Table 4: E1 AULC (trapezoid over 5–100 %, normalised by the axis width)")

    t5 = levels(rm, "e2", LEVEL_METRICS + PER_POLYP, by=("slot", "regime")).rename(columns={"slot": "stratum"})
    put("table5_e2_morphology", t5, "Table 5: E2 by morphology stratum (sun_full models, frozen τ)",
        compact_levels(t5, ("auroc", "auprc", "recall", "specificity", "balanced_accuracy") + PER_POLYP, ("stratum", "regime_label")))
    t5b = contrasts(sm, a == "flat - polypoid gap", ["regime", "comparison", "metric"])
    put("table5b_e2_gap", t5b, "E2: AUROC gap flat − polypoid per regime and between regimes")

    t6 = levels(rm, "e3", LEVEL_METRICS, by=("regime",))
    put("table6_e3_zeroshot", t6, "Table 6: E3 zero-shot on polypgen_clean_test", compact_levels(t6, LEVEL_METRICS, ("regime_label",)))

    t7 = levels(rm, "neg", ("auroc", "auprc", "specificity", "balanced_accuracy"), by=("slot", "regime")).rename(columns={"slot": "negative_source"})
    put("table7_negative_source", t7, "Table 7: negative-source decomposition (50-frame split, common positives, zero-shot)",
        compact_levels(t7, ("auroc", "specificity", "balanced_accuracy"), ("negative_source", "regime_label")))
    put("table7_negative_source_contrasts", contrasts(sm, a == "negative-source decomposition", ["regime", "comparison", "metric"]),
        "Table 7b: sequence − isolated negatives per regime")
    counts = negsource_counts(res)
    put("table7_count_checks", counts, "Table 7c: frame counts of the decomposition per run",
        counts.groupby(["negative_source"]).agg(positives_min=("positives_common", "min"), positives_max=("positives_common", "max"),
                                                negatives_min=("negatives", "min"), negatives_max=("negatives", "max"),
                                                runs=("k", "size")).reset_index() if not counts.empty else counts)

    t8 = levels(rm, "e4", E4_LEVEL_METRICS, by=("variant", "slot", "regime")).rename(columns={"variant": "budget", "slot": "arm"})
    put("table8_e4_arms", t8, "Table 8: E4, four arms by budget",
        compact_levels(t8, ("auroc", "recall", "specificity", "balanced_accuracy", "f1", "brier", "ece"), ("budget", "arm", "regime_label")),
        "AP is not tabulated as a level: it is compared only within a pack (between arms and regimes on the same "
        "evaluation split), see table8b and table9 (SAP section 6, revision c).")
    t8b = contrasts(sm, (a == "regime comparison") & (sm.get("experiment") == "e4") & (sm.get("metric") == "auprc"),
                    ["variant", "slot", "comparison", "metric"])
    put("table8b_e4_ap_within_pack", t8b, "Table 8b: E4 AP differences between regimes within each pack (per budget and arm)")
    t9 = contrasts(sm, (a == "E4 arm contrast"), ["variant", "regime", "comparison", "metric"])
    put("table9_e4_arm_contrasts", t9, "Table 9: E4 arm contrasts per budget (C − A = adapted vs zero-shot; D − C = recalibration after adaptation)",
        t9[t9["metric"].isin(["auroc", "auprc", "balanced_accuracy", "recall", "specificity"])] if not t9.empty else t9)

    sec = contrasts(sm, a == "regime comparison", ["experiment", "variant", "slot", "comparison", "metric"])
    put("table_secondary_regime_comparisons", sec, "Secondary: SSL-colon vs each comparator by experiment, budget, stratum, arm and metric",
        sec[sec["metric"].isin(["auroc", "auprc", "balanced_accuracy", "brier", "ece"] + list(PER_POLYP))] if not sec.empty else sec,
        "Descriptive (no multiplicity correction). Markdown shows a subset of metrics; the CSV has all.")

    deg_rows = []
    for (fam, variant, slot, regime), g in rm.groupby(["family", "variant", "slot", "regime"], dropna=False):
        if fam == "neg" or (fam == "e2" and slot != "overall"):
            continue
        deg_rows.append({"experiment": fam, "variant": variant, "slot": slot, "regime": regime, "regime_label": LABEL.get(regime, regime),
                         "n_runs": len(g), "n_degenerate": int(g["degenerate"].sum()), "proportion": float(g["degenerate"].mean())})
    deg = pd.DataFrame(deg_rows)
    put("table_degenerate", deg, "Degenerate predictors (test specificity or sensitivity < 0.10 at the operating threshold)")

    cal = pd.concat([
        levels(rm, "sun_full", CALIBRATION, by=("variant", "slot", "regime")).assign(experiment="sun_full"),
        levels(rm, "e3", CALIBRATION, by=("variant", "slot", "regime")).assign(experiment="e3"),
        levels(rm, "e4", CALIBRATION, by=("variant", "slot", "regime")).assign(experiment="e4"),
    ], ignore_index=True)
    cal = cal[["experiment"] + [c for c in cal.columns if c != "experiment"]]
    put("table_calibration", cal, "Calibration: Brier and ECE (15 equal-mass bins); E4 per arm",
        compact_levels(cal, CALIBRATION, ("experiment", "variant", "slot", "regime_label")),
        "Arms A/B (and C/D) share the same probabilities, so their calibration is identical by construction.")
    put("table_reliability", reliability(res), "Reliability diagrams (mean over runs, equal-mass bins)")

    pp = pd.concat([levels(rm, "e1", PER_POLYP, by=("variant", "slot", "regime")).assign(experiment="e1"),
                    levels(rm, "e2", PER_POLYP, by=("variant", "slot", "regime")).assign(experiment="e2")], ignore_index=True)
    pp = pp[["experiment"] + [c for c in pp.columns if c != "experiment"]]
    put("table_per_polyp", pp, "Per-polyp sensitivity (≥ 1 and ≥ 5 of 33 frames above τ), E1 and E2",
        compact_levels(pp, PER_POLYP, ("experiment", "variant", "slot", "regime_label")))

    put("table_sensitivity_k1_no_s500", contrasts(sm, a.isin(["k = 1 only", "without s500"]), ["id", "analysis", "hypothesis", "comparison", "metric"]),
        "Sensitivity: H1–H4 on partition k = 1 only; H6 and H7 without s500")
    put("table_loco_e3", contrasts(sm, a == "E3 leave-one-centre-out", ["excluded", "comparison", "metric"]),
        "Sensitivity: E3 leave-one-centre-out (no bootstrap)")
    put("table_loco_negative_source", contrasts(sm, a == "negative-source leave-one-centre-out",
                                                ["excluded", "slot", "regime", "comparison", "metric", "hypothesis"]),
        "Sensitivity: negative-source decomposition leave-one-centre-out (no bootstrap)")
    loc = contrasts(sm, a == "E2 flat leave-one-case-out", ["id", "excluded", "partition", "comparison", "metric", "hypothesis"])
    if not loc.empty:
        part_mean = []
        for cid, kk in zip(loc["id"], loc["partition"]):
            d = [v for (k, _), v in res.per_run[cid].items() if k == kk]
            part_mean.append(float(np.mean(d)) if d else float("nan"))
        loc.insert(loc.columns.get_loc("mean"), "partition_mean", part_mean)
    put("table_locase_e2_flat", loc, "Sensitivity: E2 flat stratum leave-one-case-out (per partition; mean = all runs with that case removed)")
    return T


def write_tables(res: Result, out: Path) -> List[Path]:
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    written = []
    tables = build_tables(res)
    md = ["# Tables", ""]
    for name, t in tables.items():
        path = out / "tables" / f"{name}.csv"
        t["df"].to_csv(path, index=False, float_format="%.10g")
        (out / "tables" / f"{name}.md").write_text(to_markdown(t["md"], t["title"], t["note"]))
        md.append(to_markdown(t["md"], t["title"], t["note"]))
        written.append(path)
    (out / "tables" / "ALL_TABLES.md").write_text("\n".join(md))
    res.summary.to_csv(out / "contrast_summary.csv", index=False, float_format="%.10g")
    paired_differences(res).to_csv(out / "paired_differences.csv", index=False, float_format="%.10g")
    res.run_metrics.to_csv(out / "run_metrics.csv", index=False, float_format="%.10g")
    return written
