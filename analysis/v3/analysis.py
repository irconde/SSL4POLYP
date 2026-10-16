"""SAP v3.2c analysis: per-run metrics, paired differences, mixed models, cluster bootstrap, Holm and the
double confirmation rule, secondary and sensitivity analyses."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import mixed
from .contrasts import COLON, hypothesis_contrasts, secondary_contrasts, sensitivity_contrasts
from .design import Contrast, View, ViewKey, bootstrap, build_views, ci95, complete_terms, negsource_rows
from .loader import Campaign, completeness, load_campaign
from .sap import (
    ALPHA, B_BOOT, BOOT_SEED, CALIBRATION, COMPARATORS, HYPOTHESES, HYPOTHESIS_TEXT, MATCHED, METRICS,
    NEGSOURCE_BUDGET, PER_POLYP, sap_status,
)

LOCO_METRICS = ("auroc", "auprc", "f1", "balanced_accuracy", "recall", "specificity")
LOCASE_METRICS = ("auroc", "auprc", "f1", "balanced_accuracy")
RUN_METRIC_COLUMNS = METRICS + CALIBRATION + PER_POLYP + ("tp", "fp", "tn", "fn", "n_pos", "n_neg", "degenerate")


@dataclass
class Result:
    campaign: Campaign
    views: Dict[ViewKey, View]
    point: Dict[ViewKey, Dict[str, float]]
    contrasts: Dict[str, Contrast]
    per_run: Dict[str, Dict[Tuple[int, int], float]]
    summary: pd.DataFrame          # one row per contrast
    hypotheses: pd.DataFrame       # H1–H7 with Holm and the confirmation verdict
    run_metrics: pd.DataFrame      # one row per view
    notes: Dict[str, str]
    info: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- sensitivity views
def loco_views(c: Campaign) -> Tuple[Dict[ViewKey, View], List[str]]:
    """E3 and negative-source views with one centre left out (no bootstrap)."""

    views: Dict[ViewKey, View] = {}
    centers: set = set()
    idx = c.index()
    for run in c.runs:
        if run.experiment == "e3":
            centers |= {x for x in run.frames["test"].col("center_id") if x}
    centers_sorted = sorted(centers)
    for run in c.runs:
        k, s, g = run.k, run.seed, run.regime
        if run.experiment == "e3":
            f = run.frames["test"]
            for cen in centers_sorted:
                key = ("loco_e3", cen, g, k, s, "all")
                views[key] = View(key, f, run.tau["primary"], np.flatnonzero(f.col("center_id") != cen), "center", "polypgen")
        if run.experiment == "e4" and run.variant == NEGSOURCE_BUDGET:
            e3 = idx.get(("e3", None, g, k, s))
            if e3 is None:
                continue
            clean = set(e3.frames["test"].col("frame_id").tolist())
            for cen in centers_sorted:
                for source in ("isolated", "sequence"):
                    key = ("loco_neg", cen, g, k, s, source)
                    rows = negsource_rows(run.frames["zeroshot"], clean, source, exclude_center=cen)
                    views[key] = View(key, run.frames["zeroshot"], run.tau["A"], rows, "e4", "polypgen")
    return views, centers_sorted


def locase_views(c: Campaign) -> Tuple[Dict[ViewKey, View], Dict[int, List[str]]]:
    """E2 flat stratum with one flat case left out, per partition (no bootstrap)."""

    views: Dict[ViewKey, View] = {}
    flat_cases: Dict[int, set] = {}
    for run in c.runs:
        if run.experiment != "e2":
            continue
        f = run.frames["test"]
        morph = np.char.lower(f.col("morphology").astype(str))
        flat_cases.setdefault(run.k, set()).update(f.col("case_id")[(f.labels == 1) & (morph == "flat")].tolist())
    for run in c.runs:
        if run.experiment != "e2":
            continue
        f = run.frames["test"]
        pos = f.labels == 1
        morph = np.char.lower(f.col("morphology").astype(str))
        flat = ~pos | (pos & (morph == "flat"))
        for case in sorted(flat_cases.get(run.k, ())):
            key = ("locase", f"k{run.k}:{case}", run.regime, run.k, run.seed, "flat")
            rows = np.flatnonzero(flat & (f.col("case_id") != case))
            views[key] = View(key, f, run.tau["primary"], rows, "case_morph", ("sun", run.k), per_polyp=False)
    return views, {k: sorted(v) for k, v in flat_cases.items()}


def loco_contrasts(
    views: Dict[ViewKey, View], centers: Sequence[str], locase: Dict[int, List[str]],
    regimes: Sequence[str], partitions: Sequence[int], seeds: Sequence[int],
) -> List[Contrast]:
    runs = [(k, s) for k in partitions for s in seeds]
    comparators = [g for g in COMPARATORS if g in regimes]
    out: List[Contrast] = []

    def add(cid: str, meta: Dict[str, Any], make) -> None:
        terms = complete_terms(views, runs, make)
        if terms:
            out.append(Contrast(cid, "sensitivity", meta, terms))

    for cen in centers:
        for base in comparators:
            for m in LOCO_METRICS:
                add(f"loco_e3|{cen}|{COLON}-{base}|{m}",
                    {"analysis": "E3 leave-one-centre-out", "excluded": cen, "comparison": f"{COLON} - {base}", "metric": m},
                    lambda k, s, cen=cen, base=base, m=m: [(("loco_e3", cen, COLON, k, s, "all"), m, 1.0),
                                                           (("loco_e3", cen, base, k, s, "all"), m, -1.0)])
        n = len(regimes)
        add(f"loco_neg|{cen}|H5",
            {"analysis": "negative-source leave-one-centre-out", "excluded": cen, "comparison": "sequence - isolated (mean of regimes)",
             "metric": "auroc", "hypothesis": "H5"},
            lambda k, s, cen=cen: [t for g in regimes for t in [(("loco_neg", cen, g, k, s, "sequence"), "auroc", 1.0 / n),
                                                                  (("loco_neg", cen, g, k, s, "isolated"), "auroc", -1.0 / n)]])
        for g in regimes:
            add(f"loco_neg|{cen}|{g}|seq-iso",
                {"analysis": "negative-source leave-one-centre-out", "excluded": cen, "regime": g,
                 "comparison": "sequence - isolated", "metric": "auroc"},
                lambda k, s, cen=cen, g=g: [(("loco_neg", cen, g, k, s, "sequence"), "auroc", 1.0),
                                            (("loco_neg", cen, g, k, s, "isolated"), "auroc", -1.0)])
        for source in ("isolated", "sequence"):
            for base in comparators:
                for m in ("auroc", "auprc"):
                    add(f"loco_neg|{cen}|{source}|{COLON}-{base}|{m}",
                        {"analysis": "negative-source leave-one-centre-out", "excluded": cen, "slot": source,
                         "comparison": f"{COLON} - {base}", "metric": m},
                        lambda k, s, cen=cen, source=source, base=base, m=m:
                        [(("loco_neg", cen, COLON, k, s, source), m, 1.0), (("loco_neg", cen, base, k, s, source), m, -1.0)])
    for kk, cases in locase.items():
        for case in cases:
            tag = f"k{kk}:{case}"
            for base in comparators:
                for m in LOCASE_METRICS:
                    def make(k, s, tag=tag, kk=kk, base=base, m=m):
                        if k == kk:
                            a, b = ("locase", tag, COLON, k, s, "flat"), ("locase", tag, base, k, s, "flat")
                        else:
                            a, b = ("e2", None, COLON, k, s, "flat"), ("e2", None, base, k, s, "flat")
                        return [(a, m, 1.0), (b, m, -1.0)]

                    add(f"locase|{tag}|{COLON}-{base}|{m}",
                        {"analysis": "E2 flat leave-one-case-out", "excluded": tag, "partition": kk,
                         "comparison": f"{COLON} - {base}", "metric": m,
                         "hypothesis": "H3" if (base == "sup_imnet" and m == "auroc") else ""},
                        make)
    return out


# --------------------------------------------------------------------------- main entry
def run_analysis(
    roots: Iterable[Path],
    B: int = B_BOOT,
    boot_seed: int = BOOT_SEED,
    workers: int = 1,
    secondary: bool = True,
    sensitivity: bool = True,
    log=print,
) -> Result:
    t0 = time.time()
    campaign = load_campaign(roots)
    if not campaign.runs:
        raise FileNotFoundError(f"no complete runs under {list(campaign.roots)}")
    regimes, partitions, seeds = campaign.regimes(), campaign.partitions(), campaign.seeds()
    log(f"[load] {len(campaign.runs)} runs; regimes={regimes}; partitions={partitions}; seeds={seeds} ({time.time() - t0:.0f}s)")
    failed = [ch for ch in campaign.checks if not ch["passed"]]
    if failed:
        raise RuntimeError(f"{len(failed)} integrity checks failed, first: {failed[0]}")

    views = build_views(campaign)
    extra: Dict[ViewKey, View] = {}
    centers: List[str] = []
    locase: Dict[int, List[str]] = {}
    if sensitivity:
        lv, centers = loco_views(campaign)
        cv, locase = locase_views(campaign)
        extra = {**lv, **cv}
    all_views = {**views, **extra}

    hyp, notes = hypothesis_contrasts(views, regimes, partitions, seeds)
    contrasts: List[Contrast] = [c for c in hyp.values() if c is not None]
    if secondary:
        contrasts += secondary_contrasts(views, regimes, partitions, seeds)
    if sensitivity:
        contrasts += sensitivity_contrasts(views, regimes, partitions, seeds)
        contrasts += loco_contrasts(all_views, centers, locase, regimes, partitions, seeds)
    by_id = {c.id: c for c in contrasts}
    if len(by_id) != len(contrasts):
        raise ValueError("duplicate contrast ids")
    # With the secondary analyses every view is reported (run_metrics.csv); otherwise only the ones used.
    used = set(all_views) if secondary else {vk for c in contrasts for vk in c.views()}
    point = {vk: all_views[vk].metrics() for vk in all_views if vk in used}
    log(f"[metrics] {len(point)} views ({time.time() - t0:.0f}s)")

    metric_of = lambda vk, m: point[vk][m]  # noqa: E731
    per_run = {c.id: c.per_run(metric_of) for c in contrasts}
    fits = mixed.fit_many(per_run).set_index("id")
    log(f"[models] {len(fits)} fits ({time.time() - t0:.0f}s)")

    boot: Dict[str, Tuple[float, float, int]] = {}
    families = sorted({c.family for c in contrasts if c.family != "sensitivity"})
    for fam in families:
        members = [c for c in contrasts if c.family == fam]
        reps, counts = bootstrap(views, members, B=B, seed=boot_seed, workers=workers)
        for j, c in enumerate(members):
            col = reps[:, j]
            lo, hi = ci95(col)
            boot[c.id] = (lo, hi, int(np.isfinite(col).sum()))
        log(f"[bootstrap] {fam}: {len(members)} contrasts, units {counts} ({time.time() - t0:.0f}s)")

    rows = []
    for c in contrasts:
        d = per_run[c.id]
        sp = mixed.spread(d)
        f = fits.loc[c.id] if c.id in fits.index else None
        lo, hi, nb = boot.get(c.id, (float("nan"), float("nan"), 0))
        row = {"id": c.id, "family": c.family, **{k: v for k, v in c.meta.items()}, **sp,
               "boot_lo": lo, "boot_hi": hi, "boot_B": nb if c.id in boot else 0}
        if f is not None:
            row.update({
                "model_method": f["method"], "model_estimate": f["estimate"], "model_se": f["se"], "model_df": f["df"],
                "model_p": f["p"], "model_ci_lo": f["ci_lo"], "model_ci_hi": f["ci_hi"],
                "var_partition": f["var_partition"], "var_seed": f["var_seed"],
                "singular": f["singular"], "converged": f["converged"], "model_note": f["note"],
            })
        rows.append(row)
    summary = pd.DataFrame(rows)

    hypotheses = confirm(summary, hyp, notes)
    run_metrics = run_metrics_table({vk: v for vk, v in views.items() if vk in point}, point)
    info = {
        "sap": sap_status(),
        "roots": [str(r) for r in campaign.roots],
        "regimes": regimes, "partitions": partitions, "seeds": seeds,
        "n_runs": len(campaign.runs), "n_views": len(point), "n_contrasts": len(contrasts),
        "bootstrap": {"B": B, "seed": boot_seed}, "loco_centers": centers,
        "locase_flat_cases": {str(k): v for k, v in locase.items()},
        "integrity_checks": {"n": len(campaign.checks), "failed": len(failed)},
        "completeness": completeness(campaign, regimes, partitions, seeds).to_dict(orient="records"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    return Result(campaign, all_views, point, by_id, per_run, summary, hypotheses, run_metrics, notes, info)


def confirm(summary: pd.DataFrame, hyp: Dict[str, Optional[Contrast]], notes: Dict[str, str], alpha: float = ALPHA) -> pd.DataFrame:
    """Holm over the available hypotheses (all seven once H2 exists) and the double confirmation rule."""

    s = summary.set_index("id")
    available = [h for h in HYPOTHESES if hyp.get(h) is not None and h in s.index]
    pvals = {h: float(s.loc[h, "model_p"]) for h in available}
    adjusted = mixed.holm(pvals)
    complete = len(available) == len(HYPOTHESES)
    family = ",".join(available)
    rows = []
    for h in HYPOTHESES:
        row: Dict[str, Any] = {"hypothesis": h, "definition": HYPOTHESIS_TEXT[h], "note": notes.get(h, "")}
        if h not in available:
            row.update(status="pending" if "pending" in notes.get(h, "") else "missing", confirmed=False)
            rows.append(row)
            continue
        r = s.loc[h]
        p_holm = adjusted[h]
        excludes_zero = bool(np.isfinite(r["boot_lo"]) and (r["boot_lo"] > 0 or r["boot_hi"] < 0))
        holm_ok = bool(np.isfinite(p_holm) and p_holm < alpha)
        row.update({
            "estimate": r["mean"], "sd_partition": r["sd_partition"], "sd_seed": r["sd_seed"], "n_runs": r["n_runs"],
            "n_partitions": r["n_partitions"], "model_method": r["model_method"], "model_estimate": r["model_estimate"],
            "model_se": r["model_se"], "model_df": r["model_df"], "model_ci_lo": r["model_ci_lo"], "model_ci_hi": r["model_ci_hi"],
            "var_partition": r["var_partition"], "var_seed": r["var_seed"], "singular": r["singular"],
            "p": r["model_p"], "p_holm": p_holm, "holm_family": family, "holm_m": len(available),
            "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"], "boot_B": r["boot_B"],
            "holm_significant": holm_ok, "boot_excludes_zero": excludes_zero,
            "confirmed": holm_ok and excludes_zero,
            "status": "final" if complete else "provisional (Holm without H2)",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_metrics_table(views: Dict[ViewKey, View], point: Dict[ViewKey, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for vk, v in views.items():
        fam, variant, regime, k, s, slot = vk
        m = point[vk]
        rows.append({"family": fam, "variant": variant, "regime": regime, "k": k, "seed": s, "slot": slot, "tau": v.tau,
                     "n_frames": int(v.rows.size), **{c: m.get(c, float("nan")) for c in RUN_METRIC_COLUMNS}})
    return pd.DataFrame(rows)


def paired_differences(res: Result) -> pd.DataFrame:
    rows = []
    for cid, values in res.per_run.items():
        for (k, s), d in sorted(values.items()):
            rows.append({"id": cid, "k": k, "seed": s, "d": d})
    return pd.DataFrame(rows)
