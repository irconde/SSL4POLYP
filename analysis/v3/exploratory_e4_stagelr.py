#!/usr/bin/env python3
"""Exploratory E4-stagelr analysis (docs/v3/exploratory_e4_stagelr_plan.md): E4 arm contrasts with the declared stage
learning rates.

Not registered in the SAP: no Holm correction, no confirmation verdicts. The registered analysis is unchanged and remains
the confirmatory one; registered results are read, never written.

Arms (balanced accuracy at each arm's τ, as in the campaign):

* A, B (unadapted) and C, D (adapted, registered schedule): the registered runs (``checkpoints_v3/campaign``).
* C', D': the same adaptations with the declared schedule, variant ``declared`` (``--e4-stage-lr-persist`` in the four
  regimes plus ``--e4-unfreeze-sequential-blocks`` for SUP-imnet), ``checkpoints_v3/exp_e4_stagelr``.
* C'p, D'p: SUP-imnet with ``--e4-stage-lr-persist`` only, variant ``persistonly``,
  ``checkpoints_v3/exp_e4_stagelr_persistonly``. In SUP-imnet, C' − C mixes the effect of the rates and of the block
  unfreezing; C'p − C isolates the rates (head-only adaptation, as registered) and C' − C'p the unfreezing.

Contrasts (per run = mean over its (regime, budget) pairs of the paired arm differences, the registered definition of
H7 in ``analysis/v3/contrasts.py``): global (4 regimes × 4 budgets), per budget (mean of regimes), per regime (mean of
budgets) and per regime × budget, for C' − A (H7'), D' − A, D' − C', C' − C, D' − D, and the registered C − A (= H7),
D − A, D − C as references; SUP-imnet only: C'p − A, D'p − A, D'p − C'p, C'p − C, D'p − D, C' − C'p, D' − D'p.

Inference with the registered functions: t-test on the 5 partition means (``mixed.fit_many``, variant ``ttest``) and
cluster bootstrap of the E4 units (``design.bootstrap``, B = 2000, seed 12345). The bootstrap space is checked to equal
the registered e4 space, so the draws are those of the registered analysis. The interpretation rules of the plan are
applied to the three global contrasts H7', D' − A and D' − C' only; the same rule applied to the registered
contrast gives its reference category, and the report says whether the new category is maintained, weakened or
strengthened (plan revision 1).

Identity check (before any new run is analysed): ``--identity`` uses the registered root in place of both new roots.
H7 and the per-budget arm contrasts (pooled and per regime) must then reproduce ``results_v3/analysis/contrast_summary.csv``
(estimate, t-test CI, bootstrap CI, B), and every new − registered contrast must be exactly 0 in the point estimate and
in every replicate.

    PYTHONPATH=. python -m analysis.v3.exploratory_e4_stagelr --out results_v3/analysis_exploratory_e4_stagelr
    PYTHONPATH=. python -m analysis.v3.exploratory_e4_stagelr --identity --out results_v3/verification/exp_e4_stagelr_analysis_check

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
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.v3 import mixed  # noqa: E402
from analysis.v3.design import Contrast, Space, View, bootstrap, build_views, ci95, complete_terms  # noqa: E402
from analysis.v3.loader import Campaign, load_campaign  # noqa: E402
from analysis.v3.sap import B_BOOT, BOOT_SEED, E4_BUDGETS, MODEL_VARIANT, REGIMES, REPO  # noqa: E402
from analysis.v3.tables import to_markdown  # noqa: E402

REG_ROOT = REPO / "checkpoints_v3/campaign/classification"
DECLARED_ROOT = REPO / "checkpoints_v3/exp_e4_stagelr/classification"
PERSISTONLY_ROOT = REPO / "checkpoints_v3/exp_e4_stagelr_persistonly/classification"
METRIC = "balanced_accuracy"
SUP = "sup_imnet"
# internal arm tags of the new views → labels used in the outputs
NEW_ARMS = {"declared": {"C": "Cs", "D": "Ds"}, "persistonly": {"C": "Cp", "D": "Dp"}}
LABEL = {"A": "A", "B": "B", "C": "C", "D": "D", "Cs": "C'", "Ds": "D'", "Cp": "C'p", "Dp": "D'p"}
DECLARED_PAIRS = (("Cs", "A"), ("Ds", "A"), ("Ds", "Cs"), ("Cs", "C"), ("Ds", "D"))
REGISTERED_PAIRS = (("C", "A"), ("D", "A"), ("D", "C"))
PERSISTONLY_PAIRS = (("Cp", "A"), ("Dp", "A"), ("Dp", "Cp"), ("Cp", "C"), ("Dp", "D"), ("Cs", "Cp"), ("Ds", "Dp"))
# the three global contrasts with an interpretation rule, and the registered contrast whose sign they are compared with
RULED = {("Cs", "A"): ("C", "A"), ("Ds", "A"): ("D", "A"), ("Ds", "Cs"): ("D", "C")}
NAME = {("Cs", "A"): "H7' (C' - A)", ("Ds", "A"): "D' - A", ("Ds", "Cs"): "D' - C'"}
# plan r1: order of the categories, to report whether the new category is maintained, weakened or strengthened
STRENGTH = {"reversed": 0, "not supported": 1, "attenuated": 2, "robust": 3}


def cid(treat: str, base: str, budget, regime: str) -> str:
    return f"{LABEL[treat]}-{LABEL[base]}|{budget}|{regime}"


def load_new(root: Path, variant: str, reg: Campaign, reg_views: Dict, regimes: Sequence[str]) -> Tuple[Dict, dict]:
    """Views C', D' (or C'p, D'p) of one variant, with structural checks against the registered runs."""

    camp = reg if root.resolve() == REG_ROOT.resolve() else load_campaign([root])
    e4 = [r for r in camp.runs if r.experiment == "e4"]
    expected_regimes = [g for g in regimes if variant == "declared" or g == SUP]
    expected = {(b, g, k, s) for b in E4_BUDGETS for g in expected_regimes for k in reg.partitions() for s in reg.seeds()}
    found = {(r.variant, r.regime, r.k, r.seed) for r in e4}
    if variant == "persistonly":
        e4 = [r for r in e4 if r.regime == SUP]
        found = {x for x in found if x[1] == SUP}
    if found != expected:
        raise RuntimeError(f"{variant}: {len(found)} E4 runs under {root}, expected {len(expected)}; "
                           f"missing {sorted(expected - found)[:3]}, extra {sorted(found - expected)[:3]}")
    own = [c for c in camp.checks if c["check"] in ("e4_zeroshot_and_adapted_same_split", "paired_runs_same_frames")
           and "'e4'" in c["detail"]]
    failed = [c for c in own if not c["passed"]]
    if failed:
        raise RuntimeError(f"{variant}: integrity checks failed: {failed[:2]}")
    tags = NEW_ARMS[variant]
    views, info = {}, {"runs": len(e4), "zeroshot_probs_equal_registered": 0, "frames_equal_registered": 0}
    for run in e4:
        k, s, g, b = run.k, run.seed, run.regime, run.variant
        regC = reg_views[("e4", b, g, k, s, "C")]
        regA = reg_views[("e4", b, g, k, s, "A")]
        test, zs = run.frames["test"], run.frames["zeroshot"]
        same_frames = (np.array_equal(test.col("frame_id"), regC.frames.col("frame_id"))
                       and np.array_equal(test.labels, regC.frames.labels))
        if not same_frames:
            raise RuntimeError(f"{variant} {run.key}: evaluation frames differ from the registered run")
        info["frames_equal_registered"] += 1
        if not (run.tau["A"] == regA.tau and run.tau["C"] == regC.tau):
            raise RuntimeError(f"{variant} {run.key}: inherited τ differs from the registered run")
        info["zeroshot_probs_equal_registered"] += int(np.array_equal(zs.probs, regA.frames.probs)
                                                       and np.array_equal(zs.col("frame_id"), regA.frames.col("frame_id")))
        for arm, tag in tags.items():
            key = ("e4", b, g, k, s, tag)
            views[key] = View(key, test, run.tau[arm], np.arange(len(test)), "e4", "polypgen")
    info["kept_inherited_tau_D"] = int(sum(r.info.get("kept_inherited_tau_D", 0) for r in e4))
    return views, info


def definitions(regimes: Sequence[str]) -> List[dict]:
    out = []

    def add(pairs, regs, group):
        for t, a in pairs:
            out.append({"treat": t, "base": a, "budget": "all", "regime": "pooled" if len(regs) > 1 else regs[0],
                        "regimes": tuple(regs), "budgets": tuple(E4_BUDGETS), "group": group})
            for b in E4_BUDGETS:
                out.append({"treat": t, "base": a, "budget": b, "regime": "pooled" if len(regs) > 1 else regs[0],
                            "regimes": tuple(regs), "budgets": (b,), "group": group})
            if len(regs) > 1:
                for g in regs:
                    out.append({"treat": t, "base": a, "budget": "all", "regime": g, "regimes": (g,),
                                "budgets": tuple(E4_BUDGETS), "group": group})
                    for b in E4_BUDGETS:
                        out.append({"treat": t, "base": a, "budget": b, "regime": g, "regimes": (g,), "budgets": (b,),
                                    "group": group})

    add(DECLARED_PAIRS, list(regimes), "declared")
    add(REGISTERED_PAIRS, list(regimes), "registered")
    add(PERSISTONLY_PAIRS, [SUP], "persistonly")
    for d in out:
        d["id"] = cid(d["treat"], d["base"], d["budget"], d["regime"])
    return out


def make_terms(d: dict):
    m = len(d["regimes"]) * len(d["budgets"])

    def make(k: int, s: int):
        terms = []
        for g in d["regimes"]:
            for b in d["budgets"]:
                terms.append((("e4", b, g, k, s, d["treat"]), METRIC, 1.0 / m))
                terms.append((("e4", b, g, k, s, d["base"]), METRIC, -1.0 / m))
        return terms

    return make


def side(lo: float, hi: float) -> int:
    """+1 / -1 when the interval lies entirely above / below zero, 0 otherwise."""

    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def interpret(est: float, t_lo: float, t_hi: float, b_lo: float, b_hi: float, reg_est: float) -> str:
    """Plan rule: robust / attenuated / not supported / reversed, relative to the sign of the registered contrast."""

    if not all(math.isfinite(x) for x in (est, t_lo, t_hi, b_lo, b_hi, reg_est)) or reg_est == 0:
        return "undefined"
    sign = 1 if reg_est > 0 else -1
    if est * sign > 0:
        return "robust" if side(t_lo, t_hi) == sign and side(b_lo, b_hi) == sign else "attenuated"
    return "reversed" if side(t_lo, t_hi) == -sign and side(b_lo, b_hi) == -sign else "not supported"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--declared-root", type=Path, default=DECLARED_ROOT)
    ap.add_argument("--persistonly-root", type=Path, default=PERSISTONLY_ROOT)
    ap.add_argument("--identity", action="store_true", help="registered root in place of both new roots (check)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    if args.identity:
        args.declared_root = args.persistonly_root = REG_ROOT
    t0 = time.time()

    reg = load_campaign([REG_ROOT])
    failed = [x for x in reg.checks if not x["passed"]]
    if failed:
        raise RuntimeError(f"{len(failed)} registered integrity checks failed, first: {failed[0]}")
    regimes = [g for g in REGIMES if g in reg.regimes()]
    partitions, seeds = reg.partitions(), reg.seeds()
    runs = [(k, s) for k in partitions for s in seeds]
    views = build_views(reg)
    reg_e4 = sorted([vk for vk in views if vk[0] == "e4"], key=repr)
    infos = {}
    for variant, root in (("declared", args.declared_root), ("persistonly", args.persistonly_root)):
        new_views, infos[variant] = load_new(Path(root), variant, reg, views, regimes)
        views.update(new_views)
        print(f"[load] {variant}: {infos[variant]} ({time.time() - t0:.0f}s)", flush=True)

    defs = definitions(regimes)
    contrasts = []
    for d in defs:
        terms = complete_terms(views, runs, make_terms(d))
        if len(terms) != len(runs):
            raise RuntimeError(f"{d['id']}: {len(terms)} complete runs, expected {len(runs)}")
        contrasts.append(Contrast(d["id"], "e4", {}, terms))
    needed = {vk for c in contrasts for vk in c.views()}
    if Space([views[vk] for vk in needed]).flat != Space([views[vk] for vk in reg_e4]).flat:
        raise RuntimeError("bootstrap space differs from the registered e4 space")

    point = {vk: views[vk].metrics() for vk in needed}
    reps, counts = bootstrap(views, contrasts, B=B_BOOT, seed=BOOT_SEED, workers=args.workers)
    print(f"[bootstrap] {len(contrasts)} contrasts, units {counts} ({time.time() - t0:.0f}s)", flush=True)

    rows, per_run, fits_in = [], [], {}
    for j, (d, c) in enumerate(zip(defs, contrasts)):
        dvals = c.per_run(lambda vk, m: point[vk][m])
        fits_in[d["id"]] = dvals
        for (k, s), v in sorted(dvals.items()):
            per_run.append({"id": d["id"], "k": k, "seed": s, "d": v})
        lo, hi = ci95(reps[:, j])
        rows.append({"id": d["id"], "group": d["group"], "comparison": f"{LABEL[d['treat']]} - {LABEL[d['base']]}",
                     "budget": d["budget"], "regime": d["regime"], "n_runs": len(dvals),
                     "n_partitions": len({k for k, _ in dvals}), "estimate": float(np.mean([dvals[r] for r in sorted(dvals)])),
                     "boot_lo": lo, "boot_hi": hi, "boot_B": int(np.isfinite(reps[:, j]).sum())})
    res = pd.DataFrame(rows)
    fits = mixed.fit_many(fits_in, MODEL_VARIANT).set_index("id")
    for col, src in (("t_ci_lo", "ci_lo"), ("t_ci_hi", "ci_hi"), ("p_unadjusted", "p"), ("method", "method")):
        res[col] = res["id"].map(fits[src])
    res_i = res.set_index("id")

    # interpretation rules (plan): the three global contrasts only
    interp = []
    for (t, a), (rt, ra) in RULED.items():
        r, ref = res_i.loc[cid(t, a, "all", "pooled")], res_i.loc[cid(rt, ra, "all", "pooled")]
        new_cat = interpret(r["estimate"], r["t_ci_lo"], r["t_ci_hi"], r["boot_lo"], r["boot_hi"], ref["estimate"])
        # plan r1: the same rule applied to the registered contrast gives its reference category
        reg_cat = interpret(ref["estimate"], ref["t_ci_lo"], ref["t_ci_hi"], ref["boot_lo"], ref["boot_hi"], ref["estimate"])
        rank_new, rank_reg = STRENGTH.get(new_cat), STRENGTH.get(reg_cat)
        change = ("undefined" if rank_new is None or rank_reg is None else
                  "maintained" if rank_new == rank_reg else "strengthened" if rank_new > rank_reg else "weakened")
        interp.append({"contrast": NAME[(t, a)], "id": cid(t, a, "all", "pooled"), "registered_id": cid(rt, ra, "all", "pooled"),
                       "registered_estimate": ref["estimate"], "registered_t_ci_lo": ref["t_ci_lo"],
                       "registered_t_ci_hi": ref["t_ci_hi"], "registered_boot_lo": ref["boot_lo"],
                       "registered_boot_hi": ref["boot_hi"], "registered_category": reg_cat,
                       "estimate": r["estimate"], "t_ci_lo": r["t_ci_lo"],
                       "t_ci_hi": r["t_ci_hi"], "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"], "boot_B": r["boot_B"],
                       "category": new_cat, "change": change})
    interp = pd.DataFrame(interp)

    # degenerate predictors (registered rule, cut 0.10) per arm, budget and regime
    deg_rows = []
    for tag in ("C", "D", "Cs", "Ds", "Cp", "Dp"):
        for b in E4_BUDGETS:
            for g in regimes:
                keys = [("e4", b, g, k, s, tag) for k, s in runs if ("e4", b, g, k, s, tag) in point]
                if keys:
                    deg_rows.append({"arm": LABEL[tag], "budget": b, "regime": g, "views": len(keys),
                                     "degenerate": int(sum(bool(point[vk]["degenerate"]) for vk in keys))})
    deg = pd.DataFrame(deg_rows)
    deg_tot = deg.groupby("arm", sort=False)[["views", "degenerate"]].sum().reset_index()
    rm = pd.read_csv(REPO / "results_v3/analysis/run_metrics.csv")
    rm = rm[(rm["family"] == "e4") & rm["slot"].isin(["C", "D"])]
    reg_deg = rm.groupby("slot")["degenerate"].sum().astype(int).to_dict()
    if {a: int(deg_tot.set_index("arm").loc[a, "degenerate"]) for a in ("C", "D")} != reg_deg:
        raise RuntimeError(f"registered degenerate counts not reproduced: {reg_deg}")

    # identity check
    check, zero = [], {}
    if args.identity:
        cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
        targets = {cid("Cs", "A", "all", "pooled"): "H7"}
        for t, a in (("Cs", "A"), ("Ds", "A"), ("Ds", "Cs")):
            rt, ra = {"Cs": "C", "Ds": "D", "A": "A"}[t], {"Cs": "C", "Ds": "D", "A": "A"}[a]
            for b in E4_BUDGETS:
                for g in ["pooled", *regimes]:
                    targets[cid(t, a, b, g)] = f"arm|{b}|{g}|{rt}-{ra}|{METRIC}"
        for mine, ref_id in targets.items():
            r, ref = res_i.loc[mine], cs.loc[ref_id]
            for q, rq in (("estimate", "mean"), ("t_ci_lo", "model_ci_lo"), ("t_ci_hi", "model_ci_hi"),
                          ("boot_lo", "boot_lo"), ("boot_hi", "boot_hi"), ("boot_B", "boot_B")):
                diff = abs(float(r[q]) - float(ref[rq]))
                check.append({"id": mine, "registered_id": ref_id, "quantity": q, "recomputed": r[q], "registered": ref[rq],
                              "abs_diff": diff, "exact": diff <= 5e-10 * max(1.0, abs(float(ref[rq])))})
        for j, d in enumerate(defs):
            if (d["treat"], d["base"]) in (("Cs", "C"), ("Ds", "D"), ("Cp", "C"), ("Dp", "D"), ("Cs", "Cp"), ("Ds", "Dp")):
                vals = reps[:, j][np.isfinite(reps[:, j])]
                zero[d["id"]] = {"estimate": float(res_i.loc[d["id"], "estimate"]), "max_abs_replicate": float(np.abs(vals).max()),
                                 "finite_replicates": int(vals.size)}
        check = pd.DataFrame(check)
        n_exact = int(check["exact"].sum())
        zero_ok = all(v["estimate"] == 0.0 and v["max_abs_replicate"] == 0.0 for v in zero.values())
        if n_exact != len(check) or not zero_ok:
            raise RuntimeError(f"identity check failed: {n_exact}/{len(check)} exact, zero contrasts ok={zero_ok}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "contrasts.csv", index=False, float_format="%.10g")
    interp.to_csv(out / "interpretation.csv", index=False, float_format="%.10g")
    deg.to_csv(out / "degenerate_by_budget_regime.csv", index=False)
    deg_tot.to_csv(out / "degenerate_totals.csv", index=False)
    pd.DataFrame(per_run).to_csv(out / "per_run.csv", index=False, float_format="%.10g")
    info = {"mode": "identity" if args.identity else "analysis",
            "roots": {"registered": str(REG_ROOT.relative_to(REPO)), "declared": str(Path(args.declared_root).resolve().relative_to(REPO)),
                      "persistonly": str(Path(args.persistonly_root).resolve().relative_to(REPO))},
            "regimes": regimes, "partitions": partitions, "seeds": seeds, "B": B_BOOT, "boot_seed": BOOT_SEED,
            "model_variant": MODEL_VARIANT, "contrasts": len(defs), "bootstrap_units": {str(g): v for g, v in counts.items()},
            "bootstrap_space_equals_registered_e4": True, "new_runs": infos,
            "degenerate_totals": deg_tot.set_index("arm")["degenerate"].to_dict(), "registered_degenerate_run_metrics": reg_deg,
            "interpretation": interp.set_index("contrast")[["registered_category", "category", "change"]].to_dict("index"),
            "elapsed_s": round(time.time() - t0, 1)}
    md = ["# E4-stagelr: arm contrasts with the declared stage learning rates", "",
          "Exploratory analysis, not registered: no Holm correction, no confirmation verdicts. Balanced accuracy.", "",
          to_markdown(interp, "Global contrasts with an interpretation rule"),
          to_markdown(res[res["regime"] == "pooled"][["id", "estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi", "boot_B"]],
                      "Global and per-budget contrasts (mean of regimes)"),
          to_markdown(res[(res["group"] == "persistonly")][["id", "estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi", "boot_B"]],
                      "SUP-imnet: declared vs persist-only"),
          to_markdown(deg_tot, "Degenerate predictors (specificity or sensitivity < 0.10)")]
    if args.identity:
        check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
        info["registered_check"] = {"n": len(check), "exact": n_exact, "max_abs_diff": float(check["abs_diff"].max())}
        info["zero_contrasts"] = zero
        md.append(f"Identity check: {n_exact}/{len(check)} quantities equal to contrast_summary.csv; "
                  f"{len(zero)} new − registered contrasts exactly 0 in the estimate and every replicate.")
    (out / "contrasts.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(json.dumps({k: info[k] for k in ("mode", "contrasts", "new_runs", "degenerate_totals", "interpretation",
                                            "elapsed_s")}, indent=1, default=str))
    if args.identity:
        print(json.dumps(info["registered_check"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
