#!/usr/bin/env python3
"""E4 sensitivity: support size vs type of negatives (docs/v3/sensitivity_e4_negative_type_plan.md).

Post hoc sensitivity analysis, not registered: registered results are read, never written.

Three conditions, all evaluated on the common evaluation set of each pack (partition k, seed s), which is the
500-frame evaluation split (``results_v3/verification/e4_common_eval/``):

* ``200iso``: registered 200-frame pack (isolated negatives in the support), restricted to the common set;
* ``200seq``: new 200-frame pack with the same 100 positives and 100 sequence negatives
  (``data_packs/v3/polypgen_fewshot_s200seq``), runs under ``--seq-root``; its test split is the common set;
* ``500``: registered 500-frame pack (its evaluation split is the common set).

Each condition uses its own runs: arms A and B on the run's zero-shot outputs, C and D on its adapted outputs, with the
operating thresholds stored by the run (A and C inherit the sun_full τ; nothing is re-estimated here). Contrasts, mean
of the four regimes: B − A, C − A, D − C, D − A per condition (balanced accuracy, sensitivity, specificity, AUROC), and
the key contrasts

* type of negatives: [X − A](200seq) − [X − A](200iso),
* support size:      [X − A](500) − [X − A](200seq),     for X = D and X = C.

Inference as registered (``analysis/v3``): paired estimate (mean of the 15 per-run values), t-test on the 5 partition
means (``mixed``, variant ``ttest``), cluster bootstrap of the E4 units (B = 2000, seed 12345). The bootstrap space is
anchored on the registered E4 units (a zero-weight contrast on the registered 50-frame views, which hold every unit), so
the draws are those of the registered e4 family and of ``e4_common_eval``; the script checks that the space equals the
one built from every registered E4 view, and that the 200iso and 500 contrasts equal
``results_v3/verification/e4_common_eval/contrasts_long.csv`` (common set).

``--dry-seq`` runs the whole pipeline with the registered 200-frame runs standing in for 200seq (code test before the
new runs exist): the type contrasts must then be exactly 0.

    PYTHONPATH=. python scripts/v3/e4_negtype.py --seq-root checkpoints_v3/sens_e4_negtype/classification \
        --out results_v3/verification/e4_negtype
    PYTHONPATH=. python scripts/v3/e4_negtype.py --dry-seq --out <scratch dir>
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
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.v3 import mixed  # noqa: E402
from analysis.v3.design import Contrast, Space, View, bootstrap, build_views, ci95, complete_terms  # noqa: E402
from analysis.v3.loader import load_campaign  # noqa: E402
from analysis.v3.sap import B_BOOT, BOOT_SEED, E4_BUDGETS, LABEL, MODEL_VARIANT, REPO, THRESHOLD_FREE  # noqa: E402
from analysis.v3.tables import to_markdown  # noqa: E402

REG_ROOT = REPO / "checkpoints_v3/campaign/classification"
COMMON_EVAL = REPO / "results_v3/verification/e4_common_eval/contrasts_long.csv"
CONDITIONS = ("200iso", "200seq", "500")
ARMS = ("A", "B", "C", "D")
METRICS = ("balanced_accuracy", "recall", "specificity", "auroc")
COMPARISONS = (("B", "A"), ("C", "A"), ("D", "C"), ("D", "A"))
KEY = {"type": ("200seq", "200iso"), "size": ("500", "200seq")}
ANCHOR = "anchor"


def restrict(v: View, key, ids: np.ndarray, whole: bool) -> View:
    rows = np.flatnonzero(np.isin(v.frames.col("frame_id"), ids))
    if rows.size != ids.size:
        raise RuntimeError(f"{v.key}: {rows.size} of the {ids.size} common frames")
    if whole and rows.size != len(v.frames):
        raise RuntimeError(f"{v.key}: evaluation split is not the common set ({len(v.frames)} frames)")
    return View(key, v.frames, v.tau, rows, v.unit_kind, v.group)


def seq_campaign(root: Path, reg, regimes, runs):
    """Load the 200seq runs and check them against the registered campaign (no sun_full parent under this root)."""

    camp = load_campaign([root])
    failed = [x for x in camp.checks if not x["passed"] and x["check"] not in ("parent_present",)]
    if failed:
        raise RuntimeError(f"{len(failed)} integrity checks failed, first: {failed[0]}")
    keys = {r.key for r in camp.runs}
    expected = {("e4", 200, g, k, s) for g in regimes for k, s in runs}
    if keys != expected:
        raise RuntimeError(f"200seq runs: {len(keys & expected)} of {len(expected)} expected, {len(keys - expected)} unexpected")
    idx = reg.index()
    for r in camp.runs:
        parent = idx[("sun_full", None, r.regime, r.k, r.seed)]
        if not (r.tau["A"] == parent.tau["primary"] == r.tau["C"]):
            raise RuntimeError(f"{r.key}: inherited τ differs from the registered sun_full τ")
    return camp


def arm_terms(cond: str, treat: str, base: str, metric: str, regimes, k: int, s: int, sign: float = 1.0):
    n = len(regimes)
    return [t for g in regimes for t in [(("e4", cond, g, k, s, treat), metric, sign / n),
                                         (("e4", cond, g, k, s, base), metric, -sign / n)]]


def make_contrasts(views, runs, regimes) -> List[Contrast]:
    out: List[Contrast] = []

    def add(cid, meta, make):
        terms = complete_terms(views, runs, make)
        if len(terms) != len(runs):
            raise RuntimeError(f"{cid}: {len(terms)} complete runs of {len(runs)}")
        out.append(Contrast(cid, "e4", meta, terms))

    for cond in CONDITIONS:
        for treat, base in COMPARISONS:
            for metric in METRICS:
                if {treat, base} in ({"A", "B"}, {"C", "D"}) and metric in THRESHOLD_FREE:
                    continue  # identical probabilities: the AUROC difference is 0 by construction
                add(f"arm|{cond}|{treat}-{base}|{metric}", {"kind": "arm", "condition": cond, "comparison": f"{treat} - {base}", "metric": metric},
                    lambda k, s, c=cond, t=treat, b=base, m=metric: arm_terms(c, t, b, m, regimes, k, s))
    for name, (hi, lo) in KEY.items():
        for treat in ("D", "C"):
            for metric in METRICS:
                add(f"key|{name}|{treat}-A|{metric}",
                    {"kind": name, "condition": f"{hi} - {lo}", "comparison": f"[{treat} - A]({hi}) - [{treat} - A]({lo})", "metric": metric},
                    lambda k, s, t=treat, m=metric, hi=hi, lo=lo: arm_terms(hi, t, "A", m, regimes, k, s)
                    + arm_terms(lo, t, "A", m, regimes, k, s, sign=-1.0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-root", type=Path, default=None, help="checkpoint root of the 200seq runs")
    ap.add_argument("--dry-seq", action="store_true", help="registered 200-frame runs stand in for 200seq (code test)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    if args.dry_seq == (args.seq_root is not None):
        ap.error("give exactly one of --seq-root and --dry-seq")
    t0 = time.time()
    reg = load_campaign([REG_ROOT])
    failed = [x for x in reg.checks if not x["passed"]]
    if failed:
        raise RuntimeError(f"{len(failed)} integrity checks failed, first: {failed[0]}")
    reg_views = build_views(reg)
    regimes, partitions, seeds = reg.regimes(), reg.partitions(), reg.seeds()
    runs = [(k, s) for k in partitions for s in seeds]
    idx = reg.index()
    common = {(k, s): reduce(np.intersect1d, [idx[("e4", b, g, k, s)].frames["test"].col("frame_id")
                                             for b in E4_BUDGETS for g in regimes]) for k, s in runs}

    if args.dry_seq:
        seq_views, seq_info = reg_views, {"source": "registered 200-frame runs (dry run)"}
        seq_runs = {r.key: r for r in reg.runs if r.experiment == "e4" and r.variant == 200}
    else:
        seq = seq_campaign(args.seq_root, reg, regimes, runs)
        seq_views, seq_info = build_views(seq), {"source": str(args.seq_root)}
        seq_runs = seq.index()

    # condition views on the common set; the 500 and 200seq evaluation splits must be the common set itself
    views: Dict = {}
    for g in regimes:
        for k, s in runs:
            for arm in ARMS:
                for cond, src, budget, whole in (("200iso", reg_views, 200, False), ("500", reg_views, 500, True),
                                                 ("200seq", seq_views, 200, not args.dry_seq)):
                    key = ("e4", cond, g, k, s, arm)
                    views[key] = restrict(src[("e4", budget, g, k, s, arm)], key, common[(k, s)], whole)
            views[("e4", ANCHOR, g, k, s, "A")] = reg_views[("e4", 50, g, k, s, "A")]

    point = {vk: v.metrics() for vk, v in views.items() if vk[1] != ANCHOR}
    mine = make_contrasts(views, runs, regimes)
    anchor_c = Contrast("anchor", "e4", {}, {(k, s): [(("e4", ANCHOR, regimes[0], k, s, "A"), "auroc", 0.0)] for k, s in runs})
    # the bootstrap space of these contrasts = the space of every registered E4 view (registered draws)
    full = Space([v for vk, v in reg_views.items() if vk[0] == "e4"])
    anchored = Space([views[vk] for vk in {vk for c in mine + [anchor_c] for vk in c.views()}])
    if full.flat != anchored.flat:
        raise RuntimeError("bootstrap space differs from the registered E4 space")
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in mine}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    reps, counts = bootstrap(views, mine + [anchor_c], B=B_BOOT, seed=BOOT_SEED, workers=args.workers)
    print(f"[bootstrap] e4: {len(mine)} contrasts, units {counts} ({time.time() - t0:.0f}s)")
    rows = []
    for j, c in enumerate(mine):
        f, sp = fits.loc[c.id], mixed.spread(per_run[c.id])
        lo, hi = ci95(reps[:, j])
        rows.append({"id": c.id, **c.meta, "estimate": sp["mean"], "sd_partition": sp["sd_partition"], "sd_seed": sp["sd_seed"],
                     "n_runs": sp["n_runs"], "method": f["method"], "t_ci_lo": f["ci_lo"], "t_ci_hi": f["ci_hi"],
                     "p_unadjusted": f["p"], "boot_lo": lo, "boot_hi": hi, "boot_B": int(np.isfinite(reps[:, j]).sum())})
    long = pd.DataFrame(rows)

    # check: 200iso and 500 = e4_common_eval (common set), all quantities
    ref = pd.read_csv(COMMON_EVAL, float_precision="round_trip")
    ref = ref[ref["set"] == "common"].set_index(["budget", "comparison", "metric"])
    check = []
    for _, r in long[(long["kind"] == "arm") & long["condition"].isin(["200iso", "500"])].iterrows():
        rr = ref.loc[(int(r["condition"][:3]), r["comparison"], r["metric"])]
        for q in ("estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi", "boot_B"):
            d = abs(float(r[q]) - float(rr[q]))
            check.append({"contrast": r["id"], "quantity": q, "recomputed": r[q], "e4_common_eval": rr[q], "abs_diff": d,
                          "exact": d <= 5e-10 * max(1.0, abs(float(rr[q])))})
    check = pd.DataFrame(check)
    n_exact = int(check["exact"].sum())
    if n_exact != len(check):
        raise RuntimeError(f"200iso / 500 do not reproduce e4_common_eval: {n_exact}/{len(check)} exact")
    if args.dry_seq:
        typ = long[long["kind"] == "type"]
        if not (typ[["estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"]].abs().max().max() < 1e-12):
            raise RuntimeError("dry run: type contrasts are not 0")

    # per-run metrics and τ per condition; zero-shot outputs of 200seq vs the registered 500 run (same frames)
    run_rows, zs_diff = [], []
    for (_, cond, g, k, s, arm), v in sorted(((vk, v) for vk, v in views.items() if vk[1] != ANCHOR), key=lambda x: repr(x[0])):
        src = seq_runs[("e4", 200, g, k, s)] if cond == "200seq" else idx[("e4", 200 if cond == "200iso" else 500, g, k, s)]
        run_rows.append({"condition": cond, "regime": g, "k": k, "seed": s, "arm": arm, "tau": v.tau, "n": int(v.rows.size),
                         "kept_inherited_tau_D": src.info.get("kept_inherited_tau_D") if arm == "D" else None,
                         **{m: point[("e4", cond, g, k, s, arm)][m] for m in METRICS}})
        if cond == "200seq" and arm == "A":
            r5 = idx[("e4", 500, g, k, s)].frames["zeroshot"]
            a = pd.Series(v.frames.probs, index=v.frames.col("frame_id"))
            b = pd.Series(r5.probs, index=r5.col("frame_id"))
            zs_diff.append(float(np.abs(a.loc[b.index].to_numpy() - b.to_numpy()).max()))
    run_metrics = pd.DataFrame(run_rows)
    arm_means = run_metrics.groupby(["condition", "arm"])[list(METRICS)].mean().reset_index()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run_metrics.to_csv(out / "run_metrics.csv", index=False, float_format="%.10g")
    arm_means.to_csv(out / "arm_means.csv", index=False, float_format="%.10g")
    long.to_csv(out / "contrasts_long.csv", index=False, float_format="%.10g")
    check.to_csv(out / "common_eval_check.csv", index=False, float_format="%.17g")
    info = {"registered_root": str(REG_ROOT.relative_to(REPO)), "seq": seq_info, "dry_seq": args.dry_seq, "regimes": regimes,
            "partitions": partitions, "seeds": seeds, "B": B_BOOT, "boot_seed": BOOT_SEED, "model_variant": MODEL_VARIANT,
            "bootstrap_units": {str(g): v for g, v in counts.items()}, "bootstrap_space_equals_registered_e4": True,
            "n_contrasts": len(mine),
            "common_eval_check": {"n": len(check), "exact": n_exact, "max_abs_diff": float(check["abs_diff"].max())},
            "zeroshot_200seq_vs_registered_500_max_abs_prob_diff": max(zs_diff),
            "kept_inherited_tau_D": run_metrics[run_metrics["arm"] == "D"].groupby("condition")["kept_inherited_tau_D"].sum().astype(int).to_dict(),
            "elapsed_s": round(time.time() - t0, 1)}
    show = long[["kind", "condition", "comparison", "metric", "estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"]]
    md = ["# E4: support size vs type of negatives (common evaluation set)", "",
          "Post hoc sensitivity analysis, not registered. Mean of the four regimes (" + ", ".join(LABEL.get(g, g) for g in regimes)
          + "). Estimate = mean of the paired per-run values (5 partitions × 3 seeds); t-test CI on the 5 partition means; "
          "cluster-bootstrap CI (B = 2000, seed 12345, the registered draws)." + (" DRY RUN: 200seq = registered 200." if args.dry_seq else ""),
          "", to_markdown(arm_means, "Arm means (15 runs × 4 regimes)"), to_markdown(show, "Contrasts"),
          f"200iso and 500 recomputed: {n_exact}/{len(check)} quantities equal to e4_common_eval (max |diff| "
          f"{info['common_eval_check']['max_abs_diff']:.3g})."]
    (out / "contrasts.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
