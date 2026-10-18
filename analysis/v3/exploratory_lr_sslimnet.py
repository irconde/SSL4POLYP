#!/usr/bin/env python3
"""Exploratory comparison: SSL-imnet fine-tuned with lr 1e-4 vs campaign v3 (lr 3e-4), E1 budgets and AULC.

Plan: ``docs/v3/plan_exploratorio_lr_sslimnet.md`` (exploratory, decided after seeing results; no Holm, no verdicts).
Uses ``analysis/v3`` unchanged (loader, views, AULC weights, t-test on the K partition means, cluster bootstrap).

For each comparator X in {SSL-colon, SSL-imnet-matched} (campaign v3 runs), metric in {AUPRC, AUROC}, and each E1
budget and the AULC over 5–100 %, three versions over the same paired runs (partition × seed):
  lr1e-4 : X − SSL-imnet(lr 1e-4)
  v3     : X − SSL-imnet(lr 3e-4, campaign v3)
  diff   : lr1e-4 − v3 = SSL-imnet(3e-4) − SSL-imnet(1e-4)   (X cancels; same for both comparators)
Both campaigns use the same partition test sets, so the bootstrap shares the sample of cases per partition.

    PYTHONPATH=. python -m analysis.v3.exploratory_lr_sslimnet --out results_v3/analysis_exploratory_lr_sslimnet
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import mixed  # noqa: E402
from .analysis import run_metrics_table  # noqa: E402
from .design import Contrast, View, aulc_weights, bootstrap, build_views, ci95  # noqa: E402
from .loader import Campaign, load_campaign  # noqa: E402
from .sap import B_BOOT, BOOT_SEED, E1_PERCENTS, MODEL_VARIANT, REPO  # noqa: E402
from .tables import levels, to_markdown  # noqa: E402

IMNET, NEW = "ssl_imnet", "ssl_imnet@lr1e-4"
COMPARATORS = ("ssl_colon", "ssl_imnet_matched")
METRICS = ("auprc", "auroc")
V3_ROOT = REPO / "checkpoints_v3/campaign/classification"
NEW_ROOT = REPO / "checkpoints_v3/campaign_lr_sslimnet_1e-4/classification"


def _views(c: Campaign, regimes, rename=None) -> Dict[tuple, View]:
    sub = Campaign([r for r in c.runs if r.regime in regimes and r.experiment in ("sun_full", "e1")], c.roots, c.checks)
    out = {}
    for (fam, var, reg, k, s, slot), v in build_views(sub).items():
        reg2 = rename.get(reg, reg) if rename else reg
        key = (fam, var, reg2, k, s, slot)
        out[key] = replace(v, key=key)
    return out


def build_contrasts(partitions, seeds) -> List[Contrast]:
    runs = [(k, s) for k in partitions for s in seeds]
    w = dict(zip(E1_PERCENTS, aulc_weights(E1_PERCENTS)))
    out: List[Contrast] = []
    for x in COMPARATORS:
        for m in METRICS:
            slots = [(f"p{p}", {p: 1.0}) for p in E1_PERCENTS] + [("AULC", w)]
            for label, weights in slots:
                def terms(base, weights=weights, m=m, x=x):
                    return {(k, s): [t for p, wt in weights.items() for t in
                                     ((("e1", p, x, k, s, "all"), m, wt), (("e1", p, base, k, s, "all"), m, -wt))] for k, s in runs}
                t_new, t_v3 = terms(NEW), terms(IMNET)
                t_diff = {rid: t_new[rid] + [(v, mm, -wt) for v, mm, wt in t_v3[rid]] for rid in runs}
                meta = {"comparison": f"{x} - ssl_imnet", "metric": m if label != "AULC" else f"AULC({m})", "budget": label}
                base_id = f"{x}-ssl_imnet|{label}|{m}"
                for tag, t in (("lr1e-4", t_new), ("v3", t_v3), ("diff", t_diff)):
                    out.append(Contrast(f"{tag}|{base_id}", "e1", {**meta, "base_id": base_id, "version": tag}, t))
    return out


def run(v3_roots: List[Path], new_root: Path, B: int, seed: int, workers: int, log=print):
    t0 = time.time()
    c3, cn = load_campaign(v3_roots), load_campaign([new_root])
    for name, c in (("v3", c3), ("lr 1e-4", cn)):
        failed = [x for x in c.checks if not x["passed"]]
        if failed:
            raise RuntimeError(f"{name}: {len(failed)} integrity checks failed, first: {failed[0]}")
    v3_views = _views(c3, (IMNET,) + COMPARATORS)
    new_views = _views(cn, (IMNET,), rename={IMNET: NEW})
    partitions, seeds = c3.partitions(), c3.seeds()
    expected = {("e1", p, k, s) for p in E1_PERCENTS for k in partitions for s in seeds}
    got = {(f, v, k, s) for (f, v, r, k, s, _) in new_views if f == "e1"}
    if got != expected:
        raise RuntimeError(f"lr 1e-4 campaign incomplete: {len(expected - got)} E1 runs missing")
    for (f, v, r, k, s, slot), view in new_views.items():
        ref = v3_views[(f, v, IMNET, k, s, slot)]
        if not np.array_equal(view.frames.col("frame_id")[view.rows], ref.frames.col("frame_id")[ref.rows]):
            raise RuntimeError(f"frames differ for {(f, v, k, s)}")
    views = {**v3_views, **new_views}
    contrasts = build_contrasts(partitions, seeds)
    point = {vk: views[vk].metrics() for vk in views}
    log(f"[load] {len(views)} views, {len(contrasts)} contrasts ({time.time() - t0:.0f}s)")
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in contrasts}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    reps, counts = bootstrap(views, contrasts, B=B, seed=seed, workers=workers)
    log(f"[bootstrap] {counts} ({time.time() - t0:.0f}s)")
    rows = []
    for j, c in enumerate(contrasts):
        f, sp = fits.loc[c.id], mixed.spread(per_run[c.id])
        lo, hi = ci95(reps[:, j])
        rows.append({"id": c.id, **c.meta, "estimate": sp["mean"], "sd_partition": sp["sd_partition"], "sd_seed": sp["sd_seed"],
                     "n_runs": sp["n_runs"], "method": f["method"], "t_ci_lo": f["ci_lo"], "t_ci_hi": f["ci_hi"],
                     "p_unadjusted": f["p"], "boot_lo": lo, "boot_hi": hi, "boot_B": int(np.isfinite(reps[:, j]).sum())})
    long = pd.DataFrame(rows)
    rm = pd.concat([run_metrics_table(v3_views, point).assign(campaign="v3"), run_metrics_table(new_views, point).assign(campaign="lr1e-4")])
    info = {"v3_roots": [str(r) for r in v3_roots], "new_root": str(new_root), "partitions": partitions, "seeds": seeds, "B": B,
            "boot_seed": seed, "model_variant": MODEL_VARIANT, "n_base_contrasts": len(contrasts) // 3,
            "elapsed_s": round(time.time() - t0, 1)}
    return long, rm, info


def wide(long: pd.DataFrame) -> pd.DataFrame:
    base = long[long["version"] == "v3"][["base_id", "comparison", "budget", "metric"]].set_index("base_id")
    for tag in ("lr1e-4", "v3", "diff"):
        sub = long[long["version"] == tag].set_index("base_id")
        for col in ("estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"):
            base[f"{tag}_{col}"] = sub[col]
    base["n_runs"] = long[long["version"] == "diff"].set_index("base_id")["n_runs"]
    return base.reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--v3-roots", type=Path, nargs="+", default=[V3_ROOT])
    ap.add_argument("--new-root", type=Path, default=NEW_ROOT)
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    long, rm, info = run(args.v3_roots, args.new_root, args.B, args.boot_seed, args.workers)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "contrasts_long.csv", index=False, float_format="%.10g")
    w = wide(long)
    w.to_csv(out / "contrasts_lr1e-4_vs_v3.csv", index=False, float_format="%.10g")
    rm.to_csv(out / "run_metrics.csv", index=False, float_format="%.10g")
    lv = []
    for camp, regime in (("v3", IMNET), ("lr1e-4", NEW)):
        r = rm[(rm["campaign"] == camp) & (rm["regime"] == regime)].copy()
        r["regime"] = f"ssl_imnet ({'lr 3e-4, v3' if camp == 'v3' else 'lr 1e-4'})"
        lv.append(levels(r, "e1", ("auprc", "auroc"), by=("variant", "regime")))
    lev = pd.concat(lv, ignore_index=True).sort_values(["variant", "regime"])
    lev.to_csv(out / "levels_ssl_imnet_e1.csv", index=False, float_format="%.10g")
    note = ("Exploratory (not registered in the SAP): no Holm, no verdicts. Estimate = mean of the paired per-run "
            "differences (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, "
            "seed 12345). diff = (X − SSL-imnet lr 1e-4) − (X − SSL-imnet lr 3e-4) = SSL-imnet(3e-4) − SSL-imnet(1e-4).")
    cols = ["comparison", "budget", "metric", "lr1e-4_estimate", "lr1e-4_t_ci_lo", "lr1e-4_t_ci_hi", "lr1e-4_boot_lo", "lr1e-4_boot_hi",
            "v3_estimate", "v3_t_ci_lo", "v3_t_ci_hi", "diff_estimate", "diff_t_ci_lo", "diff_t_ci_hi", "diff_boot_lo", "diff_boot_hi"]
    md = ["# Exploratory: SSL-imnet lr 1e-4 vs campaign v3 (lr 3e-4), E1", "", note, "", to_markdown(w[cols], "Contrasts"),
          to_markdown(lev[[c for c in lev.columns if not c.endswith("_sd_partition") and not c.endswith("_sd_seed")]],
                      "SSL-imnet E1 levels (mean over runs)")]
    (out / "contrasts_lr1e-4_vs_v3.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
