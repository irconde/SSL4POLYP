#!/usr/bin/env python3
"""Exploratory comparison: common learning rate (1e-3) vs the per-regime learning rates of campaign v3.

Plan: ``docs/v3/plan_exploratorio_lr_comun.md`` (exploratory, decided after seeing the campaign v3 results; not
used to confirm hypotheses). Uses the analysis code of ``analysis/v3`` unchanged (loader, views, contrasts,
t-test on the K partition means, cluster bootstrap) and adds only the pairing of the two campaigns:

* Regimes SUP-imnet, SSL-imnet and SSL-colon in both campaigns (the v3 values are recomputed on these three
  regimes, so H5 and H7, which average over regimes, differ from the registered four-regime values).
* For every contrast, three versions over the same paired runs (partition × seed): common lr, v3, and
  common − v3. Both campaigns use the same test sets and packs, so one bootstrap sample of units per replicate
  is shared by the two campaigns (same groups as in the SAP analysis), and the difference has its own t-test
  CI (on the K partition means of the per-run differences) and bootstrap CI.
* No Holm adjustment and no confirmation verdicts.

    PYTHONPATH=. python -m analysis.v3.exploratory_lr_common --out results_v3/analysis_exploratory_lr_common
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import mixed  # noqa: E402
from .analysis import run_metrics_table  # noqa: E402
from .contrasts import hypothesis_contrasts, secondary_contrasts  # noqa: E402
from .design import Contrast, View, bootstrap, build_views, ci95  # noqa: E402
from .loader import Campaign, load_campaign  # noqa: E402
from .sap import B_BOOT, BOOT_SEED, CALIBRATION, LABEL, METRICS, MODEL_VARIANT, REPO  # noqa: E402
from .tables import compact_levels, levels, to_markdown  # noqa: E402

REGIMES3 = ("sup_imnet", "ssl_imnet", "ssl_colon")
HYPS = ("H1", "H3", "H4", "H5", "H7")
PREFIX = "lrc:"
V3_ROOT = REPO / "checkpoints_v3/campaign/classification"
LRC_ROOT = REPO / "checkpoints_v3/campaign_lr_common/classification"


def _restrict(c: Campaign) -> Campaign:
    return Campaign([r for r in c.runs if r.regime in REGIMES3], c.roots, c.checks)


def _rename_views(views: Dict[tuple, View]) -> Dict[tuple, View]:
    out = {}
    for (fam, variant, regime, k, s, slot), v in views.items():
        key = (fam, variant, PREFIX + regime, k, s, slot)
        out[key] = replace(v, key=key, units=v.units)
    return out


def _lrc_terms(terms):
    return [((fam, var, PREFIX + reg, k, s, slot), m, w) for (fam, var, reg, k, s, slot), m, w in terms]


def build_contrasts(views_v3: Dict[tuple, View], partitions, seeds) -> List[Contrast]:
    hyp, _ = hypothesis_contrasts(views_v3, REGIMES3, partitions, seeds)
    base = [hyp[h] for h in HYPS if hyp.get(h) is not None]
    sec = secondary_contrasts(views_v3, REGIMES3, partitions, seeds)
    keep = []
    for c in sec:
        a, exp = c.meta.get("analysis"), c.meta.get("experiment")
        if a == "regime comparison" and exp in ("e1", "e2", "e3", "e4"):
            keep.append(c)
        elif a in ("AULC comparison", "AULC level", "flat - polypoid gap"):
            keep.append(c)
        elif a == "E4 arm contrast" and c.meta.get("comparison") in ("arm B - arm A", "arm C - arm A") \
                and c.meta.get("metric") in ("auroc", "balanced_accuracy"):
            keep.append(c)
    out: List[Contrast] = []
    for c in base + keep:
        terms_lc = {rid: _lrc_terms(t) for rid, t in c.terms.items()}
        terms_v3 = dict(c.terms)
        terms_diff = {rid: terms_lc[rid] + [(v, m, -w) for v, m, w in terms_v3[rid]] for rid in terms_v3}
        for tag, terms in (("common", terms_lc), ("v3", terms_v3), ("diff", terms_diff)):
            out.append(Contrast(f"{tag}|{c.id}", c.family, {**c.meta, "base_id": c.id, "version": tag}, terms))
    return out


def run(v3_root: Path, lrc_root: Path, B: int, seed: int, workers: int, log=print):
    t0 = time.time()
    c3 = _restrict(load_campaign([v3_root]))
    cl = _restrict(load_campaign([lrc_root]))
    for name, c in (("v3", c3), ("common lr", cl)):
        failed = [x for x in c.checks if not x["passed"]]
        if failed:
            raise RuntimeError(f"{name}: {len(failed)} integrity checks failed, first: {failed[0]}")
    k3 = {(r.experiment, r.variant, r.regime, r.k, r.seed) for r in c3.runs}
    kl = {(r.experiment, r.variant, r.regime, r.k, r.seed) for r in cl.runs}
    if k3 != kl:
        raise RuntimeError(f"run sets differ: only v3 {len(k3 - kl)}, only common {len(kl - k3)}")
    v3_views, lc_views = build_views(c3), _rename_views(build_views(cl))
    # paired runs must score the same frames (same test sets and packs)
    for (fam, var, reg, k, s, slot), v in v3_views.items():
        w = lc_views[(fam, var, PREFIX + reg, k, s, slot)]
        if not np.array_equal(v.frames.col("frame_id")[v.rows], w.frames.col("frame_id")[w.rows]):
            raise RuntimeError(f"frames differ between campaigns for {(fam, var, reg, k, s, slot)}")
    views = {**v3_views, **lc_views}
    log(f"[load] v3 {len(c3.runs)} runs, common lr {len(cl.runs)} runs, {len(views)} views ({time.time() - t0:.0f}s)")
    partitions, seeds = c3.partitions(), c3.seeds()
    contrasts = build_contrasts(v3_views, partitions, seeds)
    used = {vk for c in contrasts for vk in c.views()}
    point = {vk: views[vk].metrics() for vk in views}
    log(f"[metrics] {len(point)} views; {len(contrasts)} contrasts ({time.time() - t0:.0f}s)")
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in contrasts}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    boot: Dict[str, tuple] = {}
    for fam in sorted({c.family for c in contrasts}):
        members = [c for c in contrasts if c.family == fam]
        reps, counts = bootstrap(views, members, B=B, seed=seed, workers=workers)
        for j, c in enumerate(members):
            lo, hi = ci95(reps[:, j])
            boot[c.id] = (lo, hi, int(np.isfinite(reps[:, j]).sum()))
        log(f"[bootstrap] {fam}: {len(members)} contrasts ({time.time() - t0:.0f}s)")
    rows = []
    for c in contrasts:
        f = fits.loc[c.id]
        sp = mixed.spread(per_run[c.id])
        lo, hi, nb = boot[c.id]
        rows.append({"id": c.id, **{k: v for k, v in c.meta.items()}, "estimate": sp["mean"], "sd_partition": sp["sd_partition"],
                     "sd_seed": sp["sd_seed"], "n_runs": sp["n_runs"], "method": f["method"], "t_ci_lo": f["ci_lo"],
                     "t_ci_hi": f["ci_hi"], "p_unadjusted": f["p"], "boot_lo": lo, "boot_hi": hi, "boot_B": nb})
    long = pd.DataFrame(rows)
    rm_v3 = run_metrics_table(v3_views, point)
    rm_lc = run_metrics_table(lc_views, point)
    rm_lc["regime"] = rm_lc["regime"].str.replace(PREFIX, "", regex=False)
    info = {"v3_root": str(v3_root), "lrc_root": str(lrc_root), "regimes": list(REGIMES3), "partitions": partitions,
            "seeds": seeds, "B": B, "boot_seed": seed, "model_variant": MODEL_VARIANT, "n_contrasts_per_version": len(contrasts) // 3,
            "integrity_checks": {"v3": len(c3.checks), "common": len(cl.checks)}, "elapsed_s": round(time.time() - t0, 1)}
    return long, rm_v3, rm_lc, info


def wide(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["base_id", "analysis", "hypothesis", "experiment", "variant", "slot", "regime", "comparison", "metric"]
    keys = [k for k in keys if k in long.columns]
    base = long[long["version"] == "v3"][keys].drop_duplicates("base_id").set_index("base_id")
    for tag in ("common", "v3", "diff"):
        sub = long[long["version"] == tag].set_index("base_id")
        for col in ("estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"):
            base[f"{tag}_{col}"] = sub[col]
    base["n_runs"] = long[long["version"] == "diff"].set_index("base_id")["n_runs"]
    base["diff_boot_B"] = long[long["version"] == "diff"].set_index("base_id")["boot_B"]
    return base.reset_index()


def level_tables(rm_v3: pd.DataFrame, rm_lc: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    specs = {
        "table3_e1_performance": ("e1", METRICS + CALIBRATION, ("variant", "regime")),
        "table5_e2_morphology": ("e2", METRICS, ("slot", "regime")),
        "table6_e3_zeroshot": ("e3", METRICS + CALIBRATION, ("regime",)),
        "table7_negative_source": ("neg", ("auroc", "specificity", "balanced_accuracy"), ("slot", "regime")),
        "table8_e4_arms": ("e4", tuple(m for m in METRICS + CALIBRATION if m != "auprc"), ("variant", "slot", "regime")),
        "table_sun_full": ("sun_full", METRICS + CALIBRATION, ("regime",)),
    }
    out = {}
    for name, (fam, metrics, by) in specs.items():
        lc, v3 = levels(rm_lc, fam, metrics, by), levels(rm_v3, fam, metrics, by)
        keys = [c for c in list(by) + ["regime_label"] if c in lc.columns]
        cols = [f"{m}_mean" for m in metrics]
        m = lc[keys + cols].merge(v3[keys + cols], on=keys, suffixes=("_common", "_v3"), validate="one_to_one")
        for metric in metrics:
            m[f"{metric}_diff"] = m[f"{metric}_mean_common"] - m[f"{metric}_mean_v3"]
        out[name] = m
    return out


def figure(w: pd.DataFrame, path: Path) -> None:
    rows = w[w["base_id"].isin(["H1", "H3", "H4"])].set_index("base_id").loc[["H1", "H3", "H4"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = {"common": "#eb6834", "v3": "#2a78d6"}
    labels = {"common": "common lr 1e-3", "v3": "campaign v3 (per-regime lr)"}
    for i, (h, r) in enumerate(rows.iterrows()):
        y = len(rows) - 1 - i
        for dy, tag, marker in ((0.13, "common", "s"), (-0.13, "v3", "o")):
            ax.plot([r[f"{tag}_t_ci_lo"], r[f"{tag}_t_ci_hi"]], [y + dy] * 2, color=colors[tag], linewidth=3, solid_capstyle="round")
            ax.plot([r[f"{tag}_boot_lo"], r[f"{tag}_boot_hi"]], [y + dy - 0.06] * 2, color=colors[tag], linewidth=1, alpha=0.7)
            ax.plot(r[f"{tag}_estimate"], y + dy, marker=marker, color=colors[tag], markersize=7, linestyle="none",
                    label=labels[tag] if i == 0 else None)
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{h}: {r['metric']} ({r['comparison']})" for h, r in rows.iloc[::-1].iterrows()], fontsize=8)
    ax.set_xlabel("SSL-colon − SUP-imnet (thick: t-test 95 % CI; thin: bootstrap 95 % CI)")
    ax.set_title("Exploratory: common learning rate vs campaign v3 (H1, H3, H4 contrasts)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--v3-root", type=Path, default=V3_ROOT)
    ap.add_argument("--lrc-root", type=Path, default=LRC_ROOT)
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    long, rm_v3, rm_lc, info = run(args.v3_root, args.lrc_root, args.B, args.boot_seed, args.workers)
    out = args.out
    (out / "tables").mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "contrasts_long.csv", index=False, float_format="%.10g")
    w = wide(long)
    w.to_csv(out / "contrasts_common_vs_v3.csv", index=False, float_format="%.10g")
    pd.concat([rm_lc.assign(campaign="common_lr"), rm_v3.assign(campaign="v3")]).to_csv(out / "run_metrics.csv", index=False, float_format="%.10g")
    note = ("Exploratory (not registered in the SAP): no Holm adjustment, no confirmation verdicts. Estimate = mean of the "
            "paired per-run differences (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI "
            "(B = 2000, seed 12345, samples shared by both campaigns). diff = common − v3 per paired run. v3 recomputed on "
            "three regimes (H5 and H7 average three regimes, not four).")
    md_cols = ["base_id", "metric", "comparison", "common_estimate", "common_t_ci_lo", "common_t_ci_hi", "v3_estimate",
               "v3_t_ci_lo", "v3_t_ci_hi", "diff_estimate", "diff_t_ci_lo", "diff_t_ci_hi", "diff_boot_lo", "diff_boot_hi"]
    hyp = w[w["base_id"].isin(HYPS)].set_index("base_id").loc[[h for h in HYPS if h in set(w["base_id"])]].reset_index()
    parts = ["# Exploratory experiment: common learning rate 1e-3 vs campaign v3", "", note, "",
             to_markdown(hyp[[c for c in md_cols if c in hyp]], "Contrasts of H1, H3, H4, H5 and H7")]
    sec = w[~w["base_id"].isin(HYPS) & w["metric"].isin(["auroc", "auprc", "balanced_accuracy", "AULC(auprc)", "AULC(auroc)"])]
    parts.append(to_markdown(sec[[c for c in md_cols + ["experiment", "variant", "slot", "regime"] if c in sec]],
                             "Secondary contrasts (subset of metrics; all in contrasts_common_vs_v3.csv)"))
    (out / "contrasts_common_vs_v3.md").write_text("\n".join(parts))
    for name, df in level_tables(rm_v3, rm_lc).items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False, float_format="%.10g")
        (out / "tables" / f"{name}.md").write_text(to_markdown(df, f"{name}: levels, common lr vs v3 (mean over runs)"))
    for name, mask in (("table4_e1_aulc", w["analysis"].isin(["AULC level", "AULC comparison"])),
                       ("table9_e4_arm_contrasts", w["analysis"] == "E4 arm contrast"),
                       ("table5b_e2_gap", w["analysis"] == "flat - polypoid gap")):
        df = w[mask]
        df.to_csv(out / "tables" / f"{name}.csv", index=False, float_format="%.10g")
        (out / "tables" / f"{name}.md").write_text(to_markdown(df[[c for c in md_cols + ["regime", "variant"] if c in df]], name))
    figure(w, out / "fig_h1_h3_h4_common_vs_v3.png")
    (out / "summary.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
