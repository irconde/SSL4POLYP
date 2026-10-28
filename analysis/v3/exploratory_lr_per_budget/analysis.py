#!/usr/bin/env python3
"""Exploratory: learning rate selected per regime and E1 budget vs the registered single learning rate.

Plan: ``docs/v3/exploratory_lr_per_budget_plan.md`` (exploratory, decided after seeing results; no Holm, no verdicts).
Uses ``analysis/v3`` unchanged: loader, views, AULC weights, t-test on the K partition means (``mixed``) and the cluster
bootstrap (``design.bootstrap``, B = 2000, seed 12345).

Runs per regime and learning rate (E1, 5 partitions × 3 seeds × 5 budgets), from ``SOURCES``:
  registered learning rate      checkpoints_v3/campaign/classification (campaigns 3 and 4)
  reused (revision 1)           campaign_lr_common (1e-3: SUP-imnet, SSL-imnet, SSL-colon),
                                campaign_lr_sslimnet_1e-4 (1e-4: SSL-imnet)
  new                           exp_lr_per_budget/lr<lr>/classification
Every run's recorded ``run.lr`` must equal its source's learning rate.

Selection (per regime and budget): highest mean validation AP (``val.auprc`` of the best-epoch ``_last.metrics.json``,
checked against its recomputation from the validation CSV) over the 15 runs; the registered learning rate is kept when
the best mean exceeds it by less than 1e-4. The test split takes no part.

Estimands, for SSL-colon − {SUP-imnet, SSL-imnet, SSL-imnet-matched}: AULC of AUPRC (5–100 %) and AUPRC per budget, in
three versions over the same pairs (partition × seed): ``selected`` (each regime at its selected learning rate per
budget), ``registered`` and ``diff`` = selected − registered. Every contrast references only E1 views, so the bootstrap
draws the same case resamples as the registered E1 family; the registered version must reproduce the registered
analysis (``results_v3/analysis/contrast_summary.csv``) exactly.

    PYTHONPATH=. python -m analysis.v3.exploratory_lr_per_budget --out results_v3/analysis_exploratory_lr_per_budget

``--lrs`` restricts the candidate learning rates (the registered one is always a candidate); ``--selection registered``
disables the selection (code check before the new runs exist).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from .. import mixed  # noqa: E402
from ..design import Contrast, View, aulc_weights, bootstrap, build_views, ci95  # noqa: E402
from ..loader import Campaign, load_campaign, load_frames  # noqa: E402
from ..metrics import binary  # noqa: E402
from ..sap import B_BOOT, BOOT_SEED, E1_PERCENTS, MODEL_VARIANT, REPO  # noqa: E402
from ..tables import to_markdown  # noqa: E402

REGIMES = ("sup_imnet", "ssl_imnet", "ssl_colon", "ssl_imnet_matched")
COLON = "ssl_colon"
COMPARATORS = ("sup_imnet", "ssl_imnet", "ssl_imnet_matched")
LRS = ("1e-4", "3e-4", "1e-3")
TIE = 1e-4  # keep the registered learning rate when the best mean validation AP exceeds it by less than this
REGISTERED_ROOT = "checkpoints_v3/campaign/classification"
SOURCES = {  # (regime, lr) -> root, for the non-registered learning rates
    ("sup_imnet", "1e-3"): "checkpoints_v3/campaign_lr_common/classification",
    ("ssl_imnet", "1e-3"): "checkpoints_v3/campaign_lr_common/classification",
    ("ssl_colon", "1e-3"): "checkpoints_v3/campaign_lr_common/classification",
    ("ssl_imnet", "1e-4"): "checkpoints_v3/campaign_lr_sslimnet_1e-4/classification",
    ("sup_imnet", "1e-4"): "checkpoints_v3/exp_lr_per_budget/lr1e-4/classification",
    ("ssl_colon", "3e-4"): "checkpoints_v3/exp_lr_per_budget/lr3e-4/classification",
    ("ssl_imnet_matched", "3e-4"): "checkpoints_v3/exp_lr_per_budget/lr3e-4/classification",
    ("ssl_imnet_matched", "1e-3"): "checkpoints_v3/exp_lr_per_budget/lr1e-3/classification",
}
# registered per-regime contrast ids (results_v3/analysis/contrast_summary.csv)
REGISTERED_IDS = {("AULC", base): ("H1" if base == "sup_imnet" else f"aulc|{COLON}-{base}|auprc") for base in COMPARATORS}
REGISTERED_IDS.update({(f"p{p}", base): f"cmp|e1|{p}|all|{COLON}-{base}|auprc" for p in E1_PERCENTS for base in COMPARATORS})


def lr_label(value: float) -> str:
    return next(lr for lr in LRS if abs(float(lr) - value) < 1e-12)


def registered_lrs() -> Dict[str, str]:
    """Learning rate of each regime in the registered campaign (``finetune_lr`` of config/model)."""

    return {g: lr_label(float(yaml.safe_load((REPO / "config/model" / f"{g}.yaml").read_text())["model"]["finetune_lr"]))
            for g in REGIMES}


def sources(lrs) -> Dict[Tuple[str, str], Path]:
    reg = registered_lrs()
    out = {(g, reg[g]): REPO / REGISTERED_ROOT for g in REGIMES}
    out.update({key: REPO / root for key, root in SOURCES.items() if key[1] in lrs and key not in out})
    return out


def load(src: Dict[Tuple[str, str], Path], log=print):
    """E1 views keyed with regime ``<regime>@<lr>`` and one validation row per run."""

    campaigns: Dict[Path, Campaign] = {}
    views: Dict[tuple, View] = {}
    val_rows = []
    for (regime, lr), root in sorted(src.items()):
        if root not in campaigns:
            campaigns[root] = load_campaign([root])
            failed = [x for x in campaigns[root].checks if not x["passed"]]
            if failed:
                raise RuntimeError(f"{root}: {len(failed)} integrity checks failed, first: {failed[0]}")
            log(f"[load] {root.relative_to(REPO)}: {len(campaigns[root].runs)} runs")
        c = campaigns[root]
        runs = [r for r in c.runs if r.regime == regime and r.experiment == "e1"]
        expected = {(p, k, s) for p in E1_PERCENTS for k in (1, 2, 3, 4, 5) for s in (13, 29, 47)}
        got = {(r.variant, r.k, r.seed) for r in runs}
        if got != expected:
            raise RuntimeError(f"{regime} lr {lr} ({root}): {len(expected - got)} E1 runs missing, {len(got - expected)} extra")
        name = f"{regime}@{lr}"
        for run in runs:
            payload = json.loads(next(run.run_dir.rglob("*_last.metrics.json")).read_text())
            if lr_label(float(payload["run"]["lr"])) != lr:
                raise RuntimeError(f"{run.run_dir}: run.lr {payload['run']['lr']} != {lr}")
            val = load_frames(next(run.run_dir.rglob("*_val_outputs.csv")))
            val_rows.append({"regime": regime, "lr": lr, "budget": run.variant, "k": run.k, "seed": run.seed,
                             "val_ap": float(payload["val"]["auprc"]),
                             "val_ap_csv": float(binary.metrics_at(val.probs, val.labels, 0.5)["auprc"]),
                             "evaluated_epoch": int(payload["evaluated_epoch"]), "run_dir": str(run.run_dir.relative_to(REPO))})
        sub = Campaign(runs, c.roots, c.checks)
        for (fam, var, reg, k, s, slot), v in build_views(sub).items():
            key = (fam, var, name, k, s, slot)
            views[key] = replace(v, key=key)
    return views, pd.DataFrame(val_rows)


def select(val: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], str]]:
    reg = registered_lrs()
    table = (val.groupby(["regime", "budget", "lr"]).agg(val_ap_mean=("val_ap", "mean"), val_ap_sd=("val_ap", "std"),
                                                         n_runs=("val_ap", "size")).reset_index())
    chosen: Dict[Tuple[str, int], str] = {}
    for (g, p), grp in table.groupby(["regime", "budget"]):
        means = dict(zip(grp["lr"], grp["val_ap_mean"]))
        best = max(means, key=means.get)
        keep = mode == "registered" or best == reg[g] or means[best] - means[reg[g]] < TIE
        chosen[(g, p)] = reg[g] if keep else best
    table["registered"] = [lr == reg[g] for g, lr in zip(table["regime"], table["lr"])]
    table["selected"] = [chosen[(g, p)] == lr for g, p, lr in zip(table["regime"], table["budget"], table["lr"])]
    return table, chosen


def build_contrasts(chosen: Dict[Tuple[str, int], str], partitions, seeds) -> List[Contrast]:
    reg = registered_lrs()
    runs = [(k, s) for k in partitions for s in seeds]
    w = dict(zip(E1_PERCENTS, aulc_weights(E1_PERCENTS)))
    out: List[Contrast] = []
    for base in COMPARATORS:
        for label, weights in [("AULC", w)] + [(f"p{p}", {p: 1.0}) for p in E1_PERCENTS]:
            def terms(lr_of, weights=weights, base=base):
                return {(k, s): [t for p, wt in weights.items() for t in
                                 ((("e1", p, f"{COLON}@{lr_of(COLON, p)}", k, s, "all"), "auprc", wt),
                                  (("e1", p, f"{base}@{lr_of(base, p)}", k, s, "all"), "auprc", -wt))] for k, s in runs}
            t_sel = terms(lambda g, p: chosen[(g, p)])
            t_reg = terms(lambda g, p: reg[g])
            t_diff = {rid: t_sel[rid] + [(v, m, -wt) for v, m, wt in t_reg[rid]] for rid in runs}
            base_id = f"{COLON}-{base}|{label}"
            meta = {"comparison": f"{COLON} - {base}", "budget": label,
                    "metric": "AULC(auprc)" if label == "AULC" else "auprc",
                    "registered_id": REGISTERED_IDS[(label, base)]}
            for tag, t in (("selected", t_sel), ("registered", t_reg), ("diff", t_diff)):
                out.append(Contrast(f"{tag}|{base_id}", "e1", {**meta, "base_id": base_id, "version": tag}, t))
    return out


def run(lrs, mode: str, B: int, seed: int, workers: int, log=print):
    t0 = time.time()
    views, val = load(sources(lrs), log)
    table, chosen = select(val, mode)
    partitions, seeds = sorted({k for (_, _, _, k, _, _) in views}), sorted({s for (_, _, _, _, s, _) in views})
    contrasts = build_contrasts(chosen, partitions, seeds)
    needed = {vk for c in contrasts for vk in c.views()}
    point = {vk: views[vk].metrics() for vk in needed}
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in contrasts}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    reps, counts = bootstrap(views, contrasts, B=B, seed=seed, workers=workers)
    log(f"[bootstrap] {len(contrasts)} contrasts, units {counts} ({time.time() - t0:.0f}s)")
    rows = []
    for j, c in enumerate(contrasts):
        f, sp = fits.loc[c.id], mixed.spread(per_run[c.id])
        lo, hi = ci95(reps[:, j])
        rows.append({"id": c.id, **c.meta, "estimate": sp["mean"], "sd_partition": sp["sd_partition"], "sd_seed": sp["sd_seed"],
                     "n_runs": sp["n_runs"], "method": f["method"], "t_ci_lo": f["ci_lo"], "t_ci_hi": f["ci_hi"],
                     "p_unadjusted": f["p"], "boot_lo": lo, "boot_hi": hi, "boot_B": int(np.isfinite(reps[:, j]).sum())})
    long = pd.DataFrame(rows)
    check = check_registered(long)
    info = {"mode": mode, "candidate_lrs": sorted(lrs), "sources": {f"{g}@{lr}": str(r.relative_to(REPO)) for (g, lr), r in sources(lrs).items()},
            "registered_lrs": registered_lrs(), "selected": {f"{g}|p{p}": lr for (g, p), lr in sorted(chosen.items())},
            "partitions": partitions, "seeds": seeds, "B": B, "boot_seed": seed, "model_variant": MODEL_VARIANT,
            "val_ap_json_vs_csv_max_abs_diff": float((val["val_ap"] - val["val_ap_csv"]).abs().max()),
            "registered_check": {"n": len(check), "exact": int(check["exact"].sum()),
                                 "max_abs_diff": float(check["abs_diff"].max())},
            "elapsed_s": round(time.time() - t0, 1)}
    return long, table, val, check, info


def check_registered(long: pd.DataFrame) -> pd.DataFrame:
    """The ``registered`` version against the registered analysis (estimate, t-test CI, bootstrap CI)."""

    cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
    rows = []
    for _, r in long[long["version"] == "registered"].iterrows():
        ref = cs.loc[r["registered_id"]]
        for mine, theirs in (("estimate", "mean"), ("t_ci_lo", "model_ci_lo"), ("t_ci_hi", "model_ci_hi"),
                             ("boot_lo", "boot_lo"), ("boot_hi", "boot_hi"), ("boot_B", "boot_B")):
            diff = abs(float(r[mine]) - float(ref[theirs]))
            rows.append({"base_id": r["base_id"], "registered_id": r["registered_id"], "quantity": mine,
                         "recomputed": r[mine], "registered": ref[theirs], "abs_diff": diff,
                         # contrast_summary.csv stores 10 significant digits
                         "exact": diff <= 5e-10 * max(1.0, abs(float(ref[theirs])))})
    return pd.DataFrame(rows)


def wide(long: pd.DataFrame) -> pd.DataFrame:
    base = long[long["version"] == "registered"][["base_id", "comparison", "budget", "metric"]].set_index("base_id")
    for tag in ("selected", "registered", "diff"):
        sub = long[long["version"] == tag].set_index("base_id")
        for col in ("estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"):
            base[f"{tag}_{col}"] = sub[col]
    base["n_runs"] = long[long["version"] == "diff"].set_index("base_id")["n_runs"]
    return base.reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lrs", nargs="+", default=list(LRS), choices=LRS,
                    help="candidate learning rates besides each regime's registered one (sources must exist)")
    ap.add_argument("--only-sources", nargs="*", default=None, metavar="REGIME@LR",
                    help="restrict the non-registered candidates to these regime@lr pairs")
    ap.add_argument("--selection", choices=("per-budget", "registered"), default="per-budget")
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    if args.only_sources is not None:
        keep = {tuple(s.split("@")) for s in args.only_sources}
        for key in list(SOURCES):
            if key not in keep:
                del SOURCES[key]
    long, table, val, check, info = run(set(args.lrs), args.selection, args.B, args.boot_seed, args.workers)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "contrasts_long.csv", index=False, float_format="%.10g")
    w = wide(long)
    w.to_csv(out / "contrasts_selected_vs_registered.csv", index=False, float_format="%.10g")
    table.to_csv(out / "validation_ap_by_lr.csv", index=False, float_format="%.10g")
    val.to_csv(out / "validation_ap_runs.csv", index=False, float_format="%.10g")
    check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
    sel = table[table["selected"]][["regime", "budget", "lr", "registered"]].rename(columns={"lr": "selected_lr"})
    sel.to_csv(out / "selected_lrs.csv", index=False)
    note = ("Exploratory (not registered in the SAP): no Holm, no verdicts. Estimate = mean of the paired per-run "
            "differences (5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, "
            "seed 12345). selected: each regime at the learning rate selected for each budget on validation AP; "
            "registered: each regime at its registered learning rate; diff = selected − registered.")
    vt = table.copy()
    vt["mark"] = (pd.Series(np.where(vt["selected"], "selected", ""), index=vt.index)
                  + pd.Series(np.where(vt["registered"], " (registered)", ""), index=vt.index)).str.strip()
    cols = ["comparison", "budget", "metric", "selected_estimate", "selected_t_ci_lo", "selected_t_ci_hi", "selected_boot_lo",
            "selected_boot_hi", "registered_estimate", "registered_t_ci_lo", "registered_t_ci_hi", "diff_estimate",
            "diff_t_ci_lo", "diff_t_ci_hi", "diff_boot_lo", "diff_boot_hi"]
    md = ["# Exploratory: learning rate selected per E1 budget vs registered", "", note, "",
          to_markdown(vt[["regime", "budget", "lr", "val_ap_mean", "val_ap_sd", "n_runs", "mark"]],
                      "Mean validation AP (best-epoch weights) by regime, budget and learning rate"),
          to_markdown(w[cols], "Contrasts (AUPRC on the SUN test split)"),
          f"Registered values recomputed: {info['registered_check']['exact']}/{info['registered_check']['n']} equal to "
          f"results_v3/analysis/contrast_summary.csv (max |diff| {info['registered_check']['max_abs_diff']:.3g})."]
    (out / "contrasts_selected_vs_registered.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "sources"}, indent=1, default=str))
    return 0
