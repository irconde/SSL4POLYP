#!/usr/bin/env python3
"""Possible leakage in the learning-rate selection: case overlap between selection validation and campaign tests.

Analysis of existing data only (nothing is trained; registered results are read, never written).

1. Overlap, by case (``case_id``) with positives and negatives counted separately; also by frame and, for negatives,
   by source video (``source_case_id``: negative cases are 33-frame chunks of a few long negative videos):
   a) validation split of the registered sweep (v2 ``data_packs/sun_full``) vs the test split of each partition
      k1..k5 (``data_packs/v3/k<k>/base``) and vs the union of every test split of the campaign (E1/E2 SUN tests of
      k1..k5, E3 PolypGen clean test, E4 few-shot tests);
   b) validation of partition k vs test of partition j (5 × 5), the splits used by the per-budget experiment.
2. Registered sweep without leakage: validation AP of the best-epoch weights recomputed from the stored per-frame
   validation CSVs (4 regimes × 3 learning rates × 3 seeds) after removing the frames of every case that appears in
   any campaign test; mean per regime and learning rate; decision rule of SAP section 10 (highest mean validation AP).
   Sensitivity: also removing negatives whose source video appears in any campaign test.
3. Per-budget experiment without leakage: the registered selection rule (highest mean validation AP; registered
   learning rate kept when the gain is < 1e-4) applied within each partition, on the mean of its 3 seeds
   (``validation_ap_runs.csv`` of the per-budget analysis), compared with the pooled selection. If any selection
   differs, H1 and the AULC contrasts are recomputed with the per-partition selection by the per-budget analysis code
   (loader, views, t-test, cluster bootstrap B = 2000 seed 12345), each pair (partition, seed) taking the learning
   rates selected on its own partition.

    PYTHONPATH=. python scripts/v3/lr_selection_leakage.py --out results_v3/verification/lr_selection_leakage
Run from the repository root in ``ssl4polyp-analysis``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.v3 import mixed  # noqa: E402
from analysis.v3.design import Contrast, aulc_weights, bootstrap, ci95  # noqa: E402
from analysis.v3.exploratory_lr_per_budget import analysis as ex  # noqa: E402
from analysis.v3.metrics import binary  # noqa: E402
from analysis.v3.sap import B_BOOT, BOOT_SEED, E1_PERCENTS, MODEL_VARIANT  # noqa: E402

PACKS = REPO / "data_packs"
K = (1, 2, 3, 4, 5)
SEEDS = (13, 29, 47)
SWEEP_ROOT = REPO / "checkpoints_v3/lr_sweep/classification/k1"
SWEEP_LRS = {"1.0e-4": "1e-4", "3.0e-4": "3e-4", "1.0e-3": "1e-3"}
TAG = {"sup_imnet": "SUPImNet", "ssl_imnet": "SSLImNet", "ssl_colon": "SSLColon", "ssl_imnet_matched": "SSLImNetMatched"}
PER_BUDGET = REPO / "results_v3/analysis_exploratory_lr_per_budget"


def split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["case_id", "frame_id", "label", "source_case_id"], dtype=str)


def campaign_tests() -> dict:
    """Every test split of the campaign: SUN base test of k1..k5 (E1, E2), PolypGen clean test (E3), E4 few-shot tests."""

    plan = pd.read_csv(REPO / "results_v3/verification/campaign_plan_all4.csv", dtype=str)
    packs = sorted(set(plan["test_pack"].dropna()))
    return {p: split(PACKS / p / "test.csv") for p in packs}


def overlap_row(name: str, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    row = {"test": name}
    for lab, tag in (("1", "pos"), ("0", "neg")):
        v, t = val[val["label"] == lab], test[test["label"] == lab]
        shared = set(v["case_id"]) & set(t["case_id"])
        row[f"{tag}_cases_val"] = v["case_id"].nunique()
        row[f"{tag}_cases_shared"] = len(shared)
        row[f"{tag}_frames_val_in_shared_cases"] = int(v["case_id"].isin(shared).sum())
        row[f"{tag}_frames_shared"] = len(set(v["frame_id"]) & set(t["frame_id"]))
    v, t = val[val["label"] == "0"], test[test["label"] == "0"]
    src = set(v["source_case_id"]) & set(t["source_case_id"])
    row["neg_source_videos_val"] = v["source_case_id"].nunique()
    row["neg_source_videos_shared"] = len(src)
    row["neg_cases_val_from_shared_sources"] = v.loc[v["source_case_id"].isin(src), "case_id"].nunique()
    return row


def part1(tests: dict):
    sweep_val = split(PACKS / "sun_full/val.csv")
    sun_tests = {k: split(PACKS / f"v3/k{k}/base/test.csv") for k in K}
    union = pd.concat(tests.values(), ignore_index=True)
    rows = [overlap_row(f"k{k}", sweep_val, sun_tests[k]) for k in K]
    rows.append(overlap_row("union of all campaign tests", sweep_val, union))
    a = pd.DataFrame(rows)
    b = []
    for k in K:
        val_k = split(PACKS / f"v3/k{k}/base/val.csv")
        for j in K:
            b.append({"val_partition": k, "test_partition": j, **overlap_row(f"k{j}", val_k, sun_tests[j])})
    b = pd.DataFrame(b).drop(columns="test")
    return sweep_val, union, a, b


def sweep_csv(model: str, lr: str, seed: int) -> Path:
    return SWEEP_ROOT / f"lr{lr}" / f"sun_full_{model}_seed{seed}" / "sun_baselines" / f"{TAG[model]}_SUNFull_s{seed}_val_outputs.csv"


def part2(sweep_val: pd.DataFrame, union: pd.DataFrame):
    """Sweep validation AP on all frames (check), without cases in any test, and without negative source videos too."""

    reg_runs = pd.concat([pd.read_csv(REPO / f"results_v3/verification/lr_sweep_runs{t}.csv", dtype={"lr": str})
                          for t in ("", "_matched")], ignore_index=True)
    leak_cases = set(union["case_id"])
    leak_sources = set(union.loc[union["label"] == "0", "source_case_id"])
    rows = []
    for model in TAG:
        for lr in SWEEP_LRS:
            for seed in SEEDS:
                f = pd.read_csv(sweep_csv(model, lr, seed), dtype={"case_id": str, "sequence_id": str})
                if set(f["case_id"]) != set(sweep_val["case_id"]) or len(f) != len(sweep_val):
                    raise RuntimeError(f"{model} {lr} {seed}: validation CSV is not the v2 sun_full validation split")
                drop_case = f["case_id"].isin(leak_cases)
                drop_src = drop_case | ((f["label"] == 0) & f["sequence_id"].isin(leak_sources))
                reg = reg_runs[(reg_runs["model"] == model) & (reg_runs["lr"] == lr) & (reg_runs["seed"] == seed)]
                row = {"regime": model, "lr": SWEEP_LRS[lr], "seed": seed, "val_ap_registered": float(reg["val_ap"].iloc[0])}
                for tag, keep in (("all", np.ones(len(f), bool)), ("no_shared_cases", ~drop_case.to_numpy()),
                                  ("no_shared_cases_or_neg_sources", ~drop_src.to_numpy())):
                    sub = f[keep]
                    row[f"{tag}_pos_cases"] = sub.loc[sub["label"] == 1, "case_id"].nunique()
                    row[f"{tag}_neg_cases"] = sub.loc[sub["label"] == 0, "case_id"].nunique()
                    row[f"{tag}_pos_frames"] = int((sub["label"] == 1).sum())
                    row[f"{tag}_neg_frames"] = int((sub["label"] == 0).sum())
                    ok = sub["label"].nunique() == 2
                    row[f"{tag}_val_ap"] = (float(binary.metrics_at(sub["prob"].to_numpy(float), sub["label"].to_numpy(int), 0.5)["auprc"])
                                            if ok else np.nan)
                rows.append(row)
    runs = pd.DataFrame(rows)
    worst = float((runs["all_val_ap"] - runs["val_ap_registered"]).abs().max())
    reg_lrs = ex.registered_lrs()
    table = runs.groupby(["regime", "lr"]).agg(
        **{f"{t}_mean": (f"{t}_val_ap", "mean") for t in ("all", "no_shared_cases", "no_shared_cases_or_neg_sources")},
        **{f"{t}_sd": (f"{t}_val_ap", "std") for t in ("all", "no_shared_cases", "no_shared_cases_or_neg_sources")}).reset_index()
    sel = []
    for g in TAG:
        sub = table[table["regime"] == g].set_index("lr")
        r = {"regime": g, "registered_lr": reg_lrs[g]}
        for t in ("all", "no_shared_cases", "no_shared_cases_or_neg_sources"):
            m = sub[f"{t}_mean"]
            r[f"{t}_selected_lr"] = m.idxmax() if m.notna().all() else ""
        r["changes_without_shared_cases"] = r["no_shared_cases_selected_lr"] != r["registered_lr"]
        r["changes_without_shared_cases_or_neg_sources"] = (r["no_shared_cases_or_neg_sources_selected_lr"] not in ("", r["registered_lr"]))
        sel.append(r)
    return runs, table, pd.DataFrame(sel), worst


def part3():
    val = pd.read_csv(PER_BUDGET / "validation_ap_runs.csv", dtype={"lr": str})
    pooled = pd.read_csv(PER_BUDGET / "selected_lrs.csv", dtype={"selected_lr": str})
    pooled = {(r.regime, int(r.budget)): r.selected_lr for r in pooled.itertuples()}
    reg = ex.registered_lrs()
    m = val.groupby(["regime", "budget", "k", "lr"])["val_ap"].agg(["mean", "size"]).reset_index()
    if (m["size"] != 3).any():
        raise RuntimeError("expected 3 seeds per regime, budget, partition and learning rate")
    rows, chosen = [], {}
    for (g, p, k), grp in m.groupby(["regime", "budget", "k"]):
        means = dict(zip(grp["lr"], grp["mean"]))
        best = max(means, key=means.get)
        keep = best == reg[g] or means[best] - means[reg[g]] < ex.TIE
        chosen[(g, int(p), int(k))] = lr = reg[g] if keep else best
        rows.append({"regime": g, "budget": int(p), "k": int(k), **{f"val_ap_mean_{x}": means[x] for x in ex.LRS},
                     "registered_lr": reg[g], "pooled_lr": pooled[(g, int(p))], "partition_lr": lr,
                     "matches_pooled": lr == pooled[(g, int(p))]})
    return pd.DataFrame(rows), chosen, pooled


def contrasts(chosen_k: dict, pooled: dict, partitions, seeds):
    """SSL-colon − comparator AULC (and per budget) with the per-partition selection, as ``build_contrasts``."""

    reg = ex.registered_lrs()
    runs = [(k, s) for k in partitions for s in seeds]
    w = dict(zip(E1_PERCENTS, aulc_weights(E1_PERCENTS)))
    out = []
    for base in ex.COMPARATORS:
        for label, weights in [("AULC", w)] + [(f"p{p}", {p: 1.0}) for p in E1_PERCENTS]:
            def terms(lr_of, weights=weights, base=base):
                return {(k, s): [t for p, wt in weights.items() for t in
                                 ((("e1", p, f"{ex.COLON}@{lr_of(ex.COLON, p, k)}", k, s, "all"), "auprc", wt),
                                  (("e1", p, f"{base}@{lr_of(base, p, k)}", k, s, "all"), "auprc", -wt))] for k, s in runs}
            t = {"partition": terms(lambda g, p, k: chosen_k[(g, p, k)]),
                 "pooled": terms(lambda g, p, k: pooled[(g, p)]),
                 "registered": terms(lambda g, p, k: reg[g])}
            t["partition_minus_registered"] = {r: t["partition"][r] + [(v, m, -x) for v, m, x in t["registered"][r]] for r in runs}
            t["partition_minus_pooled"] = {r: t["partition"][r] + [(v, m, -x) for v, m, x in t["pooled"][r]] for r in runs}
            base_id = f"{ex.COLON}-{base}|{label}"
            meta = {"comparison": f"{ex.COLON} - {base}", "budget": label, "metric": "AULC(auprc)" if label == "AULC" else "auprc",
                    "registered_id": ex.REGISTERED_IDS[(label, base)]}
            for tag, terms_ in t.items():
                out.append(Contrast(f"{tag}|{base_id}", "e1", {**meta, "base_id": base_id, "version": tag}, terms_))
    return out


def part4(chosen_k: dict, pooled: dict, workers: int, log=print):
    t0 = time.time()
    views, _ = ex.load(ex.sources(set(ex.LRS)), log)
    partitions, seeds = sorted({k for (_, _, _, k, _, _) in views}), sorted({s for (_, _, _, _, s, _) in views})
    cs = contrasts(chosen_k, pooled, partitions, seeds)
    point = {vk: views[vk].metrics() for vk in {vk for c in cs for vk in c.views()}}
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in cs}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    reps, counts = bootstrap(views, cs, B=B_BOOT, seed=BOOT_SEED, workers=workers)
    log(f"[bootstrap] {len(cs)} contrasts, units {counts} ({time.time() - t0:.0f}s)")
    rows = []
    for j, c in enumerate(cs):
        f, sp = fits.loc[c.id], mixed.spread(per_run[c.id])
        lo, hi = ci95(reps[:, j])
        rows.append({"id": c.id, **c.meta, "estimate": sp["mean"], "n_runs": sp["n_runs"], "method": f["method"],
                     "t_ci_lo": f["ci_lo"], "t_ci_hi": f["ci_hi"], "boot_lo": lo, "boot_hi": hi,
                     "boot_B": int(np.isfinite(reps[:, j]).sum())})
    long = pd.DataFrame(rows)
    # checks: registered = registered analysis; pooled = committed per-budget "selected" version
    reg_check = ex.check_registered(long)
    prev = pd.read_csv(PER_BUDGET / "contrasts_long.csv", float_precision="round_trip")
    prev = prev[prev["version"] == "selected"].set_index("base_id")
    mine = long[long["version"] == "pooled"].set_index("base_id")
    pooled_diff = max(float((mine[c] - prev.loc[mine.index, c]).abs().max()) for c in ("estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"))
    return long, reg_check, pooled_diff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "results_v3/verification/lr_selection_leakage")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tests = campaign_tests()
    sweep_val, union, a, b = part1(tests)
    a.to_csv(out / "overlap_sweep_val_vs_tests.csv", index=False)
    b.to_csv(out / "overlap_partition_val_vs_test.csv", index=False)
    runs, table, sel, worst = part2(sweep_val, union)
    runs.to_csv(out / "sweep_val_ap_runs.csv", index=False, float_format="%.10g")
    table.to_csv(out / "sweep_val_ap_by_lr.csv", index=False, float_format="%.10g")
    sel.to_csv(out / "sweep_selection.csv", index=False)
    per_k, chosen_k, pooled = part3()
    per_k.to_csv(out / "per_budget_selection_by_partition.csv", index=False, float_format="%.10g")
    info = {"campaign_test_packs": sorted(tests), "sweep_val_ap_all_frames_vs_registered_max_abs_diff": worst,
            "sweep_selection_changes_without_shared_cases": sel.loc[sel["changes_without_shared_cases"], "regime"].tolist(),
            "per_partition_cells": len(per_k), "per_partition_matches_pooled": int(per_k["matches_pooled"].sum())}
    if not per_k["matches_pooled"].all():
        long, reg_check, pooled_diff = part4(chosen_k, pooled, args.workers)
        long.to_csv(out / "contrasts_partition_selection_long.csv", index=False, float_format="%.10g")
        reg_check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
        info.update({"registered_check": {"n": len(reg_check), "exact": int(reg_check["exact"].sum()),
                                          "max_abs_diff": float(reg_check["abs_diff"].max())},
                     "pooled_vs_committed_per_budget_max_abs_diff": pooled_diff, "B": B_BOOT, "boot_seed": BOOT_SEED})
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
