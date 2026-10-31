#!/usr/bin/env python3
"""Unadapted transfer by negative source: AUROC of the positives against sequence, isolated or all negatives.

Post hoc sensitivity analysis, not registered: nothing is trained and registered results are read, never written.

Frames: the per-frame outputs of the unadapted models used by H5 (``*_zeroshot_outputs.csv`` of the E4 50-frame runs).
Rows, as in H5 (``design.negsource_rows``): the positives also present in ``polypgen_clean_test`` (common positives)
plus the negatives of one population: (a) sequence negatives, (b) isolated negatives, (c) all negatives. (a) and (b)
are the registered views ``("neg", 50, regime, k, s, "sequence" | "isolated")``; (c) is their union.

Per population: the AUROC level of each regime and the contrasts SSL-colon − SUP-imnet (the comparison of H4),
SSL-colon − SSL-imnet, SSL-colon − SSL-imnet-matched and SSL-imnet − SUP-imnet, with the registered inference
(``analysis/v3``): paired estimate (mean of the 15 per-run values), t-test on the 5 partition means (``mixed``, variant
``ttest``) and cluster bootstrap of the E4 units (B = 2000, seed 12345). The (c) views hold no unit outside the (a) and
(b) views, so the bootstrap space and draws are those of the registered ``neg`` family; the SSL-colon contrasts in (a)
and (b) must reproduce the registered values (``cmp|neg|50|<source>|ssl_colon-<base>|auroc``) before anything is
written.

    PYTHONPATH=. python scripts/v3/zeroshot_by_negative_source.py --out results_v3/verification/zeroshot_by_negative_source
Run from the repository root in ``ssl4polyp-analysis``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.v3 import mixed  # noqa: E402
from analysis.v3.design import Contrast, Space, View, bootstrap, build_views, ci95, complete_terms  # noqa: E402
from analysis.v3.loader import load_campaign  # noqa: E402
from analysis.v3.sap import B_BOOT, BOOT_SEED, LABEL, MODEL_VARIANT, NEGSOURCE_BUDGET, REPO  # noqa: E402
from analysis.v3.tables import to_markdown  # noqa: E402

REG_ROOT = REPO / "checkpoints_v3/campaign/classification"
POPULATIONS = ("sequence", "isolated", "all")
COMPARISONS = (("ssl_colon", "sup_imnet"), ("ssl_colon", "ssl_imnet"), ("ssl_colon", "ssl_imnet_matched"), ("ssl_imnet", "sup_imnet"))
METRIC = "auroc"


def key(g, k, s, pop):
    return ("neg", NEGSOURCE_BUDGET, g, k, s, pop)


def make_contrasts(views, runs, regimes) -> List[Contrast]:
    out: List[Contrast] = []

    def add(cid, meta, make):
        terms = complete_terms(views, runs, make)
        if len(terms) != len(runs):
            raise RuntimeError(f"{cid}: {len(terms)} complete runs of {len(runs)}")
        out.append(Contrast(cid, "neg", meta, terms))

    for pop in POPULATIONS:
        for g in regimes:
            add(f"level|{pop}|{g}|{METRIC}", {"kind": "level", "population": pop, "comparison": g, "metric": METRIC},
                lambda k, s, g=g, pop=pop: [(key(g, k, s, pop), METRIC, 1.0)])
        for treat, base in COMPARISONS:
            add(f"cmp|{pop}|{treat}-{base}|{METRIC}", {"kind": "contrast", "population": pop, "comparison": f"{treat} - {base}", "metric": METRIC},
                lambda k, s, t=treat, b=base, pop=pop: [(key(t, k, s, pop), METRIC, 1.0), (key(b, k, s, pop), METRIC, -1.0)])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    t0 = time.time()
    camp = load_campaign([REG_ROOT])
    failed = [x for x in camp.checks if not x["passed"]]
    if failed:
        raise RuntimeError(f"{len(failed)} integrity checks failed, first: {failed[0]}")
    views = build_views(camp)
    regimes, partitions, seeds = camp.regimes(), camp.partitions(), camp.seeds()
    runs = [(k, s) for k in partitions for s in seeds]

    # (c): common positives + all negatives = union of the registered sequence and isolated views
    comp_rows = []
    for g in regimes:
        for k, s in runs:
            seq, iso = views[key(g, k, s, "sequence")], views[key(g, k, s, "isolated")]
            if seq.frames is not iso.frames:
                raise RuntimeError(f"{g} k{k} s{s}: sequence and isolated views on different frames")
            rows = np.union1d(seq.rows, iso.rows)
            views[key(g, k, s, "all")] = View(key(g, k, s, "all"), seq.frames, seq.tau, rows, seq.unit_kind, seq.group)
            f = seq.frames
            for pop in POPULATIONS:
                r = views[key(g, k, s, pop)].rows
                lab, src = f.labels[r], f.col("negative_source")[r]
                comp_rows.append({"k": k, "seed": s, "regime": g, "population": pop, "n": int(r.size),
                                  "positives": int((lab == 1).sum()), "neg_sequence": int(((lab == 0) & (src == "sequence")).sum()),
                                  "neg_isolated": int(((lab == 0) & (src == "isolated")).sum()),
                                  "frame_ids_sha": hash(tuple(f.col("frame_id")[r]))})
    comp = pd.DataFrame(comp_rows)
    # paired regimes evaluate the same frames in every population and pack
    if comp.groupby(["k", "seed", "population"])["frame_ids_sha"].nunique().max() != 1:
        raise RuntimeError("regimes of a pack evaluate different frames")
    comp = comp.drop(columns="frame_ids_sha")
    pack = comp[comp["regime"] == regimes[0]].pivot(index=["k", "seed"], columns="population",
                                                    values=["positives", "neg_sequence", "neg_isolated"])
    pack = pd.DataFrame({"positives": pack[("positives", "all")], "neg_sequence": pack[("neg_sequence", "all")],
                         "neg_isolated": pack[("neg_isolated", "all")]}).reset_index()

    mine = make_contrasts(views, runs, regimes)
    needed = {vk for c in mine for vk in c.views()}
    registered_space = Space([v for vk, v in views.items() if vk[0] == "neg" and vk[5] in ("sequence", "isolated")])
    if Space([views[vk] for vk in needed]).flat != registered_space.flat:
        raise RuntimeError("bootstrap space differs from the registered neg family space")
    point = {vk: views[vk].metrics() for vk in needed}
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in mine}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    reps, counts = bootstrap(views, mine, B=B_BOOT, seed=BOOT_SEED, workers=args.workers)
    print(f"[bootstrap] neg: {len(mine)} contrasts, units {counts} ({time.time() - t0:.0f}s)")
    rows = []
    for j, c in enumerate(mine):
        f, sp = fits.loc[c.id], mixed.spread(per_run[c.id])
        lo, hi = ci95(reps[:, j])
        rows.append({"id": c.id, **c.meta, "estimate": sp["mean"], "sd_partition": sp["sd_partition"], "sd_seed": sp["sd_seed"],
                     "n_runs": sp["n_runs"], "method": f["method"], "t_ci_lo": f["ci_lo"], "t_ci_hi": f["ci_hi"],
                     "p_unadjusted": f["p"], "boot_lo": lo, "boot_hi": hi, "boot_B": int(np.isfinite(reps[:, j]).sum())})
    long = pd.DataFrame(rows)

    # check: SSL-colon contrasts in (a) and (b) = registered values (estimate, t-test CI, bootstrap CI, B)
    cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
    check = []
    for _, r in long[(long["kind"] == "contrast") & long["population"].isin(["sequence", "isolated"])
                     & long["comparison"].str.startswith("ssl_colon")].iterrows():
        reg_id = f"cmp|neg|{NEGSOURCE_BUDGET}|{r['population']}|{r['comparison'].replace(' - ', '-')}|{METRIC}"
        ref = cs.loc[reg_id]
        for q, rq in (("estimate", "mean"), ("t_ci_lo", "model_ci_lo"), ("t_ci_hi", "model_ci_hi"),
                      ("boot_lo", "boot_lo"), ("boot_hi", "boot_hi"), ("boot_B", "boot_B")):
            d = abs(float(r[q]) - float(ref[rq]))
            check.append({"contrast": r["id"], "registered_id": reg_id, "quantity": q, "recomputed": r[q],
                          "registered": ref[rq], "abs_diff": d, "exact": d <= 5e-10 * max(1.0, abs(float(ref[rq])))})
    check = pd.DataFrame(check)
    n_exact = int(check["exact"].sum())
    if n_exact != len(check):
        raise RuntimeError(f"registered contrasts not reproduced: {n_exact}/{len(check)} exact")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "composition_by_pack_regime.csv", index=False)
    pack.to_csv(out / "composition_by_pack.csv", index=False)
    run_auroc = pd.DataFrame([{"regime": vk[2], "k": vk[3], "seed": vk[4], "population": vk[5], "n": int(views[vk].rows.size),
                               "auroc": point[vk][METRIC]} for vk in sorted(needed, key=repr)])
    run_auroc.to_csv(out / "run_auroc.csv", index=False, float_format="%.10g")
    long.to_csv(out / "contrasts_long.csv", index=False, float_format="%.10g")
    check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
    info = {"root": str(REG_ROOT.relative_to(REPO)), "budget": NEGSOURCE_BUDGET, "regimes": regimes, "partitions": partitions,
            "seeds": seeds, "B": B_BOOT, "boot_seed": BOOT_SEED, "model_variant": MODEL_VARIANT,
            "bootstrap_units": {str(g): v for g, v in counts.items()}, "bootstrap_space_equals_registered_neg": True,
            "boot_B_valid": {"min": int(long["boot_B"].min()), "max": int(long["boot_B"].max())},
            "n_contrasts": len(mine), "registered_check": {"n": len(check), "exact": n_exact, "max_abs_diff": float(check["abs_diff"].max())},
            "elapsed_s": round(time.time() - t0, 1)}
    show = long[["kind", "population", "comparison", "estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi", "boot_B"]].copy()
    show["comparison"] = show["comparison"].replace(LABEL)
    md = ["# Unadapted transfer by negative source (AUROC, E4 50-frame split, common positives)", "",
          "Post hoc sensitivity analysis, not registered. Estimate = mean of the paired per-run values (5 partitions × 3 "
          "seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345, the registered draws).",
          "", to_markdown(pack, "Composition per pack (same frames in every regime)"), to_markdown(show, "AUROC levels and contrasts"),
          f"Registered SSL-colon contrasts in (a) and (b) recomputed: {n_exact}/{len(check)} quantities equal to "
          f"results_v3/analysis/contrast_summary.csv (max |diff| {info['registered_check']['max_abs_diff']:.3g})."]
    (out / "zeroshot_by_negative_source.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
