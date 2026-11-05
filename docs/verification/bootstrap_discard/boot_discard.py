#!/usr/bin/env python3
"""Discarded bootstrap replicates of H5, H6 and H7: counts, causes, probabilities and intervals without discarding.

Post hoc check, not registered: nothing is trained and registered results are read, never written.

* **H6, H7** (E4 family). Every per-run, per-budget arm difference (mean of the four regimes) is a single-run contrast
  over the registered E4 bootstrap space, so the draws are those of the registered e4 family (checked). The registered
  replicate is the mean over runs of the mean over the four budgets; it is undefined (discarded) when any term is. The
  alternative averages, in each run, the budgets that are defined in that replicate (no replicate is discarded).
* **H5** (negative-source family): the registered contrast over the registered negative-source space (checked).
* **Causes.** E4 negatives are resampled as 24 units (sequences and centres with isolated negatives), drawn with
  replacement. A replicate of a view is undefined when the drawn units contain none of the view's negative units. The
  exact probability that at least one of a set of views is undefined follows by inclusion–exclusion,
  P(views in T all undefined) = ((N − |∪ units of T|) / N)^N.

The registered replicate counts and intervals (results_v3/analysis/contrast_summary.csv, 10 significant digits) must be
reproduced exactly at that precision before anything is written.

    PYTHONPATH=. python docs/verification/bootstrap_discard/boot_discard.py --out <dir>
Run from the repository root in ``ssl4polyp-analysis``; it needs the registered campaign under
``checkpoints_v3/campaign/classification``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.v3.contrasts import hypothesis_contrasts  # noqa: E402
from analysis.v3.design import Contrast, Space, bootstrap, build_views, ci95  # noqa: E402
from analysis.v3.loader import load_campaign  # noqa: E402
from analysis.v3.sap import B_BOOT, BOOT_SEED, E4_BUDGETS, REPO  # noqa: E402

REG_ROOT = REPO / "checkpoints_v3/campaign/classification"
METRIC = "balanced_accuracy"
ARMS = {"H6": ("B", "A"), "H7": ("C", "A")}


def p_any_undefined(unit_sets, n_units: int) -> float:
    """P(at least one set has none of its units among n_units draws with replacement from n_units)."""
    sets = [frozenset(s) for s in unit_sets]
    total = 0.0
    for r in range(1, len(sets) + 1):
        for comb in itertools.combinations(sets, r):
            union = frozenset().union(*comb)
            total += (-1) ** (r + 1) * ((n_units - len(union)) / n_units) ** n_units
    return total


def compare(cid: str, q: str, val, rv) -> dict:
    """contrast_summary.csv stores 10 significant digits: compare at that precision (raw difference kept)."""
    return {"id": cid, "quantity": q, "recomputed": val, "registered": rv, "raw_abs_diff": abs(float(val) - float(rv)),
            "abs_diff": abs(float(f"{float(val):.10g}") - float(rv))}


def neg_units(view) -> frozenset:
    return frozenset(name for stratum, name in view.units if stratum == "neg")


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
    cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ H6, H7
    e4 = sorted([vk for vk in views if vk[0] == "e4"], key=repr)
    single, index = [], {}
    for h, (treat, base) in ARMS.items():
        for b in E4_BUDGETS:
            for k, s in runs:
                terms = [t for g in regimes for t in ((("e4", b, g, k, s, treat), METRIC, 1.0 / len(regimes)),
                                                     (("e4", b, g, k, s, base), METRIC, -1.0 / len(regimes)))]
                index[(h, b, k, s)] = len(single)
                single.append(Contrast(f"{h}#{b}#{k}#{s}", "e4", {}, {(k, s): terms}))
    needed = {vk for c in single for vk in c.views()}
    if Space([views[vk] for vk in needed]).flat != Space([views[vk] for vk in e4]).flat:
        raise RuntimeError("bootstrap space differs from the registered E4 space")
    e4_space = Space([views[vk] for vk in e4])
    n_neg = e4_space.counts()["neg"]
    reps, counts = bootstrap(views, single, B=B_BOOT, seed=BOOT_SEED, workers=args.workers)
    print(f"[bootstrap] e4: {len(single)} single-run contrasts, units {counts} ({time.time() - t0:.0f}s)")

    rows, check = [], []
    for h in ARMS:
        cube = np.stack([np.stack([reps[:, index[(h, b, k, s)]] for b in E4_BUDGETS], axis=1) for k, s in runs], axis=1)
        registered = cube.mean(axis=2).mean(axis=1)            # [B]; NaN if any term is NaN
        averaged = np.nanmean(cube, axis=2).mean(axis=1)       # per run: mean of the defined budgets
        for variant, rep in (("registered (discard)", registered), ("average of defined budgets", averaged)):
            lo, hi = ci95(rep)
            rows.append({"hypothesis": h, "variant": variant, "boot_B": int(np.isfinite(rep).sum()), "boot_lo": lo, "boot_hi": hi})
        ref = cs.loc[h]
        lo, hi = ci95(registered)
        for q, val, rv in (("boot_lo", lo, ref["boot_lo"]), ("boot_hi", hi, ref["boot_hi"]),
                           ("boot_B", int(np.isfinite(registered).sum()), ref["boot_B"])):
            check.append(compare(h, q, val, rv))
        undefined_by_budget = {b: int(np.isnan(cube[:, :, i]).any(axis=1).sum()) for i, b in enumerate(E4_BUDGETS)}
        rows[-1]["undefined_by_budget"] = rows[-2]["undefined_by_budget"] = json.dumps(undefined_by_budget)

    # causes (H7 and H6 share the draws): per 500-frame pack, its negative units; undefined iff none drawn
    h = "H7"
    cube500 = np.stack([reps[:, index[(h, 500, k, s)]] for k, s in runs], axis=1)
    pack_units = {(k, s): neg_units(views[("e4", 500, regimes[0], k, s, "C")]) for k, s in runs}
    for (k, s), u in pack_units.items():
        for g in regimes:
            for arm in "ABCD":
                if neg_units(views[("e4", 500, g, k, s, arm)]) != u:
                    raise RuntimeError(f"negative units differ within pack k{k} s{s}")
    pack_rows = []
    for j, (k, s) in enumerate(runs):
        pack_rows.append({"k": k, "seed": s, "neg_units": len(pack_units[(k, s)]), "units": " ".join(sorted(pack_units[(k, s)])),
                          "replicates_undefined": int(np.isnan(cube500[:, j]).sum()),
                          "p_undefined": ((n_neg - len(pack_units[(k, s)])) / n_neg) ** n_neg})
    n_disc = int(np.isnan(cube500).any(axis=1).sum())
    p_e4 = p_any_undefined(list(pack_units.values()), n_neg)

    # ------------------------------------------------------------------ H5
    hyp, _ = hypothesis_contrasts(views, regimes, partitions, seeds)
    h5 = hyp["H5"]
    neg_all = sorted([vk for vk in views if vk[0] == "neg"], key=repr)
    if Space([views[vk] for vk in h5.views()]).flat != Space([views[vk] for vk in neg_all]).flat:
        raise RuntimeError("H5 bootstrap space differs from the registered negative-source space")
    neg_space = Space([views[vk] for vk in neg_all])
    reps5, counts5 = bootstrap(views, [h5], B=B_BOOT, seed=BOOT_SEED, workers=args.workers)
    lo, hi = ci95(reps5[:, 0])
    b5 = int(np.isfinite(reps5[:, 0]).sum())
    ref = cs.loc["H5"]
    for q, val, rv in (("boot_lo", lo, ref["boot_lo"]), ("boot_hi", hi, ref["boot_hi"]), ("boot_B", b5, ref["boot_B"])):
        check.append(compare("H5", q, val, rv))
    rows.append({"hypothesis": "H5", "variant": "registered (discard)", "boot_B": b5, "boot_lo": lo, "boot_hi": hi,
                 "undefined_by_budget": json.dumps({50: B_BOOT - b5})})
    iso_units = {neg_units(views[vk]) for vk in neg_all if vk[5] == "isolated"}
    seq_units = {neg_units(views[vk]) for vk in neg_all if vk[5] == "sequence"}
    if len(iso_units) != 1 or len(seq_units) != 1:
        raise RuntimeError("isolated or sequence negative units differ between H5 views")
    iso, seq = next(iter(iso_units)), next(iter(seq_units))
    n5 = neg_space.counts()["neg"]

    check = pd.DataFrame(check)
    if float(check["abs_diff"].max()) != 0.0:
        raise RuntimeError(f"registered values not reproduced exactly:\n{check}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "intervals.csv", index=False, float_format="%.10g")
    pd.DataFrame(pack_rows).to_csv(out / "packs_500.csv", index=False, float_format="%.6g")
    check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
    shift = {}
    for hh in ARMS:
        pair = res[res["hypothesis"] == hh]
        shift[hh] = float(max(abs(pair.iloc[0][c] - pair.iloc[1][c]) for c in ("boot_lo", "boot_hi")))
    info = {
        "B": B_BOOT, "boot_seed": BOOT_SEED, "elapsed_s": None,
        "e4": {"units": counts, "neg_units": n_neg, "discarded_replicates": n_disc,
               "discarded_by_budget": json.loads(res.iloc[0]["undefined_by_budget"]),
               "cause": "a 500-frame pack draws none of its negative units",
               "pack_neg_units": sorted({r["neg_units"] for r in pack_rows}),
               "p_pack_undefined": sorted({round(r["p_undefined"], 6) for r in pack_rows}),
               "p_any_pack_undefined_exact": p_e4, "expected_discards": p_e4 * B_BOOT,
               "max_limit_shift_average_vs_discard": shift},
        "h5": {"units": counts5, "neg_units": n5, "isolated_units": sorted(iso), "sequence_units": len(seq),
               "discarded_replicates": B_BOOT - b5, "cause": "no centre with isolated negatives drawn",
               "p_no_isolated_exact": ((n5 - len(iso)) / n5) ** n5, "p_no_sequence_exact": ((n5 - len(seq)) / n5) ** n5,
               "expected_discards": ((n5 - len(iso)) / n5) ** n5 * B_BOOT},
        "registered_check": {"n": len(check), "max_abs_diff_at_10_digits": float(check["abs_diff"].max()),
                             "max_raw_abs_diff": float(check["raw_abs_diff"].max())},
    }
    info["elapsed_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(info, indent=1) + "\n")
    print(res.to_string(index=False))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
