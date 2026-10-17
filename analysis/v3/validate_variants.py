#!/usr/bin/env python3
"""Extended validation: three variants of the primary analysis on the same simulated campaigns.

    V1  SAP v3.2c: lmer, Satterthwaite df, fallback t-test on the K partition means when singular/not converged
    V2  lmer, Kenward–Roger df (pbkrtest), same fallback rule
    V3  always the one-sample t-test on the K partition means (df = K − 1)

Each repetition simulates one campaign (``simulate.py``, minimal files), computes the per-run differences of
H1–H7 and their cluster bootstrap once (the bootstrap does not depend on the variant), fits the three variants,
and applies Holm over H1–H7 and the double confirmation rule for each. Repetitions run in parallel processes
(one process per repetition, no nested pools) and are appended to ``variant_reps.csv`` as they finish, so an
interrupted run resumes where it stopped. Usage::

    PYTHONPATH=. nice -n 19 python -m analysis.v3.validate_variants --out <dir> --null-reps 200 --effect-reps 50 --procs 12
    PYTHONPATH=. python -m analysis.v3.validate_variants --out <dir> --summarize-only
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

from . import mixed  # noqa: E402
from . import simulate as S  # noqa: E402
from .analysis import confirm, run_analysis  # noqa: E402
from .sap import HYPOTHESES  # noqa: E402

VARIANT_OF = {"V1": "satterthwaite", "V2": "kenward-roger", "V3": "ttest"}
MODEL_COLS = {"method": "model_method", "estimate": "model_estimate", "se": "model_se", "df": "model_df", "p": "model_p",
              "ci_lo": "model_ci_lo", "ci_hi": "model_ci_hi", "singular": "singular", "converged": "converged"}

# Decision rule fixed by the user before the run (2026-10-17)
FWER_UPPER_MAX = 0.08
COVERAGE_MIN = 0.93


def one_rep(task: Tuple[str, int, str, int]) -> List[dict]:
    scenario_name, seed, work, B = task
    scenario = S.effect_scenario() if scenario_name == "effects" else S.null_scenario()
    d = Path(work) / f"{scenario_name}_{seed}"
    t0 = time.time()
    truth = S.simulate_campaign(scenario, d, sim_seed=seed, minimal=True)
    res = run_analysis(truth["roots"], B=B, workers=1, secondary=False, sensitivity=False, log=lambda *_: None)
    hyp = {h: res.contrasts.get(h) for h in HYPOTHESES}
    per_run_h = {h: res.per_run[h] for h in HYPOTHESES if h in res.per_run}
    rows = []
    for vname, variant in VARIANT_OF.items():
        fits = mixed.fit_many(per_run_h, variant).set_index("id")
        summary = res.summary.copy().set_index("id")
        for src, dst in MODEL_COLS.items():
            summary[dst] = summary[dst].astype(object)
            summary.loc[fits.index, dst] = fits[src].astype(object)
        out = confirm(summary.reset_index(), hyp, res.notes).set_index("hypothesis")
        for h in HYPOTHESES:
            r = out.loc[h]
            effect = truth["hypotheses"][h]["effect"]
            rows.append({
                "scenario": scenario_name, "sim_seed": seed, "variant": vname, "hypothesis": h, "effect": effect,
                "realised": truth["hypotheses"][h]["realised_mean"], "clipped": truth["clipped_targets"],
                "estimate": r["estimate"], "model_method": r["model_method"], "model_df": r["model_df"],
                "model_ci_lo": r["model_ci_lo"], "model_ci_hi": r["model_ci_hi"], "p": r["p"], "p_holm": r["p_holm"],
                "holm_m": r["holm_m"], "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"], "singular": r["singular"],
                "confirmed": bool(r["confirmed"]),
                "model_covers": bool(r["model_ci_lo"] <= effect <= r["model_ci_hi"]),
                "seconds": round(time.time() - t0, 1),
            })
    shutil.rmtree(d, ignore_errors=True)
    return rows


def clopper_pearson(x: int, n: int) -> Tuple[float, float]:
    lo = float(beta.ppf(0.025, x, n - x + 1)) if x > 0 else 0.0
    hi = float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0
    return lo, hi


def read_reps(path: Path) -> pd.DataFrame:
    """``scenario`` holds the literal "null", which pandas would read as NaN by default."""

    return pd.read_csv(path, keep_default_na=False, na_values=["", "NaN", "nan"], dtype={"scenario": str, "variant": str, "hypothesis": str})


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {"variants": {}}
    for v, g in df.groupby("variant"):
        null, eff = g[g["scenario"] == "null"], g[g["scenario"] == "effects"]
        n_null = null["sim_seed"].nunique()
        fam = int(null.groupby("sim_seed")["confirmed"].any().sum())
        lo, hi = clopper_pearson(fam, n_null) if n_null else (float("nan"), float("nan"))
        per_h = {}
        for h in HYPOTHESES:
            nh, eh = null[null["hypothesis"] == h], eff[eff["hypothesis"] == h]
            per_h[h] = {
                "false_confirmation_null": float(nh["confirmed"].mean()) if len(nh) else float("nan"),
                "coverage_null": float(nh["model_covers"].mean()) if len(nh) else float("nan"),
                "coverage_effects": float(eh["model_covers"].mean()) if len(eh) else float("nan"),
                "power_effects": float(eh["confirmed"].mean()) if len(eh) else float("nan"),
            }
        min_cov = min(min(x["coverage_null"], x["coverage_effects"]) for x in per_h.values())
        out["variants"][v] = {
            "null_reps": int(n_null), "effect_reps": int(eff["sim_seed"].nunique()),
            "family_wise_false": fam, "family_wise_rate": fam / n_null if n_null else float("nan"),
            "family_wise_ci95": [lo, hi], "min_coverage": min_cov,
            "mean_power": float(np.mean([x["power_effects"] for x in per_h.values()])),
            "meets_conditions": bool(hi <= FWER_UPPER_MAX and min_cov >= COVERAGE_MIN),
            "per_hypothesis": per_h,
        }
    V = out["variants"]
    if "V1" in V and V["V1"]["meets_conditions"]:
        choice, why = "V1", "V1 meets both conditions: SAP unchanged"
    else:
        ok = [v for v in ("V2", "V3") if v in V and V[v]["meets_conditions"]]
        if ok:
            best = max(V[v]["mean_power"] for v in ok)
            tied = [v for v in ok if V[v]["mean_power"] == best]
            choice = tied[0] if len(tied) == 1 else "tie:" + ",".join(tied)
            why = f"V1 fails; among V2/V3 meeting both conditions the highest mean power is {best:.4f} ({', '.join(tied)})"
        else:
            choice, why = "V3", "V1 fails and neither V2 nor V3 meets both conditions: V3 by default"
    # identical decisions across variants? (expected for complete balanced designs)
    piv = df.pivot_table(index=["scenario", "sim_seed", "hypothesis"], columns="variant", values="p", aggfunc="first")
    conf = df.pivot_table(index=["scenario", "sim_seed", "hypothesis"], columns="variant", values="confirmed", aggfunc="first")
    out["agreement"] = {
        "max_abs_p_difference": {f"{a}-{b}": float((piv[a] - piv[b]).abs().max()) for a, b in (("V1", "V2"), ("V1", "V3"), ("V2", "V3"))
                                 if a in piv and b in piv},
        "confirmation_disagreements": {f"{a}-{b}": int((conf[a] != conf[b]).sum()) for a, b in (("V1", "V2"), ("V1", "V3"), ("V2", "V3"))
                                       if a in conf and b in conf},
    }
    out["rule"] = {"fwer_upper_max": FWER_UPPER_MAX, "coverage_min": COVERAGE_MIN, "choice": choice, "reason": why}
    out["clipped_targets_total"] = int(df.groupby(["scenario", "sim_seed"])["clipped"].first().sum())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--null-reps", type=int, default=200)
    ap.add_argument("--effect-reps", type=int, default=50)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--work", type=Path, default=None, help="scratch directory for the simulated campaigns")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    reps_csv = args.out / "variant_reps.csv"
    if not args.summarize_only:
        work = args.work or (args.out / "_work")
        work.mkdir(parents=True, exist_ok=True)
        done = set()
        if reps_csv.exists():
            prev = read_reps(reps_csv)
            done = set(zip(prev["scenario"], prev["sim_seed"]))
        tasks = [("null", s, str(work), args.B) for s in range(1000, 1000 + args.null_reps)]
        tasks += [("effects", s, str(work), args.B) for s in range(2000, 2000 + args.effect_reps)]
        tasks = [t for t in tasks if (t[0], t[1]) not in done]
        print(f"{len(tasks)} repetitions to run ({len(done)} already done), {args.procs} processes", flush=True)
        with mp.get_context("spawn").Pool(args.procs, maxtasksperchild=1) as pool:
            for i, rows in enumerate(pool.imap_unordered(one_rep, tasks), 1):
                pd.DataFrame(rows).to_csv(reps_csv, mode="a", header=not reps_csv.exists(), index=False, float_format="%.10g")
                print(f"[{i}/{len(tasks)}] {rows[0]['scenario']} seed {rows[0]['sim_seed']} ({rows[0]['seconds']} s)", flush=True)
        shutil.rmtree(work, ignore_errors=True)
    df = read_reps(reps_csv)
    summary = summarize(df)
    summary["settings"] = {"B": args.B, "null_reps_requested": args.null_reps, "effect_reps_requested": args.effect_reps}
    (args.out / "variant_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "variants"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
