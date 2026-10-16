#!/usr/bin/env python3
"""Monte Carlo validation of the analysis on simulated campaigns (no real data).

Scenarios (``simulate.py``):
  effects   known effect in every hypothesis, partition and seed variability in each difference
  null      no effect, same variability; repeated with several simulation seeds
  no_partition_variance   effects, sd_partition = 0 (the fallback rule should take over when the fit is singular)

Each repetition simulates a campaign and runs the hypothesis analysis (H1–H7, Holm, bootstrap, double rule).
Repetitions use a reduced bootstrap B (default 200) to keep the runtime manageable; the full analysis uses
B = 2000. Usage::

    PYTHONPATH=. python -m analysis.v3.validate --out <dir> [--null-reps 20] [--effect-reps 6] [--B 200]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import simulate as S  # noqa: E402
from .analysis import run_analysis  # noqa: E402
from .sap import HYPOTHESES  # noqa: E402


def run_reps(scenario: S.Scenario, seeds: List[int], work: Path, B: int, workers: int, keep: bool = False) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        d = work / f"{scenario.name}_{seed}"
        truth = S.simulate_campaign(scenario, d, sim_seed=seed, minimal=True)
        res = run_analysis(truth["roots"], B=B, workers=workers, secondary=False, sensitivity=False, log=lambda *_: None)
        for _, r in res.hypotheses.iterrows():
            h = r["hypothesis"]
            t = truth["hypotheses"][h]
            rows.append({
                "scenario": scenario.name, "sim_seed": seed, "hypothesis": h, "effect": t["effect"],
                "realised": t["realised_mean"], "clipped": truth["clipped_targets"],
                **{c: r.get(c) for c in ("estimate", "model_method", "model_ci_lo", "model_ci_hi", "p", "p_holm", "holm_m",
                                         "boot_lo", "boot_hi", "singular", "var_partition", "holm_significant",
                                         "boot_excludes_zero", "confirmed", "status")},
            })
        if not keep:
            shutil.rmtree(d, ignore_errors=True)
    df = pd.DataFrame(rows)
    df["model_covers"] = (df["model_ci_lo"] <= df["effect"]) & (df["effect"] <= df["model_ci_hi"])
    df["boot_covers"] = (df["boot_lo"] <= df["effect"]) & (df["effect"] <= df["boot_hi"])
    return df


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for sc, g in df.groupby("scenario"):
        per_h = {}
        for h, gh in g.groupby("hypothesis"):
            per_h[h] = {
                "reps": int(len(gh)), "effect": float(gh["effect"].iloc[0]),
                "mean_estimate": float(gh["estimate"].mean()), "bias": float((gh["estimate"] - gh["effect"]).mean()),
                "model_ci_coverage": float(gh["model_covers"].mean()), "boot_ci_coverage": float(gh["boot_covers"].mean()),
                "confirmation_rate": float(gh["confirmed"].mean()), "holm_significant_rate": float(gh["holm_significant"].mean()),
                "fallback_rate": float((gh["model_method"] == "fallback_t").mean()),
            }
        reps = g.groupby("sim_seed")
        out[sc] = {
            "reps": int(reps.ngroups),
            "family_wise_any_confirmed": float(reps["confirmed"].any().mean()),
            "confirmations_per_test": float(g["confirmed"].mean()),
            "clipped_targets_total": int(reps["clipped"].first().sum()),
            "per_hypothesis": per_h,
        }
    return out


def scenarios(null_seeds: List[int], effect_seeds: List[int], nopart_seeds: List[int]):
    return [
        (S.effect_scenario(), effect_seeds),
        (S.null_scenario(), null_seeds),
        (S.Scenario("no_partition_variance", dict(S.EFFECTS), {h: 0.0 for h in HYPOTHESES}, {h: 0.008 for h in HYPOTHESES}), nopart_seeds),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--null-reps", type=int, default=20)
    ap.add_argument("--effect-reps", type=int, default=6)
    ap.add_argument("--nopart-reps", type=int, default=4)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_work"
    frames = []
    for sc, seeds in scenarios(list(range(100, 100 + args.null_reps)), list(range(200, 200 + args.effect_reps)),
                               list(range(300, 300 + args.nopart_reps))):
        df = run_reps(sc, seeds, work, args.B, args.workers)
        frames.append(df)
        print(f"{sc.name}: {len(seeds)} reps done", flush=True)
    allrows = pd.concat(frames, ignore_index=True)
    allrows.to_csv(args.out / "validation_reps.csv", index=False, float_format="%.8g")
    summary = summarize(allrows)
    summary["settings"] = {"B": args.B, "null_reps": args.null_reps, "effect_reps": args.effect_reps, "nopart_reps": args.nopart_reps}
    (args.out / "validation_summary.json").write_text(json.dumps(summary, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_hypothesis"} for k, v in summary.items() if k != "settings"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
