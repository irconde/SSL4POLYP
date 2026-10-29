#!/usr/bin/env python3
"""Exploratory: H1–H7 with SSL-colon re-pretrained in the SSL-imnet-matched environment (SSL-colon-r).

Plan: ``docs/v3/exploratory_sslcolon_repretrain_plan.md`` (exploratory, not in the SAP; no Holm, no verdicts).
Uses ``analysis/v3`` unchanged: loader and integrity checks, views, ``hypothesis_contrasts`` (H1–H7 exactly as
registered), the t-test on the K partition means (``mixed``, variant ``ttest``) and the cluster bootstrap per family
(B = 2000, seed 12345).

Campaigns: the registered campaign (``checkpoints_v3/campaign/classification``) and the SSL-colon-r campaign
(``checkpoints_v3/exp_sslcolon_r/classification``, regime ``ssl_colon_r``: 180 runs, the same experiments, variants,
partitions and seeds as SSL-colon). The SSL-colon-r runs replace the SSL-colon runs and every hypothesis is rebuilt
with the registered code, so H5–H7 (means over the four regimes) change too. Three versions per hypothesis, over the
same pairs (partition × seed) and the same bootstrap resamples:
  r           H with SSL-colon-r in place of SSL-colon
  registered  H as registered (must reproduce ``results_v3/analysis/contrast_summary.csv``)
  diff        r − registered
The diffs of H1 (= of H2), H3 and H4 are the paired SSL-colon-r − SSL-colon differences in AULC of AUPRC, flat AUROC
and zero-shot (E3) AUROC. Pretraining loss curves: ``--loss-r`` vs ``--loss-matched`` (``log.txt`` of main_pretrain).

Check before the data exist: ``--r-root checkpoints_v3/campaign/classification --r-regime ssl_colon`` substitutes the
original SSL-colon for itself; ``r`` and ``registered`` must then both equal the registered H1–H7 and ``diff`` be 0.

    PYTHONPATH=. python -m analysis.v3.exploratory_sslcolon_r --out results_v3/analysis_exploratory_sslcolon_r
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

from . import mixed  # noqa: E402
from .analysis import run_metrics_table  # noqa: E402
from .contrasts import COLON, hypothesis_contrasts, secondary_contrasts  # noqa: E402
from .design import Contrast, View, bootstrap, build_views, ci95  # noqa: E402
from .loader import Campaign, integrity_checks, load_campaign  # noqa: E402
from .sap import B_BOOT, BOOT_SEED, HYPOTHESES, HYPOTHESIS_TEXT, MODEL_VARIANT, REPO  # noqa: E402
from .tables import to_markdown  # noqa: E402

R = "ssl_colon_r"  # name of the SSL-colon-r views next to the registered ones
REG_ROOT = REPO / "checkpoints_v3/campaign/classification"
R_ROOT = REPO / "checkpoints_v3/exp_sslcolon_r/classification"
LOSS_R = REPO / "checkpoints_v3/pretrain/mae_sslcolon_r/ckpts/log.txt"
LOSS_MATCHED = REPO / "checkpoints_v3/pretrain/mae_imnet_matched/ckpts/log.txt"
PAIRED = {"H1": "AULC(auprc), E1", "H3": "AUROC, E2 flat stratum", "H4": "AUROC, E3 zero-shot (polypgen_clean_test)"}


def _checked(c: Campaign, name: str) -> Campaign:
    failed = [x for x in c.checks if not x["passed"]]
    if failed:
        raise RuntimeError(f"{name}: {len(failed)} integrity checks failed, first: {failed[0]}")
    return c


def load(reg_roots: List[Path], r_root: Path, r_regime: str, log=print):
    """Registered views plus the SSL-colon-r views (regime ``R``), and the substituted campaign's views."""

    reg = _checked(load_campaign(reg_roots), "registered")
    rc = _checked(load_campaign([r_root]), "SSL-colon-r")
    r_runs = [r for r in rc.runs if r.regime == r_regime]
    want = {(r.experiment, r.variant, r.k, r.seed) for r in reg.runs if r.regime == COLON}
    got = {(r.experiment, r.variant, r.k, r.seed) for r in r_runs}
    if got != want:
        raise RuntimeError(f"SSL-colon-r runs: {len(want - got)} missing, {len(got - want)} extra (of {len(want)})")
    sub = Campaign([r for r in reg.runs if r.regime != COLON] + [replace(r, regime=COLON) for r in r_runs], reg.roots + [r_root])
    sub.checks = integrity_checks(sub)  # paired frames, inherited τ and E4 splits with SSL-colon-r in place
    _checked(sub, "substituted campaign")
    log(f"[load] registered {len(reg.runs)} runs, SSL-colon-r {len(r_runs)} runs ({r_root})")
    v_reg = build_views(reg)
    v_sub = build_views(sub)
    views: Dict[tuple, View] = dict(v_reg)
    for key, v in v_sub.items():
        if key[2] == COLON:
            nk = key[:2] + (R,) + key[3:]
            views[nk] = replace(v, key=nk)
    return reg, sub, v_reg, v_sub, views


def _rename(c: Contrast, cid: str) -> Contrast:
    terms = {rid: [((v[:2] + (R,) + v[3:]) if v[2] == COLON else v, m, w) for v, m, w in ts] for rid, ts in c.terms.items()}
    return Contrast(cid, c.family, {**c.meta, "version": "r"}, terms)


def build(reg: Campaign, v_reg, v_sub) -> Tuple[List[Contrast], List[Contrast]]:
    """(reported contrasts: r / registered / diff per hypothesis, registered secondary contrasts of the same families)."""

    regimes, partitions, seeds = reg.regimes(), reg.partitions(), reg.seeds()
    hyp_reg, _ = hypothesis_contrasts(v_reg, regimes, partitions, seeds)
    hyp_sub, _ = hypothesis_contrasts(v_sub, regimes, partitions, seeds)
    out: List[Contrast] = []
    for h in HYPOTHESES:
        a, b = hyp_sub[h], hyp_reg[h]
        if a is None or b is None or set(a.terms) != set(b.terms):
            raise RuntimeError(f"{h}: incomplete or unpaired")
        r = _rename(a, f"r|{h}")
        g = Contrast(f"registered|{h}", b.family, {**b.meta, "version": "registered"}, b.terms)
        d = Contrast(f"diff|{h}", b.family, {**b.meta, "version": "diff"},
                     {rid: r.terms[rid] + [(v, m, -w) for v, m, w in g.terms[rid]] for rid in g.terms})
        out += [r, g, d]
    # The registered bootstrap drew each family's resamples over the views of all its contrasts; the secondary
    # contrasts of the same families are bootstrapped alongside (not reported) so the resampling spaces are the same.
    families = {c.family for c in out}
    extra = [c for c in secondary_contrasts(v_reg, regimes, partitions, seeds) if c.family in families]
    return out, extra


def infer(views, contrasts: List[Contrast], extra: List[Contrast], B: int, seed: int, workers: int, log=print) -> pd.DataFrame:
    t0 = time.time()
    point = {vk: views[vk].metrics() for vk in {vk for c in contrasts for vk in c.views()}}
    per_run = {c.id: c.per_run(lambda vk, m: point[vk][m]) for c in contrasts}
    fits = mixed.fit_many(per_run, MODEL_VARIANT).set_index("id")
    boot: Dict[str, Tuple[float, float, int]] = {}
    for fam in sorted({c.family for c in contrasts}):
        members = [c for c in contrasts if c.family == fam] + [c for c in extra if c.family == fam]
        reps, counts = bootstrap(views, members, B=B, seed=seed, workers=workers)
        for j, c in enumerate(members):
            lo, hi = ci95(reps[:, j])
            boot[c.id] = (lo, hi, int(np.isfinite(reps[:, j]).sum()))
        log(f"[bootstrap] {fam}: {len(members)} contrasts, units {counts} ({time.time() - t0:.0f}s)")
    rows = []
    for c in contrasts:
        f, sp = fits.loc[c.id], mixed.spread(per_run[c.id])
        lo, hi, nb = boot[c.id]
        rows.append({"id": c.id, "hypothesis": c.meta["hypothesis"], "version": c.meta["version"], "family": c.family,
                     "definition": HYPOTHESIS_TEXT[c.meta["hypothesis"]], "comparison": c.meta.get("comparison", ""),
                     "metric": c.meta.get("metric", ""), "estimate": sp["mean"], "sd_partition": sp["sd_partition"],
                     "sd_seed": sp["sd_seed"], "n_runs": sp["n_runs"], "method": f["method"], "t_ci_lo": f["ci_lo"],
                     "t_ci_hi": f["ci_hi"], "p_unadjusted": f["p"], "boot_lo": lo, "boot_hi": hi, "boot_B": nb})
    return pd.DataFrame(rows)


def check_registered(long: pd.DataFrame, versions=("registered",)) -> pd.DataFrame:
    """Versions against the registered analysis (estimate, t-test CI, bootstrap CI, B), 10 significant digits."""

    cs = pd.read_csv(REPO / "results_v3/analysis/contrast_summary.csv", float_precision="round_trip").set_index("id")
    rows = []
    for _, r in long[long["version"].isin(versions)].iterrows():
        ref = cs.loc[r["hypothesis"]]
        for mine, theirs in (("estimate", "mean"), ("t_ci_lo", "model_ci_lo"), ("t_ci_hi", "model_ci_hi"),
                             ("boot_lo", "boot_lo"), ("boot_hi", "boot_hi"), ("boot_B", "boot_B")):
            diff = abs(float(r[mine]) - float(ref[theirs]))
            rows.append({"hypothesis": r["hypothesis"], "version": r["version"], "quantity": mine, "recomputed": r[mine],
                         "registered": ref[theirs], "abs_diff": diff,
                         "exact": diff <= 5e-10 * max(1.0, abs(float(ref[theirs])))})
    return pd.DataFrame(rows)


def loss_curves(r_log: Path, matched_log: Path) -> pd.DataFrame:
    """Mean train loss per epoch of both pretrainings (``log.txt``: one JSON object per epoch)."""

    def read(p: Path) -> pd.Series:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        s = pd.Series({int(r["epoch"]): float(r["train_loss"]) for r in rows})
        if list(s.index) != list(range(len(s))):
            raise RuntimeError(f"{p}: epochs not 0..{len(s) - 1} exactly once")
        return s

    r, m = read(r_log), read(matched_log)
    df = pd.DataFrame({"epoch": m.index, "train_loss_matched": m.values}).merge(
        pd.DataFrame({"epoch": r.index, "train_loss_r": r.values}), on="epoch", how="outer")
    df["r_minus_matched"] = df["train_loss_r"] - df["train_loss_matched"]
    return df


def wide(long: pd.DataFrame) -> pd.DataFrame:
    base = long[long["version"] == "registered"][["hypothesis", "definition"]].set_index("hypothesis")
    for tag in ("r", "registered", "diff"):
        sub = long[long["version"] == tag].set_index("hypothesis")
        for col in ("estimate", "t_ci_lo", "t_ci_hi", "boot_lo", "boot_hi"):
            base[f"{tag}_{col}"] = sub[col]
    return base.reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reg-roots", type=Path, nargs="+", default=[REG_ROOT])
    ap.add_argument("--r-root", type=Path, default=R_ROOT)
    ap.add_argument("--r-regime", default="ssl_colon_r", help="regime name of the SSL-colon-r runs under --r-root")
    ap.add_argument("--loss-r", type=Path, default=LOSS_R)
    ap.add_argument("--loss-matched", type=Path, default=LOSS_MATCHED)
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    t0 = time.time()
    reg, sub, v_reg, v_sub, views = load(args.reg_roots, args.r_root, args.r_regime)
    contrasts, extra = build(reg, v_reg, v_sub)
    long = infer(views, contrasts, extra, args.B, args.boot_seed, args.workers)
    identity = args.r_regime == COLON and args.r_root.resolve() in {r.resolve() for r in args.reg_roots}
    check = check_registered(long, ("r", "registered") if identity else ("registered",))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "hypotheses_long.csv", index=False, float_format="%.10g")
    w = wide(long)
    w.to_csv(out / "hypotheses_r_vs_registered.csv", index=False, float_format="%.10g")
    check.to_csv(out / "registered_check.csv", index=False, float_format="%.17g")
    rm = run_metrics_table({k: v for k, v in views.items() if k[2] in (COLON, R)},
                           {k: v.metrics() for k, v in views.items() if k[2] in (COLON, R)})
    rm.to_csv(out / "run_metrics_ssl_colon_and_r.csv", index=False, float_format="%.10g")
    paired = long[(long["version"] == "diff") & long["hypothesis"].isin(PAIRED)].copy()
    paired.insert(1, "paired_difference", paired["hypothesis"].map(lambda h: f"SSL-colon-r − SSL-colon, {PAIRED[h]}"))
    paired.to_csv(out / "paired_r_minus_colon.csv", index=False, float_format="%.10g")
    info = {"reg_roots": [str(r) for r in args.reg_roots], "r_root": str(args.r_root), "r_regime": args.r_regime,
            "partitions": reg.partitions(), "seeds": reg.seeds(), "B": args.B, "boot_seed": args.boot_seed,
            "model_variant": MODEL_VARIANT, "n_contrasts": len(contrasts), "n_extra_bootstrapped": len(extra),
            "registered_check": {"n": len(check), "exact": int(check["exact"].sum()), "max_abs_diff": float(check["abs_diff"].max())}}
    md = ["# Exploratory: H1–H7 with SSL-colon-r in place of SSL-colon", "",
          "Exploratory (not registered in the SAP): no Holm, no verdicts. Estimate = mean of the paired per-run values "
          "(5 partitions × 3 seeds); t-test CI on the 5 partition means; cluster-bootstrap CI (B = 2000, seed 12345). "
          "r: SSL-colon-r replaces SSL-colon in every hypothesis (H5–H7 are means over the four regimes); registered: as "
          "registered; diff = r − registered.", "", to_markdown(w, "H1–H7")]
    if args.loss_r.exists() and args.loss_matched.exists():
        lc = loss_curves(args.loss_r, args.loss_matched)
        lc.to_csv(out / "pretraining_loss_r_vs_matched.csv", index=False, float_format="%.10g")
        milestones = lc[lc["epoch"].isin([0, 50, 100, 150, 200, 250, 300, 350, 399])]
        md.append(to_markdown(milestones, "Pretraining mean train loss (selected epochs)"))
        info["loss_epochs"] = {"r": int(lc["train_loss_r"].notna().sum()), "matched": int(lc["train_loss_matched"].notna().sum())}
    else:
        info["loss_epochs"] = "not available"
    md.append(f"Registered values recomputed: {info['registered_check']['exact']}/{info['registered_check']['n']} equal to "
              f"results_v3/analysis/contrast_summary.csv (max |diff| {info['registered_check']['max_abs_diff']:.3g}).")
    (out / "hypotheses_r_vs_registered.md").write_text("\n".join(md) + "\n")
    info["elapsed_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
