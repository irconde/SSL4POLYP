#!/usr/bin/env python3
"""Per-run checks of the exploratory E4-stagelr runs (docs/v3/exploratory_e4_stagelr_plan.md); prints no metric.

For every run of a variant (``campaign_plan.stagelr_matrix``):

* **rates:** every epoch has a ``[stage-lr]`` line, and for both groups
  |lr − r_g(stage) · λ(e − 1)| / max(lr, 1e-30) < 1e-9 (interpretation I2; r_g and λ from the budget config);
* **stages:** backbone rate 0 and 0 trainable backbone parameters in stage 1; in stage 2, 7,087,872 (``head+1``) or
  14,175,744 (``head+2``) trainable backbone parameters when the blocks are unfrozen, 0 for SUP-imnet in
  ``persistonly``; runs stopped before stage 2 are counted;
* **arms A and B:** every ``*_zeroshot*_outputs.csv`` byte-identical to the registered run of the same pack, seed and
  regime;
* **parent:** the parent SHA-256 recorded by the run equals the one recorded by the registered E4 run;
* **controls** (``declared`` only): the three k = 1, seed 13 flag-on control runs are byte-identical to the controls
  (every CSV, and every checkpoint through ``exp_e4_stagelr_controls/files_manifest.csv``).

Run before the weights are deleted. Output: ``<records>/e4stagelr_<variant>/stagelr_checks.json``.

    python scripts/v3/check_e4_stagelr_runs.py --variant declared
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_plan as cp  # noqa: E402
import run_campaign as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BLOCK = 7_087_872
CONTROL_ROOT = "checkpoints_v3/exp_e4_stagelr_controls/flagon"
CONTROL_MANIFEST = REPO / "results_v3/verification/exp_e4_stagelr_controls/files_manifest.csv"
CONTROLS = {("sup_imnet", 100), ("sup_imnet", 50), ("ssl_colon", 50)}
STAGE_LR = re.compile(r"\[stage-lr\] epoch=(\d+) head=(\S+) backbone=(\S+)")
EPOCH = re.compile(r"\[epoch (\d+)\] train: .*?trainable=\(head=(\d+) backbone=(\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(size: int):
    cfg = yaml.safe_load((REPO / f"config/exp/v3/e4_budgets/s{size}.yaml").read_text())
    stages, warm, total = cfg["protocol"]["finetune_schedule"], int(cfg["scheduler"]["warmup_epochs"]), int(cfg["epochs"])

    def lam(i: int) -> float:
        if i < warm:
            return (i + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(max((i - warm) / max(1, total - warm), 0.0), 1.0)))

    return stages, lam


def check_run(run: dict, variant: str, parents: dict) -> list:
    problems = []
    k, seed, regime, size = run["k"], run["seed"], run["regime"], int(run["variant"][1:])
    out = rc.output_dir(run)
    reg = REPO / cp.CKPT_ROOT / f"k{k}/e4/s{size}/{regime}_seed{seed}"
    name = rc.run_name(run)
    log = next(out.rglob("*.log")).read_text(errors="replace")
    stages, lam = schedule(size)
    e1 = int(stages[0]["epochs"])
    rate = {1: (float(stages[0]["head_lr"]), 0.0),
            2: (float(stages[1]["head_lr"]), float(stages[1]["backbone_lr"]))}
    lrs = {int(e): (float(h), float(b)) for e, h, b in STAGE_LR.findall(log)}
    trainable = {int(e): (int(h), int(b)) for e, h, b in EPOCH.findall(log)}
    info = {"epochs": len(trainable), "reached_stage2": any(e > e1 for e in trainable)}
    if set(lrs) != set(trainable) or not lrs:
        problems.append((name, "stage_lr_lines", f"{len(lrs)} [stage-lr] lines for {len(trainable)} epochs"))
    worst = 0.0
    for e, (h, b) in lrs.items():
        rh, rb = rate[1 if e <= e1 else 2]
        for lr, r in ((h, rh), (b, rb)):
            worst = max(worst, abs(lr - r * lam(e - 1)) / max(lr, 1e-30))
        if e <= e1 and b != 0.0:
            problems.append((name, "stage1_backbone_lr", f"epoch {e}: {b}"))
    if worst >= 1e-9:
        problems.append((name, "rate_formula", f"max relative error {worst:.3e}"))
    unfrozen = variant == "declared" or regime != "sup_imnet"
    expected2 = (1 if stages[1]["mode"] == "head+1" else 2) * BLOCK if unfrozen else 0
    for e, (_, tb) in trainable.items():
        want = 0 if e <= e1 else expected2
        if tb != want:
            problems.append((name, "trainable", f"epoch {e}: backbone {tb} != {want}"))
    if {h for h, _ in trainable.values()} != {1538}:
        problems.append((name, "trainable_head", str({h for h, _ in trainable.values()})))
    info["max_rel_err"] = worst
    # arms A and B
    zs = sorted(p.relative_to(reg) for p in reg.rglob("*_zeroshot*_outputs.csv"))
    if len(zs) != 3:
        problems.append((name, "zeroshot_files", f"{len(zs)} registered zero-shot CSVs"))
    for rel in zs:
        new = out / rel
        if not new.exists() or new.read_bytes() != (reg / rel).read_bytes():
            problems.append((name, "zeroshot_identical", str(rel)))
    # parent
    payload = json.loads(next(out.rglob("*_last.metrics.json")).read_text())
    parent_sha = payload["provenance"]["parent_run"]["checkpoint_sha256"]
    if parents.get(cp.parent_path(k, seed, regime, cp.CKPT_ROOT)) != {parent_sha}:
        problems.append((name, "parent_sha256", parent_sha))
    # repeated control runs
    if variant == "declared" and k == 1 and seed == 13 and (regime, size) in CONTROLS:
        ctrl = REPO / CONTROL_ROOT / f"k1/e4/s{size}/{regime}_seed13"
        manifest = {row["path"]: row["sha256"] for row in csv.DictReader(CONTROL_MANIFEST.open())}
        compared = 0
        for f in sorted(p for p in ctrl.rglob("*.csv")):
            rel = f.relative_to(ctrl)
            compared += 1
            if not (out / rel).exists() or (out / rel).read_bytes() != f.read_bytes():
                problems.append((name, "control_csv_identical", str(rel)))
        prefix = str(ctrl.relative_to(REPO)) + "/"
        for path, digest in manifest.items():
            if path.startswith(prefix) and path.endswith(".pth"):
                compared += 1
                new = out / path[len(prefix):]
                if not new.exists() or sha256(new) != digest:
                    problems.append((name, "control_pth_sha256", path[len(prefix):]))
        info["control_files_compared"] = compared
    return problems, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=cp.STAGELR_VARIANTS, required=True)
    args = ap.parse_args()
    runs = cp.stagelr_matrix(cp.REGIMES, args.variant)
    parents = cp.registered_parent_sha256()
    problems, infos = [], {}
    for run in runs:
        if not rc.is_complete(run):
            problems.append((rc.run_name(run), "complete", "no RUN_COMPLETE.json"))
            continue
        p, info = check_run(run, args.variant, parents)
        problems += p
        infos[rc.run_name(run)] = info
    summary = {
        "variant": args.variant, "expected_runs": len(runs), "checked_runs": len(infos),
        "reached_stage2": sum(i["reached_stage2"] for i in infos.values()),
        "stopped_in_stage1": [n for n, i in infos.items() if not i["reached_stage2"]],
        "epochs_checked": sum(i["epochs"] for i in infos.values()),
        "max_rel_err": max((i["max_rel_err"] for i in infos.values()), default=float("nan")),
        "control_runs": {n: i["control_files_compared"] for n, i in infos.items() if "control_files_compared" in i},
        "problems_by_check": dict(Counter(c for _, c, _ in problems)),
        "problems": [{"run": n, "check": c, "evidence": e} for n, c, e in problems],
    }
    out = REPO / rc.STAGELR_RECORDS / f"e4stagelr_{args.variant}" / "stagelr_checks.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1))
    for p in summary["problems"][:20]:
        print("PROBLEM", p)
    return 0 if not problems and len(infos) == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
