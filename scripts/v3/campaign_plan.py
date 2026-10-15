#!/usr/bin/env python3
"""Dry run of the campaign v3 run matrix (no training, no GPU).

For every experiment, partition, seed and regime it resolves the run exactly as
``train_classification`` would (``prepare_run``): packs, splits, learning rate, threshold policy,
parent checkpoint and output layout. Each resolved run is checked against the rules of the SAP
version currently registered in ``results_v3/RUNLOG.md``. It also reports the composition of every
few-shot pack (positives, isolated and sequence negatives in the evaluation split and the support).

Outputs (``--out``): ``campaign_plan.csv``, ``campaign_plan_checks.json``, ``fewshot_composition.csv``.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ssl4polyp.classification import train_classification as tc

REPO = Path(__file__).resolve().parents[2]
PACKS = REPO / "data_packs"
V3 = PACKS / "v3"
PARTITIONS = (1, 2, 3, 4, 5)
SEEDS = (13, 29, 47)
REGIMES = ("sup_imnet", "ssl_imnet", "ssl_colon")
PENDING_REGIMES = ("ssl_imnet_matched",)
PERCENTS = (5, 10, 25, 50, 100)
SIZES = (50, 100, 200, 500)
SELECTED_LR = {"sup_imnet": 3.0e-4, "ssl_imnet": 3.0e-4, "ssl_colon": 1.0e-4}  # SAP section 10
TAG = {"sup_imnet": "SUPImNet", "ssl_imnet": "SSLImNet", "ssl_colon": "SSLColon"}
CKPT_ROOT = "checkpoints_v3/campaign/classification"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_sap() -> Dict[str, str]:
    """Last SAP row registered in RUNLOG.md: ``| <name> | `path` | `sha` | ...``."""

    rows = re.findall(r"^\| (SAP[^|]*)\| `([^`]+)` \| `([0-9a-f]{64})`", (REPO / "results_v3/RUNLOG.md").read_text(), re.M)
    name, path, digest = rows[-1]
    actual = sha256(REPO / path)
    return {"name": name.strip(), "path": path, "registered_sha256": digest, "file_sha256": actual, "match": actual == digest}


def resolve(argv: List[str]) -> Dict[str, object]:
    sys.argv = ["train_classification", *argv]
    args = tc.get_args()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        prepared = tc.prepare_run(args)
    return {"args": args, **prepared}


def parent_path(k: int, seed: int, regime: str) -> str:
    return f"{CKPT_ROOT}/k{k}/sun_full/{regime}_seed{seed}/sun_baselines/{TAG[regime]}_SUNFull_s{seed}.pth"


def manifest(pack: str) -> dict:
    return yaml.safe_load((PACKS / pack / "manifest.yaml").read_text()) or {}


def case_set(path: Path) -> set:
    with path.open(newline="") as handle:
        return {row["case_id"] for row in csv.DictReader(handle)}


def run_matrix() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for k in PARTITIONS:
        for seed in SEEDS:
            for regime in REGIMES:
                base = ["--partition", str(k), "--seed", str(seed), "--model-key", regime, "--roots", "data/roots.json"]
                runs.append({"experiment": "sun_full", "k": k, "seed": seed, "regime": regime, "variant": "",
                             "argv": ["--exp-config", "exp/v3/sun_full.yaml", *base,
                                      "--output-dir", f"{CKPT_ROOT}/k{k}/sun_full/{regime}_seed{seed}"]})
                for p in PERCENTS:
                    runs.append({"experiment": "E1", "k": k, "seed": seed, "regime": regime, "variant": f"p{p}",
                                 "argv": ["--exp-config", "exp/v3/e1.yaml", *base, "--dataset-percent", str(p),
                                          "--output-dir", f"{CKPT_ROOT}/k{k}/e1/p{p}/{regime}_seed{seed}"]})
                parent = parent_path(k, seed, regime)
                runs.append({"experiment": "E2", "k": k, "seed": seed, "regime": regime, "variant": "",
                             "argv": ["--exp-config", "exp/v3/e2.yaml", *base, "--parent-checkpoint", parent,
                                      "--output-dir", f"{CKPT_ROOT}/k{k}/e2/{regime}_seed{seed}"]})
                runs.append({"experiment": "E3", "k": k, "seed": seed, "regime": regime, "variant": "",
                             "argv": ["--exp-config", "exp/v3/e3.yaml", *base, "--parent-checkpoint", parent,
                                      "--output-dir", f"{CKPT_ROOT}/k{k}/e3/{regime}_seed{seed}"]})
                for size in SIZES:
                    runs.append({"experiment": "E4", "k": k, "seed": seed, "regime": regime, "variant": f"s{size}",
                                 "argv": ["--exp-config", f"exp/v3/e4_budgets/s{size}.yaml", *base, "--parent-checkpoint", parent,
                                          "--output-dir", f"{CKPT_ROOT}/k{k}/e4/s{size}/{regime}_seed{seed}"]})
    return runs


def check_run(run: Dict[str, object], resolved: Dict[str, object], index: dict) -> List[str]:
    """Return the list of SAP-rule violations for one resolved run."""

    a = resolved["args"]
    exp, k, seed, regime = run["experiment"], run["k"], run["seed"], run["regime"]
    base = f"v3/k{k}/base"
    errors: List[str] = []

    def expect(cond: bool, message: str) -> None:
        if not cond:
            errors.append(message)

    expect(a.partition == k, f"partition {a.partition} != {k}")
    expect(int(a.seed) == seed, f"training seed {a.seed} != {seed}")
    if exp in ("sun_full", "E1"):
        expect(a.finetune_mode == "full", f"finetune {a.finetune_mode}")
        expect(abs(float(a.lr) - SELECTED_LR[regime]) < 1e-12, f"lr {a.lr} != {SELECTED_LR[regime]}")
        expect(str(a.lr_source).startswith("model config"), f"lr source {a.lr_source}")
        expect(a.val_pack == base and a.test_pack == base, f"val/test {a.val_pack}/{a.test_pack}")
        expect(a.threshold_policy == "f1_opt_on_val", f"policy {a.threshold_policy}")
        expect(a.parent_checkpoint in (None, ""), "unexpected parent")
    if exp == "sun_full":
        expect(a.train_pack == base, f"train {a.train_pack}")
        gen = manifest(base).get("generator") or {}
        expect("sun morph" in str(gen.get("command")), "base not built by `polypdb sun morph`")
        expect(gen.get("seed") == index["partitions"][k]["partition_seed"], f"partition seed {gen.get('seed')}")
    if exp == "E1":
        pct = int(str(run["variant"])[1:])
        pack = f"v3/k{k}/sun_subsets/sun_p{pct:02d}"
        expect(a.train_pack == pack, f"train {a.train_pack} != {pack}")
        gen = manifest(pack).get("generator") or {}
        expect("subsets-canonical" in str(gen.get("command")) and gen.get("seed") == 13, "subset generator/seed")
    if exp in ("E2", "E3", "E4"):
        expect(a.parent_checkpoint == parent_path(k, seed, regime), f"parent {a.parent_checkpoint}")
    if exp in ("E2", "E3"):
        expect(a.finetune_mode == "none" and not a.train_pack, f"E2/E3 must not train ({a.finetune_mode}, {a.train_pack})")
        expect(a.threshold_policy == "sun_val_frozen", f"policy {a.threshold_policy}")
    if exp == "E2":
        expect(a.test_pack == base and a.test_split == "test", f"test {a.test_pack}/{a.test_split}")
        expect(list(a.morphology_eval or []) == ["flat", "polypoid"], f"morphology_eval {a.morphology_eval}")
    if exp == "E3":
        expect(a.test_pack == "polypgen_clean_test", f"test {a.test_pack}")
    if exp == "E4":
        size = int(str(run["variant"])[1:])
        pack = f"v3/k{k}/polypgen_fewshot/seed{seed}/s{size}"
        expect(a.train_pack == pack and a.val_pack == pack and a.test_pack == pack, f"packs {a.train_pack}")
        gen = manifest(pack).get("generator") or {}
        expect(gen.get("seed") == 1000 * k + seed, f"pack seed {gen.get('seed')} != {1000 * k + seed}")
        expect(index["partitions"][k]["fewshot_pack_seeds"][seed] == 1000 * k + seed, "index pack seed")
        expect(a.lr_source == "finetune_schedule", f"lr source {a.lr_source}")
        expect(a.threshold_policy == "sun_val_frozen", f"policy {a.threshold_policy}")
    return errors


def partition_checks(index: dict) -> List[str]:
    errors: List[str] = []
    seeds = [index["partitions"][k]["partition_seed"] for k in PARTITIONS]
    failed = list(index.get("failed_partition_seeds") or [])
    expected = [s for s in range(101, 121) if s not in failed][:4]
    if seeds != [47, *expected]:
        errors.append(f"partition seeds {seeds} != [47, {expected}]")
    for s in ("train", "val", "test"):
        if sha256(V3 / f"k1/base/{s}.csv") != sha256(PACKS / f"sun_morphology/{s}.csv"):
            errors.append(f"k1 base {s} != current sun_morphology")
    for k in PARTITIONS:
        base_cases = {s: case_set(V3 / f"k{k}/base/{s}.csv") for s in ("train", "val", "test")}
        if base_cases["train"] & base_cases["val"] or base_cases["train"] & base_cases["test"] or base_cases["val"] & base_cases["test"]:
            errors.append(f"k{k}: case leakage between base splits")
        with (V3 / f"k{k}/base/test.csv").open(newline="") as handle:
            morph = Counter((r["case_id"], r["morphology"]) for r in csv.DictReader(handle) if r["label"] == "1")
        flat = len({c for c, m in morph if m == "flat"})
        polypoid = len({c for c, m in morph if m == "polypoid"})
        if (flat, polypoid) != (5, 10):
            errors.append(f"k{k}: test flat/polypoid cases {flat}/{polypoid}")
        previous: Optional[set] = None
        for p in PERCENTS:
            cases = case_set(V3 / f"k{k}/sun_subsets/sun_p{p:02d}/train.csv")
            if not cases <= base_cases["train"]:
                errors.append(f"k{k} p{p}: subset not inside partition train")
            if previous is not None and not previous <= cases:
                errors.append(f"k{k} p{p}: subsets not nested")
            previous = cases
        if sha256(V3 / f"k{k}/sun_subsets/sun_p100/train.csv") != sha256(V3 / f"k{k}/base/train.csv"):
            errors.append(f"k{k}: p100 != base train")
    return errors


def fewshot_composition() -> List[Dict[str, object]]:
    def count(path: Path) -> Counter:
        c: Counter = Counter()
        with path.open(newline="") as handle:
            for r in csv.DictReader(handle):
                if r["label"] == "1":
                    c["pos"] += 1
                elif tc._resolve_polypgen_sequence(r["frame_id"], None) is not None:
                    c["neg_sequence"] += 1
                else:
                    c["neg_isolated"] += 1
        return c

    with (PACKS / "polypgen_clean_test/test.csv").open(newline="") as handle:
        clean_pos = {r["frame_id"] for r in csv.DictReader(handle) if r["label"] == "1"}
    rows = []
    for k in PARTITIONS:
        for seed in SEEDS:
            for size in SIZES:
                d = V3 / f"k{k}/polypgen_fewshot/seed{seed}/s{size}"
                ev, tr, va = count(d / "test.csv"), count(d / "train.csv"), count(d / "val.csv")
                row = {
                    "k": k, "seed": seed, "size": size, "pack_seed": manifest(str(d.relative_to(PACKS))).get("generator", {}).get("seed"),
                    "eval_pos": ev["pos"], "eval_neg_isolated": ev["neg_isolated"], "eval_neg_sequence": ev["neg_sequence"],
                    "support_pos": tr["pos"] + va["pos"], "support_neg_isolated": tr["neg_isolated"] + va["neg_isolated"],
                    "support_neg_sequence": tr["neg_sequence"] + va["neg_sequence"],
                    "support_train_frames": sum(tr.values()), "support_val_frames": sum(va.values()),
                }
                if size == 50:
                    with (d / "test.csv").open(newline="") as handle:
                        eval_pos = {r["frame_id"] for r in csv.DictReader(handle) if r["label"] == "1"}
                    row["eval_pos_common_with_clean_test"] = len(eval_pos & clean_pos)
                    row["flag_lt120_isolated_neg"] = ev["neg_isolated"] < 120
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "results_v3/verification")
    args = parser.parse_args()
    index = yaml.safe_load((V3 / "index.yaml").read_text())
    sap = current_sap()

    plan_rows, failures = [], []
    for run in run_matrix():
        try:
            resolved = resolve(run["argv"])
            errors = check_run(run, resolved, index)
            a = resolved["args"]
        except Exception as exc:  # report, do not stop the dry run
            errors, a = [f"resolution failed: {exc}"], None
        if errors:
            failures.append({k: run[k] for k in ("experiment", "k", "seed", "regime", "variant")} | {"errors": errors})
        plan_rows.append({
            "experiment": run["experiment"], "k": run["k"], "seed": run["seed"], "regime": run["regime"], "variant": run["variant"],
            "config": run["argv"][1], "train_pack": getattr(a, "train_pack", None) or "", "val_pack": getattr(a, "val_pack", None) or "",
            "test_pack": getattr(a, "test_pack", None) or "", "lr": getattr(a, "lr", ""), "lr_source": getattr(a, "lr_source", ""),
            "finetune": getattr(a, "finetune_mode", ""), "threshold_policy": getattr(a, "threshold_policy", ""),
            "parent": getattr(a, "parent_checkpoint", None) or "", "output_dir": getattr(a, "output_dir", ""),
            "stem": getattr(a, "run_stem", ""), "sap_ok": not errors,
        })
    pending = [{"regime": r, "status": "pending: no model config until the MAE pretraining and its learning-rate sweep (SAP section 10, rule 4)"} for r in PENDING_REGIMES]
    part_errors = partition_checks(index)
    composition = fewshot_composition()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("campaign_plan.csv", plan_rows), ("fewshot_composition.csv", composition)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    counts = Counter(r["experiment"] for r in plan_rows)
    summary = {"sap": sap, "runs": dict(counts), "total_runs": len(plan_rows), "run_failures": failures,
               "partition_check_failures": part_errors, "pending_regimes": pending,
               "fewshot_s50_lt120_isolated": [r for r in composition if r.get("flag_lt120_isolated_neg")]}
    (args.out / "campaign_plan_checks.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("sap", "runs", "total_runs")}, indent=2))
    print("run failures:", len(failures), failures[:3])
    print("partition check failures:", part_errors)
    print("pending:", [p["regime"] for p in pending])
    return 0 if sap["match"] and not failures and not part_errors else 1


if __name__ == "__main__":
    sys.exit(main())
