#!/usr/bin/env python3
"""Check the spec v3.1 section 6 criteria on finished runs.

For each ``*.metrics.json`` given (the canonical ``*_last.metrics.json`` of training runs or the
``*.metrics.json`` of evaluation-only runs) it checks:

  reload      the log records the reload of the best checkpoint (training runs)
  epoch       ``threshold_epoch_check`` passed and ``thresholds.primary.epoch`` equals the epoch
              of the weights it belongs to (evaluated checkpoint, or parent for a frozen SUN τ)
  ckpt_sha    the SHA-256 stored for the evaluated checkpoint matches the file on disk
  csv_columns every per-frame CSV exists and has all the columns of spec 1.3
  pred        the CSV ``pred`` column equals ``prob >= tau``
  json_vs_csv every metric block in the JSON equals the recomputation from the CSV (|diff| < 1e-6):
              ``test_primary``, ``val``, ``zero_shot.test_primary`` and the four E4 arms
  tau         τ recomputed from the CSVs equals the stored τ (training τ, E4 arms B and D)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ssl4polyp.classification.analysis.frame_outputs import FRAME_OUTPUT_COLUMNS, load_frame_outputs
from ssl4polyp.classification.metrics import thresholds
from ssl4polyp.classification.metrics.binary import METRICS, metrics_at

TOL = 1e-6
COUNT_KEYS = ("tp", "fp", "tn", "fn")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Checker:
    def __init__(self, name: str):
        self.name = name
        self.results: List[dict] = []

    def add(self, criterion: str, passed: bool, evidence: str) -> None:
        self.results.append({"run": self.name, "criterion": criterion, "passed": bool(passed), "evidence": evidence})


def compare_block(block: Dict, csv_path: Path, tau: float) -> Tuple[bool, str]:
    outputs = load_frame_outputs(csv_path)
    recomputed = metrics_at(outputs.probs, outputs.labels, tau)
    worst = 0.0
    compared = 0
    for key in METRICS + COUNT_KEYS:
        if key not in block:
            continue
        expected, actual = float(block[key]), float(recomputed[key])
        if math.isnan(actual) and math.isnan(expected):
            continue
        worst = max(worst, abs(expected - actual))
        compared += 1
    ok = compared > 0 and worst < TOL
    return ok, f"{compared} metrics vs {csv_path.name} at tau={tau:.9g}: max |diff|={worst:.3g}"


def check_csv(checker: Checker, csv_path: Path, tau: Optional[float]) -> None:
    if not csv_path.exists():
        checker.add(f"csv_columns:{csv_path.name}", False, "missing")
        return
    outputs = load_frame_outputs(csv_path)
    header = list(outputs.columns.keys())
    missing = [c for c in FRAME_OUTPUT_COLUMNS if c not in header]
    empty_required = [c for c in ("frame_id", "prob", "label", "pred") if not all(outputs.col(c))]
    checker.add(
        f"csv_columns:{csv_path.name}",
        not missing and not empty_required,
        f"{len(outputs)} rows; missing columns={missing or 'none'}; empty required={empty_required or 'none'}",
    )
    if tau is not None:
        mismatches = int(np.sum(outputs.preds != (outputs.probs >= tau).astype(int)))
        checker.add(f"pred:{csv_path.name}", mismatches == 0, f"{mismatches} rows with pred != (prob >= {tau:.9g})")


def verify(metrics_path: Path) -> List[dict]:
    payload = json.loads(metrics_path.read_text())
    run_dir = metrics_path.parent
    stem = metrics_path.name.replace("_last.metrics.json", "").replace(".metrics.json", "")
    checker = Checker(stem)
    training = not payload.get("eval_only", False)

    log_path = run_dir / f"{stem}.log"
    if training:
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        line = next((l for l in log_text.splitlines() if l.startswith("Reloaded best checkpoint")), "")
        checker.add("reload", bool(line), line or f"no reload line in {log_path.name}")

    check = payload.get("threshold_epoch_check") or {}
    primary = (payload.get("thresholds") or {}).get("primary") or {}
    expected_epoch = check.get("weights_epoch")
    checker.add(
        "epoch",
        bool(check.get("passed")) and primary.get("epoch") == expected_epoch,
        f"thresholds.primary.epoch={primary.get('epoch')} policy={primary.get('policy')} "
        f"vs {check.get('compared_against')} epoch={expected_epoch}; evaluated_epoch={payload.get('evaluated_epoch')}",
    )

    evaluated = payload.get("evaluated_checkpoint") or {}
    ckpt_path = Path(evaluated.get("path", ""))
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path
    if ckpt_path.exists():
        actual = sha256(ckpt_path)
        checker.add("ckpt_sha", actual == evaluated.get("sha256"), f"{ckpt_path.name} sha256={actual[:16]}… stored={str(evaluated.get('sha256'))[:16]}…")
    else:
        checker.add("ckpt_sha", False, f"evaluated checkpoint not found: {ckpt_path}")

    tau = float(payload["test_primary"]["tau"])
    test_csv = run_dir / f"{stem}_test_outputs.csv"
    val_csv = run_dir / f"{stem}_val_outputs.csv"
    check_csv(checker, test_csv, tau)
    ok, ev = compare_block(payload["test_primary"], test_csv, tau)
    checker.add("json_vs_csv:test_primary", ok, ev)
    if training:
        check_csv(checker, val_csv, tau)
        ok, ev = compare_block(payload["val"], val_csv, tau)
        checker.add("json_vs_csv:val", ok, ev)
        if primary.get("policy") == "f1_opt_on_val":
            val = load_frame_outputs(val_csv)
            recomputed = thresholds.compute_policy_threshold(
                val.probs, val.labels, policy="f1_opt_on_val", split_name="val", epoch=0
            ).tau
            checker.add("tau:primary", recomputed == float(primary["tau"]), f"recomputed={recomputed:.9g} stored={float(primary['tau']):.9g}")

    arms = payload.get("e4_arms") or {}
    if arms:
        zs_csv = run_dir / f"{stem}_zeroshot_outputs.csv"
        support = {
            "zeroshot_train": run_dir / f"{stem}_zeroshot_train_outputs.csv",
            "zeroshot_val": run_dir / f"{stem}_zeroshot_val_outputs.csv",
            "train": run_dir / f"{stem}_train_outputs.csv",
        }
        check_csv(checker, zs_csv, float(arms["A"]["metrics"]["tau"]))
        for path in support.values():
            check_csv(checker, path, float(arms["A"]["metrics"]["tau"]))
        zero_primary = (payload.get("zero_shot") or {}).get("test_primary")
        if zero_primary:
            ok, ev = compare_block(zero_primary, zs_csv, float(zero_primary["tau"]))
            checker.add("json_vs_csv:zero_shot.test_primary", ok, ev)
        for arm, csv_path in (("A", zs_csv), ("B", zs_csv), ("C", test_csv), ("D", test_csv)):
            block = arms[arm]["metrics"]
            ok, ev = compare_block(block, csv_path, float(block["tau"]))
            checker.add(f"json_vs_csv:e4_arm_{arm}", ok, ev)
        zs_train, zs_val = load_frame_outputs(support["zeroshot_train"]), load_frame_outputs(support["zeroshot_val"])
        tau_b = thresholds.compute_policy_threshold(
            np.concatenate([zs_train.probs, zs_val.probs]),
            np.concatenate([zs_train.labels, zs_val.labels]),
            policy="f1_opt_on_val", split_name="support", epoch=0,
        ).tau
        stored_b = float(arms["B"]["metrics"]["tau"])
        checker.add("tau:e4_arm_B", tau_b == stored_b, f"recomputed={tau_b:.9g} stored={stored_b:.9g} (n_support={arms['B'].get('n_support')})")
        adapted_val = load_frame_outputs(val_csv)
        stored_d = float(arms["D"]["metrics"]["tau"])
        if np.unique(adapted_val.labels).size < 2:
            ok = stored_d == float(arms["C"]["metrics"]["tau"])
            checker.add("tau:e4_arm_D", ok, f"single-class support-val; kept inherited tau={stored_d:.9g}")
        else:
            tau_d = thresholds.compute_policy_threshold(
                adapted_val.probs, adapted_val.labels, policy="f1_opt_on_val", split_name="val", epoch=0
            ).tau
            checker.add("tau:e4_arm_D", tau_d == stored_d, f"recomputed={tau_d:.9g} stored={stored_d:.9g} (n_val={len(adapted_val)})")
    return checker.results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics_json", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Write the results as JSON here")
    args = parser.parse_args()
    results: List[dict] = []
    for path in args.metrics_json:
        results.extend(verify(path))
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['run']:<40} {r['criterion']:<36} {r['evidence']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
