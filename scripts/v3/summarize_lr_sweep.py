#!/usr/bin/env python3
"""Summarise the SAP v3.1 learning-rate sweep (validation only) and apply the decision rule.

Per run: validation AP of the best-epoch weights (``val.auprc`` of the canonical metrics JSON,
recomputed from ``*_val_outputs.csv`` as a check), first-epoch train/validation loss, best epoch,
last epoch, and first-epoch training stability (peak batch loss, batches above 2·ln 2, grad norm)
from the run's stdout. Test metrics are never read.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from ssl4polyp.classification.analysis.frame_outputs import load_frame_outputs
from ssl4polyp.classification.metrics.binary import metrics_at

REPO = Path(__file__).resolve().parents[2]
TAG = {"sup_imnet": "SUPImNet", "ssl_imnet": "SSLImNet", "ssl_colon": "SSLColon", "ssl_imnet_matched": "SSLImNetMatched"}
LRS = ("1.0e-4", "3.0e-4", "1.0e-3")
SEEDS = (13, 29, 47)
EPOCH_RE = re.compile(r"\[epoch (\d+)\] train: loss=([\d.]+).*?val: loss=([\d.]+)")
BATCH_RE = re.compile(r"Train Epoch: 1 \[(\d+)/\d+ [^\]]*\]\tLoss: ([\d.]+).*?GradNorm: ([\d.]+)")


def run_row(ckpt_root: Path, results_root: Path, model: str, lr: str, seed: int) -> dict:
    tag = TAG[model]
    run_dir = ckpt_root / f"lr{lr}" / f"sun_full_{model}_seed{seed}" / "sun_baselines"
    payload = json.loads((run_dir / f"{tag}_SUNFull_s{seed}_last.metrics.json").read_text())
    assert not payload.get("test_primary"), "test metrics present in a validation-only run"
    val = load_frame_outputs(run_dir / f"{tag}_SUNFull_s{seed}_val_outputs.csv")
    ap_csv = metrics_at(val.probs, val.labels, 0.5)["auprc"]
    log = (run_dir / f"{tag}_SUNFull_s{seed}.log").read_text(errors="replace")
    epochs = [(int(e), float(t), float(v)) for e, t, v in EPOCH_RE.findall(log)]
    stdout = (results_root / f"lr{lr}_{model}_seed{seed}.stdout").read_text(errors="replace").replace("\r", "\n")
    batches = [(int(n), float(l), float(g)) for n, l, g in BATCH_RE.findall(stdout)]
    losses = [b[1] for b in batches]
    return {
        "model": model,
        "lr": lr,
        "seed": seed,
        "val_ap": float(payload["val"]["auprc"]),
        "val_ap_csv": ap_csv,
        "val_auroc": float(payload["val"]["auroc"]),
        "epoch1_train_loss": epochs[0][1],
        "epoch1_val_loss": epochs[0][2],
        "best_epoch": int(payload["evaluated_epoch"]),
        "best_val_loss": min(e[2] for e in epochs),
        "last_epoch": epochs[-1][0],
        "stopped_before_epoch6": epochs[-1][0] < 6,
        "best_epoch_before_6": int(payload["evaluated_epoch"]) < 6,
        "epoch1_peak_batch_loss": max(losses) if losses else math.nan,
        "epoch1_batches_above_2ln2": sum(1 for l in losses if l > 2 * math.log(2)),
        "epoch1_n_batches": len(losses),
        "epoch1_peak_grad_norm": max(b[2] for b in batches) if batches else math.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-root", type=Path, default=REPO / "checkpoints_v3/lr_sweep/classification/k1")
    parser.add_argument("--results-root", type=Path, default=REPO / "results_v3/lr_sweep/k1")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results_v3/verification")
    parser.add_argument("--regimes", nargs="+", default=["sup_imnet", "ssl_imnet", "ssl_colon"], choices=list(TAG))
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    args = parser.parse_args()

    rows = [run_row(args.ckpt_root, args.results_root, m, lr, s) for m in args.regimes for lr in LRS for s in SEEDS]
    worst = max(abs(r["val_ap"] - r["val_ap_csv"]) for r in rows)
    print(f"val AP JSON vs CSV: max |diff| = {worst:.2e}")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["lr"])].append(r)
    summary = []
    for (model, lr), rs in groups.items():
        aps = [r["val_ap"] for r in rs]
        summary.append({
            "model": model, "lr": lr,
            "val_ap_mean": st.mean(aps), "val_ap_sd": st.stdev(aps),
            "val_ap_seeds": " / ".join(f"{a:.4f}" for a in aps),
            "epoch1_train_loss": " / ".join(f"{r['epoch1_train_loss']:.3f}" for r in rs),
            "epoch1_val_loss": " / ".join(f"{r['epoch1_val_loss']:.3f}" for r in rs),
            "best_epoch": " / ".join(str(r["best_epoch"]) for r in rs),
            "last_epoch": " / ".join(str(r["last_epoch"]) for r in rs),
            "n_stopped_before_epoch6": sum(r["stopped_before_epoch6"] for r in rs),
            "n_best_epoch_before_6": sum(r["best_epoch_before_6"] for r in rs),
            "epoch1_peak_batch_loss": " / ".join(f"{r['epoch1_peak_batch_loss']:.2f}" for r in rs),
            "epoch1_batches_above_2ln2": " / ".join(f"{r['epoch1_batches_above_2ln2']}/{r['epoch1_n_batches']}" for r in rs),
        })

    decision = {}
    for model in args.regimes:
        cands = sorted((s for s in summary if s["model"] == model), key=lambda s: -s["val_ap_mean"])
        decision[model] = cands[0]["lr"]
    if set(args.regimes) == {"sup_imnet", "ssl_imnet", "ssl_colon"}:  # SAP section 10, rules 1-3
        common = all(lr == "1.0e-3" for lr in decision.values())
        outcome = "keep common lr 1e-3" if common else "per-regime learning rates"
    else:  # SAP v3.2 section 10, rule 4: the regime's own highest mean validation AP
        outcome = "rule 4: highest mean validation AP"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in ((f"lr_sweep_runs{args.tag}.csv", rows), (f"lr_sweep_summary{args.tag}.csv", summary)):
        with (args.out_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (args.out_dir / f"lr_sweep_decision{args.tag}.json").write_text(
        json.dumps({"chosen_lr": decision, "outcome": outcome, "rule": "SAP v3.1 section 10"}, indent=2)
    )
    for s in sorted(summary, key=lambda s: (s["model"], float(s["lr"]))):
        print(f"{s['model']:9} lr={s['lr']}  AP={s['val_ap_mean']:.4f}±{s['val_ap_sd']:.4f}  "
              f"e1 train={s['epoch1_train_loss']}  best={s['best_epoch']}  last={s['last_epoch']}  "
              f"stop<6={s['n_stopped_before_epoch6']} best<6={s['n_best_epoch_before_6']}  "
              f"peak={s['epoch1_peak_batch_loss']}  >2ln2={s['epoch1_batches_above_2ln2']}")
    print("decision:", json.dumps(decision), "->", outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
