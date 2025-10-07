from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ssl4polyp.classification.analysis import exp4_report


def _write_outputs_csv(path: Path) -> None:
    rows = [
        {"frame_id": "frame0", "prob": 0.9, "label": 1, "pred": 1, "case_id": "case0", "morphology": "flat"},
        {"frame_id": "frame1", "prob": 0.2, "label": 0, "pred": 0, "case_id": "case0", "morphology": "flat"},
        {"frame_id": "frame2", "prob": 0.8, "label": 1, "pred": 1, "case_id": "case1", "morphology": "polypoid"},
        {"frame_id": "frame3", "prob": 0.3, "label": 0, "pred": 0, "case_id": "case1", "morphology": "polypoid"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_run(
    root: Path,
    *,
    model: str,
    percent: float,
    seed: int,
    auprc: float,
    f1_score: float,
) -> None:
    base = f"{model}_p{int(percent):02d}_s{seed}"
    metrics_path = root / f"{base}_last.metrics.json"
    outputs_path = root / f"{base}_test_outputs.csv"
    payload = {
        "seed": seed,
        "epoch": 5,
        "train_loss": 0.1,
        "monitor_value": 0.2,
        "monitor_metric": "val_loss",
        "val": {"auprc": auprc - 0.1, "auroc": auprc - 0.05},
        "test_primary": {
            "auprc": auprc,
            "f1": f1_score,
            "tau": 0.5,
            "precision": 0.6,
            "recall": 0.6,
            "balanced_accuracy": 0.6,
            "loss": 0.3,
        },
        "test_sensitivity": {
            "overall": {"auprc": auprc, "f1": f1_score, "tp": 2, "fp": 0, "tn": 2, "fn": 0}
        },
        "provenance": {
            "model": model,
            "subset_percent": percent,
            "train_seed": seed,
            "pack_seed": 42,
            "split": "test",
        },
        "thresholds": {"policy": "youden", "values": {"demo": 0.5}},
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_outputs_csv(outputs_path)


@pytest.mark.parametrize("bootstrap", [0])
def test_exp4_summary_pipeline(tmp_path: Path, bootstrap: int) -> None:
    runs_root = tmp_path
    specs = [
        ("sup_imnet", 50.0, 13, 0.55, 0.50),
        ("sup_imnet", 100.0, 13, 0.58, 0.52),
        ("ssl_imnet", 50.0, 13, 0.60, 0.56),
        ("ssl_imnet", 100.0, 13, 0.62, 0.58),
        ("ssl_colon", 50.0, 13, 0.70, 0.65),
        ("ssl_colon", 100.0, 13, 0.82, 0.72),
    ]
    for model, percent, seed, auprc, f1_score in specs:
        _write_run(runs_root, model=model, percent=percent, seed=seed, auprc=auprc, f1_score=f1_score)

    runs = exp4_report.discover_runs(runs_root)
    assert set(runs.keys()) == {"sup_imnet", "ssl_imnet", "ssl_colon"}
    colon_run = runs["ssl_colon"][50.0][13]
    assert colon_run.metrics["auprc"] == pytest.approx(0.70)

    summary = exp4_report.summarize_runs(runs, bootstrap=bootstrap, rng_seed=7)

    curves = summary["curves"]
    colon_50 = curves["auprc"]["ssl_colon"][50.0]["mean"]
    assert colon_50 == pytest.approx(0.70)

    slopes = summary["slopes"]["auprc"]["ssl_colon"]
    assert slopes["50→100"] == pytest.approx((0.82 - 0.70) / 50.0)

    targets = summary["targets"]
    assert targets["auprc"] == pytest.approx(0.62)
    assert targets["f1"] == pytest.approx(0.58)

    s_at_target = summary["s_at_target"]
    assert s_at_target["auprc"]["ssl_colon"] == pytest.approx(50.0)
    assert s_at_target["auprc"]["ssl_imnet"] == pytest.approx(100.0)

    pairwise_sup = summary["pairwise"]["auprc"]["sup_imnet"][50.0]
    assert pairwise_sup["delta"] == pytest.approx(0.70 - 0.55)
    assert pairwise_sup["replicates"] == []

    aulc_delta = summary["aulc_delta"]["auprc"]["sup_imnet"]
    assert aulc_delta["delta"] == pytest.approx(0.76 - 0.565, rel=1e-6)

    report = exp4_report.generate_report(runs_root, bootstrap=bootstrap, rng_seed=11)
    assert "Experiment 4 learning curve summary" in report
    assert "SSL-Colon" in report
    assert "ΔAULC" in report
