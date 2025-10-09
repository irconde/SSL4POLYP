from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from ssl4polyp.classification.analysis import exp4_report
from ssl4polyp.classification.analysis.exp4_report import EXPECTED_SEEDS  # type: ignore[import]


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
    csv_digest: str = "deadbeefcafebabe",
) -> None:
    base = f"{model}_p{int(percent):02d}_s{seed}"
    metrics_path = root / f"{base}_last.metrics.json"
    outputs_primary = root / f"{base}_test_outputs.csv"
    outputs_with_suffix = root / f"{base}.metrics_test_outputs.csv"
    outputs_expected = metrics_path.with_name(f"{metrics_path.stem}_test_outputs.csv")
    curve_dir = root / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_path = curve_dir / f"{base}_roc_curve.csv"
    curve_path.write_text("threshold,false_positive_rate,true_positive_rate\n0.0,0.0,0.0\n1.0,1.0,1.0\n", encoding="utf-8")
    curve_digest = hashlib.sha256(curve_path.read_bytes()).hexdigest()
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
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "n_pos": 2,
            "n_neg": 2,
        },
        "test_sensitivity": {
            "auprc": auprc,
            "f1": f1_score,
            "tp": 2,
            "fp": 0,
            "tn": 2,
            "fn": 0,
            "recall": 0.6,
            "precision": 0.6,
            "balanced_accuracy": 0.6,
            "tau": 0.4,
            "loss": 0.32,
            "n_pos": 2,
            "n_neg": 2,
        },
        "provenance": {
            "model": model,
            "subset_percent": percent,
            "train_seed": seed,
            "pack_seed": 42,
            "split": "test",
            "sun_full_test_csv_sha256": csv_digest,
            "splits": {
                "sun_full_test": {
                    "csv_sha256": csv_digest,
                }
            },
        },
        "thresholds": {
            "policy": "f1_opt_on_val",
            "values": {"demo": 0.5},
            "primary": {
                "policy": "f1_opt_on_val",
                "tau": 0.5,
                "split": "sun_full/val",
            },
            "sensitivity": {
                "policy": "youden_on_val",
                "tau": 0.4,
            },
        },
        "curve_exports": {
            "test": {
                "path": curve_path.relative_to(metrics_path.parent).as_posix(),
                "sha256": curve_digest,
            }
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for path in {outputs_primary, outputs_with_suffix, outputs_expected}:
        _write_outputs_csv(path)


@pytest.mark.parametrize("bootstrap", [0])
def test_exp4_summary_pipeline(tmp_path: Path, bootstrap: int) -> None:
    runs_root = tmp_path
    base_specs = [
        ("sup_imnet", 50.0, 0.55, 0.50),
        ("sup_imnet", 100.0, 0.58, 0.52),
        ("ssl_imnet", 50.0, 0.60, 0.56),
        ("ssl_imnet", 100.0, 0.62, 0.58),
        ("ssl_colon", 50.0, 0.70, 0.65),
        ("ssl_colon", 100.0, 0.82, 0.72),
    ]
    for seed in EXPECTED_SEEDS:
        for model, percent, auprc, f1_score in base_specs:
            _write_run(runs_root, model=model, percent=percent, seed=seed, auprc=auprc, f1_score=f1_score)

    runs = exp4_report.discover_runs(runs_root)
    assert set(runs.keys()) == {"sup_imnet", "ssl_imnet", "ssl_colon"}
    colon_run = runs["ssl_colon"][50.0][13]
    assert colon_run.metrics["auprc"] == pytest.approx(0.70)

    summary = exp4_report.summarize_runs(runs, bootstrap=bootstrap, rng_seed=7)

    composition = summary["test_composition"]
    assert composition["n_pos"] == pytest.approx(2)
    assert composition["n_neg"] == pytest.approx(2)
    assert composition["prevalence"] == pytest.approx(0.5)

    primary = summary["primary"]
    primary_table = primary["learning_table"]
    colon_entry = next(
        entry
        for entry in primary_table
        if entry["model"] == "ssl_colon" and entry["percent"] == 50.0 and entry["metric"] == "auprc"
    )
    assert colon_entry["mean"] == pytest.approx(0.70)

    curves = primary["curves"]
    colon_50 = curves["auprc"]["ssl_colon"][50.0]["mean"]
    assert colon_50 == pytest.approx(0.70)

    slopes = primary["slopes"]["auprc"]["ssl_colon"]
    assert slopes["50→100"] == pytest.approx((0.82 - 0.70) / 50.0)

    targets = primary["targets"]
    assert targets["auprc"] == pytest.approx(0.62)
    assert targets["f1"] == pytest.approx(0.58)

    s_at_target = primary["s_at_target"]
    assert s_at_target["auprc"]["ssl_colon"] == pytest.approx(50.0)
    assert s_at_target["auprc"]["ssl_imnet"] == pytest.approx(100.0)

    assert summary["validated_seeds"] == list(EXPECTED_SEEDS)
    pairwise_sup = primary["pairwise"]["auprc"]["sup_imnet"][50.0]
    assert pairwise_sup["delta"] == pytest.approx(0.70 - 0.55)
    assert pairwise_sup["replicates"] == []

    aulc_delta = primary["aulc_delta"]["auprc"]["sup_imnet"]
    assert aulc_delta["delta"] == pytest.approx(0.76 - 0.565, rel=1e-6)

    report = exp4_report.generate_report(runs_root, bootstrap=bootstrap, rng_seed=11)
    assert "Experiment 4 — Label-efficiency on SUN" in report
    assert "## T1 — SUN-test composition" in report
    assert "### ΔAULC vs baselines" in report


def test_exp4_csv_digest_guardrail(tmp_path: Path) -> None:
    _write_run(tmp_path, model="sup_imnet", percent=50.0, seed=13, auprc=0.55, f1_score=0.5, csv_digest="aaaabbbb")
    _write_run(tmp_path, model="ssl_colon", percent=50.0, seed=29, auprc=0.70, f1_score=0.65, csv_digest="ccccdddd")
    with pytest.raises(RuntimeError):
        exp4_report.discover_runs(tmp_path)
