from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("numpy")

from ssl4polyp.classification.analysis import exp1_report


def _write_outputs(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "case_id", "prob", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _metrics_payload(*, seed: int, tau_primary: float, tau_sensitivity: float) -> dict[str, object]:
    val_path = "sun_full/val.csv"
    return {
        "seed": seed,
        "data": {
            "train": {"path": "sun_full/train.csv", "sha256": "train-digest"},
            "val": {"path": val_path, "sha256": "val-digest"},
            "test": {"path": "sun_full/test.csv", "sha256": "test-digest"},
        },
        "val": {"loss": 0.2, "auroc": 0.9, "auprc": 0.8},
        "test_primary": {
            "tau": tau_primary,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "n_pos": 1,
            "n_neg": 1,
            "prevalence": 0.5,
            "loss": 0.1,
            "auprc": 0.7,
            "auroc": 0.8,
        },
        "test_sensitivity": {
            "tau": tau_sensitivity,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "n_pos": 1,
            "n_neg": 1,
            "prevalence": 0.5,
            "loss": 0.11,
            "auprc": 0.69,
            "auroc": 0.79,
        },
        "thresholds": {
            "primary": {
                "policy": "f1_opt_on_val",
                "tau": tau_primary,
                "split": val_path,
                "epoch": 3,
            },
            "sensitivity": {
                "policy": "youden_on_val",
                "tau": tau_sensitivity,
                "split": val_path,
                "epoch": 3,
            },
        },
        "provenance": {
            "model": "ssl_imnet",
            "test_outputs_csv_sha256": "deadbeef",
        },
    }


def _write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(*, seed: int, primary: float, baseline: bool = False) -> exp1_report.Exp1Run:
    return exp1_report.Exp1Run(
        model="ssl_imnet" if not baseline else "sup_imnet",
        seed=seed,
        primary_metrics={"f1": primary},
        sensitivity_metrics={"f1": primary},
        tau_primary=0.5,
        tau_sensitivity=0.4,
        frames=tuple(),
        cases={},
        metrics_path=Path(f"seed{seed}.metrics.json"),
        curves={},
        provenance={},
    )


def test_compute_delta_summaries_reports_sample_std() -> None:
    treatment_runs = {
        13: _make_run(seed=13, primary=0.8),
        29: _make_run(seed=29, primary=0.7),
    }
    baseline_runs = {
        13: _make_run(seed=13, primary=0.5, baseline=True),
        29: _make_run(seed=29, primary=0.6, baseline=True),
    }

    summaries = exp1_report._compute_delta_summaries(
        treatment_runs,
        baseline_runs,
        metrics=("f1",),
        bootstrap=0,
        rng_seed=None,
        block="primary",
    )

    summary = summaries.get("f1")
    assert summary is not None
    assert math.isfinite(summary.std)
    expected_std = math.sqrt(0.02)
    assert summary.std == pytest.approx(expected_std)


def test_discover_runs_supports_last_metrics_outputs(tmp_path: Path) -> None:
    runs_root = tmp_path / "exp1"
    metrics_path = runs_root / "ssl_imnet_s13_last.metrics.json"
    payload = _metrics_payload(seed=13, tau_primary=0.5, tau_sensitivity=0.4)
    _write_metrics(metrics_path, payload)
    outputs_path = runs_root / "ssl_imnet_s13_last_test_outputs.csv"
    rows = [
        {"frame_id": "f1", "case_id": "case1", "prob": 0.9, "label": 1},
        {"frame_id": "f2", "case_id": "case2", "prob": 0.1, "label": 0},
    ]
    _write_outputs(outputs_path, rows)

    runs = exp1_report.discover_runs(runs_root)

    assert "ssl_imnet" in runs
    run = runs["ssl_imnet"].get(13)
    assert run is not None
    assert run.metrics_path == metrics_path
    assert [frame.frame_id for frame in run.frames] == ["f1", "f2"]
