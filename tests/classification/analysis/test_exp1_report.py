from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

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


def _make_run(
    *,
    seed: int,
    pos_probs: Sequence[float],
    neg_probs: Sequence[float],
    baseline: bool = False,
    tau_primary: float = 0.5,
    tau_sensitivity: float = 0.4,
) -> exp1_report.Exp1Run:
    frames: list[exp1_report.EvalFrame] = []
    cases: dict[str, list[exp1_report.EvalFrame]] = {}

    for index, prob in enumerate(pos_probs):
        case_id = f"case_pos_{index}"
        frame = exp1_report.EvalFrame(
            frame_id=f"frame_pos_{seed}_{index}",
            case_id=case_id,
            prob=float(prob),
            label=1,
        )
        frames.append(frame)
        cases.setdefault(case_id, []).append(frame)

    for index, prob in enumerate(neg_probs):
        case_id = f"case_neg_{index}"
        frame = exp1_report.EvalFrame(
            frame_id=f"frame_neg_{seed}_{index}",
            case_id=case_id,
            prob=float(prob),
            label=0,
        )
        frames.append(frame)
        cases.setdefault(case_id, []).append(frame)

    frame_tuple = tuple(frames)
    case_map = {case: tuple(items) for case, items in cases.items()}
    primary_metrics = exp1_report._metrics_from_frames(frame_tuple, tau_primary)
    sensitivity_metrics = exp1_report._metrics_from_frames(frame_tuple, tau_sensitivity)

    run = exp1_report.Exp1Run(
        model="ssl_imnet" if not baseline else "sup_imnet",
        seed=seed,
        primary_metrics=dict(primary_metrics),
        sensitivity_metrics=dict(sensitivity_metrics),
        tau_primary=float(tau_primary),
        tau_sensitivity=float(tau_sensitivity),
        frames=frame_tuple,
        cases=case_map,
        metrics_path=Path(f"seed{seed}.metrics.json"),
        curves={},
        provenance={},
    )
    return run


def test_compute_delta_summaries_reports_sample_std() -> None:
    treatment_runs = {
        13: _make_run(seed=13, pos_probs=[0.9], neg_probs=[0.1]),
        29: _make_run(seed=29, pos_probs=[0.4], neg_probs=[0.6]),
    }
    baseline_runs = {
        13: _make_run(seed=13, pos_probs=[0.4], neg_probs=[0.6], baseline=True),
        29: _make_run(seed=29, pos_probs=[0.4], neg_probs=[0.6], baseline=True),
    }

    summaries = exp1_report._compute_delta_summaries(
        treatment_runs,
        baseline_runs,
        metrics=("recall",),
        bootstrap=0,
        rng_seed=None,
        block="primary",
    )

    summary = summaries.get("recall")
    assert summary is not None
    assert math.isfinite(summary.std)
    expected_std = math.sqrt(0.5)
    assert summary.std == pytest.approx(expected_std)


def test_compute_delta_summaries_ignores_serialised_metrics() -> None:
    treatment_runs = {
        13: _make_run(seed=13, pos_probs=[0.9], neg_probs=[0.1]),
        29: _make_run(seed=29, pos_probs=[0.9], neg_probs=[0.1]),
    }
    baseline_runs = {
        13: _make_run(seed=13, pos_probs=[0.4], neg_probs=[0.6], baseline=True),
        29: _make_run(seed=29, pos_probs=[0.4], neg_probs=[0.6], baseline=True),
    }

    # Corrupt the stored metrics to ensure the recomputed deltas drive the summary.
    for run in treatment_runs.values():
        run.primary_metrics["recall"] = 0.0
    for run in baseline_runs.values():
        run.primary_metrics["recall"] = 1.0

    summaries = exp1_report._compute_delta_summaries(
        treatment_runs,
        baseline_runs,
        metrics=("recall",),
        bootstrap=0,
        rng_seed=None,
        block="primary",
    )

    summary = summaries.get("recall")
    assert summary is not None
    # All treatment runs achieve recall 1 while baselines sit at 0.
    assert summary.mean == pytest.approx(1.0)


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
