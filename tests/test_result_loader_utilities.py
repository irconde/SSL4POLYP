from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ssl4polyp.classification.analysis.result_loader import (
    CurveMetadata,
    LoadedResult,
    ResultLoader,
    compute_file_sha256,
)


def _write_curve(tmp_path: Path) -> Path:
    curve_path = tmp_path / "curve.csv"
    curve_path.write_text("x\n0.1\n0.2\n0.3\n", encoding="utf-8")
    return curve_path


def _make_payload(curve_path: Path) -> dict[str, object]:
    return {
        "seed": 13,
        "thresholds": {
            "primary": {"policy": "f1_opt_on_val"},
            "sensitivity": {"policy": "youden_on_val"},
        },
        "test": {
            "auprc": 0.75,
            "auroc": 0.81,
            "tau": 0.42,
            "tp": 40,
            "fp": 5,
            "tn": 85,
            "fn": 10,
            "n_pos": 50,
            "n_neg": 90,
        },
        "test_secondary": {
            "auprc": 0.71,
            "tp": 40,
            "fp": 5,
            "tn": 85,
            "fn": 10,
            "n_pos": 50,
            "n_neg": 90,
        },
        "provenance": {"eval_csv_sha256": "deadbeef"},
        "curve_exports": {
            "pr": {"path": curve_path.name, "sha256": compute_file_sha256(curve_path)},
        },
    }


def test_result_loader_extracts_metrics_and_curves(tmp_path: Path) -> None:
    curve_path = _write_curve(tmp_path)
    payload = _make_payload(curve_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = ResultLoader(require_sensitivity=True)
    result = loader.load(metrics_path)

    assert isinstance(result, LoadedResult)
    assert np.isclose(result.primary_metrics["auprc"], 0.75)
    assert np.isclose(result.primary_metrics["tau"], 0.42)
    assert np.isclose(result.sensitivity_metrics["auprc"], 0.71)
    assert "pr" in result.curves
    curve_meta = result.curves["pr"]
    assert isinstance(curve_meta, CurveMetadata)
    assert curve_meta.path == curve_path.resolve()
    assert curve_meta.sha256 == compute_file_sha256(curve_path)


def test_result_loader_normalises_legacy_blocks() -> None:
    loader = ResultLoader(require_sensitivity=True)
    payload = {
        "test": {"auroc": 0.5},
        "test_secondary": {"auroc": 0.4},
        "thresholds": {
            "primary": {"policy": "f1_opt_on_val"},
            "sensitivity": {"policy": "youden_on_val"},
        },
        "provenance": {"eval_csv_sha256": "abc"},
    }
    normalised = loader.normalise_payload(payload)
    assert "test_primary" in normalised and "test_sensitivity" in normalised
    assert normalised["test_primary"]["auroc"] == 0.5
    assert normalised["test_sensitivity"]["auroc"] == 0.4
