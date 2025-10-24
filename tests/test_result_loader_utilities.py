from __future__ import annotations

# pyright: reportMissingImports=false

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest  # type: ignore[import-not-found]

from ssl4polyp.classification.analysis.result_loader import (  # type: ignore[import-not-found]
    CurveMetadata,
    GuardrailViolation,
    LoadedResult,
    ResultLoader,
    compute_file_sha256,
)


def _write_curve(tmp_path: Path) -> Path:
    curve_path = tmp_path / "curve.csv"
    curve_path.write_text("x\n0.1\n0.2\n0.3\n", encoding="utf-8")
    return curve_path


def _make_payload(curve_path: Path) -> Dict[str, Any]:
    val_path = "sun_full/val.csv"
    curve_digest = compute_file_sha256(curve_path)
    return {
        "seed": 13,
        "data": {
            "train": {"path": "sun_full/train.csv", "sha256": "train-digest"},
            "val": {"path": val_path, "sha256": "val-digest"},
            "test": {"path": "sun_full/test.csv", "sha256": "test-digest"},
        },
        "thresholds": {
            "primary": {
                "policy": "f1_opt_on_val",
                "tau": 0.42,
                "split": val_path,
                "epoch": 7,
            },
            "sensitivity": {
                "policy": "youden_on_val",
                "tau": 0.38,
                "split": val_path,
                "epoch": 7,
            },
        },
        "test_primary": {
            "auprc": 0.75,
            "auroc": 0.81,
            "tau": 0.42,
            "tp": 40,
            "fp": 5,
            "tn": 85,
            "fn": 10,
            "n_pos": 50,
            "n_neg": 90,
            "prevalence": float(50) / float(140),
        },
        "test_sensitivity": {
            "auprc": 0.71,
            "tau": 0.38,
            "tp": 40,
            "fp": 5,
            "tn": 85,
            "fn": 10,
            "n_pos": 50,
            "n_neg": 90,
            "prevalence": float(50) / float(140),
        },
        "curve_exports": {
            "pr": {"path": curve_path.name, "sha256": curve_digest},
        },
    }


def test_result_loader_extracts_metrics_and_curves(tmp_path: Path) -> None:
    curve_path = _write_curve(tmp_path)
    payload = _make_payload(curve_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = ResultLoader(exp_id="exp1", required_curve_keys=("pr",))
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


def test_result_loader_allows_missing_curves_when_not_enforced(tmp_path: Path) -> None:
    curve_path = _write_curve(tmp_path)
    payload = _make_payload(curve_path)
    payload.pop("curve_exports", None)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = ResultLoader(
        exp_id="exp1",
        required_curve_keys=("pr",),
        enforce_curve_exports=False,
    )
    result = loader.load(metrics_path)

    assert isinstance(result, LoadedResult)
    assert result.curves == {}


def test_result_loader_ignores_debug_sections(tmp_path: Path) -> None:
    curve_path = _write_curve(tmp_path)
    payload = _make_payload(curve_path)
    payload.setdefault("debug", {})["notes"] = {"info": "legacy"}
    payload["thresholds"]["primary"]["debug"] = {
        "val_snapshot_prev": {
            "epoch": 3,
            "tau_used": 0.37,
            "averaging": "macro",
            "metrics": {"recall_macro": 0.7},
        }
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = ResultLoader(exp_id="exp1")
    result = loader.load(metrics_path)

    assert "debug" not in result.payload
    primary_block = result.payload["thresholds"]["primary"]
    assert "debug" not in primary_block


def test_result_loader_normalises_legacy_blocks() -> None:
    loader = ResultLoader(exp_id="exp1")
    payload = {
        "data": {
            "train": {"path": "sun_full/train.csv", "sha256": "train-digest"},
            "val": {"path": "sun_full/val.csv", "sha256": "val-digest"},
            "test": {"path": "sun_full/test.csv", "sha256": "test-digest"},
        },
        "thresholds": {
            "primary": {
                "policy": "f1_opt_on_val",
                "tau": 0.5,
                "split": "sun_full/val.csv",
                "epoch": 1,
            },
            "sensitivity": {
                "policy": "youden_on_val",
                "tau": 0.5,
                "split": "sun_full/val.csv",
                "epoch": 1,
            },
        },
        "test": {"auroc": 0.5},
        "test_secondary": {"auroc": 0.4},
    }
    with pytest.raises(GuardrailViolation):
        loader.validate(Path("metrics.json"), payload)
