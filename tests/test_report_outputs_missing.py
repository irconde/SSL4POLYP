import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ssl4polyp.classification.analysis import (  # type: ignore[import]
    exp1_report,
    exp2_report,
    exp3_report,
    exp4_report,
    exp5a_report,
    exp5b_report,
)


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if value is None:
            target.pop(key, None)
            continue
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_update(dict(target[key]), value)
        else:
            target[key] = value
    return target


def _write_metrics(path: Path, *, overrides: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "seed": 13,
        "data": {
            "train": {"path": "sun_full/train.csv", "sha256": "train-digest"},
            "val": {"path": "sun_full/val.csv", "sha256": "val-digest"},
            "test": {"path": "sun_full/test.csv", "sha256": "test-digest"},
        },
        "val": {
            "loss": 0.4,
            "auroc": 0.8,
            "auprc": 0.7,
        },
        "test_primary": {
            "tau": 0.5,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "n_pos": 1,
            "n_neg": 1,
            "prevalence": 0.5,
        },
        "test_sensitivity": {
            "tau": 0.4,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "n_pos": 1,
            "n_neg": 1,
            "prevalence": 0.5,
        },
        "thresholds": {
            "primary": {
                "policy": "sun_val_frozen",
                "tau": 0.5,
                "source_split": "sun_full/val",
                "source_checkpoint": "checkpoint.pt",
            },
            "sensitivity": {
                "policy": "val_opt_youden",
                "tau": 0.4,
                "split": "sun_full/val.csv",
                "epoch": 3,
            },
        },
        "provenance": {
            "model": "ssl_colon",
            "train_seed": 13,
            "pack_seed": 29,
            "split": "test",
            "test_outputs_csv_sha256": "deadbeef",
        },
    }
    if overrides:
        payload = _deep_update(payload, overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve_exports = payload.get("curve_exports")
    if isinstance(curve_exports, dict):
        for entry in curve_exports.values():
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            rel_path = Path(raw_path.strip())
            curve_path = rel_path if rel_path.is_absolute() else path.parent / rel_path
            curve_path.parent.mkdir(parents=True, exist_ok=True)
            if not curve_path.exists():
                curve_path.write_text("threshold,precision,recall\n0.0,1.0,0.0\n1.0,0.0,1.0\n", encoding="utf-8")
            digest = hashlib.sha256(curve_path.read_bytes()).hexdigest()
            sha_value = entry.get("sha256")
            if not isinstance(sha_value, str) or not sha_value.strip():
                entry["sha256"] = digest
            else:
                entry["sha256"] = sha_value.strip().lower()
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "module, filename, overrides",
    (
        (
            exp1_report,
            "ssl_colon_s13_last.metrics.json",
            {
                "thresholds": {
                    "primary": {
                        "policy": "f1_opt_on_val",
                        "split": "sun_full/val.csv",
                        "epoch": 3,
                    },
                    "sensitivity": {
                        "policy": "youden_on_val",
                        "split": "sun_full/val.csv",
                        "epoch": 3,
                    },
                }
            },
        ),
        (
            exp2_report,
            "ssl_colon_s13_last.metrics.json",
            {
                "thresholds": {
                    "primary": {
                        "policy": "f1_opt_on_val",
                        "split": "sun_full/val.csv",
                        "epoch": 3,
                    },
                    "sensitivity": {
                        "policy": "youden_on_val",
                        "split": "sun_full/val.csv",
                        "epoch": 3,
                    },
                }
            },
        ),
        (
            exp3_report,
            "ssl_colon__sun_morphology_s13_last.metrics.json",
            {
                "data": {
                    "train": {"path": "sun_morphology/train.csv"},
                    "val": {"path": "sun_morphology/val.csv"},
                    "test": {"path": "sun_morphology/test.csv"},
                },
                "thresholds": {
                    "primary": {
                        "policy": "f1_opt_on_val",
                        "split": "sun_morphology/val.csv",
                        "epoch": 3,
                    },
                    "sensitivity": {
                        "policy": "youden_on_val",
                        "split": "sun_morphology/val.csv",
                        "epoch": 3,
                    },
                },
            },
        ),
        (
            exp4_report,
            "ssl_colon_p50_s13_last.metrics.json",
            {
                "provenance": {"subset_percent": 50.0},
                "thresholds": {
                    "primary": {
                        "policy": "f1_opt_on_val",
                        "split": "sun_full/val.csv",
                        "epoch": 3,
                    },
                    "sensitivity": {
                        "policy": "youden_on_val",
                        "split": "sun_full/val.csv",
                        "epoch": 3,
                    },
                },
                "curve_exports": {
                    "test": {
                        "path": "ssl_colon_p50_s13_last_test_curve.csv",
                    }
                },
            },
        ),
        (
            exp5a_report,
            "ssl_colon_seed13_last.metrics.json",
            {
                "run": {"exp": "exp5a"},
                "test_sensitivity": None,
                "thresholds": {"sensitivity": None},
            },
        ),
        (
            exp5b_report,
            "ssl_colon_s13_last.metrics.json",
            {
                "test_perturbations": {
                    "per_tag": {"clean": {"f1": 1.0}},
                    "per_case": {},
                },
                "test_sensitivity": None,
                "thresholds": {"sensitivity": None},
            },
        ),
    ),
)
def test_reports_error_when_outputs_missing(
    module: Any, filename: str, overrides: Dict[str, Any], tmp_path: Path
) -> None:
    runs_root = tmp_path / module.__name__.split(".")[-1]
    metrics_path = runs_root / filename
    _write_metrics(metrics_path, overrides=overrides)

    with pytest.raises(RuntimeError) as excinfo:
        module.discover_runs(runs_root)

    assert "missing per-frame outputs" in str(excinfo.value)
