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
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "module, filename, overrides",
    (
        (exp1_report, "ssl_colon_s13_last.metrics.json", {}),
        (exp2_report, "ssl_colon_s13_last.metrics.json", {}),
        (exp3_report, "ssl_colon__sun_morphology_s13_last.metrics.json", {}),
        (
            exp4_report,
            "ssl_colon_p50_s13_last.metrics.json",
            {"provenance": {"subset_percent": 50.0}},
        ),
        (exp5a_report, "ssl_colon_seed13_last.metrics.json", {"run": {"exp": "exp5a"}}),
        (
            exp5b_report,
            "ssl_colon_s13_last.metrics.json",
            {
                "test_perturbations": {
                    "per_tag": {"clean": {"f1": 1.0}},
                    "per_case": {},
                }
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
