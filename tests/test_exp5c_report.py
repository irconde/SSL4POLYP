import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ssl4polyp.classification.analysis import exp5c_report  # type: ignore[import]


def _write_metrics(path: Path) -> None:
    payload = {
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_runs_errors_when_outputs_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    metrics_path = runs_root / "ssl_colon_s13_last.metrics.json"
    _write_metrics(metrics_path)

    with pytest.raises(RuntimeError) as excinfo:
        exp5c_report.discover_runs(runs_root)

    assert "missing per-frame outputs" in str(excinfo.value)
