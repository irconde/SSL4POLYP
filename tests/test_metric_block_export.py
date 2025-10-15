import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("distutils", reason="train_classification depends on distutils")

from ssl4polyp.classification import train_classification as tc


def test_build_metric_block_includes_primary_statistics():
    raw_metrics = {
        "auroc": 0.91,
        "auprc": 0.88,
        "recall": 0.72,
        "precision": 0.68,
        "f1": 0.7,
        "balanced_accuracy": 0.75,
        "mcc": 0.5,
        "loss": 0.34,
        "prevalence": 0.45,
        "count": 100,
        "threshold_metrics": {"tp": 45, "fp": 10, "tn": 35, "fn": 10},
        "class_counts": [35, 45],
        "tau": 0.6,
        "tau_info": "val_opt_youden",
    }

    exported = tc._prepare_metric_export(raw_metrics)
    block = tc._build_metric_block(exported)

    # Core scalar metrics should be preserved.
    assert block["auroc"] == pytest.approx(0.91)
    assert block["auprc"] == pytest.approx(0.88)
    assert block["loss"] == pytest.approx(0.34)
    assert block["recall"] == pytest.approx(0.72)
    assert block["precision"] == pytest.approx(0.68)
    assert block["f1"] == pytest.approx(0.7)
    assert block["balanced_accuracy"] == pytest.approx(0.75)
    assert block["mcc"] == pytest.approx(0.5)
    assert block["prevalence"] == pytest.approx(0.45)
    assert block["count"] == 100

    # Confusion counts and class statistics should be added.
    assert block["tp"] == 45
    assert block["fp"] == 10
    assert block["tn"] == 35
    assert block["fn"] == 10
    assert block["n_neg"] == 35
    assert block["n_pos"] == 45
    assert block["n_total"] == 80

    # Threshold metadata should accompany the block when available.
    assert block["tau"] == pytest.approx(0.6)
    assert block["tau_info"] == "val_opt_youden"
