from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ssl4polyp.classification.train_classification import (
    ParentRunReference,
    _build_metrics_provenance,
    _compute_file_sha256,
    _resolve_parent_reference,
)


@pytest.fixture()
def parent_artifacts(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "sun_baselines" / "exp_demo"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model_demo.pth"
    checkpoint_path.write_bytes(b"parent-weights")

    metrics_payload = {
        "seed": 42,
        "data": {
            "train": {"path": "sun_full/train.csv", "sha256": "train-digest"},
            "val": {"path": "sun_full/val.csv", "sha256": "val-digest"},
            "test": {"path": "sun_full/test.csv", "sha256": "test-digest"},
        },
        "val": {"auroc": 0.88},
        "test_primary": {"auroc": 0.91, "auprc": 0.87},
        "threshold_policy": "youden",
    }
    metrics_path = checkpoint_dir / "model_demo.metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    outputs_path = checkpoint_dir / "model_demo_test_outputs.csv"
    outputs_path.write_text("case_id,probability,target,pred\ncase-1,0.9,1,1\n", encoding="utf-8")

    return checkpoint_path


def test_resolve_parent_reference_includes_hashes(parent_artifacts: Path) -> None:
    reference = _resolve_parent_reference(parent_artifacts)
    assert isinstance(reference, ParentRunReference)
    assert reference.checkpoint_path == parent_artifacts
    assert reference.checkpoint_sha256 == _compute_file_sha256(parent_artifacts)
    assert reference.metrics_path is not None
    assert reference.metrics_payload is not None
    assert reference.metrics_payload["test_primary"]["auroc"] == pytest.approx(0.91)
    assert reference.metrics_sha256 == _compute_file_sha256(reference.metrics_path)
    assert reference.outputs_path is not None
    assert reference.outputs_sha256 == _compute_file_sha256(reference.outputs_path)


def test_provenance_includes_parent_block(tmp_path: Path, parent_artifacts: Path) -> None:
    output_dir = tmp_path / "outputs" / "exp5a_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _resolve_parent_reference(parent_artifacts)

    args = SimpleNamespace(
        model_key="demo",
        model_tag="demo",
        run_stem="demo__PolypGen",
        arch="vit",
        output_dir=str(output_dir),
        active_seed=7,
        dataset_layout={},
        dataset_percent=None,
        dataset_seed=None,
        test_split="test",
        parent_reference=reference,
    )

    provenance = _build_metrics_provenance(args)
    assert "parent_run" in provenance
    parent_block = provenance["parent_run"]
    assert parent_block["checkpoint"].endswith("model_demo.pth")
    assert parent_block["checkpoint_sha256"] == reference.checkpoint_sha256

    metrics_info = parent_block.get("metrics")
    assert isinstance(metrics_info, dict)
    assert metrics_info["path"].endswith("model_demo.metrics.json")
    assert metrics_info["sha256"] == reference.metrics_sha256
    assert metrics_info["payload"]["test_primary"]["auroc"] == pytest.approx(0.91)

    outputs_info = parent_block.get("outputs")
    assert isinstance(outputs_info, dict)
    assert outputs_info["path"].endswith("model_demo_test_outputs.csv")
    assert outputs_info["sha256"] == reference.outputs_sha256
