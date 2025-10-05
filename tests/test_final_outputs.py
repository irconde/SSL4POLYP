import csv
from pathlib import Path
from typing import Optional

import pytest
import torch

from ssl4polyp.classification import train_classification as tc
from ssl4polyp.classification.metrics import performance


class StaticModel(torch.nn.Module):
    def __init__(self, outputs: torch.Tensor):
        super().__init__()
        self.register_buffer("_outputs", outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - deterministic mapping
        batch_size = inputs.shape[0]
        return self._outputs[:batch_size]


@pytest.mark.parametrize("tau", [0.5])
def test_test_writes_per_frame_outputs(tmp_path: Path, tau: float):
    logits = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    metadata = [
        {"frame_id": f"frame-{idx}", "origin": "polypgen", "case_id": f"case-{idx // 2}"}
        for idx in range(len(targets))
    ]
    inputs = torch.zeros((len(targets), 3, 8, 8), dtype=torch.float32)
    test_loader = [(inputs, targets, metadata)]
    model = StaticModel(logits)
    log_path = tmp_path / "eval.log"
    outputs_path = tmp_path / "outputs.csv"

    result = tc.test(
        model,
        rank=0,
        test_loader=test_loader,
        epoch=1,
        perf_fn=performance.meanAUROC(n_class=2),
        log_path=str(log_path),
        metric_fns={},
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        loss_mode="binary_bce",
        split_name="Test",
        tau=tau,
        tau_info="fixed",
        save_outputs_path=outputs_path,
    )

    assert outputs_path.exists(), "per-frame outputs should be written"
    with outputs_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == len(targets)
    first_row = rows[0]
    assert first_row["frame_id"] == "frame-0"
    assert first_row["origin"] == "polypgen"
    assert first_row["sequence_id"] == "case-0"
    assert int(first_row["label"]) == 1
    assert int(first_row["pred"]) == 1
    assert pytest.approx(float(first_row["prob"]), rel=1e-6) == torch.sigmoid(torch.tensor(1.0)).item()

    threshold_metrics = result.get("threshold_metrics")
    assert threshold_metrics is not None
    assert threshold_metrics["tp"] == 2
    assert threshold_metrics["tn"] == 2
    assert threshold_metrics["fp"] == 0
    assert threshold_metrics["fn"] == 0
    assert threshold_metrics["prevalence"] == pytest.approx(0.5)
    assert threshold_metrics["mcc"] == pytest.approx(1.0)
    assert result["prevalence"] == pytest.approx(0.5)
    assert result["mcc"] == pytest.approx(1.0)

    log_text = log_path.read_text()
    assert "confusion @τ" in log_text


def test_curve_exports_write_csvs(tmp_path: Path):
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.6, 0.4],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    ckpt_stem = tmp_path / "demo_model"

    exports = tc._export_curve_sets(
        ckpt_stem,
        "Test",
        probabilities=probabilities,
        targets=targets,
        grid_points=5,
    )

    roc_path = exports["roc_csv"]
    pr_path = exports["pr_csv"]

    assert roc_path.exists()
    assert pr_path.exists()
    assert exports["grid_points"] == 5

    def _read_csv(path: Path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    roc_rows = _read_csv(roc_path)
    pr_rows = _read_csv(pr_path)

    assert len(roc_rows) == 5
    assert len(pr_rows) == 5

    def as_float(value: str) -> Optional[float]:
        return None if value == "" else float(value)

    first_roc = roc_rows[0]
    assert as_float(first_roc["threshold"]) == pytest.approx(0.0)
    assert as_float(first_roc["tpr"]) == pytest.approx(1.0)
    assert as_float(first_roc["fpr"]) == pytest.approx(1.0)

    last_roc = roc_rows[-1]
    assert as_float(last_roc["threshold"]) == pytest.approx(1.0)
    assert as_float(last_roc["tpr"]) == pytest.approx(0.0)
    assert as_float(last_roc["fpr"]) == pytest.approx(0.0)

    last_pr = pr_rows[-1]
    assert as_float(last_pr["threshold"]) == pytest.approx(1.0)
    assert last_pr["precision"] == ""