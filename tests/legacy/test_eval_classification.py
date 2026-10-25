from types import SimpleNamespace

import pytest
import torch

from ssl4polyp.classification import eval_classification
from ssl4polyp.classification.metrics import performance


class ConstantMetric:
    def __call__(self, *args, **kwargs):
        return torch.tensor(0.0)


def _patch_constant_metrics(monkeypatch):
    def factory(*args, **kwargs):
        return ConstantMetric()

    for name in [
        "meanF1Score",
        "meanPrecision",
        "meanRecall",
        "meanAUROC",
        "meanBalancedAccuracy",
        "meanAUPRC",
    ]:
        monkeypatch.setattr(performance, name, factory)


class StaticModel(torch.nn.Module):
    def __init__(self, outputs: torch.Tensor):
        super().__init__()
        self.register_buffer("_outputs", outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple pass-through
        batch_size = inputs.shape[0]
        return self._outputs[:batch_size]


def _build_args(results_path: str, n_class: int) -> SimpleNamespace:
    return SimpleNamespace(
        ss_framework=None,
        arch="demo",
        pretraining="random",
        dataset="toy",
        results_file=results_path,
        n_class=n_class,
    )


def test_test_records_binary_prevalence(tmp_path, monkeypatch):
    _patch_constant_metrics(monkeypatch)
    outputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.2, 1.2],
            [0.3, 1.7],
            [1.1, 0.1],
            [0.4, 1.6],
        ]
    )
    model = StaticModel(outputs)
    targets = torch.tensor([0, 1, 1, 0, 1])
    data = torch.zeros((5, 3))
    metadata = [{"frame_id": f"frame-{i}"} for i in range(5)]
    test_loader = [(data, targets, metadata)]
    results_path = tmp_path / "binary_results.txt"
    args = _build_args(str(results_path), n_class=2)

    result = eval_classification.test(model, torch.device("cpu"), test_loader, args)

    assert result["metrics"]["prevalence"] == pytest.approx(0.6)
    log_text = results_path.read_text()
    assert "Prevalence: 0.6" in log_text


def test_test_records_multiclass_prevalence(tmp_path, monkeypatch):
    _patch_constant_metrics(monkeypatch)
    outputs = torch.tensor(
        [
            [2.0, 0.5, 0.3],
            [0.3, 1.5, 0.8],
            [0.2, 0.4, 2.1],
            [0.1, 3.2, 0.7],
        ]
    )
    model = StaticModel(outputs)
    targets = torch.tensor([0, 1, 2, 1])
    data = torch.zeros((4, 3))
    metadata = [{"frame_id": f"frame-{i}"} for i in range(4)]
    test_loader = [(data, targets, metadata)]
    results_path = tmp_path / "multiclass_results.txt"
    args = _build_args(str(results_path), n_class=3)

    result = eval_classification.test(model, torch.device("cpu"), test_loader, args)

    assert result["metrics"]["prevalence"] == pytest.approx([0.25, 0.5, 0.25])
    log_text = results_path.read_text()
    assert "Prevalence: [0.25, 0.5, 0.25]" in log_text
