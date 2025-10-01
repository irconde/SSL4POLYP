import math
from typing import Iterable, List, Sequence, Tuple

import pytest

pytest.importorskip("torch")

import torch

from ssl4polyp.classification import train_classification
from ssl4polyp.classification.metrics import performance

Batch = Tuple[torch.Tensor, torch.Tensor, dict]


class DummyDataset:
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self._size


class SyntheticLoader:
    def __init__(self, batches: Sequence[Batch]) -> None:
        self._batches: List[Batch] = list(batches)
        total = sum(batch[0].shape[0] for batch in self._batches)
        self.dataset = DummyDataset(total)

    def __iter__(self) -> Iterable[Batch]:
        for batch in self._batches:
            yield batch

    def __len__(self) -> int:
        return len(self._batches)


class SequentialOutputsModel(torch.nn.Module):
    def __init__(self, outputs: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_outputs", outputs)
        self._cursor = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial mapping
        batch_size = inputs.shape[0]
        start = self._cursor
        end = start + batch_size
        self._cursor = end
        return self._outputs[start:end]


def test_test_overruns_to_capture_missing_labels(tmp_path, capfd):
    outputs = torch.tensor(
        [
            [5.0, -5.0],
            [4.0, -4.0],
            [-4.0, 4.0],
            [-5.0, 5.0],
        ]
    )
    targets = [
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    ]
    data = [torch.zeros((1, 3, 4, 4)) for _ in range(len(targets))]
    metadata = [{} for _ in range(len(targets))]
    loader = SyntheticLoader(
        [
            (data[i], targets[i], metadata[i])
            for i in range(len(targets))
        ]
    )

    model = SequentialOutputsModel(outputs)
    perf_fn = performance.meanAUROC(n_class=2)
    log_path = tmp_path / "eval.log"

    results = train_classification.test(
        model,
        rank=0,
        test_loader=loader,
        epoch=1,
        perf_fn=perf_fn,
        log_path=str(log_path),
        metric_fns={},
        split_name="Synthetic",
        return_outputs=False,
        tau=None,
        max_batches=2,
    )

    captured = capfd.readouterr()
    assert "consumed extra batches" in captured.out
    assert math.isfinite(results["auroc"])
    assert model._cursor == outputs.size(0)
    assert model._cursor > 2  # exceeded ``max_batches`` of 2 samples
    assert log_path.read_text().strip()
