from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from ssl4polyp.classification.train_classification import (
    TensorboardLogger,
    train_epoch,
)


class DummyScaler:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:  # pragma: no cover - simple stub
        pass


class FailingWriter:
    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def add_scalar(self, *args, **kwargs) -> None:
        self.calls += 1
        raise OSError("disk full")

    def close(self) -> None:
        self.closed = True


def _make_loader() -> DataLoader:
    features = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=2)


def test_train_epoch_disables_tensorboard_on_failure(tmp_path) -> None:
    model = torch.nn.Linear(8, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = DummyScaler()
    loader = _make_loader()
    failing_writer = FailingWriter()
    tb_logger = TensorboardLogger(failing_writer)

    with pytest.warns(RuntimeWarning, match="TensorBoard logging"):
        loss, global_step = train_epoch(
            model,
            rank=0,
            world_size=1,
            train_loader=loader,
            train_sampler=None,
            optimizer=optimizer,
            epoch=1,
            loss_fn=loss_fn,
            log_path=str(tmp_path / "log.txt"),
            scaler=scaler,
            use_amp=False,
            tb_logger=tb_logger,
            log_interval=1,
            global_step=0,
            seed=0,
            device=torch.device("cpu"),
            distributed=False,
        )

    assert np.isfinite(loss)
    assert global_step == len(loader)
    assert failing_writer.closed
    assert not tb_logger
