import pytest

pytest.importorskip("torch")

import torch.nn as nn

from ssl4polyp.classification.finetune import configure_finetune_parameters


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class DummyModel(nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(DummyBlock() for _ in range(depth))
        self.lin_head = nn.Linear(4, 3)
        self.frozen = False


def _trainable_parameter_names(model: nn.Module) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def _head_parameter_names(model: DummyModel) -> set[str]:
    return {f"lin_head.{name}" for name, _ in model.lin_head.named_parameters()}


def test_configure_finetune_mode_full_enables_all_params():
    model = DummyModel(depth=4)
    configure_finetune_parameters(model, "full")

    assert all(param.requires_grad for param in model.parameters())
    assert model.frozen is False


def test_configure_finetune_mode_none_only_trains_head():
    model = DummyModel(depth=4)
    configure_finetune_parameters(model, "none")

    trainable = _trainable_parameter_names(model)
    assert trainable == _head_parameter_names(model)
    assert model.frozen is True


def test_configure_finetune_mode_head_plus_two_keeps_final_blocks():
    depth = 6
    model = DummyModel(depth=depth)
    configure_finetune_parameters(model, "head+2")

    trainable = _trainable_parameter_names(model)
    expected = set(_head_parameter_names(model))
    for idx in range(depth - 2, depth):
        block = model.blocks[idx]
        for name, _ in block.named_parameters():
            expected.add(f"blocks.{idx}.{name}")

    assert trainable == expected
    assert model.frozen is False


def test_configure_finetune_mode_head_plus_one_keeps_last_block():
    depth = 5
    model = DummyModel(depth=depth)
    configure_finetune_parameters(model, "head+1")

    trainable = _trainable_parameter_names(model)
    expected = set(_head_parameter_names(model))
    idx = depth - 1
    block = model.blocks[idx]
    for name, _ in block.named_parameters():
        expected.add(f"blocks.{idx}.{name}")

    assert trainable == expected
    assert model.frozen is False
