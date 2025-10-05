import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from ssl4polyp.classification import finetune
from ssl4polyp.classification import train_classification as tc


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin_head = nn.Linear(4, 2)
        self.head = nn.Linear(2, 2)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(4)]
        )


def _group_lr(optimizer: torch.optim.Optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    raise KeyError(name)


def test_schedule_runtime_applies_stage_modes_and_lrs():
    schedule_cfg = [
        {"mode": "none", "epochs": 3, "head_lr": 1e-3},
        {
            "mode": "full",
            "epochs": 47,
            "head_lr": 5e-4,
            "backbone_lr": 2e-5,
        },
    ]
    spec = tc._sanitize_finetune_schedule_config(schedule_cfg, default_mode="full")
    stages = tc._materialize_finetune_schedule(spec, base_lr=5e-4)
    assert stages[0].start_epoch == 1 and stages[0].end_epoch == 3
    assert stages[1].start_epoch == 4 and stages[1].end_epoch == 50
    model = DummyModel()
    param_groups = finetune.collect_finetune_param_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": param_groups["head"], "lr": 1e-3, "name": "head"},
            {"params": param_groups["backbone"], "lr": 1e-3, "name": "backbone"},
        ]
    )
    runtime = tc.FinetuneScheduleRuntime(stages)

    stage1 = runtime.apply_if_needed(model, optimizer, epoch=1, rank=0)
    assert stage1 is not None and stage1.mode == "none"
    assert all(param.requires_grad for param in model.lin_head.parameters())
    assert all(param.requires_grad for param in model.head.parameters())
    backbone_params = [
        param
        for block in model.blocks
        for param in block.parameters()
    ]
    assert all(not param.requires_grad for param in backbone_params)
    assert pytest.approx(_group_lr(optimizer, "head"), rel=1e-6) == 1e-3
    assert pytest.approx(_group_lr(optimizer, "backbone"), rel=1e-6) == 0.0

    stage2 = runtime.apply_if_needed(model, optimizer, epoch=4, rank=0)
    assert stage2 is not None and stage2.mode == "full"
    for block in model.blocks:
        assert all(param.requires_grad for param in block.parameters())
    assert pytest.approx(_group_lr(optimizer, "head"), rel=1e-6) == 5e-4
    assert pytest.approx(_group_lr(optimizer, "backbone"), rel=1e-6) == 2e-5