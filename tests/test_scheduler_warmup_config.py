from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("yaml")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "distutils" not in sys.modules:
    sys.modules["distutils"] = types.ModuleType("distutils")
if "distutils.version" not in sys.modules:
    sys.modules["distutils.version"] = types.ModuleType("distutils.version")

from ssl4polyp.classification import train_classification
from ssl4polyp.configs.layered import load_layered_config


def _baseline_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    args = train_classification.get_args()
    args.model_key = "sup_imnet"
    return args


def test_budget_scheduler_warmup_resolution(monkeypatch: pytest.MonkeyPatch):
    args = _baseline_args(monkeypatch)

    experiment_cfg = load_layered_config("exp/exp5c/budgets/s50.yaml")
    scheduler_cfg = experiment_cfg.get("scheduler") or {}
    assert scheduler_cfg.get("warmup_epochs") == 6

    train_classification.apply_experiment_config(
        args,
        experiment_cfg,
        resolved_overrides=None,
    )

    assert args.warmup_epochs == 6


def test_scheduler_warmup_falls_back_when_missing(monkeypatch: pytest.MonkeyPatch):
    args = _baseline_args(monkeypatch)
    args.warmup_epochs = 3

    experiment_cfg = load_layered_config("exp/exp5c/budgets/s50.yaml")
    stripped_cfg = copy.deepcopy(experiment_cfg)
    scheduler_cfg = stripped_cfg.get("scheduler")
    if isinstance(scheduler_cfg, dict):
        scheduler_cfg.pop("warmup_epochs", None)

    train_classification.apply_experiment_config(
        args,
        stripped_cfg,
        resolved_overrides=None,
    )

    assert args.warmup_epochs == 3
