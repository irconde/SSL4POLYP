from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


def test_polypgen_fewshot_declares_val_split():
    cfg_path = Path("config/data/polypgen_fewshot.yaml")
    config = yaml.safe_load(cfg_path.read_text())
    splits = (config.get("dataset") or {}).get("splits") or {}
    assert splits.get("val") == "val"


def test_exp5c_budgets_define_expected_schedule_modes():
    expected_modes = {
        "s50": "head+1",
        "s100": "head+2",
        "s200": "head+2",
        "s500": "head+2",
    }
    for budget, mode in expected_modes.items():
        cfg_path = Path("config/exp/exp5c/budgets") / f"{budget}.yaml"
        config = yaml.safe_load(cfg_path.read_text())
        schedule = (config.get("protocol") or {}).get("finetune_schedule") or []
        assert schedule, f"exp5c budget {budget} must declare a fine-tune schedule"
        assert schedule[0].get("mode") == "none"
        assert schedule[1].get("mode") == mode
