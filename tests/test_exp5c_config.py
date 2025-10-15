from pathlib import Path

import yaml


def test_polypgen_fewshot_declares_val_split():
    cfg_path = Path("config/data/polypgen_fewshot.yaml")
    config = yaml.safe_load(cfg_path.read_text())
    splits = (config.get("dataset") or {}).get("splits") or {}
    assert splits.get("val") == "val"


def test_exp5c_schedule_unfreezes_head_plus_two():
    cfg_path = Path("config/exp/exp5c.yaml")
    config = yaml.safe_load(cfg_path.read_text())
    schedule = (config.get("protocol") or {}).get("finetune_schedule") or []
    assert schedule, "exp5c must declare a fine-tune schedule"
    assert schedule[0].get("mode") == "none"
    assert schedule[1].get("mode") == "head+2"
