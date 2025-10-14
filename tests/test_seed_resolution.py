import sys
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "distutils" not in sys.modules:
    sys.modules["distutils"] = types.ModuleType("distutils")
if "distutils.version" not in sys.modules:
    sys.modules["distutils.version"] = types.ModuleType("distutils.version")

if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")

    def _safe_load_stub(*args, **kwargs):
        return {}

    def _safe_dump_stub(*args, **kwargs):
        stream = args[1] if len(args) > 1 else kwargs.get("stream")
        if stream is not None:
            try:
                stream.write("")
            except AttributeError:
                pass

    yaml_module.safe_load = _safe_load_stub
    yaml_module.safe_dump = _safe_dump_stub
    sys.modules["yaml"] = yaml_module

pytest.importorskip("torch")

from ssl4polyp.classification import train_classification


def test_cli_seed_precedence_over_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest", "--seed", "29"])
    args = train_classification.get_args()
    assert getattr(args, "_seed_explicit", False) is True

    experiment_cfg = {
        "dataset": {
            "name": "sun_full",
            "pack": "sun_train",
            "splits": {"train": "train", "val": "val", "test": "test"},
        },
        "models": [
            {
                "key": "sup_model",
                "arch": "vit_base",
                "pretraining": "sup",
            }
        ],
        "seeds": [13, 47],
    }

    selected_model, dataset_cfg, dataset_resolved = train_classification.apply_experiment_config(
        args,
        experiment_cfg,
        resolved_overrides=None,
    )

    layout = train_classification._resolve_run_layout(
        args,
        selected_model=selected_model,
        dataset_cfg=dataset_cfg,
        dataset_resolved=dataset_resolved,
        experiment_cfg=experiment_cfg,
    )

    assert train_classification._get_active_seed(args) == 29
    assert args.seeds and args.seeds[0] == 29
    assert layout["stem"].endswith("_s29")
