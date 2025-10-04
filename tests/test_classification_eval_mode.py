from pathlib import Path
import sys
import json
import types
import copy
from typing import Dict, List

import pytest

pytest.importorskip("torch")

import math

import torch
import torch.nn as nn
import torch.optim as optim


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

if "distutils" not in sys.modules:
    sys.modules["distutils"] = types.ModuleType("distutils")
if "distutils.version" not in sys.modules:
    sys.modules["distutils.version"] = types.ModuleType("distutils.version")
if "yaml" not in sys.modules:
    fake_yaml = types.ModuleType("yaml")

    def _yaml_safe_load(*args, **kwargs):  # pragma: no cover - defensive stub
        raise RuntimeError("yaml.safe_load is not available in this test stub")

    def _yaml_safe_dump(data, stream=None, **kwargs):  # pragma: no cover - defensive stub
        text = json.dumps(data)
        if stream is None:
            return text
        stream.write(text)
        return text

    fake_yaml.safe_load = _yaml_safe_load
    fake_yaml.safe_dump = _yaml_safe_dump
    sys.modules["yaml"] = fake_yaml

from ssl4polyp.classification import train_classification
from ssl4polyp.classification.data import packs
from ssl4polyp.configs.layered import load_layered_config


def test_eval_only_dataset_skips_training_csv(monkeypatch, tmp_path):
    assert Path("config/data/polypgen_clean_test.yaml").exists()
    dataset_cfg = {
        "name": "polypgen_clean_test",
        "pack": "polypgen_clean_test",
        "splits": {"test": "test"},
    }
    resolved = train_classification._resolve_dataset_specs(dataset_cfg)

    assert resolved["train_pack"] is None
    assert resolved["val_pack"] is None
    assert resolved["test_pack"] is not None

    requested_splits: List[List[str]] = []
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("{}")

    def fake_resolve(spec, splits, pack_root):
        requested_splits.append(list(splits))
        csv_map: Dict[str, Path] = {
            split: tmp_path / f"{split}.csv" for split in splits
        }
        for csv_path in csv_map.values():
            csv_path.write_text("frame_path,label\nexample.jpg,0\n")
        return manifest_path, csv_map

    def fake_load_pack(
        *, manifest_yaml=None, roots_map=None, pack_root=None, snapshot_dir=None, **split_csvs,
    ):
        dummy_image = tmp_path / "example.jpg"
        dummy_image.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = {}
        for split, _ in split_csvs.items():
            result[split] = ([dummy_image], [0], [{"frame_path": str(dummy_image)}])
        return result

    monkeypatch.setattr(packs, "_resolve_manifest_and_csvs", fake_resolve)
    monkeypatch.setattr(packs, "load_pack", fake_load_pack)

    loaders, datasets, samplers = packs.create_classification_dataloaders(
        train_spec=resolved["train_pack"],
        val_spec=resolved["val_pack"],
        test_spec=resolved["test_pack"],
        train_split=resolved["train_split"] or "train",
        val_split=resolved["val_split"] or "val",
        test_split=resolved["test_split"] or "test",
        batch_size=1,
        num_workers=0,
        rank=0,
        world_size=1,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=False,
        pack_root=None,
        roots_map=None,
        image_size=32,
        perturbation_splits=None,
        snapshot_dir=None,
    )

    assert loaders.get("train") is None
    assert loaders.get("test") is not None
    assert samplers.get("train") is None
    assert any("test" in group for group in requested_splits)
    assert all("train" not in group for group in requested_splits)
    assert datasets.get(resolved["test_split"] or "test") is not None


def test_cli_subset_overrides_and_batch_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    args = train_classification.get_args()
    args.dataset_percent = 10.0
    args.dataset_seed = 13
    args.limit_train_batches = 2
    args.limit_val_batches = 1
    args.limit_test_batches = 3
    args.model_key = "sup_imnet"
    args.output_dir = str(tmp_path)
    args.log_interval = 1

    experiment_cfg = load_layered_config("exp/exp1_smoke.yaml")
    _, dataset_cfg, dataset_resolved = train_classification.apply_experiment_config(
        args, experiment_cfg, resolved_overrides=None
    )

    assert dataset_cfg["name"] == "sun_subsets"
    assert dataset_cfg["percent"] == 10
    assert dataset_cfg["seed"] == 13
    assert str(dataset_resolved["train_pack"]).endswith("sun_p10")
    assert str(args.train_pack).endswith("sun_p10")
    assert args.limit_train_batches == 2
    assert args.limit_val_batches == 1
    assert args.limit_test_batches == 3

    class DummyDataset:
        def __init__(self, samples):
            self._samples = samples

        def __len__(self):
            return self._samples

    class CountingLoader:
        def __init__(self, batches, batch_size):
            self._batches = batches
            self._batch_size = batch_size
            self.dataset = DummyDataset(batches * batch_size)
            self._yielded = 0

        def __len__(self):
            return self._batches

        def __iter__(self):
            self._yielded = 0
            for _ in range(self._batches):
                self._yielded += 1
                images = torch.zeros(self._batch_size, 3, 4, 4)
                labels = torch.zeros(self._batch_size, dtype=torch.long)
                yield images, labels, {}

        @property
        def yielded(self):
            return self._yielded

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    train_loader = CountingLoader(batches=5, batch_size=2)
    train_loss, new_step = train_classification.train_epoch(
        model,
        rank=0,
        world_size=1,
        train_loader=train_loader,
        train_sampler=None,
        optimizer=optimizer,
        epoch=1,
        loss_fn=loss_fn,
        log_path=str(tmp_path / "train.log"),
        scaler=scaler,
        use_amp=False,
        tb_logger=None,
        log_interval=args.log_interval,
        global_step=0,
        seed=42,
        device=torch.device("cpu"),
        distributed=False,
        max_batches=args.limit_train_batches,
    )

    assert not math.isnan(float(train_loss))
    assert new_step == args.limit_train_batches
    assert train_loader.yielded == args.limit_train_batches

    test_loader = CountingLoader(batches=6, batch_size=2)

    def dummy_perf(probs, targets):
        return torch.tensor(0.5)

    results = train_classification.test(
        model,
        rank=0,
        test_loader=test_loader,
        epoch=1,
        perf_fn=dummy_perf,
        log_path=str(tmp_path / "test.log"),
        metric_fns={},
        split_name="Test",
        return_outputs=False,
        tau=None,
        max_batches=args.limit_test_batches,
    )

    assert "auroc" in results
    expected_batches = min(
        len(test_loader),
        args.limit_test_batches + train_classification.EVAL_MAX_ADDITIONAL_BATCHES,
    )
    assert test_loader.yielded == expected_batches


def test_seed_resolution_defaults_and_protocol_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    args_default = train_classification.get_args()
    args_default.model_key = "sup_imnet"
    args_default.output_dir = str(tmp_path / "out_default")

    base_cfg = load_layered_config("exp/exp1.yaml")
    _, default_dataset_cfg, default_dataset_resolved = train_classification.apply_experiment_config(
        args_default, base_cfg, resolved_overrides=None
    )

    assert args_default.training_seeds == [13, 29, 47]
    assert args_default.dataset_seeds == [47]
    assert default_dataset_cfg.get("seed") == 47
    assert default_dataset_resolved["seed"] == 47

    monkeypatch.setattr(sys, "argv", ["pytest"])
    args_override = train_classification.get_args()
    args_override.model_key = "sup_imnet"
    args_override.output_dir = str(tmp_path / "out_override")

    override_cfg = copy.deepcopy(base_cfg)
    override_cfg.setdefault("protocol", {})["seeds"] = [29]

    _, override_dataset_cfg, override_dataset_resolved = train_classification.apply_experiment_config(
        args_override, override_cfg, resolved_overrides=None
    )

    assert args_override.training_seeds == [29]
    assert args_override.seeds == [29]
    assert args_override.config_seed == 29
    assert args_override.dataset_seeds == [47]
    assert override_dataset_cfg.get("seed") == 47
    assert override_dataset_resolved["seed"] == 47
