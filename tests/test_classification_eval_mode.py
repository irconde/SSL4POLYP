from pathlib import Path
import sys
import json
import types
from typing import Dict, List

import pytest

pytest.importorskip("torch")

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
