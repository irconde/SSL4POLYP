import csv
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
    fake_yaml = types.ModuleType("yaml")

    def _yaml_safe_load(*args, **kwargs):  # pragma: no cover - defensive stub
        raise RuntimeError("yaml.safe_load is not available in this test stub")

    def _yaml_safe_dump(data, stream=None, **kwargs):  # pragma: no cover - defensive stub
        text = "{}"
        if stream is None:
            return text
        stream.write(text)
        return text

    fake_yaml.safe_load = _yaml_safe_load
    fake_yaml.safe_dump = _yaml_safe_dump
    sys.modules["yaml"] = fake_yaml

pytest.importorskip("torch")

from ssl4polyp.classification import train_classification
from ssl4polyp.classification.data import PackDataset
from ssl4polyp.configs import data_packs_root


class _DummyTransform:
    def __call__(self, img, row=None):  # pragma: no cover - defensive stub
        raise RuntimeError("DummyTransform should not be invoked in this test")


def _build_pack_dataset(pack_name: str, split: str, *, max_rows: int = 16) -> PackDataset:
    pack_dir = data_packs_root() / pack_name
    csv_path = pack_dir / f"{split}.csv"
    assert csv_path.exists(), f"Missing CSV for {pack_name}/{split}: {csv_path}"

    rows: list[dict[str, object]] = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            rows.append(dict(row))
            if index + 1 >= max_rows:
                break

    assert rows, f"No rows parsed from {csv_path}"

    paths = [Path(row["frame_path"]) for row in rows]
    labels = [row.get("label", "0") for row in rows]
    return PackDataset(paths=paths, labels=labels, meta=rows, transform=_DummyTransform())


def test_dataset_supports_perturbations_false_for_placeholder_pack():
    dataset = _build_pack_dataset("sun_full", "train")
    assert not train_classification._dataset_supports_perturbations(dataset)


def test_dataset_supports_perturbations_true_for_real_metadata():
    dataset = _build_pack_dataset("sun_test_perturbations", "test", max_rows=32)
    assert train_classification._dataset_supports_perturbations(dataset)
