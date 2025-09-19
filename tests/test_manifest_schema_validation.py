import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if "torch" not in sys.modules:  # pragma: no cover - test shim for environments without torch
    sys.modules["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda idx: "cpu",
        ),
        version=SimpleNamespace(cuda=None),
    )

if "yaml" not in sys.modules:  # pragma: no cover - test shim for environments without PyYAML
    def _safe_load(data):
        if hasattr(data, "read"):
            data = data.read()
        if not data:
            return {}
        return json.loads(data)

    def _safe_dump(obj, stream=None):
        serialized = json.dumps(obj)
        if stream is None:
            return serialized
        stream.write(serialized)

    sys.modules["yaml"] = SimpleNamespace(safe_load=_safe_load, safe_dump=_safe_dump)

from ssl4polyp.configs.manifests import load_pack


def _write_manifest(
    path: Path,
    fields: list[str],
    splits: dict[str, str],
    roots: dict[str, Path],
) -> None:
    manifest: dict[str, object] = {
        "row_schema": {"fields": [{"name": name} for name in fields]},
        "roots": {key: str(value) for key, value in roots.items()},
    }
    manifest.update({split: {"csv": csv_name} for split, csv_name in splits.items()})
    path.write_text(json.dumps(manifest))


@pytest.fixture
def root_with_frame(tmp_path: Path) -> Path:
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / "frame.png").write_text("data")
    return root_dir


def test_load_pack_enforces_schema_columns(tmp_path: Path, root_with_frame: Path) -> None:
    train_csv = tmp_path / "train.csv"
    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame_path", "label", "split", "dataset"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_path": "root/frame.png",
                "label": "0",
                "split": "train",
                "dataset": "demo",
            }
        )

    manifest_yaml = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_yaml,
        fields=["frame_path", "label", "split", "dataset"],
        splits={"train": "train.csv"},
        roots={"root": root_with_frame},
    )

    pack = load_pack(train=train_csv, manifest_yaml=manifest_yaml)
    paths, labels, meta = pack["train"]
    assert labels == ["0"]
    assert paths[0].name == "frame.png"
    assert meta[0]["split"] == "train"


def test_load_pack_raises_on_missing_schema_column(
    tmp_path: Path, root_with_frame: Path
) -> None:
    train_csv = tmp_path / "train.csv"
    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "label", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "frame_path": "root/frame.png",
                "label": "0",
                "split": "train",
            }
        )

    manifest_yaml = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_yaml,
        fields=["frame_path", "label", "split", "dataset"],
        splits={"train": "train.csv"},
        roots={"root": root_with_frame},
    )

    with pytest.raises(ValueError, match="dataset"):
        load_pack(train=train_csv, manifest_yaml=manifest_yaml)


def test_load_pack_rejects_split_leak(tmp_path: Path, root_with_frame: Path) -> None:
    val_csv = tmp_path / "val.csv"
    with open(val_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame_path", "label", "split", "dataset"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_path": "root/frame.png",
                "label": "0",
                "split": "train",  # Incorrect split value
                "dataset": "demo",
            }
        )

    manifest_yaml = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_yaml,
        fields=["frame_path", "label", "split", "dataset"],
        splits={"val": "val.csv"},
        roots={"root": root_with_frame},
    )

    with pytest.raises(ValueError, match="Split value mismatch"):
        load_pack(val=val_csv, manifest_yaml=manifest_yaml)

