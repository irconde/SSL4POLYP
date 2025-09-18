import csv
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
import yaml

from ssl4polyp.classification.eval_outputs import write_outputs
from ssl4polyp.classification.manifests import load_pack


def test_manifest_snapshot(tmp_path):
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / "img.png").write_text("data")

    train_csv = tmp_path / "train.csv"
    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path", "label"])
        writer.writerow(["root/img.png", "0"])

    manifest_yaml = tmp_path / "manifest.yaml"
    with open(manifest_yaml, "w") as f:
        yaml.safe_dump({"train": {"csv": "train.csv"}, "roots": {"root": str(root_dir)}}, f)

    out_dir = tmp_path / "out"
    load_pack(train=train_csv, manifest_yaml=manifest_yaml, snapshot_dir=out_dir)

    snap_dir = out_dir / "manifest_snapshot"
    assert (snap_dir / "train.csv").exists()
    assert (snap_dir / "manifest.yaml").exists()

    with open(out_dir / "roots.json") as f:
        roots = json.load(f)
    assert roots["root"] == str(root_dir)

    assert (out_dir / "git-commit.txt").read().strip()
    assert (out_dir / "pip-freeze.txt").exists()
    with open(out_dir / "cuda.json") as f:
        cuda = json.load(f)
    assert "available" in cuda


def test_write_outputs(tmp_path):
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    metadata = [{"frame_id": "f1"}, {"frame_id": "f2"}]
    out_dir = tmp_path / "preds"
    write_outputs(logits, metadata, out_dir, tau=0.5)

    with open(out_dir / "tau.json") as f:
        tau = json.load(f)
    assert tau["tau"] == 0.5

    loaded = torch.load(out_dir / "logits.pt")
    assert torch.allclose(loaded, logits)

    with open(out_dir / "metadata.jsonl") as f:
        lines = [json.loads(line) for line in f]
    assert lines == metadata
