from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

yaml = pytest.importorskip("yaml")
pytest.importorskip("distutils", reason="train_classification depends on distutils")

from ssl4polyp.classification import train_classification as tc


def test_metrics_provenance_tracks_fewshot_budget(tmp_path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    for path in (train_csv, val_csv, test_csv):
        path.write_text("id,label\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.yaml"
    manifest_payload = {
        "policy": {"fewshot_budget_S": 200},
        "generator": {"seed": 321},
    }
    manifest_path.write_text(yaml.safe_dump(manifest_payload), encoding="utf-8")

    dataset_summary = {
        "train": {
            "csv_path": str(train_csv),
            "csv_sha256": "deadbeef",
            "pack_spec": "polypgen_fewshot/polypgen_fewshot_s200_seed13",
            "manifest_path": str(manifest_path),
        },
        "val": {"csv_path": str(val_csv), "csv_sha256": "feedface"},
        "test": {"csv_path": str(test_csv), "csv_sha256": "cafebabe"},
    }

    args = SimpleNamespace(
        model_key="ssl_colon",
        arch="vit_base",
        seed=13,
        dataset_summary=dataset_summary,
        output_dir=str(tmp_path),
        latest_test_outputs_path=None,
        latest_test_outputs_sha256=None,
        test_split="polypgen_clean_test",
    )

    provenance = tc._build_metrics_provenance(args)

    assert provenance["train_seed"] == 13
    assert provenance["train_pack"] == dataset_summary["train"]["pack_spec"]
    assert provenance["train_pack_name"] == Path(dataset_summary["train"]["pack_spec"]).name
    assert provenance["fewshot_budget"] == 200
    assert provenance["pack_seed"] == 321
    assert provenance["subset_percent"] == pytest.approx(100.0)
    assert provenance["split"] == "polypgen_clean_test"
    assert provenance["val_csv_sha256"] == "feedface"
    assert provenance["test_csv_sha256"] == "cafebabe"
