"""The simulated campaign has the structure of the real one and loads with every integrity check passing."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from analysis.v3.loader import FRAME_COLUMNS, completeness, load_campaign
from analysis.v3.sap import E1_PERCENTS, E4_BUDGETS, PARTITIONS, REGIMES, SEEDS

REAL = Path(__file__).resolve().parents[3] / "checkpoints_v3/campaign/classification"


def _header(path: Path):
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def test_all_runs_load_and_checks_pass(sim_full):
    out, truth = sim_full
    c = load_campaign(truth["roots"])
    assert c.regimes() == list(REGIMES)
    assert c.partitions() == list(PARTITIONS) and c.seeds() == list(SEEDS)
    comp = completeness(c, REGIMES, PARTITIONS, SEEDS)
    assert (comp["present"] == comp["expected"]).all()
    assert len(c.runs) == len(REGIMES) * len(PARTITIONS) * len(SEEDS) * (1 + len(E1_PERCENTS) + 1 + 1 + len(E4_BUDGETS))
    assert c.checks and all(ch["passed"] for ch in c.checks)
    assert truth["clipped_targets"] == 0


def test_matched_regime_in_separate_root(sim_full):
    out, _ = sim_full
    assert list((out / "campaign_matched/classification").rglob("ssl_imnet_matched_seed13"))
    assert not list((out / "campaign/classification").rglob("ssl_imnet_matched_seed*"))


def test_csv_columns_and_files(sim_full):
    out, _ = sim_full
    csvs = list(out.rglob("*_outputs.csv"))
    kinds = {re.sub(r"^.*?_s\d+_", "", p.name) for p in csvs}
    assert kinds == {"test_outputs.csv", "val_outputs.csv", "zeroshot_outputs.csv", "zeroshot_train_outputs.csv",
                     "zeroshot_val_outputs.csv", "train_outputs.csv"}
    for p in csvs[:: max(1, len(csvs) // 200)]:
        assert tuple(_header(p)) == FRAME_COLUMNS


def test_split_sizes_match_the_real_packs(sim_full):
    out, _ = sim_full
    c = load_campaign([out / "campaign/classification"])
    run = c.get("sun_full", None, "ssl_colon", 1, 13).frames["test"]
    assert len(run) == 990 and run.labels.sum() == 495
    morph = run.col("morphology")[run.labels == 1]
    assert (morph == "flat").sum() == 5 * 33 and (morph == "polypoid").sum() == 10 * 33
    e3 = c.get("e3", None, "ssl_colon", 1, 13).frames["test"]
    assert len(e3) == 1438 and e3.labels.sum() == 1254
    assert sorted(set(e3.col("center_id"))) == ["C1", "C2", "C3", "C4", "C5", "C6"]
    zs = c.get("e4", 50, "ssl_colon", 1, 13).frames["zeroshot"]
    src = zs.col("negative_source")[zs.labels == 0]
    assert (src == "isolated").sum() == 159 and (src == "sequence").sum() == 3190 and zs.labels.sum() == 1229
    zs500 = c.get("e4", 500, "ssl_colon", 1, 13).frames["zeroshot"]
    assert (zs500.col("negative_source")[zs500.labels == 0] == "isolated").sum() == 0


@pytest.mark.skipif(not REAL.exists(), reason="real campaign not present")
def test_same_layout_and_headers_as_the_real_campaign(sim_full):
    """Compares file names and CSV headers only (no value of the real results is read)."""

    out, _ = sim_full

    def names(root: Path, regime: str):
        found = set()
        for p in root.rglob("*"):
            if not p.is_file() or "manifest_snapshot" in p.parts:
                continue
            rel = p.relative_to(root)
            if rel.parts[0] != "k1" or f"{regime}_seed13" not in rel.parts:
                continue
            if p.suffix == ".csv" and p.name.endswith("_outputs.csv") or p.name.endswith("metrics.json") or p.name == "RUN_COMPLETE.json":
                found.add(re.sub(r"(SSLColon|SUPImNet|SSLImNet)", "TAG", str(rel)))
        return found

    real, sim = names(REAL, "ssl_colon"), names(out / "campaign/classification", "ssl_colon")
    assert sim == real
    real_csv = next(REAL.glob("k1/e4/s50/ssl_colon_seed13/**/*_zeroshot_outputs.csv"))
    sim_csv = next((out / "campaign/classification").glob("k1/e4/s50/ssl_colon_seed13/**/*_zeroshot_outputs.csv"))
    assert _header(real_csv) == _header(sim_csv)
