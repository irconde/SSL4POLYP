import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_aggregate_module(repo_root: Path):
    module_path = repo_root / "scripts" / "aggregate_metrics.py"
    spec = importlib.util.spec_from_file_location("aggregate_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregate_metrics(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    metrics_root = tmp_path / "results" / "classification"
    exp_dir = metrics_root / "exp_alpha"
    exp_dir.mkdir(parents=True)

    samples = [
        {"seed": 1, "test": {"auroc": 0.9, "f1": 0.8}},
        {"seed": 2, "test": {"auroc": 0.7, "f1": 0.6}},
        {"seed": 3, "test": {"auroc": 0.85, "f1": 0.75}},
    ]
    for sample in samples:
        with open(exp_dir / f"seed{sample['seed']}.json", "w") as f:
            json.dump(sample, f)

    agg_module = load_aggregate_module(repo_root)
    expected = agg_module._aggregate_metric([s["test"]["auroc"] for s in samples], 0.95, 100, 0)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_metrics.py"),
            "--metrics-root",
            str(metrics_root),
            "--bootstrap",
            "100",
            "--rng-seed",
            "0",
        ],
        cwd=repo_root,
        check=True,
    )

    summary_path = metrics_root / "summary.csv"
    assert summary_path.exists()
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # auroc and f1

    row_map = {row["metric"]: row for row in rows}
    auroc_row = row_map["auroc"]
    assert auroc_row["experiment"] == "exp_alpha"
    assert auroc_row["n"] == "3"
    assert auroc_row["seeds"] == "1,2,3"
    assert abs(float(auroc_row["mean"]) - expected["mean"]) < 1e-9
    assert abs(float(auroc_row["std"]) - expected["std"]) < 1e-9
    assert abs(float(auroc_row["ci_lower"]) - expected["ci_lower"]) < 1e-9
    assert abs(float(auroc_row["ci_upper"]) - expected["ci_upper"]) < 1e-9
