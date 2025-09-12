import csv
import json
import subprocess
import sys
from pathlib import Path


def run_check(csv_path: Path, roots: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/check_paths.py", str(csv_path), str(roots)],
        capture_output=True,
    )


def test_check_paths_success(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "img.png").write_text("data")

    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path"])
        writer.writerow(["root/img.png"])

    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"root": str(root)}))

    res = run_check(csv_file, roots)
    assert res.returncode == 0


def test_check_paths_failure(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    # file is intentionally missing
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path"])
        writer.writerow(["root/missing.png"])

    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"root": str(root)}))

    res = run_check(csv_file, roots)
    assert res.returncode != 0
