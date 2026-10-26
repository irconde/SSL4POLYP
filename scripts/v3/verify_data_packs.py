#!/usr/bin/env python3
"""Verify the data packs used by campaign v3 against their manifests (spec v3.1, section 6).

For every pack split it checks:
  * the CSV SHA-256 against the manifest ``hashes`` block, the parent pack hashes declared by
    subsets, and the ``assertions.hash`` block of ``config/data/<pack>.yaml`` when present;
  * frame, positive and negative counts against the manifest ``counts`` block;
  * that every referenced image exists under the roots of ``data/roots.json`` and has the
    width/height recorded in the CSV;
  * for PolypGen, the image SHA-1 against ``content_sha1`` in ``data/polypgen/manifests/frames.csv``.

With ``--v3`` it verifies the campaign v3 packs under ``<packs-root>/v3/`` instead (every directory
with a ``manifest.yaml``): the same per-split checks, plus every CSV SHA-256 against
``v3/index.yaml``, no CSV missing from or absent in the index, and the parent hashes declared by
subsets (``k*/base``) and few-shot packs (``_parent/polypgen_clean_test_extended``).

Writes ``<out>/pack_hashes.csv`` and ``<out>/pack_verification.json`` (``*_v3`` with ``--v3``) and exits
non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
PACKS = {
    "sun_full": ["sun_full"],
    "sun_subsets": [f"sun_subsets/sun_p{p}" for p in ("05", "10", "25", "50", "100")],
    "sun_morphology": ["sun_morphology"],
    "polypgen_clean_test": ["polypgen_clean_test"],
    "polypgen_fewshot": [f"polypgen_fewshot/polypgen_fewshot_s{s}" for s in (50, 100, 200, 500)],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config_hashes(family: str) -> Dict[str, str]:
    path = REPO / "config" / "data" / f"{family}.yaml"
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text()) or {}
    return ((cfg.get("dataset") or {}).get("assertions") or {}).get("hash") or {}


def check_pack(pack_dir: Path, label: str, roots: dict, polypgen_sha1: Dict[str, str], skip_images: bool,
               config_hashes: Dict[str, str], index: Optional[Dict[str, str]] = None) -> tuple:
    """Per-split rows and failures of one pack directory, and its manifest."""
    rows: List[dict] = []
    failures: List[str] = []
    manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text()) or {}
    manifest_hashes = manifest.get("hashes") or {}
    counts = manifest.get("counts") or {}
    for csv_path in sorted(pack_dir.glob("*.csv")):
        split = csv_path.stem
        actual = sha256(csv_path)
        expected_manifest = manifest_hashes.get(split)
        expected_config = config_hashes.get(split)
        with csv_path.open(newline="") as handle:
            frames = list(csv.DictReader(handle))
        n_pos = sum(1 for r in frames if str(r["label"]) == "1")
        n_neg = len(frames) - n_pos
        split_counts = counts.get(split) or {}
        missing = wrong_size = sha1_mismatch = sha1_checked = 0
        if not skip_images:
            for r in frames:
                image = Path(roots[r["store_id"]]) / r["frame_path"]
                if not image.exists():
                    missing += 1
                    continue
                with Image.open(image) as img:
                    if (str(img.width), str(img.height)) != (r["width"], r["height"]):
                        wrong_size += 1
                if r["store_id"].startswith("polypgen") and r["frame_path"] in polypgen_sha1:
                    sha1_checked += 1
                    if sha1(image) != polypgen_sha1[r["frame_path"]]:
                        sha1_mismatch += 1
        row = {
            "pack": label,
            "split": split,
            "csv_sha256": actual,
            "manifest_sha256": expected_manifest or "",
            "manifest_match": expected_manifest == actual if expected_manifest else "",
            "config_sha256": expected_config or "",
            "config_match": expected_config == actual if expected_config else "",
            "frames": len(frames),
            "pos": n_pos,
            "neg": n_neg,
            "manifest_frames": split_counts.get("frames", ""),
            "counts_match": (
                (split_counts.get("frames"), split_counts.get("pos"), split_counts.get("neg"))
                == (len(frames), n_pos, n_neg)
            )
            if split_counts
            else "",
            "images_missing": missing if not skip_images else "",
            "images_wrong_size": wrong_size if not skip_images else "",
            "images_sha1_checked": sha1_checked if not skip_images else "",
            "images_sha1_mismatch": sha1_mismatch if not skip_images else "",
        }
        if index is not None:
            expected_index = index.get(f"{label}/{csv_path.name}")
            row["index_sha256"] = expected_index or ""
            row["index_match"] = expected_index == actual if expected_index else ""
            if not expected_index:
                failures.append(f"{label}/{split}: not in index.yaml")
        rows.append(row)
        for key in ("manifest_match", "config_match", "counts_match", "index_match"):
            if row.get(key) is False:
                failures.append(f"{label}/{split}: {key}")
        if not expected_manifest:
            failures.append(f"{label}/{split}: no manifest hash")
        for key in ("images_missing", "images_wrong_size", "images_sha1_mismatch"):
            if row[key] not in ("", 0):
                failures.append(f"{label}/{split}: {key}={row[key]}")
    return rows, failures, manifest


def check_parent(label: str, parent_hashes: Dict[str, str], parent_dir: Path) -> tuple:
    rows: List[dict] = []
    failures: List[str] = []
    for split, expected in parent_hashes.items():
        parent_csv = parent_dir / f"{split}.csv"
        actual = sha256(parent_csv)
        match = actual == expected
        rows.append(
            {"pack": label, "split": f"parent:{split}", "csv_sha256": actual,
             "manifest_sha256": expected, "manifest_match": match}
        )
        if not match:
            failures.append(f"{label}: parent {split} hash mismatch")
    return rows, failures


def v3_index(v3_root: Path) -> Dict[str, str]:
    index = yaml.safe_load((v3_root / "index.yaml").read_text())
    hashes = {rel: h for part in index["partitions"].values() for rel, h in part["csv_sha256"].items()}
    hashes.update(index.get("parent") or {})
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packs-root", type=Path, default=REPO / "data_packs")
    parser.add_argument("--roots", type=Path, default=REPO / "data" / "roots.json")
    parser.add_argument("--out", type=Path, default=REPO / "results_v3" / "verification")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--v3", action="store_true", help="verify data_packs/v3/ against its manifests and index.yaml")
    args = parser.parse_args()

    roots = json.loads(args.roots.read_text())
    polypgen_sha1: Dict[str, str] = {}
    frames_csv = Path(roots["polypgen_clean"]) / "manifests" / "frames.csv"
    if frames_csv.exists():
        with frames_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                polypgen_sha1[row["img_path"]] = row.get("content_sha1", "")

    rows: List[dict] = []
    failures: List[str] = []
    if args.v3:
        v3_root = args.packs_root / "v3"
        index = v3_index(v3_root)
        on_disk = {p.relative_to(v3_root).as_posix() for p in v3_root.rglob("*.csv")}
        failures += [f"{rel}: in index.yaml, file missing" for rel in sorted(set(index) - on_disk)]
        failures += [f"{rel}: CSV not in index.yaml" for rel in sorted(on_disk - set(index))]
        for manifest_path in sorted(v3_root.rglob("manifest.yaml")):
            pack_dir = manifest_path.parent
            label = pack_dir.relative_to(v3_root).as_posix()
            pack_rows, pack_failures, manifest = check_pack(
                pack_dir, label, roots, polypgen_sha1, args.skip_images, {}, index)
            rows += pack_rows
            failures += pack_failures
            parent_hashes = (manifest.get("parent_pack") or {}).get("hashes") or {}
            if parent_hashes:
                if "/sun_subsets/" in f"/{label}/":
                    parent_dir = v3_root / label.split("/")[0] / "base"
                elif "/polypgen_fewshot/" in f"/{label}/":
                    parent_dir = v3_root / "_parent" / "polypgen_clean_test_extended"
                else:
                    failures.append(f"{label}: parent pack not resolved")
                    continue
                parent_rows, parent_failures = check_parent(label, parent_hashes, parent_dir)
                rows += parent_rows
                failures += parent_failures
        summary_extra = {"index_csvs": len(index), "csvs_on_disk": len(on_disk),
                         "pack_dirs": len(list(v3_root.rglob("manifest.yaml")))}
        suffix = "_v3"
    else:
        for family, packs in PACKS.items():
            config_hashes = load_config_hashes(family)
            for pack in packs:
                pack_rows, pack_failures, manifest = check_pack(
                    args.packs_root / pack, pack, roots, polypgen_sha1, args.skip_images,
                    config_hashes if pack == family else {})
                rows += pack_rows
                failures += pack_failures
                parent_hashes = (manifest.get("parent_pack") or {}).get("hashes") or {}
                parent_rows, parent_failures = check_parent(pack, parent_hashes, args.packs_root / "sun_full")
                rows += parent_rows
                failures += parent_failures
        summary_extra = {}
        suffix = ""

    args.out.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with (args.out / f"pack_hashes{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checked_splits": len(rows), **summary_extra, "failures": failures, "passed": not failures}
    (args.out / f"pack_verification{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
