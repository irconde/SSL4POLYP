#!/usr/bin/env python3
"""Verify the data packs used by campaign v3 against their manifests (spec v3.1, section 6).

For every pack split it checks:
  * the CSV SHA-256 against the manifest ``hashes`` block, the parent pack hashes declared by
    subsets, and the ``assertions.hash`` block of ``config/data/<pack>.yaml`` when present;
  * frame, positive and negative counts against the manifest ``counts`` block;
  * that every referenced image exists under the roots of ``data/roots.json`` and has the
    width/height recorded in the CSV;
  * for PolypGen, the image SHA-1 against ``content_sha1`` in ``data/polypgen/manifests/frames.csv``.

Writes ``<out>/pack_hashes.csv`` and ``<out>/pack_verification.json`` and exits non-zero on any
mismatch.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packs-root", type=Path, default=REPO / "data_packs")
    parser.add_argument("--roots", type=Path, default=REPO / "data" / "roots.json")
    parser.add_argument("--out", type=Path, default=REPO / "results_v3" / "verification")
    parser.add_argument("--skip-images", action="store_true")
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
    for family, packs in PACKS.items():
        config_hashes = load_config_hashes(family)
        for pack in packs:
            pack_dir = args.packs_root / pack
            manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text()) or {}
            manifest_hashes = manifest.get("hashes") or {}
            parent_hashes = (manifest.get("parent_pack") or {}).get("hashes") or {}
            counts = manifest.get("counts") or {}
            for csv_path in sorted(pack_dir.glob("*.csv")):
                split = csv_path.stem
                actual = sha256(csv_path)
                expected_manifest = manifest_hashes.get(split)
                expected_config = config_hashes.get(split) if pack == family else None
                with csv_path.open(newline="") as handle:
                    frames = list(csv.DictReader(handle))
                n_pos = sum(1 for r in frames if str(r["label"]) == "1")
                n_neg = len(frames) - n_pos
                split_counts = counts.get(split) or {}
                missing = wrong_size = sha1_mismatch = sha1_checked = 0
                if not args.skip_images:
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
                    "pack": pack,
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
                    "images_missing": missing if not args.skip_images else "",
                    "images_wrong_size": wrong_size if not args.skip_images else "",
                    "images_sha1_checked": sha1_checked if not args.skip_images else "",
                    "images_sha1_mismatch": sha1_mismatch if not args.skip_images else "",
                }
                rows.append(row)
                for key in ("manifest_match", "config_match", "counts_match"):
                    if row[key] is False:
                        failures.append(f"{pack}/{split}: {key}")
                if not expected_manifest:
                    failures.append(f"{pack}/{split}: no manifest hash")
                for key in ("images_missing", "images_wrong_size", "images_sha1_mismatch"):
                    if row[key] not in ("", 0):
                        failures.append(f"{pack}/{split}: {key}={row[key]}")
            for split, expected in parent_hashes.items():
                parent_csv = args.packs_root / "sun_full" / f"{split}.csv"
                match = sha256(parent_csv) == expected
                rows.append(
                    {"pack": pack, "split": f"parent:{split}", "csv_sha256": sha256(parent_csv),
                     "manifest_sha256": expected, "manifest_match": match}
                )
                if not match:
                    failures.append(f"{pack}: parent {split} hash mismatch")

    args.out.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with (args.out / "pack_hashes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checked_splits": len(rows), "failures": failures, "passed": not failures}
    (args.out / "pack_verification.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
