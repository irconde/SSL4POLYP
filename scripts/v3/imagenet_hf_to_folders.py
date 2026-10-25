#!/usr/bin/env python3
"""Convert the ILSVRC/imagenet-1k (Hugging Face) train Parquet shards to ``<out>/<synset>/<name>.JPEG``.

* The original encoded bytes are written unchanged (no decoding or re-encoding).
* The class folder is the synset of the integer label, taken from the file name stored in the
  Parquet (``<original>_<synset>.JPEG``); the script checks that every label maps to one synset
  and that label ``i`` is the ``i``-th synset in sorted order.
* File name: the original ILSVRC name (``<original>.JPEG``, i.e. the stored name without the
  ``_<synset>`` suffix added by Hugging Face); ``shard_row.JPEG`` if no name is stored.
* Every image is then opened with PIL (``verify`` and a full ``load``); problems are recorded, never
  dropped.

Outputs in ``--records``: ``images.csv`` (shard, row, label, synset, relative path, bytes, SHA-256,
PIL format, mode, size, status), ``label_synset.csv`` and ``summary.json``.
Run in an auxiliary environment with pyarrow and Pillow (not the training environments).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

NAME_RE = re.compile(r"^(?P<orig>(?P<synset>n\d{8})_\d+)_(?P=synset)\.JPEG$")
KNOWN = {0: ("n01440764", "tench"), 1: ("n01443537", "goldfish"), 2: ("n01484850", "great white shark"),
         207: ("n02099601", "golden retriever"), 999: ("n15075141", "toilet tissue")}


def convert_shard(task: tuple) -> list:
    shard_path, out_root = task
    shard = Path(shard_path).name.split("-")[1]
    rows = []
    pf = pq.ParquetFile(shard_path)
    row = 0
    for batch in pf.iter_batches(columns=["image", "label"], batch_size=256):
        for record in batch.to_pylist():
            data, stored = record["image"]["bytes"], record["image"].get("path")
            m = NAME_RE.match(stored or "")
            synset = m.group("synset") if m else None
            name = f"{m.group('orig')}.JPEG" if m else f"{shard}_{row}.JPEG"
            rel = f"{synset or 'UNKNOWN_' + str(record['label'])}/{name}"
            dest = Path(out_root) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                status = "name_collision" if dest.read_bytes() != data else "exists_identical"
            else:
                with open(dest, "wb") as handle:
                    handle.write(data)
                status = "ok"
            rows.append([shard, row, record["label"], synset or "", stored or "", rel, len(data),
                         hashlib.sha256(data).hexdigest(), status])
            row += 1
    return rows


def check_image(task: tuple) -> tuple:
    root, rel = task
    path = Path(root) / rel
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            return rel, im.format, im.mode, f"{im.width}x{im.height}", "ok"
    except Exception as exc:  # recorded, never dropped
        return rel, "", "", "", f"error: {type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet-dir", type=Path, required=True,
                        help="directory with the 294 train-*-of-00294.parquet shards of ILSVRC/imagenet-1k")
    parser.add_argument("--out", type=Path, required=True, help="destination <out>/<synset>/<name>.JPEG")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.records.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.parquet_dir.glob("train-*-of-00294.parquet"))
    assert len(shards) == 294, len(shards)

    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, shard_rows in enumerate(pool.map(convert_shard, [(str(s), str(args.out)) for s in shards])):
            rows.extend(shard_rows)
            if (i + 1) % 25 == 0:
                print(f"converted {i + 1}/294 shards, {len(rows)} images", flush=True)

    label_to_synsets = {}
    for r in rows:
        label_to_synsets.setdefault(r[2], set()).add(r[3])
    one_to_one = all(len(v) == 1 and "" not in v for v in label_to_synsets.values())
    mapping = {lab: next(iter(v)) for lab, v in sorted(label_to_synsets.items())}
    sorted_order = [mapping[i] for i in range(len(mapping))] == sorted(mapping.values())
    known = {str(k): {"expected": v[0], "found": mapping.get(k), "name": v[1], "ok": mapping.get(k) == v[0]} for k, v in KNOWN.items()}

    with ProcessPoolExecutor(args.workers) as pool:
        checks = {rel: res for rel, *res in pool.map(check_image, [(str(args.out), r[5]) for r in rows], chunksize=512)}

    header = ["shard", "row", "label", "synset", "parquet_path", "relpath", "bytes", "sha256", "write_status",
              "pil_format", "pil_mode", "size", "pil_status"]
    with (args.records / "images.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(header)
        for r in rows:
            w.writerow(r + list(checks[r[5]]))
    with (args.records / "label_synset.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["label", "synset"])
        w.writerows(sorted(mapping.items()))

    on_disk_dirs = sorted(p.name for p in args.out.iterdir() if p.is_dir())
    on_disk_files = sum(len(os.listdir(args.out / d)) for d in on_disk_dirs)
    problems = [{"relpath": r[5], "write_status": r[8], "pil_status": checks[r[5]][3]}
                for r in rows if r[8] != "ok" or checks[r[5]][3] != "ok"]
    summary = {
        "images_in_parquet": len(rows), "class_folders": len(on_disk_dirs), "files_on_disk": on_disk_files,
        "labels": len(mapping), "label_to_synset_one_to_one": one_to_one, "label_order_is_sorted_synsets": sorted_order,
        "known_classes": known, "names_without_original": sum(1 for r in rows if not NAME_RE.match(r[4])),
        "write_status": dict(Counter(r[8] for r in rows)),
        "pil_formats": dict(Counter(c[0] for c in checks.values())), "pil_modes": dict(Counter(c[1] for c in checks.values())),
        "pil_status_ok": sum(1 for c in checks.values() if c[3] == "ok"), "problems": problems,
    }
    (args.records / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=2))
    print("problems:", len(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
