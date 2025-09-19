"""Utilities for handling dataset manifests for classification tasks.

This module provides helpers for reading CSV split files, verifying their
integrity with SHA256 hashes declared in a manifest YAML, resolving the
filesystem paths for the listed frames, and a convenience ``load_pack``
function that loads one or more splits at once.  Dataset packs are, by
default, resolved relative to the repository-level ``data_packs/`` directory
so that manifests and split CSVs can be colocated with other configuration
artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import yaml

from . import data_packs_root, resolve_config_path, resolve_data_pack_path


if not hasattr(Path, "read"):
    Path.read = Path.read_text  # type: ignore[attr-defined]

# Type aliases for clarity
Row = MutableMapping[str, str]
Paths = List[Path]
Labels = List[str]
Meta = List[Row]
SplitReturn = Tuple[Paths, Labels, Meta]
SplitCountsExpectation = Tuple[Optional[int], Dict[str, int], Dict[str, int]]

REQUIRED_COLUMNS = {"frame_path", "label"}
CANONICAL_SPLIT_NAMES = {"train", "val", "test", "eval"}


def _coerce_to_int(value: object) -> Optional[int]:
    """Attempt to coerce ``value`` into an integer."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def resolve_manifest_path(manifest: Optional[str | Path]) -> Optional[Path]:
    """Resolve ``manifest`` relative to the default configuration directory."""

    if manifest is None:
        return None
    return resolve_config_path(manifest)


def resolve_pack_asset(path: Optional[str | Path]) -> Optional[Path]:
    """Resolve ``path`` relative to the default data pack directory."""

    if path is None:
        return None
    return resolve_data_pack_path(path)


def load_split(
    csv_path: Path,
    required_columns: Optional[Sequence[str]] = None,
    split_column: Optional[str] = None,
    expected_split_value: Optional[str] = None,
) -> Meta:
    """Read a CSV split file and return its rows.

    Parameters
    ----------
    csv_path:
        Path to the CSV file to read.
    required_columns:
        Additional columns expected to be present in the CSV, beyond
        :data:`REQUIRED_COLUMNS`.
    split_column:
        Name of the column that encodes the logical split of the row.  When
        provided together with ``expected_split_value`` every row must report
        the expected value.
    expected_split_value:
        Expected value for ``split_column``.  A :class:`ValueError` is raised if
        any row disagrees.

    Returns
    -------
    Meta
        Parsed CSV rows.

    Raises
    ------
    ValueError
        If any required columns are missing or if the split column contains an
        unexpected value.
    """

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = set(REQUIRED_COLUMNS)
        if required_columns is not None:
            required.update(required_columns)
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Missing required columns {sorted(missing)} in {csv_path}")
        rows: Meta = list(reader)
    if split_column and expected_split_value is not None:
        for idx, row in enumerate(rows, start=1):
            value = row.get(split_column)
            if value != expected_split_value:
                raise ValueError(
                    "Split value mismatch in {} row {}: expected {!r} in column {!r}, got {!r}".format(
                        csv_path, idx, expected_split_value, split_column, value
                    )
                )
    return rows


def verify_hash(csv_path: Path, manifest_yaml: Optional[Path]) -> None:
    """Validate the SHA256 hash of ``csv_path`` against ``manifest_yaml``.

    ``manifest_yaml`` may define hashes for one or more CSV files.  Several
    common layouts are supported.  If no corresponding hash is found the
    function silently returns.
    """

    if manifest_yaml is None:
        return

    with open(manifest_yaml, "r") as f:
        manifest = yaml.safe_load(f) or {}

    csv_name = Path(csv_path).name
    expected: Optional[str] = None

    if isinstance(manifest, Mapping):
        # Layout 1: {split: {csv: "train.csv", sha256: "..."}}
        for entry in manifest.values():
            if isinstance(entry, Mapping):
                csv_entry = entry.get("csv")
                if csv_entry and Path(csv_entry).name == csv_name:
                    expected = entry.get("sha256")
                    break
        # Layout 2: {hashes: {"train.csv": "sha256"}}
        if expected is None and "hashes" in manifest:
            hashes = manifest.get("hashes") or {}
            if isinstance(hashes, Mapping):
                expected = hashes.get(csv_name)
        # Layout 3: {"train.csv": "sha256"}
        if expected is None:
            entry = manifest.get(csv_name)
            if isinstance(entry, str):
                expected = entry
            elif isinstance(entry, Mapping):
                expected = entry.get("sha256")

    if not expected:
        return

    actual = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SHA256 mismatch for {csv_path}: expected {expected}, got {actual}"
        )


def resolve_paths(
    rows: Sequence[Row],
    roots_map: Optional[Mapping[str, str]] = None,
    sample: int = 10,
) -> Paths:
    """Resolve ``frame_path`` entries in ``rows`` to concrete paths.

    Parameters
    ----------
    rows:
        Rows returned by :func:`load_split`.
    roots_map:
        Mapping of root identifiers (the first path component) to absolute
        filesystem paths.  Paths will be expanded according to this mapping.
    sample:
        Number of entries to sample for existence checking.
    """

    paths: Paths = []
    for row in rows:
        raw = row.get("frame_path")
        if raw is None:
            raise ValueError("Row missing 'frame_path' entry")
        p = Path(raw)
        if roots_map and p.parts:
            root = p.parts[0]
            if root in roots_map:
                p = Path(roots_map[root]) / Path(*p.parts[1:])
        paths.append(p)

    if paths:
        for p in random.sample(paths, min(sample, len(paths))):
            if not p.exists():
                raise FileNotFoundError(f"Missing file referenced in manifest: {p}")

    return paths


def load_pack(
    train: Optional[Path] = None,
    val: Optional[Path] = None,
    test: Optional[Path] = None,
    manifest_yaml: Optional[Path] = None,
    roots_map: Optional[Mapping[str, str]] = None,
    pack_root: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
) -> Dict[str, SplitReturn]:
    """Load dataset splits described by CSV manifest files.

    The function returns a mapping of split name to ``(paths, labels, meta)``.
    If ``manifest_yaml`` is provided, missing split paths are inferred from it
    and SHA256 hashes are verified when available.  Relative CSV paths are
    resolved first against the manifest location (when available) and then
    against ``pack_root`` which defaults to the repository ``data_packs``
    directory.

    When ``snapshot_dir`` is provided, the function will copy any parsed CSV
    files and ``manifest.yaml`` into ``snapshot_dir/manifest_snapshot`` and
    record provenance information (``roots.json``, git commit, ``pip freeze``,
    and CUDA details) inside ``snapshot_dir``.

    """

    splits: Dict[str, Optional[Path]] = {
        "train": train,
        "val": val,
        "test": test,
    }

    manifest: Optional[Mapping[str, object]] = None
    schema_columns: Optional[Sequence[str]] = None
    split_column_name: Optional[str] = None
    expected_counts: Dict[str, SplitCountsExpectation] = {}
    if manifest_yaml is not None and not isinstance(manifest_yaml, Path):
        manifest_yaml = Path(manifest_yaml)
    pack_root = pack_root or data_packs_root()
    manifest_parent: Optional[Path] = None
    if manifest_yaml is not None:
        with open(manifest_yaml, "r") as f:
            manifest = yaml.safe_load(f) or {}
        if isinstance(manifest, Mapping) and "eval" in manifest:
            raise ValueError(
                "Manifest defines an 'eval' split which is no longer supported; rename the split to 'test'."
            )
        if isinstance(manifest, Mapping):
            row_schema = manifest.get("row_schema")
            if isinstance(row_schema, Mapping):
                fields = row_schema.get("fields")
                columns: List[str] = []
                if isinstance(fields, Sequence):
                    for field in fields:
                        if isinstance(field, Mapping):
                            name = field.get("name")
                            if name is not None:
                                columns.append(str(name))
                        elif isinstance(field, str):
                            columns.append(field)
                if columns:
                    schema_columns = columns
                split_field = row_schema.get("split_column")
                if isinstance(split_field, str):
                    split_column_name = split_field
                elif schema_columns and "split" in schema_columns:
                    split_column_name = "split"
            counts_section = manifest.get("counts")
            if isinstance(counts_section, Mapping):
                for split_name, raw_counts in counts_section.items():
                    if not isinstance(raw_counts, Mapping):
                        continue
                    expected_rows = _coerce_to_int(raw_counts.get("frames"))
                    explicit_label_counts: Dict[str, int] = {}
                    fallback_label_counts: Dict[str, int] = {}
                    label_counts_section = raw_counts.get("label_counts")
                    if isinstance(label_counts_section, Mapping):
                        for label, value in label_counts_section.items():
                            coerced = _coerce_to_int(value)
                            if coerced is not None:
                                explicit_label_counts[str(label)] = coerced
                    for key, value in raw_counts.items():
                        if key in {"frames", "label_counts"}:
                            continue
                        key_str = str(key)
                        if "case" in key_str or "frame" in key_str:
                            continue
                        coerced = _coerce_to_int(value)
                        if coerced is None:
                            continue
                        fallback_label_counts[key_str] = coerced
                    expected_counts[str(split_name)] = (
                        expected_rows,
                        explicit_label_counts,
                        fallback_label_counts,
                    )
        for name, path in splits.items():
            if path is not None:
                continue
            if not isinstance(manifest, Mapping):
                continue
            entry = manifest.get(name)
            if entry is None:
                continue
            if isinstance(entry, Mapping):
                csv_entry = entry.get("csv")
            else:
                csv_entry = entry
            if csv_entry is None:
                continue
            csv_path = Path(csv_entry)
            if not csv_path.is_absolute():
                if manifest_parent is None:
                    manifest_parent = Path(manifest_yaml).parent
                csv_path = manifest_parent / csv_path
            splits[name] = csv_path
        if roots_map is None and isinstance(manifest, Mapping):
            roots_map = manifest.get("roots")  # type: ignore[assignment]

    if manifest_parent is None and manifest_yaml is not None:
        manifest_parent = Path(manifest_yaml).parent

    def _resolve_csv_path(path: Path) -> Path:
        if path.is_absolute():
            return path
        candidates: List[Path] = []
        if manifest_parent is not None:
            candidates.append(manifest_parent / path)
        if pack_root is not None:
            candidates.append(pack_root / path)
        candidates.append(Path.cwd() / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    result: Dict[str, SplitReturn] = {}
    used_csvs: List[Path] = []
    for name, csv_path in splits.items():
        if csv_path is None:
            continue
        csv_path = Path(csv_path)
        csv_path = _resolve_csv_path(csv_path)
        verify_hash(csv_path, manifest_yaml)
        expected_split_value: Optional[str] = None
        if split_column_name is not None and name in CANONICAL_SPLIT_NAMES:
            expected_split_value = name
        rows = load_split(
            csv_path,
            required_columns=schema_columns,
            split_column=split_column_name,
            expected_split_value=expected_split_value,
        )
        labels: Labels = [row.get("label", "") for row in rows]
        expected = expected_counts.get(name)
        if expected is not None:
            expected_rows, explicit_label_counts, fallback_label_counts = expected
            if expected_rows is not None and len(rows) != expected_rows:
                raise ValueError(
                    "Row count mismatch for split {!r} ({}): expected {}, found {}".format(
                        name, csv_path, expected_rows, len(rows)
                    )
                )
            if explicit_label_counts or fallback_label_counts:
                actual_label_counts = Counter(labels)
                enforce_full_label_set = bool(explicit_label_counts)
                if explicit_label_counts:
                    expected_label_counts = dict(explicit_label_counts)
                else:
                    expected_label_counts = dict(fallback_label_counts)
                mismatched: List[str] = []
                for label, expected_count in expected_label_counts.items():
                    actual_count = actual_label_counts.get(label, 0)
                    if actual_count != expected_count:
                        mismatched.append(
                            "{}: expected {}, found {}".format(label, expected_count, actual_count)
                        )
                if enforce_full_label_set:
                    for label in sorted(set(actual_label_counts) - set(expected_label_counts)):
                        mismatched.append(
                            "{}: expected 0, found {}".format(label, actual_label_counts[label])
                        )
                if mismatched:
                    raise ValueError(
                        "Label count mismatch for split {!r} ({}): {}".format(
                            name, csv_path, "; ".join(mismatched)
                        )
                    )
        paths = resolve_paths(rows, roots_map)
        result[name] = (paths, labels, rows)
        used_csvs.append(csv_path)

    if snapshot_dir is not None:
        manifest_snap = snapshot_dir / "manifest_snapshot"
        manifest_snap.mkdir(parents=True, exist_ok=True)
        for p in used_csvs:
            shutil.copy2(p, manifest_snap / p.name)
        if manifest_yaml is not None:
            shutil.copy2(manifest_yaml, manifest_snap / Path(manifest_yaml).name)
        if roots_map is not None:
            with open(snapshot_dir / "roots.json", "w") as f:
                json.dump(roots_map, f, indent=2)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unknown"
        with open(snapshot_dir / "git-commit.txt", "w") as f:
            f.write(commit + "\n")
        try:
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        except Exception:
            freeze = ""
        with open(snapshot_dir / "pip-freeze.txt", "w") as f:
            f.write(freeze)
        cuda_info = {"available": torch.cuda.is_available(), "version": torch.version.cuda}
        if torch.cuda.is_available():
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
        with open(snapshot_dir / "cuda.json", "w") as f:
            json.dump(cuda_info, f, indent=2)

    return result

