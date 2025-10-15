"""Pack-centric dataset and dataloader utilities for classification experiments."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ssl4polyp.configs import data_packs_root
from ssl4polyp.configs.manifests import load_pack

from .transforms import ClassificationTransforms, DEFAULT_HMAC_KEY


logger = logging.getLogger(__name__)


@dataclass
class PackDataset(Dataset):
    """Dataset backed by manifest-defined frame paths with metadata."""

    paths: Sequence[Path]
    labels: Optional[Sequence[object]]
    meta: Sequence[MutableMapping[str, object]]
    transform: ClassificationTransforms
    return_meta: bool = True
    provenance: Optional[Mapping[str, Any]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._paths = [str(Path(p)) for p in self.paths]
        self._labels = self._prepare_labels(self.labels)
        self._meta = [dict(row) for row in self.meta] if self.meta else [{} for _ in self._paths]
        provenance_dict = dict(self.provenance or {})
        self.provenance = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in provenance_dict.items()
        }

    @staticmethod
    def _prepare_labels(labels: Optional[Sequence[object]]) -> Optional[list[int]]:
        if labels is None:
            return None
        prepared: list[int] = []
        for label in labels:
            if label in (None, ""):
                raise ValueError("Encountered empty label while preparing PackDataset.")
            try:
                prepared.append(int(label))
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"Labels must be convertible to int, got {label!r}") from exc
        return prepared

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._paths)

    @property
    def labels_list(self) -> Optional[list[int]]:
        return self._labels

    @property
    def metadata(self) -> Sequence[MutableMapping[str, object]]:
        return self._meta

    def __getitem__(self, index: int):  # type: ignore[override]
        path = self._paths[index]
        row = self._meta[index] if index < len(self._meta) else {}
        with Image.open(path) as img:
            tensor = self.transform(img, row=row)
        label = None if self._labels is None else self._labels[index]
        if label is None:
            return tensor, row
        if self.return_meta:
            return tensor, label, row
        return tensor, label


def pack_collate(batch: Sequence[Tuple]) -> Tuple[torch.Tensor, ...]:
    if not batch:
        raise ValueError("Cannot collate empty batch")
    images: list[torch.Tensor] = []
    labels: list[int] = []
    metadata: list[MutableMapping[str, object]] = []
    has_labels = len(batch[0]) == 3
    for sample in batch:
        if has_labels:
            if len(sample) != 3:
                raise ValueError("Mixed batch with and without labels is not supported")
            image, label, row = sample
            labels.append(int(label))
        else:
            if len(sample) != 2:
                raise ValueError("Unexpected sample structure in batch")
            image, row = sample
        images.append(image)
        metadata.append(row)
    batched_images = torch.stack(images, dim=0)
    if has_labels:
        batched_labels = torch.tensor(labels, dtype=torch.long)
        return batched_images, batched_labels, metadata
    return batched_images, metadata  # type: ignore[return-value]


def _resolve_base_path(spec: str | Path, pack_root: Optional[Path]) -> Path:
    candidate = Path(spec)
    candidates = []
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if pack_root is not None:
        candidates.append(Path(pack_root) / candidate)
    default_root = data_packs_root()
    candidates.append(default_root / candidate)
    candidates.append(candidate)
    for option in candidates:
        if option.exists():
            return option
    raise FileNotFoundError(f"Could not resolve data pack '{spec}'")


def _resolve_manifest_and_csvs(
    spec: str | Path,
    splits: Iterable[str],
    pack_root: Optional[Path],
) -> Tuple[Optional[Path], Dict[str, Path]]:
    base_path = _resolve_base_path(spec, pack_root)
    splits = list(splits)
    csv_map: Dict[str, Path] = {}
    manifest: Optional[Path]
    if base_path.is_file():
        suffix = base_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            manifest = base_path
            base_dir = manifest.parent
            for split in splits:
                candidate = base_dir / f"{split}.csv"
                if not candidate.exists():
                    raise FileNotFoundError(f"Missing CSV for split '{split}' under {base_dir}")
                csv_map[split] = candidate
        elif suffix == ".csv":
            if len(splits) != 1:
                raise ValueError("CSV specification requires exactly one split")
            manifest = None
            csv_map[splits[0]] = base_path
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported pack specification '{spec}'")
    else:
        manifest = base_path / "manifest.yaml"
        if not manifest.exists():
            manifest = None
        for split in splits:
            candidate = base_path / f"{split}.csv"
            if not candidate.exists():
                raise FileNotFoundError(f"Missing CSV for split '{split}' under {base_path}")
            csv_map[split] = candidate
    return manifest, csv_map


def _normalise_stage(split: str) -> str:
    split_lower = split.lower()
    if split_lower in {"train", "training"}:
        return "train"
    if split_lower in {"val", "valid", "validation"}:
        return "val"
    return "test"


def _build_transforms(
    splits: Iterable[str],
    image_size: int,
    hmac_key: bytes,
    perturbation_splits: Sequence[str] | None,
    overrides: Optional[Mapping[str, ClassificationTransforms]] = None,
) -> Dict[str, ClassificationTransforms]:
    overrides = overrides or {}
    perturbation_set = {s.lower() for s in (perturbation_splits or [])}
    transforms_map: Dict[str, ClassificationTransforms] = {}
    for split in splits:
        if split in overrides:
            transforms_map[split] = overrides[split]
            continue
        stage = _normalise_stage(split)
        enable_perturb = split.lower() in perturbation_set if perturbation_set else False
        transforms_map[split] = ClassificationTransforms(
            stage=stage,
            image_size=image_size,
            enable_augmentations=(stage == "train"),
            enable_perturbations=enable_perturb,
            hmac_key=hmac_key,
        )
    return transforms_map


def _load_datasets(
    spec_per_split: Mapping[str, str | Path],
    *,
    pack_root: Optional[Path],
    roots_map: Optional[Mapping[str, str]],
    image_size: int,
    hmac_key: bytes,
    perturbation_splits: Sequence[str] | None,
    transform_overrides: Optional[Mapping[str, ClassificationTransforms]],
    snapshot_dir: Optional[Path],
) -> Dict[str, PackDataset]:
    datasets: Dict[str, PackDataset] = {}
    grouped: Dict[str, list[str]] = defaultdict(list)
    spec_lookup: Dict[str, str | Path] = {}
    for split, spec in spec_per_split.items():
        key = str(spec)
        grouped[key].append(split)
        spec_lookup[key] = spec
    transforms_map = _build_transforms(
        spec_per_split.keys(),
        image_size=image_size,
        hmac_key=hmac_key,
        perturbation_splits=perturbation_splits,
        overrides=transform_overrides,
    )
    snapshot_consumed = False
    csv_hash_cache: Dict[Path, str] = {}
    for spec_key, splits in grouped.items():
        manifest, csv_map = _resolve_manifest_and_csvs(spec_lookup[spec_key], splits, pack_root)
        load_kwargs = {split: csv_map[split] for split in splits}
        pack_dict = load_pack(
            manifest_yaml=manifest,
            roots_map=roots_map,
            pack_root=pack_root or data_packs_root(),
            snapshot_dir=(snapshot_dir if snapshot_dir and not snapshot_consumed else None),
            **load_kwargs,
        )
        snapshot_consumed = snapshot_consumed or snapshot_dir is not None
        for split in splits:
            if split not in pack_dict:
                raise KeyError(f"Split '{split}' not returned for pack '{spec_lookup[spec_key]}'")
            paths, labels, meta = pack_dict[split]
            csv_path = Path(csv_map[split])
            csv_hash = csv_hash_cache.get(csv_path)
            if csv_hash is None:
                hasher = hashlib.sha256()
                with open(csv_path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        if not chunk:
                            break
                        hasher.update(chunk)
                csv_hash = hasher.hexdigest()
                csv_hash_cache[csv_path] = csv_hash
            provenance = {
                "split": split,
                "csv_path": str(csv_path),
                "csv_sha256": csv_hash,
                "pack_spec": str(spec_lookup.get(spec_key, spec_key)),
            }
            if manifest is not None:
                provenance["manifest_path"] = str(manifest)
            datasets[split] = PackDataset(
                paths=paths,
                labels=labels,
                meta=meta,
                transform=transforms_map[split],
                provenance=provenance,
            )
    return datasets


def create_classification_dataloaders(
    *,
    train_spec: str | Path | None,
    val_spec: Optional[str | Path] = None,
    test_spec: Optional[str | Path] = None,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    batch_size: int = 32,
    num_workers: int = 8,
    rank: int = 0,
    world_size: int = 1,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    pack_root: Optional[Path] = None,
    roots_map: Optional[Mapping[str, str]] = None,
    image_size: int = 224,
    perturbation_splits: Optional[Sequence[str]] = None,
    hmac_key: bytes = DEFAULT_HMAC_KEY,
    transform_overrides: Optional[Mapping[str, ClassificationTransforms]] = None,
    snapshot_dir: Optional[Path] = None,
    train_drop_last: Optional[bool] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, PackDataset], Dict[str, Optional[DistributedSampler]]]:
    alias_to_split: Dict[str, str] = {}
    spec_per_split: Dict[str, str | Path] = {}
    if train_spec is not None:
        alias_to_split["train"] = train_split
        spec_per_split[train_split] = train_spec
    if val_spec is not None:
        alias_to_split["val"] = val_split
        spec_per_split[val_split] = val_spec
    if test_spec is not None:
        alias_to_split["test"] = test_split
        spec_per_split[test_split] = test_spec

    datasets = _load_datasets(
        spec_per_split,
        pack_root=pack_root,
        roots_map=roots_map,
        image_size=image_size,
        hmac_key=hmac_key,
        perturbation_splits=perturbation_splits,
        transform_overrides=transform_overrides,
        snapshot_dir=snapshot_dir,
    )

    loaders: Dict[str, DataLoader] = {}
    samplers: Dict[str, Optional[DistributedSampler]] = {}

    def _build_loader(split_alias: str, actual_split: str, drop_last: bool) -> None:
        dataset = datasets.get(actual_split)
        if dataset is None:
            return
        dataset_size = len(dataset)
        train_drop_last = drop_last if split_alias == "train" else False
        if split_alias == "train" and drop_last:
            replicas = world_size if world_size > 1 else 1
            required_examples = batch_size * replicas
            if dataset_size < required_examples:
                logger.debug(
                    "Disabling drop_last for train split '%s': %d samples < required %d",
                    actual_split,
                    dataset_size,
                    required_examples,
                )
                train_drop_last = False
        sampler: Optional[DistributedSampler] = None
        shuffle = False
        dataset_len = len(dataset)
        if split_alias == "train":
            train_drop_last_flag = drop_last
            if dataset_len < batch_size:
                train_drop_last_flag = False
            elif world_size > 1 and train_drop_last_flag:
                per_replica_samples = dataset_len // world_size
                if per_replica_samples < batch_size:
                    train_drop_last_flag = False
        else:
            train_drop_last_flag = False
        if split_alias == "train":
            if world_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    rank=rank,
                    num_replicas=world_size,
                    shuffle=True,
                    drop_last=train_drop_last_flag,
                )
            else:
                shuffle = True
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle if sampler is None else False,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": train_drop_last_flag if split_alias == "train" else False,
            "collate_fn": pack_collate,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = persistent_workers
        else:
            loader_kwargs["persistent_workers"] = False
        loader = DataLoader(**loader_kwargs)
        if split_alias == "train" and len(loader) == 0:
            raise RuntimeError(
                "Training dataloader constructed zero batches; reduce batch_size or disable drop_last. "
                f"Samples available={dataset_len}, batch_size={batch_size}, world_size={world_size}."
            )
        loaders[split_alias] = loader
        samplers[split_alias] = sampler

    train_drop_last_pref = True if train_drop_last is None else bool(train_drop_last)
    for alias, split_name in alias_to_split.items():
        drop_last_pref = train_drop_last_pref if alias == "train" else False
        _build_loader(alias, split_name, drop_last_pref)

    return loaders, datasets, samplers


__all__ = [
    "PackDataset",
    "create_classification_dataloaders",
    "pack_collate",
]
