import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ssl4polyp import utils
from ssl4polyp.classification.data import create_classification_dataloaders
from ssl4polyp.classification.metrics import performance, thresholds as threshold_utils
from ssl4polyp.configs import data_packs_root


CHECKPOINT_NAME_RE = re.compile(
    r"""
    ^
    (?P<model_tag>[0-9A-Za-z]+)
    __
    (?P<data_tag>[0-9A-Za-z]+)
    (?P<qualifiers>(?:_[0-9A-Za-z]+)*)
    _s(?P<seed>\d+)
    (?:
        _e(?P<epoch>\d+)
        _(?P<best_tag>[0-9A-Za-z]+)
        (?:\+(?P<digest>[0-9A-Za-z]+))?
    )?
    $
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class CheckpointCandidate:
    """Metadata extracted from a canonical checkpoint path."""

    path: Path
    model_tag: str
    data_tag: str
    seed: int
    qualifiers: Tuple[str, ...]
    best_tag: Optional[str]
    epoch: Optional[int]
    digest: Optional[str]
    relative_dir: Tuple[str, ...]


def _normalise_canonical(tag: Optional[str]) -> Optional[str]:
    if tag is None:
        return None
    text = re.sub(r"[^0-9A-Za-z]+", "", str(tag)).lower()
    return text or None


def _normalise_path_segment(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", text.lower()).strip("_")
    return cleaned


def _normalise_pack_to_segments(pack: Optional[str]) -> set[str]:
    segments: set[str] = set()
    if not pack:
        return segments
    pieces = str(pack).replace("\\", "/").split("/")
    for piece in pieces:
        segment = _normalise_path_segment(piece)
        if segment:
            segments.add(segment)
    return segments


def _parse_checkpoint_stem(stem: str) -> Optional[Dict[str, Any]]:
    match = CHECKPOINT_NAME_RE.match(stem)
    if not match:
        return None
    qualifiers_raw = match.group("qualifiers") or ""
    qualifiers: Tuple[str, ...] = tuple(
        part for part in qualifiers_raw.split("_") if part
    )
    epoch = match.group("epoch")
    digest = match.group("digest")
    return {
        "model_tag": match.group("model_tag"),
        "data_tag": match.group("data_tag"),
        "seed": int(match.group("seed")),
        "qualifiers": qualifiers,
        "best_tag": match.group("best_tag"),
        "epoch": int(epoch) if epoch is not None else None,
        "digest": digest,
    }


def _discover_checkpoints(root: Path) -> List[CheckpointCandidate]:
    root = Path(root).expanduser()
    if not root.exists():
        return []
    seen: set[Path] = set()
    raw: list[CheckpointCandidate] = []
    for path in sorted(root.rglob("*.pth")):
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        if not resolved.exists():
            continue
        metadata = _parse_checkpoint_stem(resolved.stem)
        if not metadata:
            continue
        seen.add(resolved)
        try:
            relative_parts = path.relative_to(root).parts[:-1]
        except ValueError:
            try:
                relative_parts = resolved.relative_to(root).parts[:-1]
            except ValueError:
                relative_parts = resolved.parent.parts
        candidate = CheckpointCandidate(
            path=resolved,
            relative_dir=tuple(relative_parts),
            **metadata,
        )
        raw.append(candidate)

    grouped: Dict[Tuple[str, str, int], List[CheckpointCandidate]] = {}
    for item in raw:
        key = (item.model_tag, item.data_tag, item.seed)
        grouped.setdefault(key, []).append(item)

    result: List[CheckpointCandidate] = []
    for items in grouped.values():
        with_best = [entry for entry in items if entry.best_tag]
        if with_best:
            result.extend(with_best)
        else:
            result.extend(items)

    result.sort(
        key=lambda c: (
            _normalise_canonical(c.model_tag) or "",
            _normalise_canonical(c.data_tag) or "",
            c.seed,
            c.epoch if c.epoch is not None else -1,
            str(c.path),
        )
    )
    return result


def _format_candidate(candidate: CheckpointCandidate) -> str:
    parts = [candidate.model_tag, candidate.data_tag, f"s{candidate.seed}"]
    if candidate.best_tag:
        parts.append(candidate.best_tag)
    rel_dir = "/".join(candidate.relative_dir)
    descriptor = ", ".join(parts)
    if rel_dir:
        return f"{candidate.path} ({rel_dir}: {descriptor})"
    return f"{candidate.path} ({descriptor})"


def _filter_candidates(
    candidates: Iterable[CheckpointCandidate],
    *,
    model_filter: Optional[str],
    data_filter: Optional[str],
    seed_filter: Optional[int],
    best_filter: Optional[str],
    pack_filters: set[str],
) -> List[CheckpointCandidate]:
    filtered: List[CheckpointCandidate] = []
    pack_matched: List[CheckpointCandidate] = []
    pack_unmatched: List[CheckpointCandidate] = []
    for candidate in candidates:
        if model_filter and (
            _normalise_canonical(candidate.model_tag) != model_filter
        ):
            continue
        if data_filter and (
            _normalise_canonical(candidate.data_tag) != data_filter
        ):
            continue
        if seed_filter is not None and candidate.seed != seed_filter:
            continue
        if best_filter is not None:
            best_tag = _normalise_canonical(candidate.best_tag)
            if best_tag != best_filter:
                continue
        if pack_filters:
            segments = {
                _normalise_path_segment(part) for part in candidate.relative_dir
            }
            if segments & pack_filters:
                pack_matched.append(candidate)
            else:
                pack_unmatched.append(candidate)
                continue
        filtered.append(candidate)
    if pack_filters:
        if pack_matched:
            return pack_matched
        return pack_unmatched
    return filtered


def _compose_legacy_checkpoint_path(args, checkpoint_dir: Path) -> Path:
    if args.ss_framework:
        ckpt_name = (
            f"{args.arch}-{args.pretraining}_{args.ss_framework}_"
            f"init-frozen_{str(False)}-dataset_{args.dataset}.pth"
        )
    else:
        ckpt_name = (
            f"{args.arch}-{args.pretraining}_"
            f"init-frozen_{str(False)}-dataset_{args.dataset}.pth"
        )
    return checkpoint_dir / ckpt_name


def _sanitize_path_segment(raw: Any, *, default: str = "default") -> str:
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return default
    text = text.strip("/ ")
    if "/" in text:
        text = text.split("/")[-1]
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("._-")
    return cleaned.lower() if cleaned else default


def _search_int(pattern: str, *candidates: Optional[str]) -> Optional[int]:
    for text in candidates:
        if not text:
            continue
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _resolve_thresholds_subdir(args, checkpoint: CheckpointCandidate) -> str:
    if getattr(args, "threshold_pack", None):
        segment = _sanitize_path_segment(args.threshold_pack, default="")
        if segment and segment != "default":
            return segment

    dataset_name = getattr(args, "threshold_dataset", None) or getattr(
        args, "dataset", None
    )
    dataset_key = str(dataset_name or "").lower()
    parts: List[str] = []

    if dataset_key == "sun_full":
        parts.append("sun")
    elif dataset_key:
        segment = _sanitize_path_segment(dataset_key, default="dataset")
        if segment and segment != "default":
            parts.append(segment)

    percent = _search_int(
        r"p(\d+)", args.threshold_pack, args.test_pack, checkpoint.data_tag
    )
    dataset_seed = _search_int(r"seed(\d+)", args.threshold_pack, args.test_pack)
    size = _search_int(
        r"few(\d+)", args.threshold_pack, args.test_pack, checkpoint.data_tag
    )
    if size is None:
        size = _search_int(r"_s(\d+)", args.threshold_pack, args.test_pack)

    if dataset_key.startswith("polypgen_fewshot"):
        if size is not None:
            parts.append(f"c{int(size)}")
        if dataset_seed is not None:
            parts.append(f"s{int(dataset_seed)}")
    elif dataset_key.startswith("sun_subsets"):
        if percent is not None:
            parts.append(f"p{int(percent)}")
        if dataset_seed is not None:
            parts.append(f"s{int(dataset_seed)}")

    if not parts:
        fallback = _sanitize_path_segment(checkpoint.data_tag, default="dataset")
        if fallback:
            parts.append(fallback)

    if len(parts) == 1:
        split = getattr(args, "threshold_split", None) or "val"
        parts.append(_sanitize_path_segment(split, default="val"))

    return "_".join(parts)


def _load_threshold_metadata(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Threshold metadata must be a JSON object")
    if "selected_threshold" not in payload:
        raise ValueError("Threshold metadata missing 'selected_threshold'")
    try:
        selected_threshold = float(payload["selected_threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected threshold must be numeric") from exc

    metadata: Dict[str, Any] = dict(payload)
    metadata["selected_threshold"] = selected_threshold

    threshold_key = metadata.get("threshold_key")
    if threshold_key is not None:
        metadata["threshold_key"] = str(threshold_key)

    thresholds_raw = metadata.get("thresholds") or {}
    if thresholds_raw and not isinstance(thresholds_raw, dict):
        raise ValueError("Threshold metadata 'thresholds' field must be an object")
    thresholds_map: Dict[str, float] = {}
    for key, value in (thresholds_raw or {}).items():
        try:
            thresholds_map[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid threshold value for key {key!r}: {value!r}"
            ) from exc

    metadata["thresholds"] = thresholds_map

    key = metadata.get("threshold_key")
    if key and key not in thresholds_map:
        thresholds_map[key] = selected_threshold

    if "metrics_at_threshold" in metadata:
        metrics = metadata["metrics_at_threshold"]
        if metrics is not None and not isinstance(metrics, dict):
            raise ValueError("'metrics_at_threshold' must be an object")
        if isinstance(metrics, dict):
            cleaned_metrics: Dict[str, Optional[float]] = {}
            for name, value in metrics.items():
                if value is None:
                    cleaned_metrics[str(name)] = None
                    continue
                try:
                    cleaned_metrics[str(name)] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid metric value for key {name!r}: {value!r}"
                    ) from exc
            metadata["metrics_at_threshold"] = cleaned_metrics

    if "seed" in metadata and metadata["seed"] is not None:
        try:
            metadata["seed"] = int(metadata["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Threshold metadata seed must be an integer") from exc

    metadata["path"] = Path(path)
    return metadata


def _find_threshold_metadata(
    root: Path,
    model_tag: str,
    *,
    expected_directory: Optional[str],
    pack_filters: set[str],
) -> List[Dict[str, Any]]:
    root = Path(root).expanduser()
    pattern = f"{model_tag}.json"
    matches: List[Dict[str, Any]] = []
    for path in sorted(root.rglob(pattern)):
        try:
            metadata = _load_threshold_metadata(path)
        except ValueError as exc:
            raise ValueError(f"Failed to load threshold metadata from {path}: {exc}") from exc

        model_in_file = metadata.get("model_tag")
        if model_in_file and (
            _normalise_canonical(model_in_file)
            != _normalise_canonical(model_tag)
        ):
            continue

        directory = metadata.get("directory")
        if expected_directory and directory:
            if _normalise_path_segment(directory) != _normalise_path_segment(
                expected_directory
            ):
                continue

        if pack_filters:
            candidate_pack = metadata.get("data_pack")
            if candidate_pack:
                candidate_segments = _normalise_pack_to_segments(candidate_pack)
                if not candidate_segments & pack_filters:
                    continue

        matches.append(metadata)

    return matches


def _resolve_threshold_metadata(
    args,
    checkpoint: Optional[CheckpointCandidate],
    checkpoint_dir: Path,
) -> Optional[Dict[str, Any]]:
    if checkpoint is None:
        return None
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    candidates = [checkpoint_dir / "thresholds", checkpoint_dir.parent / "thresholds"]
    thresholds_root: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            thresholds_root = candidate
            break
    if thresholds_root is None:
        thresholds_root = candidates[0]

    expected_subdir = _resolve_thresholds_subdir(args, checkpoint)
    candidate_path = thresholds_root / expected_subdir / f"{checkpoint.model_tag}.json"

    metadata: Optional[Dict[str, Any]] = None
    if candidate_path.exists():
        metadata = _load_threshold_metadata(candidate_path)
    else:
        pack_filters = _normalise_pack_to_segments(getattr(args, "threshold_pack", None))
        matches = _find_threshold_metadata(
            thresholds_root,
            checkpoint.model_tag,
            expected_directory=expected_subdir,
            pack_filters=pack_filters,
        )
        if len(matches) == 1:
            metadata = matches[0]
        elif len(matches) > 1:
            options = ", ".join(str(m["path"]) for m in matches)
            raise RuntimeError(
                "Multiple threshold metadata files matched for model tag "
                f"'{checkpoint.model_tag}': {options}"
            )

    return metadata


@torch.no_grad()
def test(
    model,
    device,
    test_loader,
    args,
    save_preds: Optional[Path] = None,
    tau: Optional[float] = None,
    tau_source: Optional[str] = None,
):
    model.eval()
    mf1 = performance.meanF1Score(n_class=args.n_class)
    mprec = performance.meanPrecision(n_class=args.n_class)
    mrec = performance.meanRecall(n_class=args.n_class)
    mauroc = performance.meanAUROC(n_class=args.n_class)
    mbalanced = performance.meanBalancedAccuracy(n_class=args.n_class)
    mauprc = performance.meanAUPRC(n_class=args.n_class)
    frame_ids = []
    preds_list = []
    apply_tau = tau is not None and args.n_class == 2
    warned_tau = False
    for i, batch in enumerate(test_loader):
        if len(batch) != 3:
            raise ValueError("Expected batches to provide (images, labels, metadata)")
        data, target, meta = batch
        frame_ids.extend([m.get("frame_id") for m in meta])
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        probs = torch.softmax(output, dim=1)
        if tau is not None and args.n_class != 2 and not warned_tau:
            print("Warning: provided tau but dataset is not binary; ignoring threshold.")
            warned_tau = True
        if apply_tau:
            pred_batch = (probs[:, 1] >= tau).long()
        else:
            pred_batch = torch.argmax(probs, 1)
        preds_list.extend(pred_batch.cpu().tolist())
        if i == 0:
            pred = pred_batch
            prob = probs
            targ = target
        else:
            pred = torch.cat((pred, pred_batch), 0)
            prob = torch.cat((prob, probs), 0)
            targ = torch.cat((targ, target), 0)

    if save_preds is not None and frame_ids:
        save_path = Path(save_preds)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "prediction"])
            for fid, p in zip(frame_ids, preds_list):
                writer.writerow([fid, p])

    metrics: dict[str, Any] = {}
    if len(frame_ids) > 0:
        if args.ss_framework:
            name = (
                f"{args.arch}-{args.pretraining}_{args.ss_framework}_"
                f"init-frozen_{str(False)}-dataset_{args.dataset}"
            )
        else:
            name = (
                f"{args.arch}-{args.pretraining}_"
                f"init-frozen_{str(False)}-dataset_{args.dataset}"
            )
        print_title = f"Classification results for {name} @ {datetime.now()}"
        metrics["balanced_accuracy"] = mbalanced(
            prob, targ, tau=tau if apply_tau else None
        ).item()
        metrics["mF1"] = mf1(pred, targ).item()
        metrics["mPrecision"] = mprec(pred, targ).item()
        metrics["mRecall"] = mrec(pred, targ).item()
        metrics["mAUROC"] = mauroc(prob, targ).item()
        metrics["mAUPRC"] = mauprc(prob, targ).item()
        metrics["accuracy"] = (pred == targ).sum().item() / len(pred)
        total_samples = targ.numel()
        if total_samples == 0:
            prevalence_value: Any = 0.0 if args.n_class == 2 else [0.0] * args.n_class
        elif args.n_class == 2:
            positive = (targ == 1).sum().item()
            prevalence_value = positive / total_samples
        else:
            counts = torch.bincount(targ.view(-1), minlength=args.n_class)
            prevalence_value = [
                counts[class_index].item() / total_samples for class_index in range(args.n_class)
            ]
        metrics["prevalence"] = prevalence_value
        prevalence_serialised = json.dumps(prevalence_value)
        print_balanced = f"Balanced accuracy: {metrics['balanced_accuracy']}"
        print_mf1 = f"mF1: {metrics['mF1']}"
        print_mprec = f"mPrecision: {metrics['mPrecision']}"
        print_mrec = f"mRecall: {metrics['mRecall']}"
        print_mauroc = f"mAUROC: {metrics['mAUROC']}"
        print_mauprc = f"mAUPRC: {metrics['mAUPRC']}"
        print_acc = f"Accuracy: {metrics['accuracy']}"
        print_prev = f"Prevalence: {prevalence_serialised}"
        print(print_title)
        print(print_balanced)
        print(print_mf1)
        print(print_mprec)
        print(print_mrec)
        print(print_mauroc)
        print(print_mauprc)
        print(print_acc)
        print(print_prev)
        if tau is not None and args.n_class == 2:
            tau_label = tau_source or "tau"
            print_tau = f"{tau_label}: {tau:.6f}"
            print(print_tau)
        results_path = Path(args.results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "a") as f:
            f.write(print_title + "\n")
            f.write(print_balanced + "\n")
            f.write(print_mf1 + "\n")
            f.write(print_mprec + "\n")
            f.write(print_mrec + "\n")
            f.write(print_mauroc + "\n")
            f.write(print_mauprc + "\n")
            f.write(print_acc + "\n")
            f.write(print_prev + "\n")
            if tau is not None and args.n_class == 2:
                f.write(f"{tau_label}: {tau:.6f}\n")

    result: dict[str, object] = {"metrics": metrics}
    if tau is not None:
        result.update({
            "tau": tau,
            "tau_source": tau_source,
        })

    return result


def build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roots_path = Path(args.roots).expanduser()
    if not roots_path.exists():
        raise FileNotFoundError(
            f"Roots mapping not found at {roots_path}. Copy data/roots.example.json to data/roots.json or provide --roots explicitly."
        )
    with open(roots_path) as f:
        roots_map = json.load(f)

    loaders, datasets, _ = create_classification_dataloaders(
        train_spec=None,
        val_spec=args.threshold_pack,
        test_spec=args.test_pack,
        train_split="train",
        val_split=args.threshold_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        pack_root=Path(args.pack_root).expanduser(),
        roots_map=roots_map,
        image_size=args.image_size,
        perturbation_splits=args.perturbation_splits,
        hmac_key=args.perturbation_key.encode("utf-8"),
    )

    test_dataloader = loaders.get("test")
    if test_dataloader is None:
        raise RuntimeError("Test dataloader missing; check --test-pack/--test-split inputs.")

    dataset = datasets.get(args.test_split)
    if dataset is None or dataset.labels_list is None:
        raise ValueError("Selected test pack does not provide labels; cannot compute metrics.")
    args.n_class = len(set(dataset.labels_list))

    threshold_loader = loaders.get("val") if args.threshold_pack else None

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        model = utils.get_MAE_backbone(None, True, args.n_class, False, None)
    elif args.pretraining == "ImageNet_class":
        model = utils.get_ImageNet_or_random_ViT(
            True, args.n_class, False, None, ImageNet_weights=True
        )
    elif args.pretraining == "random":
        model = utils.get_ImageNet_or_random_ViT(
            True, args.n_class, False, None, ImageNet_weights=False
        )
    else:
        raise ValueError(f"Unsupported pretraining option '{args.pretraining}'")

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    candidates = _discover_checkpoints(checkpoint_dir)
    selected_candidate: Optional[CheckpointCandidate] = None
    if candidates:
        model_filter = _normalise_canonical(getattr(args, "model_tag", None))
        data_filter = _normalise_canonical(getattr(args, "data_tag", None))
        seed_filter = getattr(args, "seed", None)
        best_filter = _normalise_canonical(getattr(args, "best_tag", None))
        pack_filters = _normalise_pack_to_segments(getattr(args, "test_pack", None))

        filtered = _filter_candidates(
            candidates,
            model_filter=model_filter,
            data_filter=data_filter,
            seed_filter=seed_filter,
            best_filter=best_filter,
            pack_filters=pack_filters,
        )
        if not filtered:
            available = "\n".join(f"  - {_format_candidate(c)}" for c in candidates)
            raise ValueError(
                "No canonical checkpoints matched the provided filters. "
                "Supply --model-tag/--data-tag/--seed/--best-tag to disambiguate. "
                f"Available checkpoints:\n{available}"
            )
        if len(filtered) > 1:
            options = "\n".join(f"  - {_format_candidate(c)}" for c in filtered)
            raise ValueError(
                "Multiple checkpoints matched the requested filters. "
                "Refine your selection with --model-tag, --data-tag, --seed or --best-tag. "
                f"Matches:\n{options}"
            )
        selected_candidate = filtered[0]
        ckpt_path = selected_candidate.path
    else:
        ckpt_path = _compose_legacy_checkpoint_path(args, checkpoint_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. Adjust filters or --checkpoint-dir."
            )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    thresholds_map = dict(checkpoint.get("thresholds", {}) or {})
    threshold_metadata = _resolve_threshold_metadata(args, selected_candidate, checkpoint_dir)
    if threshold_metadata:
        thresholds_map.update(threshold_metadata.get("thresholds", {}))
    tau_json = ckpt_path.with_suffix(".thresholds.json")
    if tau_json.exists():
        try:
            thresholds_map.update(threshold_utils.load_thresholds(tau_json))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to load thresholds from {tau_json}: {exc}")

    threshold_files = checkpoint.get("threshold_files") or {}
    if isinstance(threshold_files, dict) and threshold_files:
        thresholds_root = checkpoint.get("thresholds_root")
        if thresholds_root:
            base_root = Path(thresholds_root).expanduser().parent
        else:
            base_root = ckpt_path.parent
        for key, rel_path in threshold_files.items():
            candidate = Path(rel_path)
            if not candidate.is_absolute():
                candidate = base_root / rel_path
            if not candidate.exists():
                print(
                    f"Warning: threshold file referenced for {key!r} missing at {candidate}"
                )
                continue
            try:
                loaded = threshold_utils.load_thresholds(candidate)
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"Warning: failed to load thresholds from {candidate}: {exc}"
                )
                continue
            thresholds_map.update(loaded)

    return test_dataloader, threshold_loader, model, device, thresholds_map, ckpt_path


def _resolve_tau(
    args,
    model: torch.nn.Module,
    device: torch.device,
    threshold_loader,
    thresholds_map,
) -> Tuple[Optional[float], Optional[str]]:
    raw_policy = (args.threshold_policy or "auto").strip().lower()
    if raw_policy not in {"auto", "youden", "none"}:
        raise ValueError(
            f"Unsupported threshold policy '{raw_policy}'. Use 'auto', 'youden' or 'none'."
        )
    policy = raw_policy
    if policy == "auto":
        policy = "youden" if args.n_class == 2 else "none"
        if policy == "none" and args.threshold_pack:
            print(
                "Warning: threshold policy resolved to 'none'; ignoring provided threshold pack."
            )
    if policy == "none":
        args.threshold_policy = policy
        return None, None

    tau: Optional[float] = None
    tau_source: Optional[str] = None
    if args.threshold_pack:
        if threshold_loader is None:
            raise RuntimeError(
                "Threshold pack specified but validation loader could not be constructed"
            )
        tau = threshold_utils.compute_threshold_from_loader(
            model, threshold_loader, device, policy=policy
        )
        tau_source = f"refit:{args.threshold_pack}:{args.threshold_split}:{policy}"
    else:
        threshold_dataset = args.threshold_dataset or args.dataset
        key = args.threshold_key
        if key is None and threshold_dataset:
            key = threshold_utils.format_threshold_key(
                threshold_dataset, args.threshold_split, policy
            )
        tau = threshold_utils.resolve_threshold(thresholds_map, key)
        if tau is not None:
            tau_source = key
        elif key is not None:
            print(
                f"Warning: threshold '{key}' not found in checkpoint; proceeding with argmax predictions."
            )
    args.threshold_policy = policy
    return tau, tau_source


def evaluate(args):
    args.perturbation_splits = [s.lower() for s in (args.perturbation_splits or [])]
    (
        test_dataloader,
        threshold_loader,
        model,
        device,
        thresholds_map,
        _,
    ) = build(args)
    tau, tau_source = _resolve_tau(args, model, device, threshold_loader, thresholds_map)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    return test(
        model,
        device,
        test_dataloader,
        args,
        save_preds=args.predictions,
        tau=tau,
        tau_source=tau_source,
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned classification model"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["vit_b"],
        default="vit_b",
        dest="arch",
    )
    parser.add_argument(
        "--pretraining",
        type=str,
        required=True,
        choices=["Hyperkvasir", "ImageNet_class", "ImageNet_self", "random"],
    )
    parser.add_argument("--ss-framework", type=str, choices=["mae"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Canonical model tag embedded in the checkpoint filename",
    )
    parser.add_argument(
        "--data-tag",
        type=str,
        default=None,
        help="Canonical data tag embedded in the checkpoint filename",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed identifier to match within the checkpoint filename",
    )
    parser.add_argument(
        "--best-tag",
        type=str,
        default=None,
        help="Monitor tag associated with the best checkpoint selection",
    )
    parser.add_argument("--test-pack", type=str, required=True)
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument(
        "--threshold-dataset",
        type=str,
        default=None,
        help="Dataset name associated with stored thresholds (defaults to --dataset)",
    )
    parser.add_argument(
        "--threshold-pack",
        type=str,
        default=None,
        help="Pack specification used to re-fit the decision threshold",
    )
    parser.add_argument(
        "--threshold-split",
        type=str,
        default="val",
        help="Split name associated with the threshold (lookup or re-fit)",
    )
    parser.add_argument(
        "--threshold-policy",
        type=str,
        default="auto",
        choices=["auto", "youden", "none"],
        help="Threshold policy to apply when converting probabilities to labels",
    )
    parser.add_argument(
        "--threshold-key",
        type=str,
        default=None,
        help="Explicit key to load from the checkpoint threshold mapping",
    )
    parser.add_argument(
        "--pack-root",
        type=str,
        default=str(data_packs_root()),
        help="Base directory containing data packs (defaults to repository data_packs/)",
    )
    parser.add_argument(
        "--roots",
        type=str,
        default=str(Path("data") / "roots.json"),
        help="JSON file mapping manifest roots to concrete directories",
    )
    parser.add_argument(
        "--perturbation-key",
        type=str,
        default="ssl4polyp",
        help="HMAC key for deterministic per-row perturbations",
    )
    parser.add_argument(
        "--perturbation-splits",
        nargs="*",
        default=["test"],
        help="Splits (case-insensitive) where on-the-fly perturbations should be enabled",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Enable pinned memory for dataloading",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned memory",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=True,
        help="Keep worker processes alive between epochs",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Shut down dataloader workers after each epoch",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path("checkpoints")),
        help="Directory containing fine-tuned classification checkpoints",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Optional CSV path to store frame-level predictions",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=str(Path("..") / "eval_results.txt"),
        help="File where evaluation summaries are appended",
    )
    return parser.parse_args()


def main():
    args = get_args()
    evaluate(args)


if __name__ == "__main__":
    main()
