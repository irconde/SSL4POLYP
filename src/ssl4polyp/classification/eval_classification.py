import os
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch

from ssl4polyp import utils
from ssl4polyp.classification.data import create_classification_dataloaders
from ssl4polyp.classification.metrics import performance, thresholds as threshold_utils
from ssl4polyp.configs import data_packs_root


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
        print_mf1 = f"mF1: {mf1(pred, targ).item()}"
        print_mprec = f"mPrecision: {mprec(pred, targ).item()}"
        print_mrec = f"mRecall: {mrec(pred, targ).item()}"
        print_mauroc = f"mAUROC: {mauroc(prob, targ).item()}"
        print_acc = f"Accuracy: {(pred == targ).sum().item() / len(pred)}"
        print(print_title)
        print(print_mf1)
        print(print_mprec)
        print(print_mrec)
        print(print_mauroc)
        print(print_acc)
        if tau is not None and args.n_class == 2:
            tau_label = tau_source or "tau"
            print_tau = f"{tau_label}: {tau:.6f}"
            print(print_tau)
        results_path = Path(args.results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "a") as f:
            f.write(print_title + "\n")
            f.write(print_mf1 + "\n")
            f.write(print_mprec + "\n")
            f.write(print_mrec + "\n")
            f.write(print_mauroc + "\n")
            f.write(print_acc + "\n")
            if tau is not None and args.n_class == 2:
                f.write(f"{tau_label}: {tau:.6f}\n")

    if tau is not None:
        return {
            "tau": tau,
            "tau_source": tau_source,
        }

    return {}


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

    checkpoint_dir = Path(args.checkpoint_dir)
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
    ckpt_path = checkpoint_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Adjust --checkpoint-dir or provide --checkpoint manually."
        )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    thresholds_map = dict(checkpoint.get("thresholds", {}) or {})
    tau_json = ckpt_path.with_suffix(".thresholds.json")
    if tau_json.exists():
        try:
            thresholds_map.update(threshold_utils.load_thresholds(tau_json))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to load thresholds from {tau_json}: {exc}")

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
    test(
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
