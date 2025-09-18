import sys
import os
import argparse
import time
import random
import subprocess
from pathlib import Path
import json
import math

import yaml

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from ssl4polyp import utils
from ssl4polyp.classification.data import create_classification_dataloaders
from ssl4polyp.classification.metrics import performance
from ssl4polyp.configs import data_packs_root
from ssl4polyp.configs.layered import (
    extract_dataset_config,
    load_layered_config,
    resolve_model_entries,
)

import numpy as np


def set_determinism(seed: int) -> None:
    """Configure deterministic behavior for reproducibility.

    warn_only=True ensures operations without deterministic implementations
    raise a warning but continue running non-deterministically.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"Setting deterministic mode with seed {seed}")


def create_scheduler(optimizer, args):
    name = getattr(args, "scheduler", "none")
    if name is None:
        name = "none"
    name = name.lower()
    if name == "cosine":
        total_epochs = args.epochs
        warmup_epochs = getattr(args, "warmup_epochs", 0)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "plateau":
        min_lr = getattr(args, "min_lr", 1e-6)
        patience = getattr(args, "scheduler_patience", 2)
        factor = getattr(args, "scheduler_factor", 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    return None


def _select_model_config(models, desired_key: str | None):
    if not models:
        raise ValueError("Experiment configuration does not define any models")
    if desired_key:
        for model in models:
            key = model.get("key") or model.get("name")
            if key == desired_key or model.get("name") == desired_key:
                return model
        raise ValueError(
            f"Model key '{desired_key}' not found in experiment configuration. Available: {[m.get('key') or m.get('name') for m in models]}"
        )
    if len(models) == 1:
        return models[0]
    raise ValueError(
        "Experiment configuration defines multiple models; specify one with --model-key"
    )


def _resolve_dataset_specs(dataset_cfg: Dict[str, Any]):
    splits = dataset_cfg.get("splits", {})
    train_split = splits.get("train", "train")
    val_split = splits.get("val")
    test_split = splits.get("test")

    base_pack = dataset_cfg.get("pack")
    train_pack = dataset_cfg.get("train_pack", base_pack)
    val_pack = dataset_cfg.get("val_pack", dataset_cfg.get("base_pack", base_pack))
    test_pack = dataset_cfg.get("test_pack", dataset_cfg.get("base_pack", base_pack))

    if "train_pattern" in dataset_cfg:
        percent = dataset_cfg.get("percent")
        seed = dataset_cfg.get("seed")
        if percent is None or seed is None:
            raise ValueError(
                "Dataset configuration requires 'percent' and 'seed' values to resolve train_pattern"
            )
        train_pack = dataset_cfg["train_pattern"].format(percent=percent, seed=seed)

    if "pack_pattern" in dataset_cfg:
        size = dataset_cfg.get("size")
        seed = dataset_cfg.get("seed")
        if size is None or seed is None:
            raise ValueError(
                "Dataset configuration requires 'size' and 'seed' values to resolve pack_pattern"
            )
        resolved_pack = dataset_cfg["pack_pattern"].format(size=size, seed=seed)
        train_pack = dataset_cfg.get("train_pack", resolved_pack)
        if test_pack is None:
            test_pack = resolved_pack
        dataset_cfg.setdefault("pack", resolved_pack)

    return {
        "train_pack": train_pack,
        "val_pack": val_pack,
        "test_pack": test_pack,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
    }


def apply_experiment_config(args, experiment_cfg: Dict[str, Any]):
    dataset_cfg = extract_dataset_config(experiment_cfg)
    model_cfgs = resolve_model_entries(experiment_cfg.get("models", []))
    selected_model = _select_model_config(model_cfgs, getattr(args, "model_key", None))

    if "optimizer" in experiment_cfg and experiment_cfg["optimizer"].lower() != "adamw":
        raise ValueError("Only AdamW optimizer is currently supported")

    args.lr = experiment_cfg.get("lr", args.lr)
    args.weight_decay = experiment_cfg.get("weight_decay", getattr(args, "weight_decay", 0.05))
    args.batch_size = experiment_cfg.get("batch_size", args.batch_size)
    args.epochs = experiment_cfg.get("epochs", args.epochs)
    args.seed = experiment_cfg.get("seed", args.seed)
    args.image_size = experiment_cfg.get("image_size", args.image_size)
    args.workers = experiment_cfg.get("num_workers", args.workers)
    args.prefetch_factor = experiment_cfg.get("prefetch_factor", args.prefetch_factor)
    args.pin_memory = experiment_cfg.get("pin_memory", args.pin_memory)
    args.persistent_workers = experiment_cfg.get("persistent_workers", args.persistent_workers)
    args.log_interval = experiment_cfg.get("log_interval", args.log_interval)

    scheduler_cfg = experiment_cfg.get("scheduler")
    if isinstance(scheduler_cfg, str):
        args.scheduler = scheduler_cfg
        args.warmup_epochs = 0
    elif isinstance(scheduler_cfg, dict):
        args.scheduler = scheduler_cfg.get("name", getattr(args, "scheduler", "none"))
        args.warmup_epochs = scheduler_cfg.get("warmup_epochs", getattr(args, "warmup_epochs", 0))
        args.min_lr = scheduler_cfg.get("min_lr", getattr(args, "min_lr", 1e-6))
        args.scheduler_patience = scheduler_cfg.get(
            "patience", getattr(args, "scheduler_patience", 2)
        )
        args.scheduler_factor = scheduler_cfg.get(
            "factor", getattr(args, "scheduler_factor", 0.5)
        )

    early_cfg = experiment_cfg.get("early_stop", {}) or {}
    args.early_stop_monitor = early_cfg.get("monitor", getattr(args, "early_stop_monitor", "val_auroc"))
    args.early_stop_patience = early_cfg.get(
        "patience", getattr(args, "early_stop_patience", 0)
    )

    args.threshold_policy = experiment_cfg.get(
        "threshold_policy", getattr(args, "threshold_policy", None)
    )

    amp_enabled = experiment_cfg.get("amp")
    if amp_enabled is not None:
        args.precision = "amp" if amp_enabled else "fp32"

    if "output_dir" in experiment_cfg:
        args.output_dir = str(experiment_cfg["output_dir"])

    args.arch = selected_model.get("arch", args.arch)
    args.pretraining = selected_model.get("pretraining", args.pretraining)
    args.ckpt = selected_model.get("checkpoint", args.ckpt)
    args.frozen = selected_model.get("frozen", args.frozen)
    args.ss_framework = selected_model.get("ss_framework", args.ss_framework)
    args.dataset = dataset_cfg.get("name", args.dataset)

    dataset_resolved = _resolve_dataset_specs(dataset_cfg)
    args.train_pack = (
        str(dataset_resolved["train_pack"]) if dataset_resolved["train_pack"] else None
    )
    args.val_pack = (
        str(dataset_resolved["val_pack"]) if dataset_resolved["val_pack"] else None
    )
    args.test_pack = (
        str(dataset_resolved["test_pack"]) if dataset_resolved["test_pack"] else None
    )
    args.train_split = dataset_resolved["train_split"] or args.train_split
    args.val_split = dataset_resolved["val_split"] or args.val_split
    args.test_split = dataset_resolved["test_split"] or args.test_split

    return selected_model, dataset_cfg, dataset_resolved

def train_epoch(
    model,
    rank,
    world_size,
    train_loader,
    train_sampler,
    optimizer,
    epoch,
    loss_fn,
    log_path,
    scaler,
    use_amp,
    writer,
    log_interval,
    global_step,
    seed,
):
    t = time.time()
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(seed + epoch - 1)
    loss_accumulator = []
    for batch_idx, batch in enumerate(train_loader):
        if len(batch) == 3:
            data, target, _ = batch
        else:
            data, target = batch
        data, target = data.cuda(rank), target.cuda(rank)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(data)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        dist.all_reduce(loss)
        loss /= world_size
        loss_accumulator.append(loss.item())
        if rank == 0:
            if writer is not None and global_step % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("lr", lr, global_step)
            if batch_idx + 1 < len(train_loader):
                print(
                    "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * len(data) * world_size,
                        len(train_loader.dataset),
                        100.0 * (batch_idx + 1) / len(train_loader),
                        loss.item(),
                        time.time() - t,
                    ),
                    end="",
                )
            else:
                printout = "Train Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data) * world_size,
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
                print("\r" + printout)
                with open(log_path, "a") as f:
                    f.write(printout)
                    f.write("\n")
        dist.barrier()
        global_step += 1

    return np.mean(loss_accumulator), global_step


@torch.no_grad()
def test(
    model,
    rank,
    test_loader,
    epoch,
    perf_fn,
    log_path,
    metric_fns: Optional[Dict[str, Any]] = None,
    split_name: str = "Test",
    return_outputs: bool = False,
):
    if test_loader is None:
        return float("nan")
    t = time.time()
    model.eval()
    metric_fns = metric_fns or {}
    N = 0
    logits = None
    targets = None
    for batch_idx, batch in enumerate(test_loader):
        if len(batch) == 3:
            data, target, _ = batch
        elif len(batch) == 2:
            raise ValueError("Test dataloader does not provide labels; enable metadata return with labels.")
        else:  # pragma: no cover - defensive
            raise ValueError("Unexpected batch structure returned by dataloader")
        data, target = data.cuda(rank), target.cuda(rank)
        N += len(data)
        output = model(data).detach().cpu()
        target_cpu = target.detach().cpu()
        if batch_idx == 0:
            logits = output
            targets = target_cpu
        else:
            logits = torch.cat((logits, output), 0)
            targets = torch.cat((targets, target_cpu), 0)

        probs = torch.softmax(logits, dim=1)
        primary_metric = perf_fn(probs, targets).item()
        additional_metrics = {
            name: fn(torch.argmax(probs, dim=1), targets).item()
            for name, fn in metric_fns.items()
        }
        metrics_display = "\t".join(
            [f"AUROC: {primary_metric:.6f}"]
            + [f"{name.upper()}: {value:.6f}" for name, value in additional_metrics.items()]
        )
        if batch_idx + 1 < len(test_loader):
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\t{}\tTime: {:.6f}".format(
                    split_name,
                    epoch,
                    N,
                    len(test_loader.dataset),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    metrics_display,
                    time.time() - t,
                ),
                end="",
            )
        else:
            printout = "{}  Epoch: {} [{}/{} ({:.1f}%)]\t{}\tTime: {:.6f}".format(
                split_name,
                epoch,
                N,
                len(test_loader.dataset),
                100.0 * (batch_idx + 1) / len(test_loader),
                metrics_display,
                time.time() - t,
            )

            print("\r" + printout)
            with open(log_path, "a") as f:
                f.write(printout)
                f.write("\n")
    probs = torch.softmax(logits, dim=1)
    results = {
        "auroc": perf_fn(probs, targets).item(),
    }
    preds = torch.argmax(probs, dim=1)
    for name, fn in metric_fns.items():
        results[name] = fn(preds, targets).item()

    if return_outputs:
        results["logits"] = logits
        results["probabilities"] = probs
        results["targets"] = targets

    return results


def build(args, rank):

    pack_root = Path(args.pack_root).expanduser() if args.pack_root else None
    val_spec = args.val_pack or args.train_pack
    test_spec = args.test_pack or val_spec

    loaders, datasets, samplers = create_classification_dataloaders(
        train_spec=args.train_pack,
        val_spec=val_spec,
        test_spec=test_spec,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        rank=rank,
        world_size=args.world_size,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        pack_root=pack_root,
        roots_map=args.roots_map,
        image_size=args.image_size,
        perturbation_splits=args.perturbation_splits,
        hmac_key=args.perturbation_key.encode("utf-8"),
        snapshot_dir=Path(args.output_dir) if rank == 0 else None,
    )

    train_dataloader = loaders.get("train")
    if train_dataloader is None:
        raise RuntimeError("Training dataloader could not be constructed; check --train-pack and --train-split inputs.")
    val_dataloader = loaders.get("val")
    test_dataloader = loaders.get("test")
    if val_dataloader is None:
        raise RuntimeError("Validation dataloader missing; specify --val-pack/--val-split.")
    if test_dataloader is None:
        raise RuntimeError("Test dataloader missing; specify --test-pack/--test-split.")
    train_sampler = samplers.get("train")

    train_dataset = datasets.get(args.train_split)
    if train_dataset is None or train_dataset.labels_list is None:
        raise ValueError("Training dataset does not provide labels; ensure the selected pack includes labels.")
    train_labels = list(train_dataset.labels_list)
    n_class = len(set(train_labels))
    if n_class == 0:
        raise ValueError("No classes found in training dataset.")
    counts = np.bincount(train_labels, minlength=n_class)
    N_total = len(train_labels)
    class_weights = [
        (N_total / (n_class * count)) if count > 0 else 0.0 for count in counts
    ]

    # Override automatically computed class weights if provided by the user
    if args.class_weights is not None:
        class_weights = [float(w) for w in args.class_weights.split(",")]
        if len(class_weights) != n_class:
            raise ValueError("Number of class weights must match number of classes")

    if args.pretraining in ["Hyperkvasir", "ImageNet_self"]:
        assert os.path.exists(args.ckpt)
        model = utils.get_MAE_backbone(
            args.ckpt, True, n_class, args.frozen, None
        )
    elif args.pretraining == "ImageNet_class":
        model = utils.get_ImageNet_or_random_ViT(
            True, n_class, args.frozen, None, ImageNet_weights=True
        )
    elif args.pretraining == "random":
        model = utils.get_ImageNet_or_random_ViT(
            True, n_class, args.frozen, None, ImageNet_weights=False
        )
    base_name = f"{args.arch}-{args.pretraining}"
    if args.ss_framework:
        base_name += f"_{args.ss_framework}"
    base_name += f"_init-frozen_{str(args.frozen)}-dataset_{args.dataset}"
    ckpt_path = os.path.join(args.output_dir, base_name + ".pth")
    log_path = os.path.join(args.output_dir, base_name + ".txt")

    if os.path.exists(ckpt_path):
        main_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(main_dict["model_state_dict"])
        start_epoch = main_dict["epoch"] + 1
        best_val_perf = main_dict.get("val_perf")
        random.setstate(main_dict["py_state"])
        np.random.set_state(main_dict["np_state"])
        torch.set_rng_state(main_dict["torch_state"])
    else:
        start_epoch = 1
        best_val_perf = None
        open(log_path, "w")

    model.cuda(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0)
    )
    use_amp = args.precision == "amp"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = create_scheduler(optimizer, args)
    if best_val_perf is not None:
        optimizer.load_state_dict(main_dict["optimizer_state_dict"])
        scaler.load_state_dict(main_dict["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in main_dict:
            scheduler.load_state_dict(main_dict["scheduler_state_dict"])

    return (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        train_sampler,
        model,
        optimizer,
        ckpt_path,
        log_path,
        start_epoch,
        best_val_perf,
        class_weights,
        scaler,
        scheduler,
    )


def train(rank, args):

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=args.world_size,
        init_method="tcp://localhost:58472",
    )

    print(f"Rank {rank + 1}/{args.world_size} process initialized.\n")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        train_sampler,
        model,
        optimizer,
        ckpt_path,
        log_path,
        start_epoch,
        best_val_perf,
        class_weights,
        scaler,
        scheduler,
    ) = build(args, rank)
    use_amp = args.precision == "amp"

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).cuda(rank))
    perf_fn = performance.meanAUROC(n_class=len(class_weights))
    aux_metric_fns = {
        "f1": performance.meanF1Score(n_class=len(class_weights)),
        "precision": performance.meanPrecision(n_class=len(class_weights)),
        "recall": performance.meanRecall(n_class=len(class_weights)),
    }
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.output_dir, "tb"))
        with open(log_path, "a") as f:
            f.write(str(args))
            f.write("\n")
    else:
        writer = None
    dist.barrier()

    global_step = 0
    no_improve_epochs = 0
    scheduler_name = getattr(args, "scheduler", "none").lower()
    early_patience = getattr(args, "early_stop_patience", 0) or 0
    device = torch.device("cuda", rank)

    for epoch in range(start_epoch, args.epochs + 1):
        try:
            loss, global_step = train_epoch(
                model,
                rank,
                args.world_size,
                train_dataloader,
                train_sampler,
                optimizer,
                epoch,
                loss_fn,
                log_path,
                scaler,
                use_amp,
                writer,
                args.log_interval,
                global_step,
                args.seed,
            )
            if rank == 0:
                val_metrics = test(
                    model.module,
                    rank,
                    val_dataloader,
                    epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    split_name="Val",
                )
                test_metrics = test(
                    model.module,
                    rank,
                    test_dataloader,
                    epoch,
                    perf_fn,
                    log_path,
                    metric_fns=aux_metric_fns,
                    split_name="Test",
                )
                val_perf = val_metrics["auroc"]
                test_perf = test_metrics["auroc"]
                if writer is not None:
                    for name, value in val_metrics.items():
                        if name in {"logits", "probabilities", "targets"}:
                            continue
                        writer.add_scalar(f"val/{name}", value, epoch)
                    for name, value in test_metrics.items():
                        if name in {"logits", "probabilities", "targets"}:
                            continue
                        writer.add_scalar(f"test/{name}", value, epoch)
            else:
                val_perf = 0.0
                test_perf = 0.0

            if scheduler is not None:
                if scheduler_name == "plateau":
                    metric_tensor = torch.tensor([val_perf if rank == 0 else 0.0], device=device)
                    dist.broadcast(metric_tensor, src=0)
                    scheduler.step(metric_tensor.item())
                else:
                    scheduler.step()
            dist.barrier()
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)

        if rank == 0:
            improved = best_val_perf is None or val_perf > best_val_perf
            if improved:
                print("Saving...")
                with open(log_path, "a") as f:
                    f.write("Saving...")
                    f.write("\n")
                payload = {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": loss,
                    "val_perf": val_perf,
                    "val_auroc": val_perf,
                    "test_perf": test_perf,
                    "test_auroc": test_perf,
                    "py_state": random.getstate(),
                    "np_state": np.random.get_state(),
                    "torch_state": torch.get_rng_state(),
                }
                if scheduler is not None:
                    payload["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(payload, ckpt_path)
                best_val_perf = val_perf
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

        if early_patience > 0:
            stop_tensor = torch.tensor(
                [1 if (rank == 0 and no_improve_epochs >= early_patience) else 0],
                device=device,
            )
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                if rank == 0:
                    print("Early stopping triggered after reaching patience limit.")
                break

        dist.barrier()
    if writer is not None:
        writer.close()
    dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained model for classification"
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        dest="exp_config",
        help="Experiment configuration reference (relative to configs/ or absolute path)",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        dest="model_key",
        help="Model key to select when an experiment defines multiple models",
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
        default=None,
        choices=["Hyperkvasir", "ImageNet_class", "ImageNet_self", "random"],
    )
    parser.add_argument("--ss-framework", type=str, choices=["mae"])
    parser.add_argument("--checkpoint", type=str, dest="ckpt", default=None)
    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--train-pack",
        type=str,
        default=None,
        help="Pack specification (name, directory or manifest) providing the training split",
    )
    parser.add_argument(
        "--val-pack",
        type=str,
        default=None,
        help="Pack specification for validation split (defaults to --train-pack)",
    )
    parser.add_argument(
        "--test-pack",
        type=str,
        default=None,
        help="Pack specification for test split (defaults to --val-pack)",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
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
        help=("JSON file mapping manifest root identifiers to directories. Defaults to data/roots.json"),
    )
    parser.add_argument(
        "--perturbation-key",
        type=str,
        default="ssl4polyp",
        help="HMAC key used for deterministic per-row perturbations",
    )
    parser.add_argument(
        "--perturbation-splits",
        nargs="*",
        default=["test"],
        help="Splits (case-insensitive) where on-the-fly perturbations should be enabled",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Comma-separated list of class weights to override automatic computation",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "plateau"],
        default="none",
    )
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stop-monitor", type=str, default="val_auroc")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--threshold-policy", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Enable pinned memory when loading batches",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned memory when loading batches",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=True,
        help="Keep dataloader workers alive between epochs",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Shut down dataloader workers at the end of each epoch",
    )
    parser.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("checkpoints") / "classification"),
        dest="output_dir",
        help="Directory for checkpoints, logs and config snapshots",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = get_args()
    experiment_cfg = None
    dataset_cfg = None
    dataset_resolved = None
    selected_model = None

    if args.exp_config:
        experiment_cfg = load_layered_config(args.exp_config)
        selected_model, dataset_cfg, dataset_resolved = apply_experiment_config(args, experiment_cfg)

    for required in ("pretraining", "dataset", "train_pack"):
        if getattr(args, required) is None:
            raise ValueError(
                f"Missing required argument '{required}'. Provide it via --{required.replace('_', '-')} or the experiment config."
            )

    if not args.val_pack:
        args.val_pack = args.train_pack
    if not args.test_pack:
        args.test_pack = args.val_pack

    args.pack_root = str(Path(args.pack_root).expanduser()) if args.pack_root else str(data_packs_root())
    args.perturbation_splits = [s.lower() for s in (args.perturbation_splits or [])]
    args.scheduler = (args.scheduler or "none").lower()

    roots_path = Path(args.roots).expanduser()
    if not roots_path.exists():
        raise FileNotFoundError(
            f"Roots mapping not found at {roots_path}. Copy data/roots.example.json to data/roots.json or provide --roots explicitly."
        )
    with open(roots_path) as f:
        roots_map = json.load(f)
    args.roots_map = roots_map

    set_determinism(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tb"), exist_ok=True)

    def _serialize(value):
        if isinstance(value, dict):
            return {k: _serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_serialize(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    run_config = {
        "args": {k: _serialize(v) for k, v in vars(args).items() if k != "roots_map"},
    }
    if experiment_cfg is not None:
        run_config["experiment"] = _serialize(experiment_cfg)
    if dataset_cfg is not None and dataset_resolved is not None:
        run_config["dataset"] = {
            "configured": _serialize(dataset_cfg),
            "resolved": _serialize(dataset_resolved),
        }
    if selected_model is not None:
        run_config["model"] = _serialize(selected_model)
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        commit = None
    run_config["git_commit"] = commit
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f)
    args.world_size = torch.cuda.device_count()
    assert args.batch_size % args.world_size == 0
    mp.spawn(train, nprocs=args.world_size, args=(args,), join=True)


if __name__ == "__main__":
    main()
