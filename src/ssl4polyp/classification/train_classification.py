import sys
import os
import argparse
import time
import random
import subprocess
from pathlib import Path
import json

import yaml

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
def test(model, rank, test_loader, epoch, perf_fn, log_path):
    if test_loader is None:
        return float("nan")
    t = time.time()
    model.eval()
    perf_accumulator = 0
    N = 0
    for batch_idx, batch in enumerate(test_loader):
        if len(batch) == 3:
            data, target, _ = batch
        elif len(batch) == 2:
            raise ValueError("Test dataloader does not provide labels; enable metadata return with labels.")
        else:  # pragma: no cover - defensive
            raise ValueError("Unexpected batch structure returned by dataloader")
        data, target = data.cuda(rank), target.cuda(rank)
        N += len(data)
        output = model(data)
        if batch_idx == 0:
            pred = torch.argmax(output, 1)
            targ = target
        else:
            pred = torch.cat((pred, torch.argmax(output, 1)), 0)
            targ = torch.cat((targ, target), 0)
        perf = perf_fn(pred, targ).item()
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    N,
                    len(test_loader.dataset),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    perf,
                    time.time() - t,
                ),
                end="",
            )
        else:
            printout = "Test  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                epoch,
                N,
                len(test_loader.dataset),
                100.0 * (batch_idx + 1) / len(test_loader),
                perf,
                time.time() - t,
            )

            print("\r" + printout)
            with open(log_path, "a") as f:
                f.write(printout)
                f.write("\n")
    return perf


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
        prev_best_test = main_dict["val_perf"]
        random.setstate(main_dict["py_state"])
        np.random.set_state(main_dict["np_state"])
        torch.set_rng_state(main_dict["torch_state"])
    else:
        start_epoch = 1
        prev_best_test = None
        open(log_path, "w")

    model.cuda(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = args.precision == "amp"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if prev_best_test is not None:
        optimizer.load_state_dict(main_dict["optimizer_state_dict"])
        scaler.load_state_dict(main_dict["scaler_state_dict"])

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
        prev_best_test,
        class_weights,
        scaler,
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
        prev_best_test,
        class_weights,
        scaler,
    ) = build(args, rank)
    use_amp = args.precision == "amp"

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).cuda(rank))
    perf_fn = performance.meanF1Score(n_class=len(class_weights))
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.output_dir, "tb"))
        with open(log_path, "a") as f:
            f.write(str(args))
            f.write("\n")
    else:
        writer = None
    dist.barrier()

    if args.lrs:
        val_perf_ = torch.tensor(0).cuda(rank)
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5
            )
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        if prev_best_test is not None:
            sched_dict = scheduler.state_dict()
            sched_dict["best"] = prev_best_test
            sched_dict["last_epoch"] = start_epoch - 1
            sched_dict["_last_lr"] = [lr]
            scheduler.load_state_dict(sched_dict)

    global_step = 0
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
                val_perf = test(
                    model.module, rank, val_dataloader, epoch, perf_fn, log_path
                )
                if args.lrs:
                    val_perf_ = torch.tensor(val_perf).cuda(rank)
                test_perf = test(
                    model.module, rank, test_dataloader, epoch, perf_fn, log_path
                )
            dist.barrier()
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs:
            torch.distributed.broadcast(val_perf_, 0)
            scheduler.step(val_perf_)
            if rank == 0:
                if lr != optimizer.state_dict()["param_groups"][0]["lr"]:
                    lr = optimizer.state_dict()["param_groups"][0]["lr"]
                    with open(log_path, "a") as f:
                        printout = "Epoch    {}: reducing learning rate of group 0 to {}.".format(
                            epoch, lr
                        )
                        print(printout)
                        f.write(printout)
                        f.write("\n")
        if rank == 0:

            if prev_best_test == None or val_perf > prev_best_test:
                print("Saving...")
                with open(log_path, "a") as f:
                    f.write("Saving...")
                    f.write("\n")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "loss": loss,
                        "val_perf": val_perf,
                        "test_perf": test_perf,
                        "py_state": random.getstate(),
                        "np_state": np.random.get_state(),
                        "torch_state": torch.get_rng_state(),
                    },
                    ckpt_path,
                )
                prev_best_test = val_perf
        dist.barrier()
    if writer is not None:
        writer.close()
    dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained model for classification"
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
    parser.add_argument("--checkpoint", type=str, dest="ckpt")
    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--train-pack",
        type=str,
        required=True,
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
    parser.add_argument(
        "--learning-rate-scheduler", action="store_true", default=False, dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
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
    roots_path = Path(args.roots).expanduser()
    if not roots_path.exists():
        raise FileNotFoundError(
            f"Roots mapping not found at {roots_path}. Copy data/roots.example.json to data/roots.json or provide --roots explicitly."
        )
    with open(roots_path) as f:
        roots_map = json.load(f)
    args.roots_map = roots_map

    if not args.val_pack:
        args.val_pack = args.train_pack
    if not args.test_pack:
        args.test_pack = args.val_pack

    args.pack_root = str(Path(args.pack_root).expanduser()) if args.pack_root else str(data_packs_root())
    args.perturbation_splits = [s.lower() for s in (args.perturbation_splits or [])]

    set_determinism(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tb"), exist_ok=True)
    config = vars(args).copy()
    config.pop("roots_map", None)
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        commit = None
    config["git_commit"] = commit
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)
    args.world_size = torch.cuda.device_count()
    assert args.batch_size % args.world_size == 0
    mp.spawn(train, nprocs=args.world_size, args=(args,), join=True)


if __name__ == "__main__":
    main()
