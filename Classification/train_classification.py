import sys
import os
import argparse
import time
import numpy as np
import glob
import random
import subprocess

import yaml

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from Data import dataloaders
from Metrics import performance

sys.path.append("..")
import utils


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
):
    t = time.time()
    model.train()
    train_sampler.set_epoch(epoch - 1)
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
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
    t = time.time()
    model.eval()
    perf_accumulator = 0
    N = 0
    for batch_idx, (data, target) in enumerate(test_loader):
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

    simple_layout = args.simple or not os.path.exists(
        os.path.join(args.root, "labeled-images")
    )
    if simple_layout:
        class_dirs = sorted(glob.glob(os.path.join(args.root, "*/")))
        class_id = 0
        input_paths = []
        targets = []
        N_in_class = []
        N_total = 0
        for cd in class_dirs:
            contents = sorted(glob.glob(cd + "*.jpg"))
            cd_targets = [class_id for _ in range(len(contents))]
            input_paths += contents
            targets += cd_targets
            class_id += 1
            N_in_class.append(len(contents))
            N_total += len(contents)
        n_class = class_id
        class_weights = [1 / N * N_total / n_class for N in N_in_class]
    elif args.dataset.startswith("Hyperkvasir"):
        if args.dataset.endswith("pathological"):
            class_type = "pathological-findings/"
        elif args.dataset.endswith("anatomical"):
            class_type = "anatomical-landmarks/"
        base_folders = sorted(glob.glob(args.root + "/labeled-images/*/"))
        sub_folders = []
        for bf in base_folders:
            sub_folders += sorted(glob.glob(bf + "*/"))
        subsub_folders = []
        for sf in sub_folders:
            if sf.endswith(class_type):
                subsub_folders += sorted(glob.glob(sf + "*/"))
        class_id = 0
        input_paths = []
        targets = []
        N_in_class = []
        N_total = 0
        for ssf in subsub_folders:
            contents = sorted(glob.glob(ssf + "*.jpg"))
            ssf_targets = [class_id for _ in range(len(contents))]
            input_paths += contents
            targets += ssf_targets
            class_id += 1
            N_in_class.append(len(contents))
            N_total += len(contents)
        n_class = class_id
        class_weights = [1 / N * N_total / n_class for N in N_in_class]
    # Override automatically computed class weights if provided by the user
    if args.class_weights is not None:
        class_weights = [float(w) for w in args.class_weights.split(",")]
        assert len(class_weights) == n_class, "Number of class weights must match number of classes"
    train_paths = None
    val_paths = None
    test_paths = None
    if args.train_dir:
        train_paths = sorted(glob.glob(os.path.join(args.train_dir, "*.jpg")))
    if args.val_dir:
        val_paths = sorted(glob.glob(os.path.join(args.val_dir, "*.jpg")))
    if args.test_dir:
        test_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.jpg")))
    if args.train_paths is not None:
        train_paths = args.train_paths
    if args.val_paths is not None:
        val_paths = args.val_paths
    if args.test_paths is not None:
        test_paths = args.test_paths

    (
        train_dataloader,
        test_dataloader,
        val_dataloader,
        train_sampler,
    ) = dataloaders.get_dataloaders(
        rank,
        args.world_size,
        input_paths,
        targets,
        batch_size=args.batch_size // args.world_size,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
    )

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
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--simple-dataset",
        action="store_true",
        default=False,
        dest="simple",
        help="assume data_root/class_x/*.jpg structure; inferred if 'labeled-images' missing",
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--train-dir", type=str, help="directory for training images")
    parser.add_argument("--val-dir", type=str, help="directory for validation images")
    parser.add_argument("--test-dir", type=str, help="directory for test images")
    parser.add_argument("--train-paths", nargs="*", help="explicit training image paths")
    parser.add_argument("--val-paths", nargs="*", help="explicit validation image paths")
    parser.add_argument("--test-paths", nargs="*", help="explicit test image paths")
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
        "--pin-memory", action="store_true", default=True
    )
    parser.add_argument(
        "--persistent-workers", action="store_true", default=True
    )
    parser.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="Trained models", dest="output_dir")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Setting Python, NumPy and Torch seeds to {args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tb"), exist_ok=True)
    config = vars(args).copy()
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

