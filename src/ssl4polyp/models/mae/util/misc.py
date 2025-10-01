# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled: bool = True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    # Expose underlying GradScaler methods for explicit gradient accumulation
    def scale(self, loss):
        return self._scaler.scale(loss)

    def step(self, optimizer):
        self._scaler.step(optimizer)

    def update(self):
        self._scaler.update()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_name = str(epoch)
    checkpoint_path = output_dir / f"checkpoint-{epoch_name}.pth"
    if loss_scaler is not None:
        to_save = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scaler": loss_scaler.state_dict(),
            "args": args,
        }
        save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag=f"checkpoint-{epoch_name}",
            client_state=client_state,
        )

    if is_main_process() and checkpoint_path.exists():
        last_symlink = output_dir / "last.pth"
        try:
            if last_symlink.is_symlink() or last_symlink.exists():
                last_symlink.unlink()
            last_symlink.symlink_to(checkpoint_path.name)
        except OSError:
            pass


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def _get_reduce_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x, dtype=torch.float64, device=_get_reduce_device())
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    return x


def all_reduce_sum(x):
    world_size = get_world_size()
    if world_size > 1:
        tensor = torch.tensor(x, dtype=torch.float64, device=_get_reduce_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()
    return x


def all_reduce_max(x):
    world_size = get_world_size()
    if world_size > 1:
        tensor = torch.tensor(x, dtype=torch.float64, device=_get_reduce_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()
    return x


def detect_grad_anomalies(parameters):
    has_nan = False
    has_inf = False
    for param in parameters:
        grad = param.grad
        if grad is None:
            continue
        if torch.isnan(grad).any():
            has_nan = True
        if torch.isinf(grad).any():
            has_inf = True
        if has_nan and has_inf:
            break
    return has_nan, has_inf


class EpochSummary:
    def __init__(self):
        self.reset()

    def reset(self):
        self._start_time = time.time()
        self._loss_sum = 0.0
        self._loss_sq_sum = 0.0
        self._loss_count = 0
        self._batch_count = 0
        self._sample_count = 0
        self._nan_loss_steps = 0
        self._inf_loss_steps = 0
        self._nan_grad_steps = 0
        self._inf_grad_steps = 0

    def update(self, loss_value, batch_size, *, grad_nan=False, grad_inf=False):
        if loss_value is None:
            return
        loss_value = float(loss_value)
        if math.isfinite(loss_value):
            self._loss_sum += loss_value
            self._loss_sq_sum += loss_value * loss_value
            self._loss_count += 1
        self._batch_count += 1
        self._sample_count += int(batch_size)

        if math.isnan(loss_value):
            self._nan_loss_steps += 1
        if math.isinf(loss_value):
            self._inf_loss_steps += 1
        if grad_nan:
            self._nan_grad_steps += 1
        if grad_inf:
            self._inf_grad_steps += 1

    def finalize(self, epoch, lr_values, *, prefix="Train"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = time.time() - self._start_time
        device = _get_reduce_device()

        stats_tensor = torch.tensor([
            self._loss_sum,
            self._loss_sq_sum,
            self._loss_count,
            self._sample_count,
            self._batch_count,
            self._nan_loss_steps,
            self._inf_loss_steps,
            self._nan_grad_steps,
            self._inf_grad_steps,
        ], dtype=torch.float64, device=device)

        if get_world_size() > 1:
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        (loss_sum, loss_sq_sum, loss_count, sample_count, batch_count,
         nan_loss_steps, inf_loss_steps, nan_grad_steps, inf_grad_steps) = stats_tensor.tolist()

        loss_count = int(round(loss_count))
        sample_count = int(round(sample_count))
        batch_count = int(round(batch_count))
        nan_loss_steps = int(round(nan_loss_steps))
        inf_loss_steps = int(round(inf_loss_steps))
        nan_grad_steps = int(round(nan_grad_steps))
        inf_grad_steps = int(round(inf_grad_steps))

        epoch_time_tensor = torch.tensor([epoch_time], dtype=torch.float64, device=device)
        if get_world_size() > 1:
            dist.all_reduce(epoch_time_tensor, op=dist.ReduceOp.MAX)
        epoch_time = epoch_time_tensor.item()

        loss_mean = loss_sum / loss_count if loss_count else float('nan')
        loss_std = 0.0
        if loss_count > 1:
            variance = (loss_sq_sum / loss_count) - (loss_mean ** 2)
            if variance < 0.0:
                variance = 0.0
            loss_std = math.sqrt(variance)

        throughput = sample_count / epoch_time if epoch_time > 0 else 0.0

        lr_values = list(lr_values) if lr_values is not None else []
        lr_values = [float(lr) for lr in lr_values]
        lr_line = "N/A"
        if lr_values:
            min_lr = min(lr_values)
            max_lr = max(lr_values)
            if math.isclose(min_lr, max_lr, rel_tol=1e-12, abs_tol=1e-12):
                lr_line = f"{min_lr:.6f}"
            else:
                lr_line = f"min={min_lr:.6f}, max={max_lr:.6f}"

        if is_main_process():
            header = f"{prefix} epoch {epoch} summary:"
            print(header)
            if loss_count:
                loss_msg = f"  loss: mean={loss_mean:.6f}"
                if loss_count > 1:
                    loss_msg += f", std={loss_std:.6f}"
            else:
                loss_msg = "  loss: mean=N/A"
            print(loss_msg)
            print(f"  lr: {lr_line}")
            print(f"  throughput: {throughput:.2f} samples/s")
            print(f"  wall clock: {epoch_time:.2f} s")
            print(f"  data: batches={batch_count}, samples={sample_count}")
            anomaly_lines = []
            if nan_loss_steps:
                anomaly_lines.append(f"    loss NaN steps: {nan_loss_steps}")
            if inf_loss_steps:
                anomaly_lines.append(f"    loss Inf steps: {inf_loss_steps}")
            if nan_grad_steps:
                anomaly_lines.append(f"    grad NaN steps: {nan_grad_steps}")
            if inf_grad_steps:
                anomaly_lines.append(f"    grad Inf steps: {inf_grad_steps}")
            if anomaly_lines:
                print("  NaN/Inf counters:")
                for line in anomaly_lines:
                    print(line)

        return {
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "throughput": throughput,
            "epoch_time": epoch_time,
            "batches": batch_count,
            "samples": sample_count,
            "nan_loss_steps": nan_loss_steps,
            "inf_loss_steps": inf_loss_steps,
            "nan_grad_steps": nan_grad_steps,
            "inf_grad_steps": inf_grad_steps,
            "lr": lr_values,
        }
