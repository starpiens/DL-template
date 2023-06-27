import argparse
import os
from typing import Callable, List
from datetime import datetime
from argparse import Namespace

import tqdm
import yaml
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from forge import prepare_training
from utils import AverageMeter, MinMeter


def train(
    loader: DataLoader,
    model: Module,
    optim: Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    criterion_fns: List[Callable[[Tensor, Tensor], int | float]] = [],
) -> List[float]:
    model.train()
    results = [AverageMeter() for _ in range(len(criterion_fns))]

    for idx, (inp, tar) in enumerate(tqdm.tqdm(loader, desc="Training  ", ncols=80)):
        # Forward pass.
        inp = inp.cuda()
        tar = tar.cuda()
        out = model(inp)
        loss = loss_fn(out, tar)

        # Backward pass.
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Update training results.
        num_items = inp.shape[0]
        for i in range(len(criterion_fns)):
            with torch.no_grad():
                result = criterion_fns[i](out, tar)
            if type(result) is torch.Tensor:
                result = result.item()
            results[i].update(result, num_items)

    return [i.avg for i in results]


@torch.no_grad()
def validate(
    loader: DataLoader,
    model: Module,
    criterion_fns: List[Callable[[Tensor, Tensor], int | float]] = [],
) -> List[float]:
    model.eval()
    results = [AverageMeter() for _ in range(len(criterion_fns))]

    for idx, (inp, tar) in enumerate(tqdm.tqdm(loader, desc="Validating", ncols=80)):
        inp = inp.cuda()
        tar = tar.cuda()
        out = model(inp)

        # Update validation results.
        num_items = inp.shape[0]
        for i in range(len(criterion_fns)):
            result = criterion_fns[i](out, tar)
            if type(result) is torch.Tensor:
                result = result.item()
            results[i].update(result, num_items)

    return [i.avg for i in results]


def save_checkpoint(
    model: Module,
    optim: Optimizer,
    sched: LRScheduler | None,
    epoch: int,
    cfg: dict,
    args: Namespace,
    run_name: str,
    sv_path: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
            "epoch": epoch,
            "cfg": cfg,
            "tb_log_dir": args.tb_log_dir,
            "save_dir": args.save_dir,
            "run_name": run_name,
        },
        sv_path,
    )


def main():
    # Get args.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config .yaml file. "
        "This option is ignored when using option '--resume'.",
    )
    parser.add_argument(
        "--gpu", type=str, default=None, help="Indicies of gpus to be used."
    )
    parser.add_argument(
        "--tb_log_dir",
        metavar="DIR",
        type=str,
        default="runs/",
        help="Tensorboard save directory location. "
        "This option is ignored when using option '--resume'.",
    )
    parser.add_argument(
        "--save_dir",
        metavar="DIR",
        type=str,
        default="save/",
        help="PyTorch save directory location. "
        "This option is ignored when using option '--resume'.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from specified saved file. "
        "If specified, options '--config', '--tb_log_dir' and '--save_dir' "
        "are being ignored, and saved values are being used.",
    )
    args = parser.parse_args()

    # Setup environment.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    if args.gpu is not None:
        print(f"Using GPU(s): {args.gpu}.", flush=True)
    else:
        print(f"Using all {torch.cuda.device_count()} GPU(s).", flush=True)
    cudnn.benchmark = True

    # Prepare training from the beginning.
    if args.resume is None:
        # Load config.
        with open(args.config, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        if cfg is not None:
            print(f"Loaded config at: {f.name}", flush=True)
        else:
            raise IOError("Failed to load config.")
        (train_loader, val_loader, model, optim, sched, loss_fn) = prepare_training(cfg)

        start_epoch = 1
        now = datetime.now()
        run_name = "{}_{}".format(
            now.strftime("%y%m%d_%H%M%S"),
            os.path.splitext(os.path.basename(args.config))[0],
        )

    # Prepare training from the checkpoint.
    else:
        checkpoint = torch.load(args.resume)
        cfg = checkpoint["cfg"]
        (train_loader, val_loader, model, optim, sched, loss_fn) = prepare_training(cfg)

        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        if sched is not None:
            sched.load_state_dict(checkpoint["sched"])

        args.tb_log_dir = checkpoint["tb_log_dir"]
        args.save_dir = checkpoint["save_dir"]
        run_name = checkpoint["run_name"]

    print(
        "\nRun name: {}".format(run_name),
        "Prepared training with:",
        '\tTrain data: class "{}" from "{}" (num: {}).'.format(
            train_loader.dataset.__class__.__name__,
            train_loader.dataset.__class__.__module__,
            len(train_loader.dataset),  # type: ignore
        ),
        '\tVal data:   class "{}" from "{}" (num: {}).'.format(
            val_loader.dataset.__class__.__name__,
            val_loader.dataset.__class__.__module__,
            len(val_loader.dataset),  # type: ignore
        ),
        '\tModel:      class "{}" from "{}".'.format(
            model.module.__class__.__name__, model.module.__class__.__module__
        ),
        '\tOptimizer:  class "{}" from "{}".'.format(
            optim.__class__.__name__, optim.__class__.__module__
        ),
        '\tScheduler:  class "{}" from "{}".'.format(
            sched.__class__.__name__, sched.__class__.__module__
        ),
        '\tLoss:       class "{}" from "{}".'.format(
            loss_fn.__class__.__name__, loss_fn.__class__.__module__
        ),
        "",
        sep="\n",
    )

    tb_log_path = os.path.join(args.tb_log_dir, run_name)
    best_save_path = os.path.join(args.save_dir, run_name, "best.pth")
    last_save_path = os.path.join(args.save_dir, run_name, "last.pth")
    os.makedirs(os.path.join(args.save_dir, run_name), exist_ok=True)
    writer = SummaryWriter(tb_log_path)
    writer.add_text("configs", str(cfg))
    print(f"Tensorboard log path: {tb_log_path}")
    print(f"Best model save path: {best_save_path}")
    print(f"Last model save path: {last_save_path}")
    print()

    train_loss = MinMeter()
    val_loss = MinMeter()
    print(f"Starting training from epoch {start_epoch}...", flush=True)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        print(f'Epoch {epoch}/{cfg["epochs"]}')

        # Train for an epoch.
        (loss,) = train(
            train_loader,
            model,
            optim,
            loss_fn,
            criterion_fns=[loss_fn],
        )
        train_loss.update(loss)
        print("\tLoss:      {} (Best: {})".format(loss, train_loss.min))
        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("train/loss (best)", train_loss.min, epoch)

        # Validate.
        (loss,) = validate(
            val_loader,
            model,
            criterion_fns=[loss_fn],
        )
        val_loss.update(loss)
        print("\tLoss:      {} (Best: {})".format(loss, val_loss.min))
        writer.add_scalar("val/loss", loss, epoch)
        writer.add_scalar("val/loss (best)", val_loss.min, epoch)

        if sched is not None:
            sched.step()
        writer.flush()
        print()

        if val_loss.last == val_loss.min:
            save_checkpoint(
                model, optim, sched, epoch + 1, cfg, args, run_name, best_save_path
            )
        save_checkpoint(
            model, optim, sched, epoch + 1, cfg, args, run_name, last_save_path
        )


if __name__ == "__main__":
    main()
