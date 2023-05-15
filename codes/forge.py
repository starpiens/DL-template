import data
import models
import losses

import torch
from torch.utils.data import DataLoader
import torchvision


def create_dataset(dataset_cfg: dict) \
        -> torch.utils.data.Dataset:
    """Creates a dataset."""
    name = dataset_cfg['name']
    args = dataset_cfg.get('args', None)
    try:
        dataset_cls = getattr(data, name)
    except AttributeError:
        dataset_cls = getattr(torchvision.datasets, name)
    dataset = dataset_cls(**(args or {}))
    return dataset


def create_train_loader(train_dataset_cfg: dict) \
        -> torch.utils.data.DataLoader:
    """Creates a data loader for the training."""
    dataset = create_dataset(train_dataset_cfg)
    batch_size = train_dataset_cfg.get('batch_size', 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)
    return loader


def create_val_loader(val_dataset_cfg: dict) \
        -> torch.utils.data.DataLoader:
    """Creates a data loader for the validation."""
    dataset = create_dataset(val_dataset_cfg)
    batch_size = val_dataset_cfg.get('batch_size', 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True, drop_last=False)
    return loader


def create_model(model_cfg: dict) \
        -> torch.nn.Module:
    """Creates a model."""
    name = model_cfg['name']
    args = model_cfg.get('args', None)
    try:
        model_cls = getattr(models, name)
    except AttributeError:
        model_cls = getattr(torchvision.models, name)
    model = model_cls(**(args or {}))
    return model


def create_optimizer(optim_cfg: dict, model: torch.nn.Module) \
        -> torch.optim.Optimizer:
    """Creates an optimizer."""
    name = optim_cfg['name']
    args = optim_cfg.get('args', None)
    optim_cls = getattr(torch.optim, name)
    optim = optim_cls(model.parameters(), **(args or {}))
    return optim


def create_scheduler(sch_cfg: dict, optim: torch.optim.Optimizer) \
        -> torch.optim.lr_scheduler.LRScheduler:
    """Creates a learning rate scheduler."""
    name = sch_cfg['name']
    args = sch_cfg.get('args', None)
    sch_cls = getattr(torch.optim.lr_scheduler, name)
    sch = sch_cls(optim, **(args or {}))
    return sch


def create_loss_fn(loss_cfg: dict) \
        -> torch.nn.Module:
    """Creates a loss function."""
    name = loss_cfg['name']
    args = loss_cfg.get('args', None)
    try:
        loss_cls = getattr(losses, name)
    except AttributeError:
        loss_cls = getattr(torch.nn.modules.loss, name)
    loss_fn = loss_cls(**(args or {}))
    return loss_fn
