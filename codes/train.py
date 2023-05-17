import argparse
import os

import tqdm
import yaml
from torch import nn

from forge import *
from utils import AverageMeter


def train(loader, model, optim, loss_fn):
    """Train a model for an epoch."""
    model.train()
    model.cuda()
    loss_avg = AverageMeter()

    for idx, (inp, tar) in enumerate(tqdm.tqdm(loader)):
        inp = inp.cuda()
        tar = tar.cuda()
        out = model.forward(inp)
        loss = loss_fn.forward(out, tar)
        loss_avg.update(loss.item(), inp.shape[0])

        optim.zero_grad()
        loss.backward()
        optim.step()

    return loss_avg.avg()


def validate(loader, model, metric_fns):
    """Validate a model."""
    model.eval()
    model.cuda()
    if metric_fns is not list:
        metric_fns = [metric_fns]
    metric_avg = [AverageMeter() for _ in range(len(metric_fns))]

    for idx, (inp, tar) in enumerate(tqdm.tqdm(loader)):
        inp = inp.cuda()
        tar = tar.cuda()
        out = model.forward(inp)
        for i in range(len(metric_fns)):
            metric = metric_fns[i].forward(out, tar)
            metric_avg[i].update(metric.item(), inp.shape[0])

    return [m.avg() for m in metric_avg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to config .yaml file')
    parser.add_argument('gpu', type=str,
                        help='Number of gpu(s) to use')

    args = parser.parse_args()

    # Load config.
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        if cfg is not None:
            print('Loaded config.')
        else:
            raise IOError('Failed to load config.')

    # Setup environment.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cudnn.benchmark = True

    # Prepare training.
    train_loader = create_train_loader(cfg['train_dataset'])
    val_loader = create_val_loader(cfg['val_dataset'])
    model = create_model(cfg['model'])
    model = nn.DataParallel(model)
    model = model.cuda()
    optim = create_optimizer(cfg['optimizer'], model)
    sched = create_scheduler(cfg['scheduler'], optim)
    loss_fn = create_loss_fn(cfg['loss'])
    loss_fn = loss_fn.cuda()
    num_epochs = cfg['num_epochs']

    # Train.
    for epoch in range(num_epochs):
        train_loss = train(train_loader, model, optim, loss_fn)
        val_loss = validate(val_loader, model, [loss_fn])
        sched.step()


if __name__ == '__main__':
    main()
