from __future__ import print_function

import argparse
import math
import os

import numpy as np
import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
from jittor import models as jt_models
import tqdm

import adv_attack
import data
from data_aug import *
from models.wideresnet import WideResNet
from models.preact_resnet import PreActResNet18
from models.normalize import Normalize

parser = argparse.ArgumentParser(description='Jittor CIFAR TRADES Adversarial Training')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--epochs-reset', type=int, default=10, metavar='N',
                    help='number of epochs to reset perturbation')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')

parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--num-workers', default=4, type=int)

parser.add_argument('--arch', default='WideResNet', choices=['WideResNet', 'PreActResNet18', 'resnet18', 'resnet50'],
                    help='Adversarial training architecture')
parser.add_argument('--decay-steps', default=[24, 28], type=int, nargs="+")

parser.add_argument('--epsilon', default=8 / 255, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=1, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=1.0, type=float,
                    help='perturb step size')

parser.add_argument('--max-step-size', default=14, type=float,
                    help='maximum perturb step size')
parser.add_argument('--min-step-size', default=4, type=float,
                    help='minimum perturb step size')
parser.add_argument('--c', default=0.01, type=float,
                    help='hard fraction')
parser.add_argument('--beta', default=0.5, type=float,
                    help='hardness momentum')

parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='number of warmup epochs')

parser.add_argument('--model-dir', default='./results',
                    help='directory of model for saving checkpoint')

args = parser.parse_args()

epochs_reset = args.epochs_reset

args.epsilon = args.epsilon / 255
args.max_step_size = args.max_step_size / 255
args.min_step_size = args.min_step_size / 255
args.step_size = args.step_size * args.epsilon

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def _to_var(x, dtype=None):
    if isinstance(x, jt.Var):
        out = x
    else:
        out = jt.array(x)
    if dtype is not None:
        out = out.cast(dtype)
    return out


def _step_scheduler(optimizer, current_step, decay_steps_set):
    if current_step in decay_steps_set and hasattr(optimizer, 'lr'):
        optimizer.lr = optimizer.lr * 0.1


def train(args, model, train_loader, n_ex, delta, optimizer, decay_steps_set, gdnorms, epoch, global_step):
    model.train()
    total_batches = max(1, math.ceil(n_ex / args.batch_size))
    pbar = tqdm.tqdm(enumerate(train_loader), total=total_batches)

    for _, (data_batch, label_batch, index_batch) in pbar:
        nat = _to_var(data_batch, jt.float32)
        label = _to_var(label_batch)
        index = _to_var(index_batch)

        with jt.no_grad():
            if args.dataset != 'imagenet':
                delta_trans, transform_info = aug(delta[index])
                nat_trans = aug_trans(nat, transform_info)
                adv_trans = jt.clamp(delta_trans + nat_trans, 0, 1)
            else:
                delta_trans, transform_info = aug_imagenet(delta[index].cast(jt.float32))
                nat_trans = aug_trans_imagenet(nat, transform_info)
                adv_trans = jt.clamp(delta_trans + nat_trans, 0, 1)

        if epoch > args.warmup_epochs:
            next_adv_trans, gdnorm = adv_attack.get_adv_adaptive_step_size(
                model=model,
                x_nat=nat_trans,
                x_adv=adv_trans,
                y=label,
                gdnorm=gdnorms[index],
                args=args,
                epsilon=args.epsilon
            )
            gdnorms[index] = gdnorm
        else:
            next_adv_trans = adv_attack.get_adv_constant_step_size(
                model=model,
                x_nat=nat_trans,
                x_adv=adv_trans,
                y=label,
                step_size=args.step_size,
                epsilon=args.epsilon
            )

        model.train()

        loss = nn.cross_entropy_loss(model(next_adv_trans.detach()), label)
        optimizer.step(loss)

        global_step[0] += 1
        _step_scheduler(optimizer, global_step[0], decay_steps_set)

        pbar.set_postfix(loss=float(loss.data[0]))
        if args.dataset != "imagenet":
            delta[index] = inverse_aug(delta[index], next_adv_trans - nat_trans, transform_info)
        else:
            delta[index] = inverse_aug_imagenet(delta[index], (next_adv_trans - nat_trans).cast(jt.float16), transform_info).cast(jt.float16)


def main():
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = Normalize(mean, std)

    if args.dataset != 'imagenet':
        model = nn.Sequential(normalize, eval(args.arch)(num_classes=data.cls_dict[args.dataset]))
    else:
        model = nn.Sequential(normalize, getattr(jt_models, args.arch)())

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.dataset != 'imagenet':
        train_loader, test_loader, n_ex = data.load_data(args.dataset, args.batch_size, args.num_workers)
        delta = (jt.rand([n_ex] + data.shapes_dict[args.dataset], dtype=jt.float32) * 2 - 1) * args.epsilon
    else:
        train_loader, test_loader, n_ex = data.load_data_imagenet(args.dataset, args.batch_size, args.num_workers)
        delta = (jt.rand([n_ex] + data.shapes_dict[args.dataset], dtype=jt.float16) * 2 - 1) * args.epsilon

    total_batches = max(1, math.ceil(n_ex / args.batch_size))
    decay_steps = [x * total_batches for x in args.decay_steps]
    decay_steps_set = set(decay_steps)

    gdnorm = jt.zeros((n_ex,), dtype=jt.float32)
    global_step = [0]

    for epoch in range(1, args.epochs + 1):
        if epoch % epochs_reset == 0 and epoch != args.epochs:
            delta.assign((jt.rand(delta.shape, dtype=delta.dtype) * 2 - 1) * args.epsilon)
        train(args, model, train_loader, n_ex, delta, optimizer, decay_steps_set, gdnorm, epoch, global_step)

    jt.save(model.state_dict(), os.path.join(model_dir, 'last.pkl'))


if __name__ == '__main__':
    main()
