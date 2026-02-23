from __future__ import print_function

import os
import math
import argparse
import numpy as np
import tqdm

import jittor as jt
import jittor.nn as nn

import data
from models.wideresnet import WideResNet
from models.preact_resnet import PreActResNet18
from models.normalize import Normalize


def _sign(x):
    return (x > 0).cast(x.dtype) - (x < 0).cast(x.dtype)


def perturb_pgd(net, data_batch, label, steps, eps, restarts=1):
    nat = data_batch.clone().detach()
    x = nat + (jt.rand(nat.shape) - 0.5) * 2 * eps
    x = jt.clamp(x, 0, 1)
    step_size = eps / 10 * 2

    for _ in range(steps):
        x = x.detach()
        x.start_grad()
        output = net(x)
        loss = nn.cross_entropy_loss(output, label)
        grad = jt.grad(loss, x)

        x = x + step_size * _sign(grad)
        noise = x - nat
        noise = jt.minimum(noise, jt.ones_like(noise) * eps)
        noise = jt.maximum(noise, -jt.ones_like(noise) * eps)
        x = nat + noise
        x = jt.clamp(x, 0, 1)

    return x.detach()


def perturb_fgsm(net, nat, label, eps):
    nat = nat.clone().detach()
    x = nat + (jt.rand(nat.shape) - 0.5) * 2 * eps
    x = jt.clamp(x, 0, 1)
    step_size = eps

    x = x.detach()
    x.start_grad()
    output = net(x)
    loss = nn.cross_entropy_loss(output, label)
    grad = jt.grad(loss, x)

    x = x + step_size * _sign(grad)
    noise = x - nat
    noise = jt.minimum(noise, jt.ones_like(noise) * eps)
    noise = jt.maximum(noise, -jt.ones_like(noise) * eps)
    x = nat + noise
    x = jt.clamp(x, 0, 1)
    return x.detach()


parser = argparse.ArgumentParser(description='Jittor CIFAR PGD Attack Evaluation')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epsilon', default=8 / 255, type=float, help='perturbation')
parser.add_argument('--model-dir', default='./results/test', help='model for white-box attack evaluation')
parser.add_argument('--arch', default='WideResNet', help='model architecture')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--model-name', default='last', help='checkpoint name without suffix')
parser.add_argument('--max-batches', type=int, default=0, help='limit evaluation batches for quick smoke test (0 means all)')

args = parser.parse_args()
kwargs = {'num_workers': 2} if args.dataset != 'imagenet' else {'num_workers': 5}

args.epsilon = args.epsilon / 255


def main(model_name):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if args.dataset != 'imagenet':
        _, test_loader, n_test = data.load_data(args.dataset, args.batch_size, kwargs['num_workers'])
    else:
        test_loader = data.load_data_imagenet_val(args.dataset, args.batch_size, kwargs['num_workers'])
        n_test = len(test_loader)

    normalize = Normalize(mean, std)
    if args.dataset != 'imagenet':
        if args.model_dir.find("PreActResNet18") != -1:
            args.arch = "PreActResNet18"
        elif args.model_dir.find("WideResNet") != -1:
            args.arch = "WideResNet"
        model = nn.Sequential(normalize, eval(args.arch)(num_classes=data.cls_dict[args.dataset]))
    else:
        from jittor import models as jt_models
        model = nn.Sequential(normalize, getattr(jt_models, args.arch)())

    ckpt_path = os.path.join(args.model_dir, f"{model_name}.pkl")
    model.load_parameters(jt.load(ckpt_path))
    model.eval()

    accs, losses = [], []
    total_batches = max(1, math.ceil(n_test / args.batch_size))
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)

    pbar = tqdm.tqdm(enumerate(test_loader), total=total_batches)
    for i, (x_batch, y_batch) in pbar:
        if args.max_batches > 0 and i >= args.max_batches:
            break

        image, target = jt.array(x_batch), jt.array(y_batch)
        nat = image.clone().detach()
        adv_pgd10 = perturb_pgd(model, image, target, 10, args.epsilon, restarts=10)
        adv_pgd50 = perturb_pgd(model, image, target, 50, args.epsilon, restarts=10)
        adv_fgsm = perturb_fgsm(model, image, target, args.epsilon)
        eval_list = [nat, adv_pgd10, adv_pgd50, adv_fgsm]

        tmp_loss = []
        tmp_acc = []
        for adv in eval_list:
            logits = model(adv)
            loss = nn.cross_entropy_loss(logits, target, reduction='none')
            pred = jt.argmax(logits, dim=1)[0]
            acc = (pred == target)
            tmp_acc.append(acc.numpy())
            tmp_loss.append(loss.numpy())
        tmp_acc = np.stack(tmp_acc, axis=0)
        tmp_loss = np.stack(tmp_loss, axis=0)

        accs.append(tmp_acc)
        losses.append(tmp_loss)
        pbar.set_postfix(acc=tmp_acc.mean(axis=1))

    losses = np.concatenate(losses, axis=1)
    accs = np.concatenate(accs, axis=1)
    np.save(os.path.join(args.model_dir, f"{model_name}_test_loss.npy"), losses)
    np.save(os.path.join(args.model_dir, f"{model_name}_test_acc.npy"), accs)

    print('Nat Loss \t Nat Acc \t PGD10 Loss \t PGD10 Acc \t PGD50 Loss \t PGD50 Acc \t FGSM Loss \t FGSM Acc')
    output_str = []
    for i in range(len(losses)):
        output_str.append("{:.4f} \t {:.4f} ".format(losses[i].mean(), accs[i].mean()))
    print("\t".join(output_str))


if __name__ == '__main__':
    main(args.model_name)
