import jittor as jt
import jittor.nn as nn


def _sign(x):
    return (x > 0).cast(x.dtype) - (x < 0).cast(x.dtype)


def get_adv_pgd(model, x_nat, y, step_size, epsilon, num_steps):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()

    x_adv = (jt.rand(x_nat.shape) - 0.5) * 2 * epsilon + x_nat
    x_adv = jt.clamp(x_adv, 0, 1)

    for _ in range(num_steps):
        x_adv = x_adv.detach()
        x_adv.start_grad()
        loss_kl = ce_loss(model(x_adv), y)
        grad = jt.grad(loss_kl, x_adv)
        x_adv = x_adv.detach() + step_size * _sign(grad.detach())
        x_adv = jt.minimum(jt.maximum(x_adv, x_nat - epsilon), x_nat + epsilon)
        x_adv = jt.clamp(x_adv, 0.0, 1.0)

    return x_adv


def get_adv_constant_step_size(model, x_nat, x_adv, y, step_size, epsilon):
    model.eval()
    x_adv = x_adv.detach()
    x_adv.start_grad()
    logits = model(x_adv)
    loss = nn.cross_entropy_loss(logits, y)
    grad = jt.grad(loss, x_adv)
    x_adv = x_adv.detach() + step_size * _sign(grad.detach())
    x_adv = jt.minimum(jt.maximum(x_adv, x_nat - epsilon), x_nat + epsilon)
    x_adv = jt.clamp(x_adv, 0.0, 1.0)

    return x_adv


def get_adv_adaptive_step_size(model, x_nat, x_adv, y, gdnorm, args, epsilon):
    model.eval()
    x_adv = x_adv.detach()
    x_adv.start_grad()
    logits = model(x_adv)
    loss = nn.cross_entropy_loss(logits, y)
    grad = jt.grad(loss, x_adv)

    with jt.no_grad():
        cur_gdnorm = jt.norm(grad.view(len(x_adv), -1), dim=1).detach() ** 2 * (1 - args.beta) + gdnorm * args.beta
        step_sizes = 1 / (1 + jt.sqrt(cur_gdnorm) / args.c) * 2 * 8 / 255
        step_sizes = jt.clamp(step_sizes, args.min_step_size, args.max_step_size)

    step_sizes = step_sizes.view(-1, 1, 1, 1).broadcast(grad.shape)
    x_adv = x_adv.detach() + step_sizes * _sign(grad.detach())
    x_adv = jt.minimum(jt.maximum(x_adv, x_nat - epsilon), x_nat + epsilon)
    x_adv = jt.clamp(x_adv, 0.0, 1.0)

    return x_adv, cur_gdnorm
