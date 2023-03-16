#
# SOURCE: https://github.com/qwopqwop200/D-Adaptation-Adan/blob/main/opt/dadapt_adan.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any

import torch
import torch.optim

if TYPE_CHECKING:
    pass
else:
    _params_t = Any

def to_real(x):
    if torch.is_complex(x):
        return x.real
    else:
        return x

class DAdaptAdan(torch.optim.Optimizer):
    r"""
    Implements Adan with D-Adaptation automatic step-sizes. Leave LR set to 1 unless you encounter instability.
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float, flot], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.02).
        no_prox (boolean):
            how to perform the decoupled weight decay (default: False)
        log_every (int):
            Log using print every k steps, default 0 (no logging).
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.98, 0.92, 0.99),
                 eps=1e-8, weight_decay=0.02,
                 no_prox=False,
                 log_every=0, d0=1e-6,
                 growth_rate=float('inf')):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        no_prox=no_prox,
                        d = d0,
                        k=0,
                        gsq_weighted=0.0,
                        log_every=log_every,
                        growth_rate=growth_rate)
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    # Experimental implementation of Adan's restart strategy
    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['gsq_weighted'] = 0.0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(to_real(p.data), memory_format=torch.preserve_format).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        g_sq = 0.0
        sksq_weighted = 0.0
        sk_l1 = 0.0

        ngroups = len(self.param_groups)

        group = self.param_groups[0]
        gsq_weighted = group['gsq_weighted']
        d = group['d']
        lr = group['lr']
        dlr = d*lr

        no_prox = group['no_prox']
        growth_rate = group['growth_rate']
        log_every = group['log_every']

        beta1, beta2, beta3 = group['betas']

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(to_real(p.data), memory_format=torch.preserve_format).detach()

                if state['step'] == 0:
                    # Previous gradient values
                    state['pre_grad'] = grad.clone()

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_diff'], state['exp_avg_sq']
                grad_diff = grad - state['pre_grad']

                grad_grad = to_real(grad * grad.conj())
                update = grad + beta2 * grad_diff
                update_update = to_real(update * update.conj())

                exp_avg.mul_(beta1).add_(grad, alpha=dlr*(1. - beta1))
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=dlr*(1. - beta2))
                exp_avg_sq.mul_(beta3).add_(update_update, alpha=1. - beta3)

                denom = exp_avg_sq.sqrt().add_(eps)

                g_sq += grad_grad.div_(denom).sum().item()

                s = state['s']
                s.mul_(beta3).add_(grad, alpha=dlr*(1. - beta3))
                sksq_weighted += to_real(s * s.conj()).div_(denom).sum().item()
                sk_l1 += s.abs().sum().item()

            ######

        gsq_weighted = beta3*gsq_weighted + g_sq*(dlr**2)*(1-beta3)
        d_hat = d

        if lr > 0.0:
            d_hat = (sksq_weighted/(1-beta3) - gsq_weighted)/sk_l1
            d = max(d, min(d_hat, d*growth_rate))

        if log_every > 0 and k % log_every == 0:
            print(f"ng: {ngroups} lr: {lr} dlr: {dlr} d_hat: {d_hat}, d: {d}. sksq_weighted={sksq_weighted:1.1e} sk_l1={sk_l1:1.1e} gsq_weighted={gsq_weighted:1.1e}")

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['d'] = d

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_diff'], state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(eps)
                denom = denom.type(p.type())

                update = (exp_avg + beta2 * exp_avg_diff).div_(denom)

                ### Take step
                if no_prox:
                    p.data.mul_(1 - dlr * decay)
                    p.add_(update, alpha=-1)
                else:
                    p.add_(update, alpha=-1)
                    p.data.div_(1 + dlr * decay)

                state['pre_grad'].copy_(grad)

            group['k'] = k + 1

        return loss