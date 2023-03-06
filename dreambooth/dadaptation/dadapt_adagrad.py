# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
import pdb

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class DAdaptAdaGrad(torch.optim.Optimizer):
    """
    Adagrad with D-Adaptation. Leave LR set to 1 unless you encounter instability.
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        log_every (int): 
            Log using print every k steps, default 0 (no logging).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float): 
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-6).
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate. 
            Default is inf, for unrestricted.
    """

    def __init__(
        self, params: _params_t, 
        lr: float = 1.0,
        momentum: float = 0, 
        log_every: int = 0,
        weight_decay: float = 0.0,
        eps: float = 0.0,
        d0 = 1e-6, growth_rate=float('inf')
    ):
        if d0 <= 0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if momentum < 0:
            raise ValueError(f"Momentum {momentum} must be non-negative")
        if eps <= 0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, 
            momentum=momentum,
            eps=eps, 
            weight_decay=weight_decay,
            gsq_weighted=0.0, 
            log_every=log_every,
            d=d0,
            growth_rate=growth_rate,
            k = 0,
            sksq_weighted=0.0,
            skl1=0.0)
        self.d0 = d0
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = group["lr"] 
        momentum = group['momentum']
        ck = 1 - momentum
        
        log_every = group['log_every']
        growth_rate = group['growth_rate']

        gsq_weighted = group['gsq_weighted']
        sksq_weighted = group['sksq_weighted']
        skl1 = group['skl1']
        d = group['d']
        
        dlr = d*lr

        g_sq = 0.0
        sksq_weighted_change = 0.0
        skl1_change = 0.0

        for group in self.param_groups:
            eps = group["eps"]
            k = group['k']
            decay = group['weight_decay']

            ######
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                
                if "alphak" not in state:
                    state["alphak"] = torch.full_like(p.data, fill_value=1e-6).detach()
                    state['sk'] = torch.zeros_like(p.data).detach()
                    state["x0"] = torch.clone(p.data).detach()

                    if grad.is_sparse:
                        state['weighted_sk'] = torch.zeros_like(p.data).detach()

                sk = state['sk']
                alphak = state['alphak']
                
                grad_sq = 0.0
                if grad.is_sparse: 
                    weighted_sk = state['weighted_sk']

                    grad = grad.coalesce()
                    grad_vals = grad._values()
                    vk_vals = grad_vals*grad_vals

                    sk_vals = sk.sparse_mask(grad).coalesce()._values()

                    old_skl1_vals = sk_vals.abs().sum().item()

                    sk.data.add_(grad, alpha=dlr)

                    sk_vals = sk.sparse_mask(grad).coalesce()._values()
                    alphak_vals = alphak.sparse_mask(grad).coalesce()._values()
                    weighted_sk_vals = weighted_sk.sparse_mask(grad).coalesce()._values()

                    ### Update alpha before step
                    alphak_vals = alphak.sparse_mask(grad).coalesce()._values()
                    alphakp1_vals = alphak_vals + vk_vals

                    alphak_delta_vals = alphakp1_vals - alphak_vals
                    alphak_delta = torch.sparse_coo_tensor(grad.indices(), alphak_delta_vals, grad.shape)
                    alphak.add_(alphak_delta)

                    ####
                    denominator = torch.sqrt(alphakp1_vals + eps)
                    
                    grad_sq = (grad_vals * grad_vals).div(denominator).sum().item()
                    g_sq += grad_sq

                    ### Update weighted sk sq tracking
                    weighted_skp1_vals = (sk_vals * sk_vals).div(denominator)

                    sksq_weighted_change += weighted_skp1_vals.sum().item() - weighted_sk_vals.sum().item()

                    weighted_skp1_delta_vals = weighted_skp1_vals - weighted_sk_vals
                    weighted_skp1_delta = torch.sparse_coo_tensor(grad.indices(), weighted_skp1_delta_vals, grad.shape)
                    weighted_sk.add_(weighted_skp1_delta)

                    skl1_vals = sk_vals.abs().sum().item()

                    skl1_change += skl1_vals - old_skl1_vals

                else:
                    if decay != 0:
                        grad.add_(p.data, alpha=decay)

                    old_sksq_weighted_param = (sk * sk).div(torch.sqrt(alphak) + eps).sum().item()
                    old_skl1_param = sk.abs().sum().item()

                    alphak.data.add_(grad * grad)
                    grad_sq = (grad * grad).div(torch.sqrt(alphak) + eps).sum().item()
                    g_sq += grad_sq

                    sk.data.add_(grad, alpha=dlr)
                    
                    sksq_weighted_param = (sk * sk).div(torch.sqrt(alphak) + eps).sum().item()
                    skl1_param = sk.abs().sum().item()

                    sksq_weighted_change += sksq_weighted_param - old_sksq_weighted_param
                    skl1_change += skl1_param - old_skl1_param
            ######
            
        sksq_weighted = sksq_weighted + sksq_weighted_change
        skl1 = skl1 + skl1_change

        # if we have not done any progres, return
        # if we have any gradients available, will have skl1 > 0 (unless \|g\|=0)
        if skl1 == 0:
            return loss

        gsq_weighted = gsq_weighted + dlr*dlr*g_sq
        d_hat = d
        
        if lr > 0.0:
            d_hat = (sksq_weighted - gsq_weighted)/skl1
            d = group['d'] = max(d, min(d_hat, d*growth_rate))

        if log_every > 0 and k % log_every == 0:
            print(f"d_hat: {d_hat}, d: {d}. sksq_weighted={sksq_weighted:1.1e} skl1={skl1:1.1e} gsq_weighted={gsq_weighted:1.1e} lr={lr}")

        for group in self.param_groups:
            group['gsq_weighted'] = gsq_weighted
            group['skl1'] = skl1
            group['sksq_weighted'] = sksq_weighted

            group['d'] = d

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                alphak = state["alphak"]
                sk = state["sk"]
                x0 = state["x0"]

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_vals = grad._values()

                    sk_vals = sk.sparse_mask(grad).coalesce()._values()
                    alphak_vals = alphak.sparse_mask(grad).coalesce()._values()
                    x0_vals = x0.sparse_mask(grad).coalesce()._values()
                    p_vals = p.data.sparse_mask(grad).coalesce()._values()

                    loc_vals = x0_vals - sk_vals.div(torch.sqrt(alphak_vals + eps))

                    loc_delta_vals = loc_vals - p_vals
                    loc_delta = torch.sparse_coo_tensor(grad.indices(), loc_delta_vals, grad.shape) 
                    p.data.add_(loc_delta)
                    
                else:
                    z = x0 - sk.div(torch.sqrt(alphak) + eps)

                    if momentum != 0:
                        p.data.mul_(1-ck).add_(z, alpha=ck)
                    else:
                        p.data.copy_(z)
            group['k'] = k + 1
        return loss
