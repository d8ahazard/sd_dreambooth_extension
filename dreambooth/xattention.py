from __future__ import annotations

import gc
from typing import Union, Optional

import torch
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from torch.optim import Optimizer


def get_scheduler(
        name: Union[str, SchedulerType],
        optimizer: Optimizer,
        num_warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
        num_cycles: int = 1,
        power: float = 1.0,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`int`, *optional*):
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
        power (`float`, *optional*, defaults to 1.0):
            Power factor. See `POLYNOMIAL` scheduler
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def optim_to(profiler, optim: torch.optim.Optimizer, device="cpu"):
    if profiler is None:
        torch.cuda.empty_cache()
        gc.collect()

    def inplace_move(obj: torch.Tensor, target):
        if hasattr(obj, 'data'):
            obj.data = obj.data.to(target)
        if hasattr(obj, '_grad') and obj._grad is not None:
            obj._grad.data = obj._grad.data.to(target)

    if isinstance(optim, torch.optim.Optimizer):
        for group in optim.param_groups:
            for param in group['params']:
                inplace_move(param, device)
        for key, value in optim.state.items():
            if isinstance(value, torch.Tensor):
                inplace_move(value, device)
    if profiler is None:
        torch.cuda.empty_cache()
        gc.collect()
