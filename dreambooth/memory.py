# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A collection of utilities for ensuring that training can always occur. Heavily influenced by the
[toma](https://github.com/BlackHC/toma) library.
"""

import functools
import gc
import inspect
import traceback

import torch
import torch.backends.cudnn

from dreambooth import shared
from dreambooth.utils.utils import cleanup


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


profiler = None


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128,
                               starting_grad_size: int = 128, logging_dir: str = ""):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory
        starting_grad_size:
            The starting number of grad accumulation steps to use. Will be divided by 2 every loop.
        logging_dir:
            The directory to use for logging.
    """
    global profiler
    try:
        profile_memory = shared.profile_db
    except Exception:
        profile_memory = False

    torch.backends.cudnn.benchmark = not profile_memory

    if profile_memory and profiler is None:
        from torch.profiler import profile

        cleanup(True)

        profiler = profile(
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=100, repeat=100),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{logging_dir}'),
            with_stack=True,
            profile_memory=True)
        print("Starting profiler...")
        profiler.start()
    else:
        prof = None

    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size,
                                 starting_grad_size=starting_grad_size, logging_dir=logging_dir)

    batch_size = starting_batch_size
    grad_size = starting_grad_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        nonlocal grad_size
        nonlocal prof
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, grad_size, prof, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    grad_size //= 2
                    if grad_size == 0:
                        grad_size = 1
                    print(f"OOM Detected, reducing batch/grad size to {batch_size}/{grad_size}.")
                    traceback.print_exc()
                else:
                    raise

    return decorator
