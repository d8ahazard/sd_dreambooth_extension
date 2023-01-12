import sys
import inspect

import torch.cuda


class VRAMMonitor:
    def __init__(self, method_names):
        self.prev_vram_usage = 0
        self.method_stack = []
        self.method_names = method_names
        self.system_modules = sys.modules
        self.wrap_methods()

    def _get_vram_usage(self):
        return torch.cuda.memory_allocated()
        pass

    def _log_vram_change(self, method_name, change_amount):
        # Get the parent of the method
        if self.method_stack:
            parent = self.method_stack[-1]
        else:
            parent = None

        # Log the VRAM usage change
        print(f"Method '{method_name}' called by '{parent}' caused a VRAM usage change of {change_amount}")

    def _wrap_method(self, method_name, original_method):
        # Define a wrapper function that logs the VRAM usage change
        def wrapper(*args, **kwargs):
            self.method_stack.append(method_name)
            try:
                result = original_method(*args, **kwargs)
                current_vram_usage = self._get_vram_usage()
                change_amount = current_vram_usage - self.prev_vram_usage
                if change_amount != 0:
                    self._log_vram_change(method_name, change_amount)
                    self.prev_vram_usage = current_vram_usage
                return result
            finally:
                self.method_stack.pop()

        # Replace the original method with the wrapper function
        setattr(self.system_modules[original_method.__module__], method_name, wrapper)

    def _wrap_methods_in_module(self, module):
        for name, obj in vars(module).items():
            if name in self.method_names and inspect.isfunction(obj):
                self._wrap_method(name, obj)
            elif inspect.ismodule(obj):
                self._wrap_methods_in_module(obj)

    def wrap_methods(self):
        for name, module in self.system_modules.items():
            if inspect.ismodule(module):
                self._wrap_methods_in_module(module)
        sys.modules = self.system_modules
