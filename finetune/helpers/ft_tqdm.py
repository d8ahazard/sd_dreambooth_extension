from typing import Iterable

from tqdm import tqdm

from finetune.helpers.ft_state import FinetuneStatus


class fttqdm(tqdm):
    def __init__(self, iterable: Iterable = None, **kwargs):
        self.user = None
        self.target = None
        self.status_handler = None
        self.status_index = 0
        self.update_ui = True
        self.state = FinetuneStatus()
        if "total" in kwargs:
            total = kwargs["total"]
            if total is not None:
                self.state.total_jobs = kwargs["total"]
        if "desc" in kwargs:
            desc = kwargs["desc"]
            desc = desc.replace(":", "")
            if desc is not None:
                if "." not in desc and ":" not in desc:
                    desc = f"{desc}:"
                self.state.status = desc
        super().__init__(iterable=iterable, **kwargs)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""
        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable
        self.state.total_jobs = len(iterable)
        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

    def update(self, n=1):
        if self.update_ui:
            self.state.step_job(n)
        super().update(n)

    def reset(self, total=None):
        self.set_description(None)
        if total is not None and self.update_ui:
            self.state.total_jobs = total
            self.state.current_job = 0
        super().reset(total)

    def set_description(self, desc=None, refresh=True):
        if self.update_ui:
            self.state.status = desc
        super().set_description(desc, refresh)

    # Set the description without ":" appended
    def set_description_str(self, desc=None, refresh=True):
        if self.update_ui:
            self.state.status = desc
        super().set_description_str(desc, refresh)

    def pause_ui(self):
        self.update_ui = False

    def unpause_ui(self):
        self.update_ui = True
