from typing import Iterable

from tqdm import tqdm

from dreambooth import shared


class mytqdm(tqdm):
    def __init__(self, iterable: Iterable = None, **kwargs):
        self.user = None
        self.target = None
        self.status_handler = None
        self.status_index = 0
        try:
            from core.handlers.status import StatusHandler
            user = kwargs["user"] if "user" in kwargs else None
            target = kwargs["target"] if "target" in kwargs else None
            self.status_index = kwargs["index"] if "index" in kwargs else 0
            self.user = user
            self.target = target
            if user and target:
                self.status_handler = StatusHandler(user_name=user, target=target)
            if "user" in kwargs:
                del kwargs["user"]
            if "target" in kwargs:
                del kwargs["target"]
            if "index" in kwargs:
                del kwargs["index"]
        except:
            if "user" in kwargs:
                del kwargs["user"]
            if "target" in kwargs:
                del kwargs["target"]
            if "index" in kwargs:
                del kwargs["index"]
        self.update_ui = True
        if "total" in kwargs:
            total = kwargs["total"]
            if total is not None:
                if self.status_handler is not None:
                    self.status_handler.update(f"progress_{self.status_index}_total", total)
                shared.status.job_count = kwargs["total"]
        if "desc" in kwargs:
            desc = kwargs["desc"]
            desc = desc.replace(":", "")
            if desc is not None:
                if "." not in desc and ":" not in desc:
                    desc = f"{desc}:"
                if self.status_handler is not None:
                    status_title = "status" if self.status_index == 0 else "status_2"
                    self.status_handler.update(status_title, desc)
                shared.status.textinfo = desc
        super().__init__(iterable=iterable, **kwargs)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""
        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable
        shared.status.job_count = len(iterable)
        if self.status_handler is not None:
            self.status_handler.update(items={f"progress_{self.status_index}_total": len(iterable), f"progress_{self.status_index}_current": 0})
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
            shared.status.job_no += n
            if shared.status.job_no > shared.status.job_count:
                shared.status.job_no = shared.status.job_count
            if self.status_handler is not None:
                self.status_handler.update(items={f"progress_{self.status_index}_total": shared.status.job_count, f"progress_{self.status_index}_current": shared.status.job_no})
        super().update(n)

    def reset(self, total=None):
        self.set_description(None)
        if total is not None and self.update_ui:
            shared.status.job_no = 0
            shared.status.job_count = total
            if self.status_handler is not None:
                self.status_handler.update(
                    items={f"progress_{self.status_index}_total": shared.status.job_count, f"progress_{self.status_index}_current": shared.status.job_no})
        super().reset(total)

    def set_description(self, desc=None, refresh=True):
        if self.update_ui:
            shared.status.textinfo = desc
            if self.status_handler is not None and desc is not None:
                status_title = "status" if self.status_index == 0 else "status_2"
                self.status_handler.update(
                    items={status_title: desc})
        super().set_description(desc, refresh)

    # Set the description without ":" appended
    def set_description_str(self, desc=None, refresh=True):
        if self.update_ui:
            shared.status.textinfo = desc
            if self.status_handler is not None:
                status_title = "status" if self.status_index == 0 else "status_2"
                self.status_handler.update(
                    items={status_title: desc})
        super().set_description_str(desc, refresh)

    def pause_ui(self):
        self.update_ui = False

    def unpause_ui(self):
        self.update_ui = True
