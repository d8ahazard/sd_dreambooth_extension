from typing import Iterable

from tqdm import tqdm

from dreambooth import shared


class mytqdm(tqdm):
    def __init__(self, iterable: Iterable = None, **kwargs):
        self.status_handler = None
        try:
            from core.handlers.status import StatusHandler
            user = kwargs["user"] if "user" in kwargs else None
            target = kwargs["target"] if "target" in kwargs else None
            self.status_handler = StatusHandler(user_name=user, target=target)
            shared.status_handler = self.status_handler
            if "user" in kwargs:
                del kwargs["user"]
            if "target" in kwargs:
                del kwargs["target"]
        except:
            if "user" in kwargs:
                del kwargs["user"]
            if "target" in kwargs:
                del kwargs["target"]
        self.update_ui = True
        if "total" in kwargs:
            total = kwargs["total"]
            if total is not None:
                if self.status_handler is not None:
                    self.status_handler.update("progress_1_total", total)
                shared.status.job_count = kwargs["total"]
        if "desc" in kwargs:
            desc = kwargs["desc"]
            desc = desc.replace(":", "")
            if desc is not None:
                if "." not in desc and ":" not in desc:
                    desc = f"{desc}:"
                if self.status_handler is not None:
                    self.status_handler.update("status", desc)
                shared.status.textinfo = desc
        super().__init__(iterable=iterable, **kwargs)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""
        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable
        shared.status.job_count = len(iterable)
        if self.status_handler is not None:
            self.status_handler.update(items={"progress_1_total": len(iterable), "progress_1_current": 0})
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
                self.status_handler.update(items={"progress_1_total": shared.status.job_count, "progress_1_current": shared.status.job_no})
        super().update(n)

    def reset(self, total=None):
        if total is not None and self.update_ui:
            shared.status.job_no = 0
            shared.status.job_count = total
            if self.status_handler is not None:
                self.status_handler.update(
                    items={"progress_1_total": shared.status.job_count, "progress_1_current": shared.status.job_no})
        super().reset(total)

    def set_description(self, desc=None, refresh=True):
        if self.update_ui:
            shared.status.textinfo = desc
            if self.status_handler is not None:
                self.status_handler.update(
                    items={"status": desc})
        super().set_description(desc, refresh)

    def pause_ui(self):
        self.update_ui = False

    def unpause_ui(self):
        self.update_ui = True
