import datetime
import time

from modules import devices, images, shared
from modules.shared import parallel_processing_allowed, opts


class DreamState:
    interrupted = False
    interrupted_after_save = False
    interrupted_after_epoch = False
    skipped = False
    job = ""
    job_no = 0
    job_count = 0
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    textinfo = None
    textinfo2 = None
    time_start = None
    need_restart = False

    def interrupt(self):
        self.interrupted = True

    def interrupt_after_save(self):
        self.interrupted_after_save = True

    def interrupt_after_epoch(self):
        self.interrupted_after_epoch = True

    def dict(self):
        obj = {
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
            "last_status": self.textinfo
        }

        return obj

    def begin(self):
        self.sampling_step = 0
        self.job_count = -1
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent = None
        self.current_image = None
        self.current_image_sampling_step = 0
        self.interrupted = False
        self.textinfo = None
        self.time_start = time.time()
        devices.torch_gc()

    def end(self):
        self.job = ""
        self.job_count = 0
        devices.torch_gc()

    def nextjob(self):
        if opts.show_progress_every_n_steps == -1:
            self.do_set_current_image(False)

        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    """sets self.current_image from self.current_latent if enough sampling steps have been made after the last call to this"""

    def set_current_image(self):
        from_shared = False
        if shared.state.current_latent is not None and self.current_latent is None:
            self.sampling_step = shared.state.sampling_step
            self.current_image_sampling_step = shared.state.current_image_sampling_step
            self.current_latent = shared.state.current_latent
            from_shared = True
        if self.sampling_step - self.current_image_sampling_step >= opts.show_progress_every_n_steps > 0:
            self.do_set_current_image(from_shared)

    def do_set_current_image(self, from_shared):
        if not parallel_processing_allowed:
            return
        if self.current_latent is None:
            return

        if isinstance(self.current_latent, list):
            if len(self.current_latent) > 1:
                self.current_image = images.image_grid(self.current_latent)
            else:
                self.current_image = self.current_latent[0]
        if from_shared:
            self.current_image_sampling_step = shared.state.sampling_step
        else:
            self.current_image_sampling_step = self.sampling_step
        self.current_latent = None


status = DreamState()
