# One wrapper we're going to use to not depend so much on the main app.
import datetime
import math
import os
import time

import torch
from PIL import Image

dreambooth_models_path = ""
models_path = ""
script_path = os.path.dirname(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
ckpt_dir = ""
lora_models_path = ""
show_progress_every_n_steps = 10
parallel_processing_allowed = True
dataset_filename_word_regex = ""
dataset_filename_join_string = " "
device_id = None
state = None
disable_safe_unpickle = True
ckptfix = False
medvram = False
lowvram = False
CLIP_stop_at_last_layers = 2
config = os.path.join(script_path, "configs", "v1-inference.yaml")


def image_grid(imgs, batch_size=1, rows=None):
    if rows is None:
        rows = math.floor(math.sqrt(len(imgs)))
        while len(imgs) % rows != 0:
            rows -= 1
        else:
            rows = math.sqrt(len(imgs))
            rows = round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


def load_auto_settings():
    global models_path, script_path, ckpt_dir, device_id, disable_safe_unpickle, dataset_filename_word_regex, \
        dataset_filename_join_string, show_progress_every_n_steps, parallel_processing_allowed, state, ckptfix, medvram, \
        lowvram, dreambooth_models_path, lora_models_path, CLIP_stop_at_last_layers
    try:
        from modules import shared as ws, devices, images
        from modules import paths
        from modules.paths import models_path as mp, script_path as sp, sd_path as sdp
        models_path = mp
        script_path = sp
        ckpt_dir = ws.cmd_opts.ckpt_dir
        device_id = ws.cmd_opts.device_id
        CLIP_stop_at_last_layers = ws.opts.CLIP_stop_at_last_layers
        disable_safe_unpickle = ws.cmd_opts.disable_safe_unpickle
        dataset_filename_word_regex = ws.opts.dataset_filename_word_regex
        dataset_filename_join_string = ws.opts.dataset_filename_join_string
        show_progress_every_n_steps = ws.opts.show_progress_every_n_steps
        parallel_processing_allowed = ws.parallel_processing_allowed
        state = ws.state
        ckptfix = ws.cmd_opts.ckptfix
        medvram = ws.cmd_opts.medvram
        lowvram = ws.cmd_opts.lowvram
        config = ws.cmd_opts.config
        try:
            dreambooth_models_path = ws.cmd_opts.dreambooth_models_path
            lora_models_path = ws.cmd_opts.lora_models_path
        except:
            pass

    except:
        print("Exception importing SD-WebUI module.")
        pass


def get_cuda_device_string():
    if device_id is not None:
        return f"cuda:{device_id}"

    return "cuda"


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class DreamState:
    interrupted = False
    interrupted_after_save = False
    interrupted_after_epoch = False
    do_save_model = False
    do_save_samples = False
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
    sample_prompts = []
    time_start = None
    need_restart = False

    def interrupt(self):
        self.interrupted = True

    def interrupt_after_save(self):
        self.interrupted_after_save = True

    def interrupt_after_epoch(self):
        self.interrupted_after_epoch = True

    def save_samples(self):
        self.do_save_samples = True

    def save_model(self):
        self.do_save_model = True

    def dict(self):
        obj = {
            "do_save_model": self.do_save_model,
            "do_save_samples": self.do_save_samples,
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
            "last_status": self.textinfo,
            "sample_prompts": self.sample_prompts
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
        self.sample_prompts = []
        self.time_start = time.time()
        torch_gc()

    def end(self):
        self.job = ""
        self.job_count = 0
        torch_gc()

    def nextjob(self):
        if show_progress_every_n_steps == -1:
            self.do_set_current_image(False)

        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    """sets self.current_image from self.current_latent if enough sampling steps have been made after the last call to this"""

    def set_current_image(self):
        from_shared = False
        if state.current_latent is not None and self.current_latent is None:
            self.sampling_step = state.sampling_step
            self.current_image_sampling_step = state.current_image_sampling_step
            self.current_latent = state.current_latent
            from_shared = True
        if self.sampling_step - self.current_image_sampling_step >= show_progress_every_n_steps > 0:
            self.do_set_current_image(from_shared)

    def do_set_current_image(self, from_shared):
        if not parallel_processing_allowed:
            return
        if self.current_latent is None:
            return

        if isinstance(self.current_latent, list):
            if len(self.current_latent) > 1:
                self.current_image = image_grid(self.current_latent)
            else:
                self.current_image = self.current_latent[0]
        if from_shared:
            self.current_image_sampling_step = state.sampling_step
        else:
            self.current_image_sampling_step = self.sampling_step
        self.current_latent = None


def stop_safe_unpickle():
    enabled = False
    try:
        from modules import shared as ws
        enabled = not ws.cmd_opts.disable_safe_unpickle
        if enabled:
            ws.cmd_opts.disable_safe_unpickle = True
    except:
        pass
    return enabled


def start_safe_unpickle():
    try:
        from modules import shared as ws
        ws.cmd_opts.disable_safe_unpickle = False
    except:
        pass


load_auto_settings()

status = DreamState()
if state is None:
    state = status
