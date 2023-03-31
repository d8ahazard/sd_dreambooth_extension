# One wrapper we're going to use to not depend so much on the main app.
import datetime
import json
import logging
import os
import pathlib
import subprocess
import time

import PIL
import numpy
import torch
from PIL import Image
from packaging import version

logger = logging.getLogger(__name__)


def load_auto_settings():
    global models_path, script_path, ckpt_dir, device_id, disable_safe_unpickle, dataset_filename_word_regex, \
        dataset_filename_join_string, show_progress_every_n_steps, parallel_processing_allowed, state, ckptfix, medvram, \
        lowvram, dreambooth_models_path, ui_lora_models_path, CLIP_stop_at_last_layers, profile_db, debug, config, device, \
        force_cpu, embeddings_dir, sd_model
    try:
        import modules.script_callbacks
        from modules import shared as ws
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
        profile_db = ws.cmd_opts.profile_db
        debug = ws.cmd_opts.debug_db
        medvram = ws.cmd_opts.medvram
        lowvram = ws.cmd_opts.lowvram
        config = ws.cmd_opts.config
        device = ws.device
        sd_model = ws.sd_model

        def set_model(new_model):
            global sd_model
            sd_model = new_model

        # Keep a reference to loaded script
        modules.script_callbacks.on_model_loaded(set_model)

        try:
            dreambooth_models_path = ws.cmd_opts.dreambooth_models_path or dreambooth_models_path
            ui_lora_models_path = ws.cmd_opts.lora_models_path or ui_lora_models_path
            embeddings_dir = ws.cmd_opts.embeddings_dir or embeddings_dir
        except:
            pass

        try:
            force_cpu = ws.cmd_opts.force_cpu
            if force_cpu:
                device = torch.device("cpu")
        except:
            pass
        return True
    except Exception:
        return False


def get_launch_errors():
    errors = os.environ.get("ERRORS", None)
    if errors == "":
        launch_error = None
    elif errors is not None:
        errors = json.loads(errors)
        launch_error = errors
    else:
        launch_error = None
    launch_errors = ""
    if launch_error is not None:
        launch_errors = "The Dreambooth extension has been disabled because the following error(s) were detected on launch.<br>" \
                        " Please completely restart the Auto1111 web-UI.<br>" \
                        "If this error persists, please consult the <a href='https://github.com/d8ahazard/sd_dreambooth_extension/wiki'> wiki</a> for more information.<br>"
        launch_strings = "<br>".join(launch_error)
        launch_errors += f"<b>{launch_strings}</b>"
    return launch_errors


def get_cuda_device_string():
    if device_id is not None:
        return f"cuda:{device_id}"

    return "cuda"


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        logger.debug(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                            env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def format_time(seconds: float):
    date = datetime.datetime.utcfromtimestamp(seconds)
    return datetime.datetime.strftime(date, "%H:%M:%S")


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
    time_left_force_display = False
    active = False

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
            "sample_prompts": self.sample_prompts,
            "active": self.active
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
        self.textinfo2 = None
        self.time_left_force_display = False
        self.active = True
        torch_gc()

    def end(self):
        print("Duration: " + format_time(time.time() - self.time_start))
        self.job = ""
        self.job_count = 0
        self.job_no = 0
        self.active = False
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
        # If using txt2img to generate, try and grab the current latent
        if state.current_latent is not None and self.current_latent is None:
            self.sampling_step = state.sampling_step
            self.current_image_sampling_step = state.current_image_sampling_step
            self.current_latent = state.current_latent
            from_shared = True
        if self.sampling_step - self.current_image_sampling_step >= show_progress_every_n_steps > 0:
            self.do_set_current_image(from_shared)

    def do_set_current_image(self, from_shared):
        if self.current_latent is not None:
            if from_shared:
                self.current_image_sampling_step = state.sampling_step
            else:
                self.current_image_sampling_step = self.sampling_step
            self.current_image = self.current_latent
            self.current_latent = None

        if self.current_image is not None:
            if isinstance(self.current_image, list):
                to_check = self.current_image
            else:
                to_check = [self.current_image]

            real_images = []
            for check in to_check:
                if isinstance(check, (numpy.ndarray, PIL.Image.Image, pathlib.Path, str)):
                    real_images.append(check)
            self.current_image = real_images if len(real_images) > 2 else real_images[0] if len(
                real_images) == 1 else None


def tensor_to_fix(self, *args, **kwargs):
    if self.device.type != 'mps' and \
            ((len(args) > 0 and isinstance(args[0], torch.device) and args[0].type == 'mps') or
             (isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'mps')):
        self = self.contiguous()
    return orig_tensor_to(self, *args, **kwargs)


# MPS workaround for https://github.com/pytorch/pytorch/issues/80800

def layer_norm_fix(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps':
        args = list(args)
        args[0] = args[0].contiguous()
    return orig_layer_norm(*args, **kwargs)


def numpy_fix(self, *args, **kwargs):
    if self.requires_grad:
        self = self.detach()
    return orig_tensor_numpy(self, *args, **kwargs)


def cumsum_fix(input, cumsum_func, *args, **kwargs):
    if input.device.type == 'mps':
        output_dtype = kwargs.get('dtype', input.dtype)
        if any(output_dtype == broken_dtype for broken_dtype in [torch.bool, torch.int8, torch.int16, torch.int64]):
            return cumsum_func(input.cpu(), *args, **kwargs).to(input.device)
    return cumsum_func(input, *args, **kwargs)


def load_vars(root_path = None):
    global script_path, models_path, embeddings_dir, dreambooth_models_path, ckpt_dir, ui_lora_models_path, db_model_config, \
    data_path, show_progress_every_n_steps, parallel_processing_allowed, dataset_filename_word_regex, dataset_filename_join_string, \
    device_id, state, disable_safe_unpickle, ckptfix, medvram, lowvram, debug, profile_db, sub_quad_q_chunk_size, sub_quad_kv_chunk_size, \
    sub_quad_chunk_threshold, CLIP_stop_at_last_layers, sd_model, config, force_cpu, paths, is_auto, device, orig_tensor_to, orig_layer_norm, \
    orig_tensor_numpy, extension_path, orig_cumsum, orig_Tensor_cumsum, status, state

    script_path = os.sep.join(__file__.split(os.sep)[0:-4]) if root_path is None else root_path
    logger.debug(f"Script path is {script_path}")
    models_path = os.path.join(script_path, "models")
    embeddings_dir = os.path.join(script_path, "embeddings")
    dreambooth_models_path = os.path.join(models_path, "dreambooth")
    ckpt_dir = os.path.join(models_path, "Stable-diffusion")
    ui_lora_models_path = os.path.join(models_path, "Lora")
    db_model_config = None
    data_path = os.path.join(script_path, ".cache")
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
    debug = False
    profile_db = False
    sub_quad_q_chunk_size = 1024
    sub_quad_kv_chunk_size = None
    sub_quad_chunk_threshold = None
    CLIP_stop_at_last_layers = 2
    sd_model = None
    config = os.path.join(script_path, "configs", "v1-inference.yaml")
    force_cpu = False
    paths = []
    is_auto = load_auto_settings()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if getattr(torch, 'has_mps', False):
        try:
            torch.zeros(1).to(torch.device("mps"))
            device = torch.device("mps")
        except Exception:
            pass

    orig_tensor_to = torch.Tensor.to
    orig_layer_norm = torch.nn.functional.layer_norm
    # MPS workaround for https://github.com/pytorch/pytorch/issues/90532
    orig_tensor_numpy = torch.Tensor.numpy
    extension_path = os.path.join(script_path, "extensions", "sd_dreambooth_extension")

    orig_cumsum = torch.cumsum
    orig_Tensor_cumsum = torch.Tensor.cumsum

    if device.type == "mps":
        if version.parse(torch.__version__) < version.parse("1.13"):
            # PyTorch 1.13 doesn't need these fixes but unfortunately is slower and has regressions that prevent training from working
            torch.Tensor.to = tensor_to_fix
            torch.nn.functional.layer_norm = layer_norm_fix
            torch.Tensor.numpy = numpy_fix
        elif version.parse(torch.__version__) > version.parse("1.13.1"):
            if not torch.Tensor([1, 2]).to(torch.device("mps")).equal(
                    torch.Tensor([1, 1]).to(torch.device("mps")).cumsum(0, dtype=torch.int16)):
                torch.cumsum = lambda input, *args, **kwargs: (cumsum_fix(input, orig_cumsum, *args, **kwargs))
                torch.Tensor.cumsum = lambda self, *args, **kwargs: (
                    cumsum_fix(self, orig_Tensor_cumsum, *args, **kwargs))
            orig_narrow = torch.narrow
            torch.narrow = lambda *args, **kwargs: (orig_narrow(*args, **kwargs).clone())

    status = DreamState()

    if state is None:
        state = status


script_path = ""
models_path = ""
embeddings_dir = ""
dreambooth_models_path = ""
ckpt_dir = ""
ui_lora_models_path = ""
db_model_config = None
data_path = ""
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
debug = False
profile_db = False
sub_quad_q_chunk_size = 1024
sub_quad_kv_chunk_size = None
sub_quad_chunk_threshold = None
CLIP_stop_at_last_layers = 2
sd_model = None
config = ""
force_cpu = False
paths = []
device = torch.device("cpu")
orig_tensor_to = torch.Tensor.to
orig_layer_norm = torch.nn.functional.layer_norm
orig_tensor_numpy = torch.Tensor.numpy
extension_path = ""
status = None
orig_cumsum = torch.cumsum
orig_Tensor_cumsum = torch.Tensor.cumsum

load_vars()