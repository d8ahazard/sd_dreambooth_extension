from enum import Enum


class FinetuneState(Enum):
    IDLE = 0
    RUNNING = 1
    FINISHED = 2
    CANCELLED = 3


class FinetuneStatus:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not FinetuneStatus._instance:
            FinetuneStatus._instance = super(FinetuneStatus, cls).__new__(cls, *args, **kwargs)
        return FinetuneStatus._instance

    def __init__(self):
        self.total_jobs = 0
        self.current_job = 0
        self.total_steps = 0
        self.current_step = 0
        self.samples = {}  # Dictionary to store sample images
        self.state = FinetuneState.IDLE
        self.status = ""

    def start(self):
        self.total_jobs = 0
        self.current_job = 0
        self.total_steps = 0
        self.current_step = 0
        self.samples.clear()  # Clear existing samples
        self.state = FinetuneState.RUNNING

    def stop(self, description: str = None):
        self.state = FinetuneState.FINISHED
        self.current_job = self.total_jobs
        self.current_step = self.total_steps
        if description:
            self.status = description

    def cancel(self):
        self.state = FinetuneState.CANCELLED

    def step_job(self, n=1):
        self.current_job += n
        if self.current_job > self.total_jobs:
            self.total_jobs = self.current_job

    def step(self, n=1, description: str = None):
        self.current_step += n
        if self.current_step > self.total_steps:
            self.total_steps = self.current_step
        if description:
            self.status = description

    def add_sample(self, path, prompt, params=None, clear=False):
        if clear:
            self.samples.clear()
        if not params:
            params = {}
        self.samples[path] = {'prompt': prompt, 'params': params}

    def clear_samples(self):
        self.samples.clear()

    def get_samples(self):
        return self.samples

    def progress_bar_html(self):
        if self.state == "IDLE":
            return ""
        elif self.state == "RUNNING":
            return f"<div class='progress'><div class='progress-bar' role='progressbar' style='width: {self.current_step / self.total_steps * 100}%'></div></div>"
        elif self.state == "FINISHED":
            return f"<div class='progress'><div class='progress-bar bg-success' role='progressbar' style='width: 100%'></div></div>"
        elif self.state == "CANCELLED":
            return f"<div class='progress'><div class='progress-bar bg-danger' role='progressbar' style='width: 100%'></div></div>"
        else:
            return ""
