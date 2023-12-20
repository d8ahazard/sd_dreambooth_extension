import threading

from finetune.helpers.ft_state import FinetuneStatus


class DownloadProgress:
    def __init__(self, total_size: int):
        self.total_size = total_size
        self.bytes_downloaded = 0
        self.s_task = FinetuneStatus()
        self.s_task.total_steps = 100
        self.lock = threading.Lock()

    def update(self, bytes_count):
        with self.lock:
            self.bytes_downloaded += bytes_count
            percentage = (self.bytes_downloaded / self.total_size) * 100
            # Total size and bytes_downloaded are in bytes, convert them to Mb
            total_mb = self.total_size / 1024 / 1024
            downloaded_mb = self.bytes_downloaded / 1024 / 1024
            # Convert to 2 decimal places
            total_mb = round(total_mb, 2)
            downloaded_mb = round(downloaded_mb, 2)
            if percentage > 100:
                percentage = 100
            if downloaded_mb > total_mb:
                downloaded_mb = total_mb
            self.s_task.step(percentage, f"Downloaded {downloaded_mb}MB of {total_mb} MB.")
