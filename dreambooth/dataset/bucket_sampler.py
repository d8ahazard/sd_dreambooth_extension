import random
from typing import Tuple

from dreambooth.dataset.db_dataset import DbDataset


class BucketSampler:
    def __init__(self, dataset: DbDataset, batch_size, debug=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.resolutions = dataset.resolutions
        self.active_resos = []
        self.bucket_counter = BucketCounter(starting_keys=self.resolutions)
        self.batch = []
        self.current_bucket = 0
        self.current_index = 0
        self.total_samples = 0
        self.debug = debug
        self.set_buckets()

    def __iter__(self):
        while self.total_samples < len(self.dataset):
            batch = self.fill_batch()
            if len(batch) == 0:
                raise StopIteration
            yield batch
        self.total_samples = 0

    def __next__(self):
        if len(self.batch) == 0:
            self.batch = self.fill_batch()
        if len(self.batch) == 0:
            print("Well, this is bad. We have no batch data.")
            raise StopIteration
        return self.batch.pop()

    def __len__(self):
        return len(self.dataset.active_resolution) * self.batch_size

    def set_buckets(self):
        # Initialize list of bucket counts if not set
        all_resos = self.resolutions
        resos_to_use = []
        am = self.bucket_counter.missing()
        missing = am.copy()
        pop_index = 0
        # Tell the counter to check if all values are equal. If so, reset the counts to 0
        self.bucket_counter.check_reset()
        if len(missing):
            while len(resos_to_use) < len(all_resos):
                if len(missing):
                    for res, count in am.items():
                        resos_to_use.append(res)
                        missing[res] -= 1
                        if missing[res] == 0:
                            del missing[res]
                        am = missing.copy()
                else:
                    resos_to_use.append(all_resos[pop_index])
                    pop_index += 1
                    if pop_index >= len(all_resos):
                        pop_index = 0
        else:
            resos_to_use = all_resos.copy()
        if not self.debug:
            random.shuffle(resos_to_use)
        self.active_resos = resos_to_use
        self.current_bucket = 0

    def fill_batch(self):
        current_res = self.active_resos[self.current_bucket]
        self.dataset.shuffle_buckets()
        batch = []
        repeats = 0
        while len(batch) < self.batch_size:
            self.dataset.active_resolution = current_res
            img_index, img_repeats = self.dataset.get_example(current_res)
            # next_item = torch.as_tensor(next_item, device='cpu', dtype=torch.float)
            if img_repeats != 0:
                self.bucket_counter.count(current_res)
                repeats += 1
            batch.append(img_index)
            self.total_samples += 1
        if repeats != 0:
            # Increment bucket if we've 'emptied' the current one
            self.current_bucket += 1
            # If we've run through our list of resolutions, re-create it
            if self.current_bucket >= len(self.active_resos):
                self.set_buckets()
        return batch

    def __getitem__(self, index):
        if len(self.batch) == 0:
            self.batch = self.fill_batch()
        if len(self.batch) == 0:
            print("Well, this is bad. We have no batch data.")
            raise StopIteration
        return self.batch.pop()


class BucketCounter:
    def __init__(self, starting_keys=None):
        self.counts = {}
        print("Initializing bucket counter!")
        if starting_keys is not None:
            for key in starting_keys:
                self.counts[key] = 0

    def count(self, key: Tuple[int, int]):
        if key in self.counts:
            self.counts[key] += 1
        else:
            self.counts[key] = 1

    def min(self):
        return min(self.counts.values()) if len(self.counts) else 0

    def max(self):
        return max(self.counts.values()) if len(self.counts) else 0

    def get(self, key: Tuple[int, int]):
        return self.counts[key] if key in self.counts else 0

    def check_reset(self):
        if self.max() == self.min():
            for key in list(self.counts.keys()):
                self.counts[key] = 0

    def missing(self):
        out = {}
        max = self.max()
        for key in list(self.counts.keys()):
            if self.counts[key] < max:
                out[key] = max - self.counts[key]
        return out

    def print(self):
        print(f"Bucket counts: {self.counts}")
