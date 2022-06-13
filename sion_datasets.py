from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import torch
import torchvision

from datasets import BatchS3Dataset
import pysion.pysion as go_bindings
import logging_utils

LOGGER = logging_utils.initialize_logger()
DATALOG = logging_utils.get_logger("datalog")
INITIALIZE_WORKERS = 10

class SionDataset(BatchS3Dataset):
    def __init__(
        self,
        dataset_name: str,
        s3_bucket: str,
        obj_size: int = 16,
        label_idx: int = 0,
        img_transform: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__(
            bucket_name=s3_bucket,
            dataset_name=dataset_name,
            obj_size=obj_size,
            label_idx=label_idx,
            img_transform=img_transform,
        )

        # Defines object data
        self.base_keyname = f"{dataset_name}-{self.object_size}-"
        self.labels = np.ones(self.chunked_labels.shape, dtype=np.uint8)
        self.metas = np.full(len(self.chunked_labels), None)

    def __getitem__(self, idx: int):
        num_samples = len(self.chunked_fpaths[idx])
        key = f"{self.base_keyname}-{idx:05d}"
       
        try:
            meta = self.metas[idx]
            if meta is None:
                raise KeyError("Key not set")
            bytes = go_bindings.get_array_from_cache(go_bindings.GO_LIB, key, meta[0])
            # images = torch.tensor(np_arr).reshape(self.data_shape)
            np_arr = self.unwrap(bytes, meta)
            labels = self.labels[idx]
            np_arr, labels = self.shuffle(np_arr, np.copy(labels))

            # Keep load_images in try block, so we may reset it if necessary
            images = torch.stack(list(map(lambda x: self.load_image(x), np_arr)))
        except Exception as e:
            LOGGER.warn("{} Resetting image {} due to {}".format(idx, key, e))
            np_arr, labels = self.set_in_cache(idx)
            np_arr, labels = self.shuffle(np_arr, np.copy(labels))
            images = torch.stack(list(map(lambda x: self.load_image(x), np_arr)))

        data = (images, torch.tensor(labels))
        self.total_samples += num_samples
        return data

    def set_in_cache(self, idx: int):
        key = f"{self.base_keyname}-{idx:05d}"
        # LOGGER.debug("{}. setting images {}".format(idx, key))
        img_bytes, labels = self.get_s3_threaded(idx)
        arr = np.array(img_bytes)
        self.labels[idx] = labels
        obj_bytes = self.wrap(arr)
        self.metas[idx] = [len(obj_bytes), arr.dtype]
        # LOGGER.debug("Setting in cache: {} images, read {} bytes, setting {} bytes: {}".format(len(images), bytes_loaded, len(obj_bytes), list(map(lambda x: len(x), arr))))
        go_bindings.set_array_in_cache(go_bindings.GO_LIB, key, obj_bytes)
        return arr, self.labels[idx]

    def set_in_cache_threaded(self, idx: int):
        self.initial_progress[idx] = 2
        _ = self.set_in_cache(idx)
        self.initial_progress[idx] = 1
        return idx

    def track_threads(self, future):
        ret = future.result()
        self.initial_finished += 1
        start = ret - INITIALIZE_WORKERS
        if start < 0:
            start = 0
        end = ret + INITIALIZE_WORKERS
        if end > len(self.chunked_fpaths):
            end = len(self.chunked_fpaths)
        lookback = list(filter(lambda x: self.initial_progress[x] != 1, range(start, ret)))
        lookahead = list(filter(lambda x: self.initial_progress[x] == 2, range(ret, end)))
        LOGGER.debug("Initiated {}/{} objects, status {}:{}:{}".format(
            self.initial_finished,
            len(self.initial_progress),
            list(map(lambda x: -x if self.initial_progress[x] == 0 else x, lookback)),
            ret, 
            lookahead
        ))
        return ret

    def wrap(self, arr: np.ndarray) -> bytes:
        return arr.tobytes()

    def unwrap(self, bytes_arr: bytes, meta: any) -> np.ndarray:
        arr = np.frombuffer(bytes_arr, dtype=meta[1])
        # LOGGER.debug("Unwraped {} bytes: {}".format(len(bytes_arr), list(map(lambda x: len(x), arr))))
        # array from buffer is readonly. We will need to shuffle the array, so return a copy.
        return np.copy(arr)

    def initial_set_all_data(self):
        idxs = list(range(len(self.chunked_fpaths)))
        LOGGER.info("Loading {} into InfiniCache in parallel".format(self.dataset_name))

        start_time = time.time()
        self.initial_progress = np.full(len(idxs), 0)
        self.initial_finished = 0
        with ThreadPoolExecutor(max_workers=INITIALIZE_WORKERS) as executor:
            futures = [executor.submit(self.set_in_cache_threaded, idx) for idx in idxs]
            _ = [self.track_threads(future) for future in as_completed(futures)]
            LOGGER.info("DONE with initial SET into InfiniCache")
        end_time = time.time()
        time_taken = end_time - start_time

        LOGGER.info(
            "Finished Setting Data in InfiniCache. Total load time for %d samples is %.3f sec.",
            self.total_samples,
            time_taken,
        )
        return time_taken, self.total_samples

    def __str__(self):
        return f"{self.dataset_name}_SionDataset"
