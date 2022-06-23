from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from threading import Lock
from functools import partial
from random import randint

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torch.utils.data import Subset as SubDataSet
from zmq import has

import logging_utils

LOGGER = logging_utils.initialize_logger()
DATALOG = logging_utils.get_logger("datalog")
INITIALIZE_WORKERS = 10

class LoadTimes:
    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.avg = 0
        self.sum = 0
        self.lock = Lock()

    def reset(self):
        with self.lock:
            self.num_loads = 0
            self.avg = 0
            self.sum = 0

    def update(self, val: Union[float, int]):
        with self.lock:
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: Average={self.avg:.03f}\tSum={self.sum:.03f}\tCount={self.count}"

def DatasetBuildin(name: str, dataset: Dataset):
    dataset.total_samples = len(dataset)
    dataset.__dict__["__str__"] = lambda: "{}_DatasetBuildin".format(name)
    return dataset

class Subset(SubDataSet):
    def __init__(self, dataset: Dataset, indices: torch.Tensor):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.total_samples = len(indices)

    def __str__(self):
        return self.dataset.__str__()

class DatasetDisk(Dataset):
    """Simulates having to load each data point from disk every call."""

    def __init__(
        self,
        filepaths: list[str],
        label_idx: int,
        dataset_name: str,
        img_transform: Optional[torchvision.transforms.Compose] = None,
        s3_bucket: str = "",
    ):
        self.dataset_name = dataset_name
        
        if len(filepaths) == 1:
            localpath = Path(filepaths[0])

            # Download
            if s3_bucket != "":
                self.s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))   
                self.download_from_s3(s3_bucket, localpath.absolute())  # We probably don't need this since the AWS CLI is faster\
            
            # Expand
            filepaths = list(localpath.rglob("*.png"))
            filepaths.extend(list(localpath.rglob("*.jpg")))
            filepaths = list(map(lambda x: str(x), filepaths))

        self.filepaths = np.array(filepaths)
        self.label_idx = label_idx
        self.img_transform = img_transform
        self.total_samples = 0

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        label = os.path.basename(self.filepaths[idx]).split(".")[0].split("_")[self.label_idx]
        pil_img = Image.open(self.filepaths[idx])
        if self.img_transform:
            img_tensor = self.img_transform(pil_img)
        else:
            img_tensor = F.pil_to_tensor(pil_img)
            img_tensor = img_tensor.to(torch.float32).div(255)

        self.total_samples += 1

        return img_tensor, int(label)

    def __str__(self):
        return f"{self.dataset_name}_DatasetDisk"

    def download_from_s3(self, s3_path: str, local_path: str):
        """
        First need to download all images from S3 to Disk to use for training.
        """
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        paginator = s3_client.get_paginator("list_objects_v2")
        filenames = []
        for page in paginator.paginate(Bucket=s3_path):
            for content in page.get("Contents"):
                filenames.append(content["Key"])
        partial_dl = partial(self.download_file, local_path, s3_path)
        LOGGER.info("Downloading %s from S3 to Disk", self.dataset_name)
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(partial_dl, fname) for fname in filenames]
            _ = [future.result() for future in as_completed(futures)]
        LOGGER.info("Download is complete")

    def download_file(self, output_dir: str, bucket_name: str, file_name: str):
        self.s3_client.download_file(bucket_name, file_name, f"{output_dir}/{file_name}")

class BatchS3Dataset(Dataset):
    """Simulates having to load each data point from S3 every call."""

    def __init__(
        self, 
        bucket_name: str,
        dataset_name: str = None,
        obj_size: int = 16,
        label_idx: int = 0,
        img_transform: Optional[torchvision.transforms.Compose] = None,
    ):
        self.s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self.bucket_name = bucket_name
        self.dataset_name = dataset_name
        if self.dataset_name is None:
            self.dataset_name = bucket_name
            LOGGER.info("Initializing dataset from s3(%s)", bucket_name)
        else:
            LOGGER.info("Initializing dataset %s from s3(%s)", dataset_name, bucket_name)
        self.label_idx = label_idx
        self.object_size = obj_size
        self.img_transform = img_transform
        
        # Define some statistics
        self.total_samples = 0
        
        # Load metadata from S3
        paginator = self.s3_client.get_paginator("list_objects_v2")
        filenames = []
        labels = []
        for page in paginator.paginate(Bucket=bucket_name):
            for content in page.get("Contents"):
                filenames.append(content["Key"])
                labels.append(int(content["Key"].split(".")[0].split("_")[self.label_idx]))

        
        # Chunk the filenames into objects of size self.object_size where self.object_size is the
        # number of images.
        multiple_len = len(filenames) - (len(filenames) % self.object_size)
        filenames_arr = np.array(filenames[:multiple_len])
        assert len(filenames_arr) % self.object_size == 0
        labels_arr = np.array(labels[:multiple_len])

        # Needs to be a numpy array to avoid memory leaks with multiprocessing:
        #       https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # Keep original data
        self.chunked_fpaths = np.array(np.split(filenames_arr, (len(filenames_arr) // self.object_size)))
        self.chunked_labels = np.array(np.split(labels_arr, (len(labels_arr) // self.object_size)))

        LOGGER.info("Dataset {} initialized".format(self.dataset_name))

    def __len__(self):
        return len(self.chunked_fpaths)

    def __getitem__(self, idx: int):
        num_samples = len(self.chunked_fpaths[idx])

        np_arr, labels = self.get_s3_threaded(idx)
        np_arr, labels = self.shuffle(np_arr, labels)
        images = torch.stack(list(map(lambda x: self.load_image(x), np_arr)))

        data = (images, torch.tensor(labels))
        self.total_samples += num_samples
        return data

    def shuffle(self, arr1, arr2):
        """
        Permute the arrays synchornizely.
        """
        if len(arr1) != len(arr2):
            raise ValueError("Arrays to be shuffled must be the same length")
        for i in range(len(arr1)-1,0,-1):
            j = randint(0,i)
            arr1[i], arr1[j] = arr1[j], arr1[i]
            arr2[i], arr2[j] = arr2[j], arr2[i]
        
        return arr1, arr2

    def get_s3_threaded(self, idx: int):
        fpaths = self.chunked_fpaths[idx]
        with ThreadPoolExecutor(len(fpaths)) as executor:
            futures = [executor.submit(self.load_s3, f) for f in fpaths]
            # Returns tensor of shape [object_size, num_channels, H, W]
            # results = torch.stack([future.result() for future in as_completed(futures)])
            results = [future.result() for future in as_completed(futures)]
        
        # Transform bytes to tensor-ready data
        labels = torch.tensor(self.chunked_labels[idx])
        # Transform to np.array for convience
        labels = np.array(labels, dtype=np.uint8)

        return results, labels

    def load_image(self, img_bytes: bytes) -> torch.Tensor:
        pil_img = Image.open(BytesIO(img_bytes))
        if self.img_transform:
            img_tensor = self.img_transform(pil_img)
        else:
            img_tensor = F.pil_to_tensor(pil_img)
            img_tensor = img_tensor.to(torch.float32).div(255)
        return img_tensor

    def load_s3(self, s3_prefix: str) -> any:
        s3_png = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_prefix)
        img_bytes = s3_png["Body"].read()
        return img_bytes

    def __str__(self):
        return f"{self.bucket_name}_DatasetS3"