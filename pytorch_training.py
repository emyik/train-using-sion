"""
Deep learning training cycle.
"""
from __future__ import annotations
import random
import sys
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import DatasetBuildin, DatasetDisk, BatchS3Dataset, Subset
import cnn_models
import logging_utils
import utils

LOGGER = logging_utils.initialize_logger()
DATALOG = logging_utils.get_logger("datalog")

SEED = 1234
NUM_EPOCHS = 10
DEVICE = "cuda:0"
LEARNING_RATE = 0.1

random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

class ConfigObject(object):
  """A dummy module used for Synchronizer for customizing __dict__"""
  __spec__ = None

def get_config(config=None):
    config = config or {}
    config.setdefault("dataset", "cifar")
    config.setdefault("ready", False)
    config.setdefault("cpu", False)
    config.setdefault("disk_source", "")
    config.setdefault("s3_source", "")
    config.setdefault("s3_train", "")
    config.setdefault("s3_test", "")
    config.setdefault("loader", "")
    config.setdefault("model", "")
    config.setdefault("batch", 64)
    config.setdefault("minibatch", 16)
    config.setdefault("epochs", 100)
    config.setdefault("accuracy", 1.0)
    config.setdefault("benchmark", False)
    config.setdefault("workers", 0)
    config.setdefault("output", "")
    config.setdefault("prefix", "")
    config.setdefault("test_mode", False)

    return config


def training_cycle(
    model: nn.Module,
    train_dataloader: DataLoader,
    optim_func: torch.optim.Adam = None,
    loss_fn: nn.CrossEntropyLoss = None,
    scheduler=None,
    num_epochs: int = 1,
    start_epoch: int = 1,
    device: str = "cuda"
):
    num_batches = len(train_dataloader)

    validation = num_epochs == 0
    if validation:
        num_epochs = 1

    loading_time = 0.0
    for epoch in range(num_epochs):
        batch_log_start = time.time()
        batch_log_loading = 0
        iteration = 0
        running_loss = 0.0
        model.train(mode=not validation)
        history = None
        epoch_start = time.time()

        batch_start = time.time()
        for idx, (images, labels) in enumerate(train_dataloader):
            batch_loading_time = time.time() - batch_start
            loading_time += batch_loading_time
            batch_log_loading += batch_loading_time
            images = images.to(device)
            labels = labels.to(device)

            if not validation:
                logits, _ = model(images)

                loss = loss_fn(logits, labels)
                optim_func.zero_grad()
                loss.backward()
                optim_func.step()
                running_loss += float(loss.item())
            else:
                with torch.no_grad():
                    logits, _ = model(images)

            iteration += 1
            history = count_top_k_preds(logits, labels, history)

            if idx % 100 == 99:
                if not validation:
                    DATALOG.info("%d,%d,%f,%f,%f,%f,%f", logging_utils.DATALOG_BATCH, epoch + start_epoch,
                                 batch_log_start, batch_log_loading, time.time() - batch_log_start, idx + 1, num_batches)
                print(
                    (
                        f"Epoch: {epoch+1:03d}/{num_epochs:03d} |"
                        f" Batch: {idx+1:03d}/{num_batches:03d} |"
                        f" Cost: {running_loss/iteration:.4f} |"
                        f" Elapsed: {time.time() - epoch_start:.3f} secs"
                    )
                )
                batch_log_start = time.time()
                batch_log_loading = 0
                iteration = 0
                running_loss = 0.0
                sys.stdout.flush()

            # reset batch start
            batch_start = time.time()

        top1, top5, total = history
        if scheduler:
            scheduler.step()

    return loading_time, top1 / total, top5 / total


def compare_pred_vs_actual(logit_scores: torch.Tensor, labels: torch.Tensor, silent: bool = False):
    logit_scores = logit_scores.to("cpu")
    labels = labels.to("cpu")
    logit_preds = torch.argmax(logit_scores, axis=1)
    num_correct = torch.sum(logit_preds == labels)
    perc_correct = num_correct / labels.shape[0] * 100
    if not silent:
        print(
            f"Num correct is: {num_correct}/{labels.shape[0]} ({perc_correct}%)")
    return perc_correct


def count_top_k_preds(logit_scores: torch.Tensor, labels: torch.Tensor, history=None):
    top1, top5, total = 0.0, 0.0, 0.0
    if history is not None:
        top1, top5, total = history

    predictions = logit_scores.topk(5)[1].data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    total += len(labels)
    for i in range(len(labels)):
        if labels[i] == predictions[i, 0]:
            top1 += 1
            top5 += 1
            continue

        for j in range(1, 4):
            if labels[i] == predictions[i, j]:
                top5 += 1
                break

    return top1, top5, total


def run_training_get_results(
    model: nn.Module,
    data_loader: DataLoader,
    validation_loader: DataLoader,
    optim_func: torch.optim,
    loss_fn: nn.CrossEntropyLoss,
    num_epochs: int,
    device: str,
    target_accuracy: float = 1.0,
):
    validation_loading_time = 0.0
    validation_time = 0.0
    training_loading_time = 0.0
    training_time = 0.0

    if validation_loader is not None:
        validation_start = time.time()
        validation_loading_time, accuracy, top5 = training_cycle(
            model, train_dataloader=validation_loader, num_epochs=0, start_epoch=0, device=device)
        validation_time = time.time() - validation_start
        LOGGER.info("Pretrained top-1 accuracy: %.3f, top-5 accuracy %.3f",
                    accuracy * 100, top5 * 100)
        DATALOG.info("%d,%d,%f,%f,%f,%f,%f", logging_utils.DATALOG_VALIDATION, 0,
                     validation_start, validation_loading_time, validation_time, accuracy, top5)

    max_lr = 0.01
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim_func, max_lr, epochs=num_epochs, steps_per_epoch=len(data_loader))
    for epoch in range(num_epochs):
        epoch_start = time.time()
        loading_time, accuracy, top5 = training_cycle(
            model, data_loader, optim_func, loss_fn, scheduler, 1, epoch+1, device=device)
        training_time += time.time() - epoch_start
        training_loading_time += loading_time
        LOGGER.info(
            "[Epoch %3d] Training for %s with %d samples, data loading time %.3f sec, training time %.3f sec, top accuracies %.3f, %.3f.",
            epoch + 1,
            str(data_loader.dataset),
            data_loader.dataset.total_samples,
            training_loading_time,
            training_time,
            accuracy * 100,
            top5 * 100,
        )
        DATALOG.info("%d,%d,%f,%f,%f,%f,%f", logging_utils.DATALOG_TRAINING, epoch + 1,
                     epoch_start, training_loading_time, training_time, accuracy, top5)

        if validation_loader is not None:
            validation_start = time.time()
            loading_time, accuracy, top5 = training_cycle(
                model, validation_loader, num_epochs=0, start_epoch=epoch+1, device=device)
            validation_time += time.time() - validation_start
            validation_loading_time += loading_time
            LOGGER.info("[Epoch %3d] Validation top-1 accuracy: %.3f, top-5 accuracy %.3f",
                        epoch + 1, accuracy * 100, top5 * 100)
            DATALOG.info("%d,%d,%f,%f,%f,%f,%f", logging_utils.DATALOG_VALIDATION, epoch + 1,
                         validation_start, validation_loading_time, validation_time, accuracy, top5)

            if accuracy >= target_accuracy:
                LOGGER.info("Accuracy reached.")
                break


def initialize_model(
    model_type: str, num_channels: int, num_classes: int = 10, device: str = DEVICE
) -> tuple[nn.Module, nn.CrossEntropyLoss, torch.optim.Adam]:
    if model_type == "resnet":
        print("Initializing Resnet50 model")
        model = cnn_models.Resnet50(num_channels, num_classes)
    elif model_type == "efficientnet":
        print("Initializing EfficientNetB4 model")
        model = cnn_models.EfficientNetB4(num_channels, num_classes)
    elif model_type == "densenet":
        print("Initializing DenseNet161 model")
        model = cnn_models.DenseNet161(num_channels, num_classes)
    else:
        print("Initializing BasicCNN model")
        model = cnn_models.BasicCNN(num_channels, num_classes)

    model = model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optim_func = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-4)
    # optim_func = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    return model, loss_fn, optim_func


def get_dataloader_times(data_loader: DataLoader):
    idx = len(data_loader)
    start_time = time.time()
    for i, _ in enumerate(data_loader):
        if not i % 100:
            print(i)
        if i > 200:
            break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}.")
    print(f"Time taken per iter: {(end_time - start_time) / idx}.")


def train(config):
    config = get_config(config)
    args = ConfigObject()
    args.__dict__ = config

    if args.s3_source == "None":
        args.s3_source = ""
    if args.s3_train == "None":
        args.s3_train = ""
    if args.s3_test == "None":
        args.s3_test = ""

    if args.output != "":
        output = args.prefix + args.output
        DATALOG = logging_utils.get_logger(
            DATALOG.name, logging_utils.set_file_handler(output))

    img_transform = utils.normalize_image(True)
    test_transform = utils.normalize_image(True)
    if args.dataset == "cifar":
        img_transform = utils.cifar_transforms_train()
    trainset, testset = None, None
    trainloader, testloader = None, None
    print(f"Config: {args.__dict__}")

        # Define the dataset
    loadtestset = (args.s3_test != "")
    if args.loader == "":
        from torchvision import datasets as buildins
        from filelock import FileLock
        if args.dataset == "cifar":
            lockpath = "{}.lock".format(args.dataset)
            with FileLock(lockpath):
                trainset = DatasetBuildin("CIFAR10", buildins.CIFAR10(root="data", train=True, download=True, transform=img_transform))
            with FileLock(lockpath):
                testset = DatasetBuildin("CIFAR10", buildins.CIFAR10(root="data", train=False, download=False, transform=test_transform)) # Downloaded
            loadtestset = True
        else:
            raise ValueError("Unsupport buildin dataset")
    elif args.loader == "disk":
        if args.dataset != "cifar10":
            raise ValueError("Unsupport disk dataset")

        if args.disk_source == "":
            raise ValueError("No path on disk specified")

        # Downloaded
        if args.ready:
            args.s3_train = ""
            args.s3_test = ""

        start_time = time.time()
        trainset = DatasetDisk(
            filepaths=[os.path.join(args.disk_source, "training")], s3_bucket=args.s3_train, label_idx=0, dataset_name=args.dataset + "_train", img_transform=img_transform
        )
        loading_time = time.time() - start_time
        if not args.ready:
            DATALOG.info("%d,%d,%f,%f,%f,%f,%f", logging_utils.DATALOG_LOAD_TRAINING,
                        0, start_time, loading_time, loading_time, len(trainset), len(trainset))

        if loadtestset:
            start_time = time.time()
            testset = DatasetDisk(
                filepaths=[os.path.join(args.disk_source, "test")], s3_bucket=args.s3_test, label_idx=0, dataset_name=args.dataset + "_test", img_transform=test_transform
            )
            loading_time = time.time() - start_time
            if not args.ready:
                DATALOG.info("%d,%d,%f,%f,%f,%f,%f", logging_utils.DATALOG_LOAD_VALIDATION,
                            0, start_time, loading_time, loading_time, len(testset), len(testset))
    elif args.loader == "infinicache":
        from sion_datasets import SionDataset
        import pysion.pysion as go_bindings

        go_bindings.GO_LIB = go_bindings.load_go_lib(
            os.path.join(os.path.dirname(__file__), "pysion", "pysion.so"))
        go_bindings.GO_LIB.initializeVars()

        if args.s3_train == "":
            args.s3_train = args.s3_source

        trainset = SionDataset(
            args.dataset + "_train",
            args.s3_train,
            obj_size=args.minibatch,
            img_transform=img_transform,
        )
        start_time = time.time()
        loading_time, total_samples = trainset.initial_set_all_data()
        DATALOG.info("%d,%d,%f,%f,%f,%f,%f", logging_utils.DATALOG_LOAD_TRAINING,
                    0, start_time, loading_time, loading_time, total_samples, total_samples)

        if loadtestset:
            testset = SionDataset(
                args.dataset + "_test",
                args.s3_test,
                obj_size=args.minibatch,
                img_transform=test_transform,
            )
            start_time = time.time()
            loading_time, total_samples = testset.initial_set_all_data()
            DATALOG.info("%d,%d,%f,%f,%f,%f,%f", logging_utils.DATALOG_LOAD_VALIDATION,
                        0, start_time, loading_time, loading_time, total_samples, total_samples)
    else:
        if args.s3_train == "":
            args.s3_train = args.s3_source

        trainset = BatchS3Dataset(
            args.s3_train, obj_size=args.minibatch, img_transform=img_transform
        )
        if loadtestset:
            testset = BatchS3Dataset(
                args.s3_test, obj_size=args.minibatch, img_transform=test_transform
            )

    # Define the dataloader
    collate_fn = None
    batch = args.batch
    if isinstance(trainset, BatchS3Dataset):
        collate_fn = utils.infinicache_collate
        batch = args.batch/args.minibatch
    if args.test_mode:
        # If test mode is enabled, we only load a batch
        trainset = Subset(trainset, list(range(args.batch)))
        if loadtestset:
            testset = Subset(testset, list(range(args.batch)))
    trainloader = DataLoader(
        trainset, batch_size=int(batch), shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
    )
    if loadtestset:
        testloader = DataLoader(
            testset, batch_size=int(batch), shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
        )

    # Define the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cpu:
        device = "cpu"
    num_classes = 10
    if args.dataset == "places":
        num_classes = 365
    model, loss_fn, optim_func = initialize_model(
        args.model, 3, num_classes=num_classes, device=device)
    print("Running training with the {}".format(args.loader))
    if args.benchmark:
        args.epochs = 0
    run_training_get_results(
        model, trainloader, testloader, optim_func, loss_fn, args.epochs, device, target_accuracy=args.accuracy
    )
