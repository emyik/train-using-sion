from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
import os


def split_cifar_data(cifar_data_dir: str, test_fnames_path: str) -> list[str]:
    """
    Expects all files to be in a single directory with the same filenames as in the S3 bucket.
    """
    dataset_path = Path(cifar_data_dir)
    filenames = list(dataset_path.rglob("*.png"))
    filenames.extend(list(dataset_path.rglob("*.jpg")))
    with open(test_fnames_path) as f:
        test_fnames = set([fname.strip() for fname in f.readlines()])
    filestubs = set(map(lambda x: x.name, filenames))
    train_filenames = list(filestubs.difference(test_fnames))
    train_filenames = sorted(train_filenames, key=lambda filename: filename)
    train_filenames = list(map(lambda x: os.path.join(cifar_data_dir, x), train_filenames))
    test_filenames = sorted(test_fnames, key=lambda filename: filename)
    test_filenames = list(map(lambda x: os.path.join(cifar_data_dir, x), test_filenames))

    return train_filenames, test_filenames


def normalize_image(channels: bool):
    if not channels:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485), (0.229)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    return transform


def cifar_transforms_train():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def infinicache_collate(batch):
    """Custom collate function to work with the mini-objects
    """
    images, labels = zip(*batch)
    new_images = torch.cat(images, 0)
    new_labels = torch.cat(labels, 0)
    return new_images, new_labels

###############################################################################
# Visualization and Prediction
###############################################################################

CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck')

def get_class(id):
    return CLASSES[id]

def show_images(images, labels):
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    imshow(torchvision.utils.make_grid(images))
    print("Labels: ", labels)
    print("Labels: ",
          " ".join(f"{get_class(label):5s}" for label in labels))

def predict(model, images):
    outputs, _ = model(images)
    _, predictions = torch.max(outputs, 1)

    print("Predictions: ", predictions)
    print("Predictions: ",
          " ".join(f"{get_class(prediction):5s}" for prediction in predictions))
    return predictions

def predict_and_display(model, images, labels):
    show_images(images, labels)
    print()
    predictions = predict(model, images)
    print()
    num_correct = predictions.eq(labels).sum().item()
    num_total = len(images)
    print(
        f"Predicted {num_correct}/{num_total} correctly. "
        f"Accuracy: {num_correct / num_total:.0%}.")