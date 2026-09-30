"""
Dataset that returns flattened MNIST/FMNIST pixel values as regression targets
instead of integer class labels.

The INR file naming convention is:
    {split}_png_{train|test}/{digit}/{counter:06d}.pth

where counter is a per-digit sequential index assigned during generate_inrs.py.
To recover the original image, we rebuild the per-digit counter → global MNIST index
mapping at init time.
"""

from collections import defaultdict
from pathlib import Path

import torch
from torchvision import datasets

from scalegmn.src.data.mnist_inr_dataset import LabeledINRDataset

# Module-level cache: load MNIST/FMNIST images once, share across all dataset instances.
_IMAGE_CACHE = {}


def _get_image_data(data_root: str, dataset_name: str):
    """Load and cache image tensors + per-digit index mappings."""
    key = (data_root, dataset_name)
    if key not in _IMAGE_CACHE:
        ds_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
        train_ds = ds_cls(root=data_root, train=True, download=False)
        test_ds = ds_cls(root=data_root, train=False, download=False)

        train_images = train_ds.data.float().div_(255.0).reshape(-1, 784)  # [60000, 784]
        test_images = test_ds.data.float().div_(255.0).reshape(-1, 784)  # [10000, 784]

        train_digit_to_indices = defaultdict(list)
        for i, label in enumerate(train_ds.targets.tolist()):
            train_digit_to_indices[label].append(i)

        test_digit_to_indices = defaultdict(list)
        for i, label in enumerate(test_ds.targets.tolist()):
            test_digit_to_indices[label].append(i)

        _IMAGE_CACHE[key] = (train_images, test_images,
                             train_digit_to_indices, test_digit_to_indices)
    return _IMAGE_CACHE[key]


class ImageRegressionINRDataset(LabeledINRDataset):
    """INR graph dataset where label = flattened original image pixels [784]."""

    def __init__(self, data_root: str, dataset_name: str = "mnist",
                 fanin_rescale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._fanin_rescale = fanin_rescale
        self._build_index(data_root, dataset_name)

    def _build_index(self, data_root: str, dataset_name: str):
        """Build an index mapping dataset position → (image_tensor_ref, global_idx)."""
        train_images, test_images, train_d2i, test_d2i = _get_image_data(data_root, dataset_name)

        # Store indices into the shared image tensors (no per-instance copies)
        self._image_indices = []  # list of (is_train: bool, global_idx: int)
        for path_str in self.dataset["path"]:
            parts = Path(path_str).parts
            is_train = "train" in path_str
            digit = int(parts[-2])
            counter = int(Path(parts[-1]).stem)

            if is_train:
                global_idx = train_d2i[digit][counter]
            else:
                global_idx = test_d2i[digit][counter]
            self._image_indices.append((is_train, global_idx))

        self._train_images = train_images
        self._test_images = test_images

    def batch_to_graphs(self, weights, biases, input_emb=None, **kwargs):
        if self._fanin_rescale:
            weights = tuple(w * w.shape[0] for w in weights)
        return super().batch_to_graphs(weights, biases, input_emb, **kwargs)

    def get_label(self, index, state_dict, aux):
        is_train, global_idx = self._image_indices[index]
        if is_train:
            return self._train_images[global_idx]
        return self._test_images[global_idx]
