"""MNIST dataset wrapper for zkml-bench."""

from __future__ import annotations

import os
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from ..exceptions import DatasetDownloadError
from .base import BenchmarkDataset


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """Seed numpy and stdlib random inside each DataLoader worker."""
    import numpy as np

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class MNISTDataset(BenchmarkDataset):
    """Wrapper around torchvision MNIST for zkml-bench.

    Parameters
    ----------
    data_dir:
        Directory where the MNIST files will be stored / read from.
        Tilde expansion is applied automatically.
    """

    name = "mnist"
    num_classes = 10
    input_shape = (1, 28, 28)

    # Standard MNIST normalisation statistics
    _MEAN = (0.1307,)
    _STD = (0.3081,)

    def __init__(self, data_dir: str = "~/.zkml_bench/data") -> None:
        self._data_dir = os.path.expanduser(data_dir)
        os.makedirs(self._data_dir, exist_ok=True)
        self._pin_memory = torch.cuda.is_available()

        # Normalisation transform (no augmentation for MNIST)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
            ]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_train_dataset(self) -> torchvision.datasets.MNIST:
        """Download (if needed) and return the raw training dataset."""
        try:
            return torchvision.datasets.MNIST(
                root=self._data_dir,
                train=True,
                download=True,
                transform=self._transform,
            )
        except Exception as e:
            raise DatasetDownloadError(
                "Failed to download MNIST. "
                "Manual download: http://yann.lecun.com/exdb/mnist/ "
                f"to {self._data_dir}. Original error: {e}"
            ) from e

    def _load_test_dataset(self) -> torchvision.datasets.MNIST:
        """Download (if needed) and return the raw test dataset."""
        try:
            return torchvision.datasets.MNIST(
                root=self._data_dir,
                train=False,
                download=True,
                transform=self._transform,
            )
        except Exception as e:
            raise DatasetDownloadError(
                "Failed to download MNIST. "
                "Manual download: http://yann.lecun.com/exdb/mnist/ "
                f"to {self._data_dir}. Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # BenchmarkDataset interface
    # ------------------------------------------------------------------

    def train_loader(self, batch_size: int, seed: int) -> DataLoader:
        """Return a shuffled DataLoader over the MNIST training set."""
        dataset = self._load_train_dataset()
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=self._pin_memory,
            persistent_workers=True,
            worker_init_fn=_worker_init_fn,
            generator=generator,
        )

    def test_loader(self, batch_size: int) -> DataLoader:
        """Return an unshuffled DataLoader over the MNIST test set."""
        dataset = self._load_test_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=self._pin_memory,
        )

    def calibration_loader(self, n_samples: int, seed: int) -> DataLoader:
        """Return a DataLoader over *n_samples* training examples.

        The subset indices are drawn with ``seed`` for reproducibility.
        No augmentation is applied (same transform as test).
        """
        dataset = self._load_train_dataset()
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), k=n_samples)
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=n_samples,
            shuffle=False,
            num_workers=0,
            pin_memory=self._pin_memory,
        )

    def sample_input(self, batch_size: int = 1) -> torch.Tensor:
        """Return the first *batch_size* test images as a deterministic tensor."""
        dataset = self._load_test_dataset()
        indices = list(range(batch_size))
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        images, _ = next(iter(loader))
        return images
