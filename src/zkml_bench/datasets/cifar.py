"""CIFAR-10 and CIFAR-100 dataset wrappers for zkml-bench."""

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


class Cutout:
    """Randomly mask a square patch (length × length) of the input tensor with zeros.

    Applied after ToTensor + Normalize. Standard regularizer for small-image datasets.
    """

    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones(h, w, dtype=img.dtype, device=img.device)
        for _ in range(self.n_holes):
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------


class CIFAR10Dataset(BenchmarkDataset):
    """Wrapper around torchvision CIFAR-10 for zkml-bench.

    Parameters
    ----------
    data_dir:
        Directory where CIFAR-10 files will be stored / read from.
        Tilde expansion is applied automatically.
    """

    name = "cifar10"
    num_classes = 10
    input_shape = (3, 32, 32)

    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2470, 0.2435, 0.2616)

    def __init__(self, data_dir: str = "~/.zkml_bench/data") -> None:
        self._data_dir = os.path.expanduser(data_dir)
        os.makedirs(self._data_dir, exist_ok=True)
        self._pin_memory = torch.cuda.is_available()

        self._train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
                Cutout(n_holes=1, length=16),
            ]
        )
        self._test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
            ]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_train_dataset(
        self, augment: bool = True
    ) -> torchvision.datasets.CIFAR10:
        transform = self._train_transform if augment else self._test_transform
        try:
            return torchvision.datasets.CIFAR10(
                root=self._data_dir,
                train=True,
                download=True,
                transform=transform,
            )
        except Exception as e:
            raise DatasetDownloadError(
                "Failed to download CIFAR-10. "
                "Manual download: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
                f"to {self._data_dir}. Original error: {e}"
            ) from e

    def _load_test_dataset(self) -> torchvision.datasets.CIFAR10:
        try:
            return torchvision.datasets.CIFAR10(
                root=self._data_dir,
                train=False,
                download=True,
                transform=self._test_transform,
            )
        except Exception as e:
            raise DatasetDownloadError(
                "Failed to download CIFAR-10. "
                "Manual download: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
                f"to {self._data_dir}. Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # BenchmarkDataset interface
    # ------------------------------------------------------------------

    def train_loader(self, batch_size: int, seed: int) -> DataLoader:
        """Return a shuffled DataLoader over the CIFAR-10 training set."""
        dataset = self._load_train_dataset(augment=True)
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
        """Return an unshuffled DataLoader over the CIFAR-10 test set."""
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

        Uses test transforms (no augmentation) and a fixed random subset
        chosen with ``seed``.
        """
        dataset = self._load_train_dataset(augment=False)
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


# ---------------------------------------------------------------------------
# CIFAR-100
# ---------------------------------------------------------------------------


class CIFAR100Dataset(BenchmarkDataset):
    """Wrapper around torchvision CIFAR-100 for zkml-bench.

    Parameters
    ----------
    data_dir:
        Directory where CIFAR-100 files will be stored / read from.
        Tilde expansion is applied automatically.
    """

    name = "cifar100"
    num_classes = 100
    input_shape = (3, 32, 32)

    _MEAN = (0.5071, 0.4867, 0.4408)
    _STD = (0.2675, 0.2565, 0.2761)

    def __init__(self, data_dir: str = "~/.zkml_bench/data") -> None:
        self._data_dir = os.path.expanduser(data_dir)
        os.makedirs(self._data_dir, exist_ok=True)
        self._pin_memory = torch.cuda.is_available()

        self._train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
                Cutout(n_holes=1, length=16),
            ]
        )
        self._test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
            ]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_train_dataset(
        self, augment: bool = True
    ) -> torchvision.datasets.CIFAR100:
        transform = self._train_transform if augment else self._test_transform
        try:
            return torchvision.datasets.CIFAR100(
                root=self._data_dir,
                train=True,
                download=True,
                transform=transform,
            )
        except Exception as e:
            raise DatasetDownloadError(
                "Failed to download CIFAR-100. "
                "Manual download: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz "
                f"to {self._data_dir}. Original error: {e}"
            ) from e

    def _load_test_dataset(self) -> torchvision.datasets.CIFAR100:
        try:
            return torchvision.datasets.CIFAR100(
                root=self._data_dir,
                train=False,
                download=True,
                transform=self._test_transform,
            )
        except Exception as e:
            raise DatasetDownloadError(
                "Failed to download CIFAR-100. "
                "Manual download: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz "
                f"to {self._data_dir}. Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # BenchmarkDataset interface
    # ------------------------------------------------------------------

    def train_loader(self, batch_size: int, seed: int) -> DataLoader:
        """Return a shuffled DataLoader over the CIFAR-100 training set."""
        dataset = self._load_train_dataset(augment=True)
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
        """Return an unshuffled DataLoader over the CIFAR-100 test set."""
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

        Uses test transforms (no augmentation) and a fixed random subset
        chosen with ``seed``.
        """
        dataset = self._load_train_dataset(augment=False)
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
