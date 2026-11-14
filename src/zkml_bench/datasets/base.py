"""Abstract base class for benchmark datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


class BenchmarkDataset(ABC):
    """Abstract base class for all benchmark datasets.

    Subclasses must declare the ``name``, ``num_classes``, and ``input_shape``
    class attributes and implement all abstract methods.
    """

    name: str
    num_classes: int
    input_shape: tuple[int, ...]  # (C, H, W)

    @abstractmethod
    def train_loader(self, batch_size: int, seed: int) -> DataLoader:
        """Return a DataLoader over the full training set.

        Parameters
        ----------
        batch_size:
            Number of samples per batch.
        seed:
            Random seed used to initialise the sampler and worker RNGs so that
            training is reproducible across runs.
        """

    @abstractmethod
    def test_loader(self, batch_size: int) -> DataLoader:
        """Return a DataLoader over the full test set (no shuffle).

        Parameters
        ----------
        batch_size:
            Number of samples per batch.
        """

    @abstractmethod
    def calibration_loader(self, n_samples: int, seed: int) -> DataLoader:
        """Return a DataLoader over a fixed subset of the training set.

        The subset is used for EZKL calibration and is selected with the
        given ``seed`` so results are reproducible.  Augmentation transforms
        are *not* applied; only normalisation is used.

        Parameters
        ----------
        n_samples:
            Number of training samples to include in the subset.
        seed:
            Random seed used to select the subset indices.
        """

    @abstractmethod
    def sample_input(self, batch_size: int = 1) -> torch.Tensor:
        """Return a deterministic dummy input for ZK cost analysis.

        Always returns the first ``batch_size`` samples from the test set
        (without augmentation) so the tensor is identical across calls.

        Parameters
        ----------
        batch_size:
            Number of samples to return.
        """
