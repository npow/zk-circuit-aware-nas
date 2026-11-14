"""Dataset loaders for zkml-bench."""

from .base import BenchmarkDataset
from .mnist import MNISTDataset
from .cifar import CIFAR10Dataset, CIFAR100Dataset

# Registry mapping dataset name -> class
DATASET_REGISTRY: dict[str, type] = {
    "mnist": MNISTDataset,
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
}


def get_dataset(name: str, data_dir: str = "~/.zkml_bench/data") -> BenchmarkDataset:
    """Instantiate a dataset by name.

    Parameters
    ----------
    name:
        Dataset name — one of ``"mnist"``, ``"cifar10"``, ``"cifar100"``.
    data_dir:
        Root directory for dataset storage. Tilde expansion is applied.

    Raises
    ------
    ValueError
        If *name* is not in :data:`DATASET_REGISTRY`.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](data_dir=data_dir)


__all__ = [
    "BenchmarkDataset",
    "MNISTDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "DATASET_REGISTRY",
    "get_dataset",
]
