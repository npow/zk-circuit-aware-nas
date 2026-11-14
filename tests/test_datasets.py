"""Tests for dataset registry, metadata, and get_dataset."""

import pytest

# torchvision is a runtime dependency for datasets; skip entire module if absent
torchvision = pytest.importorskip("torchvision", reason="torchvision not installed")

from zkml_bench.datasets import DATASET_REGISTRY, get_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------


def test_dataset_registry_has_mnist():
    assert "mnist" in DATASET_REGISTRY


def test_dataset_registry_has_cifar10():
    assert "cifar10" in DATASET_REGISTRY


def test_dataset_registry_has_cifar100():
    assert "cifar100" in DATASET_REGISTRY


def test_dataset_registry_has_all_expected():
    assert "mnist" in DATASET_REGISTRY
    assert "cifar10" in DATASET_REGISTRY
    assert "cifar100" in DATASET_REGISTRY


# ---------------------------------------------------------------------------
# get_dataset errors
# ---------------------------------------------------------------------------


def test_get_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("imagenet")


def test_get_unknown_dataset_error_lists_available():
    with pytest.raises(ValueError) as exc_info:
        get_dataset("imagenet")
    assert "mnist" in str(exc_info.value) or "Available" in str(exc_info.value)


def test_get_unknown_dataset_tinyimagenet_raises():
    with pytest.raises(ValueError):
        get_dataset("tinyimagenet")


def test_get_unknown_dataset_empty_string_raises():
    with pytest.raises(ValueError):
        get_dataset("")


# ---------------------------------------------------------------------------
# Dataset metadata (doesn't require download)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected_classes,expected_shape", [
    ("mnist", 10, (1, 28, 28)),
    ("cifar10", 10, (3, 32, 32)),
    ("cifar100", 100, (3, 32, 32)),
])
def test_dataset_metadata(name, expected_classes, expected_shape, tmp_dir):
    """Test dataset class attributes (doesn't require download)."""
    cls = DATASET_REGISTRY[name]
    ds = cls(data_dir=tmp_dir)
    assert ds.name == name
    assert ds.num_classes == expected_classes
    assert ds.input_shape == expected_shape


def test_mnist_name_attribute(tmp_dir):
    cls = DATASET_REGISTRY["mnist"]
    ds = cls(data_dir=tmp_dir)
    assert ds.name == "mnist"


def test_cifar10_name_attribute(tmp_dir):
    cls = DATASET_REGISTRY["cifar10"]
    ds = cls(data_dir=tmp_dir)
    assert ds.name == "cifar10"


def test_cifar100_name_attribute(tmp_dir):
    cls = DATASET_REGISTRY["cifar100"]
    ds = cls(data_dir=tmp_dir)
    assert ds.name == "cifar100"


def test_mnist_num_classes(tmp_dir):
    cls = DATASET_REGISTRY["mnist"]
    ds = cls(data_dir=tmp_dir)
    assert ds.num_classes == 10


def test_cifar10_num_classes(tmp_dir):
    cls = DATASET_REGISTRY["cifar10"]
    ds = cls(data_dir=tmp_dir)
    assert ds.num_classes == 10


def test_cifar100_num_classes(tmp_dir):
    cls = DATASET_REGISTRY["cifar100"]
    ds = cls(data_dir=tmp_dir)
    assert ds.num_classes == 100


def test_mnist_input_shape(tmp_dir):
    cls = DATASET_REGISTRY["mnist"]
    ds = cls(data_dir=tmp_dir)
    assert ds.input_shape == (1, 28, 28)


def test_cifar10_input_shape(tmp_dir):
    cls = DATASET_REGISTRY["cifar10"]
    ds = cls(data_dir=tmp_dir)
    assert ds.input_shape == (3, 32, 32)


def test_cifar100_input_shape(tmp_dir):
    cls = DATASET_REGISTRY["cifar100"]
    ds = cls(data_dir=tmp_dir)
    assert ds.input_shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


def test_dataset_classes_are_benchmark_datasets(tmp_dir):
    """All registered classes subclass BenchmarkDataset."""
    from zkml_bench.datasets.base import BenchmarkDataset

    for name, cls in DATASET_REGISTRY.items():
        assert issubclass(cls, BenchmarkDataset), (
            f"{cls.__name__} should subclass BenchmarkDataset"
        )


def test_get_dataset_returns_correct_type(tmp_dir):
    """get_dataset returns an instance with the expected name attribute."""
    ds = get_dataset("mnist", data_dir=tmp_dir)
    assert ds.name == "mnist"
    assert ds.num_classes == 10


def test_get_dataset_cifar10_correct_type(tmp_dir):
    ds = get_dataset("cifar10", data_dir=tmp_dir)
    assert ds.name == "cifar10"
    assert ds.num_classes == 10
