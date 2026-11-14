"""Tests for the Trainer class, seed utilities, and device helpers."""

import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from zkml_bench.exceptions import WeightLoadError
from zkml_bench.training.trainer import Trainer, TrainingResult, set_seed, get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_loader(n=100, num_classes=10, batch_size=32):
    torch.manual_seed(0)
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def make_simple_model():
    class Net(nn.Module):
        def __init__(self, num_classes=10, input_shape=(1, 28, 28)):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, num_classes))

        def forward(self, x):
            return self.net(x)

    return Net()


# ---------------------------------------------------------------------------
# Basic training
# ---------------------------------------------------------------------------


def test_training_runs_without_error():
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=2, batch_size=32, lr=0.01)
    train_loader = make_loader()
    test_loader = make_loader()
    result, top1, top5 = trainer.train(train_loader, test_loader, seed=42)
    assert not result.training_failed
    assert result.epochs_run == 2
    assert 0.0 <= top1 <= 100.0
    assert 0.0 <= top5 <= 100.0
    assert result.train_time_s > 0


def test_training_result_is_dataclass():
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=1, batch_size=32, lr=0.01)
    train_loader = make_loader()
    test_loader = make_loader()
    result, _, _ = trainer.train(train_loader, test_loader, seed=0)
    assert isinstance(result, TrainingResult)


def test_training_result_failure_reason_none_on_success():
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=1, batch_size=32, lr=0.01)
    train_loader = make_loader()
    test_loader = make_loader()
    result, _, _ = trainer.train(train_loader, test_loader, seed=0)
    assert result.failure_reason is None


def test_training_epochs_run_matches():
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=3, batch_size=32, lr=0.01)
    train_loader = make_loader()
    test_loader = make_loader()
    result, _, _ = trainer.train(train_loader, test_loader, seed=0)
    assert result.epochs_run == 3


def test_eval_time_nonnegative():
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=1, batch_size=32, lr=0.01)
    train_loader = make_loader()
    test_loader = make_loader()
    result, _, _ = trainer.train(train_loader, test_loader, seed=0)
    assert result.eval_time_s >= 0.0


def test_top5_gte_top1():
    """Top-5 accuracy should always be >= top-1."""
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=1, batch_size=32, lr=0.01)
    train_loader = make_loader()
    test_loader = make_loader()
    _, top1, top5 = trainer.train(train_loader, test_loader, seed=0)
    assert top5 >= top1


# ---------------------------------------------------------------------------
# NaN loss divergence
# ---------------------------------------------------------------------------


def test_nan_loss_raises_divergence():
    """NaN loss from the model results in training_failed=True with divergence."""

    class NaNModel(nn.Module):
        """Model that always outputs NaN — has a parameter so SGD can be created."""

        def __init__(self, num_classes=10, input_shape=(1, 28, 28)):
            super().__init__()
            # Dummy parameter so SGD optimizer initialises without error
            self._dummy = nn.Linear(1, 1)

        def forward(self, x):
            return torch.full((x.size(0), 10), float("nan"))

    model = NaNModel()
    device = torch.device("cpu")
    trainer = Trainer(model, "nan_model", device, epochs=5, batch_size=32, lr=0.01)
    train_loader = make_loader()
    test_loader = make_loader()
    result, top1, top5 = trainer.train(train_loader, test_loader, seed=42)
    assert result.training_failed is True
    assert result.failure_reason == "divergence"
    assert top1 == 0.0
    assert top5 == 0.0


def test_nan_model_epochs_run_less_than_target():
    """NaN model should run fewer than the requested number of epochs."""

    class NaNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = nn.Linear(1, 1)

        def forward(self, x):
            return torch.full((x.size(0), 10), float("nan"))

    model = NaNModel()
    device = torch.device("cpu")
    trainer = Trainer(model, "nan_model", device, epochs=10, batch_size=32, lr=0.01)
    train_loader = make_loader()
    test_loader = make_loader()
    result, _, _ = trainer.train(train_loader, test_loader, seed=0)
    assert result.epochs_run < 10


# ---------------------------------------------------------------------------
# Poly model configuration
# ---------------------------------------------------------------------------


def test_poly_model_uses_grad_clip():
    """Poly models have grad_clip_norm=1.0 and warmup_epochs=5 by default."""
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_poly", device, epochs=1, batch_size=32, lr=0.01)
    assert trainer.grad_clip_norm == 1.0
    assert trainer.warmup_epochs == 5


def test_non_poly_model_no_grad_clip():
    """Non-poly models have no grad clip by default."""
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=1, batch_size=32, lr=0.01)
    assert trainer.grad_clip_norm is None


def test_poly_lenet5_poly_uses_grad_clip():
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "lenet5_poly", device, epochs=1, batch_size=32, lr=0.01)
    assert trainer.grad_clip_norm == 1.0


def test_non_poly_resnet20_no_grad_clip():
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "resnet20", device, epochs=1, batch_size=32, lr=0.01)
    assert trainer.grad_clip_norm is None


def test_poly_model_explicit_grad_clip_overrides_default():
    """Explicit grad_clip_norm passed at construction overrides poly default."""
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(
        model, "mlp_poly", device, epochs=1, batch_size=32, lr=0.01, grad_clip_norm=2.5
    )
    assert trainer.grad_clip_norm == 2.5


def test_non_poly_model_explicit_grad_clip_respected():
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(
        model, "mlp_small", device, epochs=1, batch_size=32, lr=0.01, grad_clip_norm=5.0
    )
    assert trainer.grad_clip_norm == 5.0


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    """Two runs with the same seed produce the same top-1 accuracy."""

    def run_with_seed(seed):
        model = make_simple_model()
        device = torch.device("cpu")
        trainer = Trainer(model, "mlp_small", device, epochs=3, batch_size=32, lr=0.01)
        tl = make_loader()
        vl = make_loader()
        result, top1, _ = trainer.train(tl, vl, seed=seed)
        return top1

    acc1 = run_with_seed(42)
    acc2 = run_with_seed(42)
    assert abs(acc1 - acc2) < 0.02


# ---------------------------------------------------------------------------
# load_weights
# ---------------------------------------------------------------------------


def test_load_weights_shape_mismatch_raises(tmp_dir):
    """Loading weights from a model with different architecture raises WeightLoadError."""
    model1 = make_simple_model()

    class DifferentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(784, 64), nn.Linear(64, 10)
            )

        def forward(self, x):
            return self.net(x)

    model2 = DifferentNet()
    device = torch.device("cpu")
    weights_path = os.path.join(tmp_dir, "weights.pt")
    torch.save(model1.state_dict(), weights_path)

    trainer = Trainer(model2, "different_net", device, epochs=1, batch_size=32, lr=0.01)
    with pytest.raises(WeightLoadError):
        trainer.load_weights(weights_path)


def test_load_weights_success(tmp_dir):
    """Loading weights from the same model architecture works without error."""
    model1 = make_simple_model()
    model2 = make_simple_model()
    device = torch.device("cpu")

    weights_path = os.path.join(tmp_dir, "weights.pt")
    torch.save(model1.state_dict(), weights_path)

    trainer = Trainer(model2, "mlp_small", device, epochs=1, batch_size=32, lr=0.01)
    trainer.load_weights(weights_path)  # should not raise


def test_load_weights_missing_file_raises(tmp_dir):
    """Loading from a non-existent file raises an appropriate error."""
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=1, batch_size=32, lr=0.01)
    with pytest.raises(Exception):  # FileNotFoundError or similar
        trainer.load_weights(os.path.join(tmp_dir, "does_not_exist.pt"))


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


def test_set_seed_deterministic():
    set_seed(42)
    x1 = torch.randn(10)
    set_seed(42)
    x2 = torch.randn(10)
    assert torch.allclose(x1, x2)


def test_set_seed_different_seeds_differ():
    set_seed(1)
    x1 = torch.randn(10)
    set_seed(2)
    x2 = torch.randn(10)
    assert not torch.allclose(x1, x2)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------


def test_get_device_cpu():
    d = get_device("cpu")
    assert d == torch.device("cpu")


def test_get_device_auto_returns_device():
    d = get_device("auto")
    assert isinstance(d, torch.device)


def test_get_device_auto_is_cpu_or_cuda_or_mps():
    d = get_device("auto")
    assert d.type in ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# WarmupLRScheduler
# ---------------------------------------------------------------------------


def test_warmup_scheduler_steps():
    from zkml_bench.training.trainer import WarmupLRScheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sched = WarmupLRScheduler(optimizer, warmup_epochs=5, lr_target=0.1, lr_start=0.0)

    assert sched.is_warmup()
    for _ in range(5):
        sched.step()
    assert not sched.is_warmup()


def test_warmup_scheduler_lr_increases():
    """LR should increase monotonically during warmup."""
    from zkml_bench.training.trainer import WarmupLRScheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sched = WarmupLRScheduler(optimizer, warmup_epochs=5, lr_target=0.1, lr_start=0.0)

    lrs = []
    for _ in range(5):
        sched.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    for i in range(len(lrs) - 1):
        assert lrs[i + 1] >= lrs[i], f"LR should not decrease during warmup: {lrs}"


def test_exploding_loss_divergence_detected():
    """Exploding loss (> 10x epoch 1 loss for 5 consecutive epochs) triggers divergence."""
    import torch.nn as nn

    # Model with very high LR → diverging loss
    class BigLossFakeModel(nn.Module):
        def __init__(self, num_classes=10, input_shape=(1, 28, 28)):
            super().__init__()
            # Always return very large values that increase with each forward pass
            self._counter = torch.zeros(1, requires_grad=True)
            self.linear = nn.Linear(784, 10)

        def forward(self, x):
            # Return very large logits to cause cross-entropy explosion
            return self.linear(x.flatten(1)) * 1e6

    # Simple approach: patch _train_epoch to return increasing losses
    model = make_simple_model()
    device = torch.device("cpu")
    trainer = Trainer(model, "mlp_small", device, epochs=20, batch_size=32, lr=0.01)

    # Monkey-patch _train_epoch to simulate exploding losses
    epoch_num = [0]
    def fake_train_epoch(loader):
        epoch_num[0] += 1
        if epoch_num[0] == 1:
            return 0.01, 0.0  # epoch 1 loss = 0.01
        else:
            return 10.0, 0.0  # all subsequent: 1000x baseline → explosion
    trainer._train_epoch = fake_train_epoch

    train_loader = make_loader()
    test_loader = make_loader()
    result, top1, top5 = trainer.train(train_loader, test_loader, seed=42)
    assert result.training_failed
    assert result.failure_reason == "divergence"
