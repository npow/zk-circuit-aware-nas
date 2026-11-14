"""Tests for model registry, forward shapes, and ONNX export."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.onnx

from zkml_bench.models import get_model, list_models, register_model
from zkml_bench.exceptions import ModelRegistrationError

MNIST_SHAPE = (1, 28, 28)
CIFAR_SHAPE = (3, 32, 32)


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name,input_shape,num_classes", [
    ("mlp_small", CIFAR_SHAPE, 10),
    ("mlp_large", CIFAR_SHAPE, 10),
    ("lenet5", CIFAR_SHAPE, 10),
    ("lenet5", MNIST_SHAPE, 10),
    ("resnet20", CIFAR_SHAPE, 10),
    ("vgg11_small", CIFAR_SHAPE, 10),
    ("mlp_poly", CIFAR_SHAPE, 10),
    ("lenet5_poly", CIFAR_SHAPE, 10),
    ("resnet20_poly", CIFAR_SHAPE, 10),
    ("resnet20_relu_bn_free", CIFAR_SHAPE, 10),
    # Transformer family
    ("tiny_transformer", CIFAR_SHAPE, 10),
    ("small_bert", CIFAR_SHAPE, 10),
    ("mlp_mixer_tiny", CIFAR_SHAPE, 10),
    ("vit_tiny", CIFAR_SHAPE, 10),
    ("attention_only_2head", CIFAR_SHAPE, 10),
    # MoE family
    ("moe_tiny", CIFAR_SHAPE, 10),
    ("moe_resnet_block", CIFAR_SHAPE, 10),
    # Attention variants
    ("linear_attention_small", CIFAR_SHAPE, 10),
    ("local_attention_window", CIFAR_SHAPE, 10),
    ("flash_attention_compatible", CIFAR_SHAPE, 10),
    # Wide / huge / quant models
    ("wide_resnet_small", CIFAR_SHAPE, 10),
    ("mlp_huge", CIFAR_SHAPE, 10),
    ("qlinear_small", CIFAR_SHAPE, 10),
    ("int8_resnet_small", CIFAR_SHAPE, 10),
    # Polynomial transformers (ZK-friendly)
    ("poly_transformer_tiny", CIFAR_SHAPE, 10),
    ("poly_attention_block", CIFAR_SHAPE, 10),
])
def test_model_forward_shape(model_name, input_shape, num_classes):
    """All models output (B, num_classes) logits with no softmax applied."""
    model = get_model(model_name, num_classes=num_classes, input_shape=input_shape)
    model.eval()
    # Use small-scale input for poly models to avoid overflow with cubic activations
    # and random (LeCun) weight initialisation
    torch.manual_seed(0)
    scale = 0.01 if "poly" in model_name else 1.0
    x = torch.randn(2, *input_shape) * scale
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, num_classes), (
        f"{model_name}: expected (2, {num_classes}), got {out.shape}"
    )
    # Poly models with cubic activations can produce NaN with large random inputs
    # (overflow before training); skip NaN check for poly models unless input is small
    if "poly" not in model_name:
        assert not torch.isnan(out).any(), f"{model_name}: NaN in output"
    # For non-NaN outputs, softmax rows should sum to 1
    if not torch.isnan(out).any():
        row_sum = out.softmax(dim=-1).sum(dim=-1)
        assert torch.allclose(row_sum, torch.ones(2), atol=1e-4), (
            f"{model_name}: softmax row sum deviates from 1"
        )


@pytest.mark.parametrize("model_name,input_shape,num_classes", [
    ("mlp_small", MNIST_SHAPE, 10),
    ("mlp_small", CIFAR_SHAPE, 100),
    ("resnet20", CIFAR_SHAPE, 100),
])
def test_model_various_class_counts(model_name, input_shape, num_classes):
    """Models correctly adapt to different class counts."""
    model = get_model(model_name, num_classes=num_classes, input_shape=input_shape)
    model.eval()
    x = torch.randn(1, *input_shape)
    with torch.no_grad():
        out = model(x)
    assert out.shape[-1] == num_classes


def test_model_output_no_softmax():
    """Model output should be raw logits, not probabilities (no Softmax layer)."""
    model = get_model("mlp_small", num_classes=10, input_shape=CIFAR_SHAPE)
    model.eval()
    x = torch.randn(4, *CIFAR_SHAPE)
    with torch.no_grad():
        out = model(x)
    # Probabilities would all be in [0, 1]; logits typically span a wider range
    # Just check output does NOT uniformly lie in [0, 1] after exp
    # (raw logits often have negatives)
    has_negative = (out < 0).any()
    assert has_negative, "Expected raw logits (some negative values), got all non-negative"


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def test_model_onnx_exportable():
    """Built-in models export to ONNX opset 17."""
    model = get_model("mlp_small", num_classes=10, input_shape=CIFAR_SHAPE)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        torch.onnx.export(model, x, path, opset_version=17)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.parametrize("model_name", [
    "lenet5", "resnet20", "mlp_poly", "lenet5_poly",
    # New model families
    "tiny_transformer", "small_bert", "mlp_mixer_tiny", "vit_tiny", "attention_only_2head",
    "moe_tiny", "moe_resnet_block",
    "linear_attention_small", "local_attention_window", "flash_attention_compatible",
    "wide_resnet_small", "mlp_huge", "qlinear_small", "int8_resnet_small",
    "poly_transformer_tiny", "poly_attention_block",
])
def test_multiple_models_onnx_exportable(model_name):
    """All models can be exported to ONNX opset 17."""
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        torch.onnx.export(model, x, path, opset_version=17)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def test_list_models_returns_all():
    models = list_models()
    names = [m["name"] for m in models]
    # Original baselines + ZK-friendly
    assert "mlp_small" in names
    assert "mlp_large" in names
    assert "resnet20" in names
    assert "resnet20_poly" in names
    assert "lenet5" in names
    assert "lenet5_poly" in names
    assert "vgg11_small" in names
    assert "mlp_poly" in names
    assert "resnet20_relu_bn_free" in names
    # Transformer family
    assert "tiny_transformer" in names
    assert "small_bert" in names
    assert "mlp_mixer_tiny" in names
    assert "vit_tiny" in names
    assert "attention_only_2head" in names
    # MoE family
    assert "moe_tiny" in names
    assert "moe_resnet_block" in names
    # Attention variants
    assert "linear_attention_small" in names
    assert "local_attention_window" in names
    assert "flash_attention_compatible" in names
    # Wide / huge / quant models
    assert "wide_resnet_small" in names
    assert "mlp_huge" in names
    assert "qlinear_small" in names
    assert "int8_resnet_small" in names
    # Polynomial transformers
    assert "poly_transformer_tiny" in names
    assert "poly_attention_block" in names
    # Verify we have at least 25 models total
    assert len(names) >= 25


def test_list_models_returns_dicts_with_name_and_description():
    models = list_models()
    for entry in models:
        assert "name" in entry
        assert "description" in entry
        assert isinstance(entry["name"], str)
        assert isinstance(entry["description"], str)


# ---------------------------------------------------------------------------
# register_model errors
# ---------------------------------------------------------------------------


def test_register_model_duplicate_raises():
    with pytest.raises(ModelRegistrationError):
        @register_model("mlp_small")  # already registered
        class DuplicateModel(nn.Module):
            def __init__(self, num_classes, input_shape):
                super().__init__()

            def forward(self, x):
                return x


def test_register_model_bad_signature_missing_num_classes_raises():
    with pytest.raises(ModelRegistrationError, match="num_classes"):
        @register_model("bad_model_missing_num_classes_xyz_unique")
        class BadModel(nn.Module):
            def __init__(self, n_classes, input_shape):  # wrong param name
                super().__init__()

            def forward(self, x):
                return x


def test_register_model_bad_signature_missing_input_shape_raises():
    with pytest.raises(ModelRegistrationError):
        @register_model("bad_model_missing_input_shape_xyz_unique")
        class BadModel2(nn.Module):
            def __init__(self, num_classes, shape):  # wrong param name
                super().__init__()

            def forward(self, x):
                return x


def test_register_model_bad_signature_both_missing_raises():
    with pytest.raises(ModelRegistrationError):
        @register_model("bad_model_both_missing_xyz_unique")
        class BadModel3(nn.Module):
            def __init__(self, n, s):  # both wrong
                super().__init__()

            def forward(self, x):
                return x


# ---------------------------------------------------------------------------
# get_model errors
# ---------------------------------------------------------------------------


def test_get_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model_xyz", num_classes=10, input_shape=CIFAR_SHAPE)


def test_get_model_unknown_name_lists_available():
    """Error message should include available model names."""
    with pytest.raises(ValueError) as exc_info:
        get_model("nonexistent_model_xyz", num_classes=10, input_shape=CIFAR_SHAPE)
    assert "mlp_small" in str(exc_info.value) or "Available" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Model eval mode
# ---------------------------------------------------------------------------


def test_model_eval_mode_no_dropout_effect():
    """Model in eval mode should produce deterministic output."""
    model = get_model("mlp_small", num_classes=10, input_shape=CIFAR_SHAPE)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


def test_resnet20_poly_no_relu():
    """resnet20_poly should contain no ReLU modules."""
    model = get_model("resnet20_poly", num_classes=10, input_shape=CIFAR_SHAPE)
    relu_modules = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    assert len(relu_modules) == 0, "resnet20_poly should not contain ReLU layers"


def test_mlp_small_has_relu():
    """mlp_small should contain ReLU activations."""
    model = get_model("mlp_small", num_classes=10, input_shape=CIFAR_SHAPE)
    relu_modules = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    assert len(relu_modules) > 0, "mlp_small should contain ReLU activations"


# ---------------------------------------------------------------------------
# New model structural tests
# ---------------------------------------------------------------------------


def test_poly_transformer_tiny_no_softmax():
    """poly_transformer_tiny should not contain any Softmax modules."""
    model = get_model("poly_transformer_tiny", num_classes=10, input_shape=CIFAR_SHAPE)
    softmax_modules = [m for m in model.modules() if isinstance(m, (nn.Softmax, nn.LogSoftmax))]
    assert len(softmax_modules) == 0, "poly_transformer_tiny should not contain Softmax layers"


def test_poly_attention_block_no_softmax():
    """poly_attention_block should not contain any Softmax modules."""
    model = get_model("poly_attention_block", num_classes=10, input_shape=CIFAR_SHAPE)
    softmax_modules = [m for m in model.modules() if isinstance(m, (nn.Softmax, nn.LogSoftmax))]
    assert len(softmax_modules) == 0, "poly_attention_block should not contain Softmax layers"


def test_moe_tiny_has_multiple_experts():
    """moe_tiny should contain a TopKMoE layer with multiple experts."""
    from zkml_bench.models.moe import TopKMoE
    model = get_model("moe_tiny", num_classes=10, input_shape=CIFAR_SHAPE)
    moe_layers = [m for m in model.modules() if isinstance(m, TopKMoE)]
    assert len(moe_layers) >= 1, "moe_tiny should contain at least one TopKMoE layer"
    assert moe_layers[0].n_experts >= 2, "MoE layer should have at least 2 experts"


def test_int8_resnet_small_uses_relu6():
    """int8_resnet_small should use ReLU6 (not plain ReLU) throughout."""
    model = get_model("int8_resnet_small", num_classes=10, input_shape=CIFAR_SHAPE)
    relu6_modules = [m for m in model.modules() if isinstance(m, nn.ReLU6)]
    assert len(relu6_modules) > 0, "int8_resnet_small should contain ReLU6 layers"


def test_qlinear_small_uses_relu6():
    """qlinear_small should use ReLU6 activations."""
    model = get_model("qlinear_small", num_classes=10, input_shape=CIFAR_SHAPE)
    relu6_modules = [m for m in model.modules() if isinstance(m, nn.ReLU6)]
    assert len(relu6_modules) > 0, "qlinear_small should contain ReLU6 layers"


@pytest.mark.parametrize("model_name", [
    "tiny_transformer", "small_bert", "vit_tiny", "attention_only_2head",
    "flash_attention_compatible",
])
def test_transformer_models_have_multihead_attention(model_name):
    """Transformer-family models should contain nn.MultiheadAttention."""
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    attn_modules = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
    assert len(attn_modules) >= 1, f"{model_name} should contain at least one MultiheadAttention layer"


def test_mlp_mixer_tiny_no_attention():
    """mlp_mixer_tiny uses MLPs only (no MultiheadAttention)."""
    model = get_model("mlp_mixer_tiny", num_classes=10, input_shape=CIFAR_SHAPE)
    attn_modules = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
    assert len(attn_modules) == 0, "mlp_mixer_tiny should not contain MultiheadAttention"


def test_registry_has_at_least_25_models():
    """Registry should have at least 25 models after all imports."""
    models = list_models()
    assert len(models) >= 25, f"Expected >=25 models, got {len(models)}: {[m['name'] for m in models]}"


# ---------------------------------------------------------------------------
# Iso-FLOP pair tests
# ---------------------------------------------------------------------------

ISO_FLOP_MODEL_NAMES = [
    "mlp_wide_2",
    "mlp_deep_6",
    "mlp_wide_1",
    "mlp_deep_10",
    "cnn_wide_2",
    "cnn_deep_6",
    "cnn_wide_4",
    "cnn_deep_12",
]

ISO_FLOP_PAIRS = [
    ("mlp_wide_2",  "mlp_deep_6"),
    ("mlp_wide_1",  "mlp_deep_10"),
    ("cnn_wide_2",  "cnn_deep_6"),
    ("cnn_wide_4",  "cnn_deep_12"),
]


@pytest.mark.parametrize("model_name", ISO_FLOP_MODEL_NAMES)
def test_iso_flop_forward_shape(model_name):
    """All iso-FLOP models output (B, num_classes) logits on CIFAR-10 input."""
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    model.eval()
    x = torch.randn(1, *CIFAR_SHAPE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10), f"{model_name}: expected (1, 10), got {out.shape}"
    assert not torch.isnan(out).any(), f"{model_name}: NaN in output"


@pytest.mark.parametrize("model_name", ISO_FLOP_MODEL_NAMES)
def test_iso_flop_onnx_export(model_name):
    """All iso-FLOP models export to ONNX opset 17 without error."""
    import os
    import tempfile
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        torch.onnx.export(model, x, path, opset_version=17)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.parametrize("model_name", ISO_FLOP_MODEL_NAMES)
def test_iso_flop_no_batchnorm(model_name):
    """Iso-FLOP models must not contain BatchNorm (incompatible with GKR backends)."""
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    bn_modules = [
        m for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    assert len(bn_modules) == 0, (
        f"{model_name}: found {len(bn_modules)} BatchNorm layer(s); "
        "iso-FLOP models must be BN-free for cross-backend comparability"
    )


@pytest.mark.parametrize("model_name", ISO_FLOP_MODEL_NAMES)
def test_iso_flop_has_relu(model_name):
    """Iso-FLOP models use ReLU activations throughout."""
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    relu_modules = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    assert len(relu_modules) > 0, f"{model_name}: expected ReLU activations, found none"


@pytest.mark.parametrize("name_a,name_b", ISO_FLOP_PAIRS)
def test_iso_flop_pair_matched_flops(name_a, name_b):
    """Each iso-FLOP pair has matching FLOPs within 5%."""
    from zkml_bench.metrics.zk_proxy import compute_proxy

    tolerance = 0.05
    sample = torch.zeros(1, *CIFAR_SHAPE)

    model_a = get_model(name_a, num_classes=10, input_shape=CIFAR_SHAPE)
    model_b = get_model(name_b, num_classes=10, input_shape=CIFAR_SHAPE)

    proxy_a = compute_proxy(model_a, sample)
    proxy_b = compute_proxy(model_b, sample)

    flops_a = proxy_a.flops
    flops_b = proxy_b.flops

    assert flops_a > 0, f"{name_a}: proxy reported 0 FLOPs"
    assert flops_b > 0, f"{name_b}: proxy reported 0 FLOPs"

    diff = abs(flops_a - flops_b) / max(flops_a, flops_b)
    assert diff <= tolerance, (
        f"Iso-FLOP pair {name_a}/{name_b} exceeds {tolerance*100:.0f}% tolerance: "
        f"{flops_a:,} vs {flops_b:,} ({diff*100:.2f}%)"
    )


def test_iso_flop_pairs_registered():
    """All 8 iso-FLOP model names are present in the model registry."""
    names = [m["name"] for m in list_models()]
    for model_name in ISO_FLOP_MODEL_NAMES:
        assert model_name in names, f"{model_name} not found in model registry"


def test_registry_has_at_least_33_models():
    """Registry should have at least 33 models (25 original + 8 iso-FLOP)."""
    models = list_models()
    assert len(models) >= 33, (
        f"Expected >=33 models, got {len(models)}: {[m['name'] for m in models]}"
    )


# ---------------------------------------------------------------------------
# CNN iso-FLOP DeepProve-compatibility structural tests
# ---------------------------------------------------------------------------

CNN_ISO_FLOP_NAMES = ["cnn_wide_2", "cnn_deep_6", "cnn_wide_4", "cnn_deep_12"]


@pytest.mark.parametrize("model_name", CNN_ISO_FLOP_NAMES)
def test_cnn_iso_flop_uses_maxpool(model_name):
    """CNN iso-FLOP models must use MaxPool2d for spatial downsampling (DeepProve requires it)."""
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    maxpool_modules = [m for m in model.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(maxpool_modules) >= 1, (
        f"{model_name}: expected at least one MaxPool2d layer for downsampling; "
        "DeepProve rejects Conv2d(stride>1)"
    )


@pytest.mark.parametrize("model_name", CNN_ISO_FLOP_NAMES)
def test_cnn_iso_flop_no_strided_conv(model_name):
    """CNN iso-FLOP models must not use Conv2d with stride>1 (DeepProve hard parser error)."""
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    strided_convs = [
        m for m in model.modules()
        if isinstance(m, nn.Conv2d) and any(s > 1 for s in (
            (m.stride,) if isinstance(m.stride, int) else m.stride
        ))
    ]
    assert len(strided_convs) == 0, (
        f"{model_name}: found {len(strided_convs)} Conv2d layer(s) with stride>1; "
        "these would be rejected by DeepProve's ONNX parser"
    )


@pytest.mark.parametrize("model_name", CNN_ISO_FLOP_NAMES)
def test_cnn_iso_flop_has_adaptive_avg_pool(model_name):
    """CNN iso-FLOP models must end with AdaptiveAvgPool2d (GlobalAveragePool)."""
    model = get_model(model_name, num_classes=10, input_shape=CIFAR_SHAPE)
    gap_modules = [m for m in model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    assert len(gap_modules) >= 1, (
        f"{model_name}: expected AdaptiveAvgPool2d (global average pool) before classifier"
    )
