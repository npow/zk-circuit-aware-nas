"""ZK circuit cost proxy metrics.

Primary method: ONNX graph analysis (preferred — matches EZKL's view of the model).
Fallback method: PyTorch forward hooks (used when ONNX export fails).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import math
import os
import struct
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@contextlib.contextmanager
def _suppress_onnx_noise():
    """Suppress the torrent of warnings/logs from torch.onnx / onnxscript / onnx_ir."""
    noisy_loggers = [
        "onnxscript",
        "onnxscript.version_converter",
        "onnxscript.optimizer",
        "onnxscript.optimizer._constant_folding",
        "onnx_ir",
        "onnx_ir.passes",
        "onnx_ir.passes.common.unused_removal",
        "onnx_ir.passes._pass_infra",
        "torch.onnx",
    ]
    old_levels = {}
    for name in noisy_loggers:
        lgr = logging.getLogger(name)
        old_levels[name] = lgr.level
        lgr.setLevel(logging.ERROR)

    # Redirect stderr+stdout to suppress absl/C++ logs (W0308) and [torch.onnx] prints
    devnull = open(os.devnull, "w")
    old_stderr, old_stdout = sys.stderr, sys.stdout
    sys.stderr = devnull
    sys.stdout = devnull
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            yield
    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        devnull.close()
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)

# ---------------------------------------------------------------------------
# Cost table
# ---------------------------------------------------------------------------

# OP_COSTS: dict[str, int] — cost per element of output tensor
# (or per MAC for linear ops: Gemm, MatMul, Conv)
OP_COSTS: dict[str, int] = {
    # Non-linear: comparison-based
    "Relu": 2,
    "LeakyRelu": 3,
    "Clip": 4,
    "Abs": 2,
    "Sign": 2,
    "HardSigmoid": 4,
    "HardSwish": 6,
    "MaxPool": 3,
    "ArgMax": 20,
    "ArgMin": 20,
    # Non-linear: transcendental
    "Sigmoid": 16,
    "Tanh": 16,
    "Exp": 16,
    "Log": 16,
    "Sqrt": 16,
    "Erf": 24,
    "Softmax": 32,
    "LogSoftmax": 32,
    # Trigonometric (same lookup-table tier as Exp/Log; used by RoPE)
    "Sin": 16,
    "Cos": 16,
    "Atan": 24,
    "Atan2": 24,
    # Negation — linear (single constraint)
    "Neg": 1,
    # Normalization
    "LayerNormalization": 24,
    "InstanceNormalization": 24,
    "GroupNormalization": 24,
    "MeanVarianceNormalization": 24,
    "LpNormalization": 20,
    # Semi-linear
    "BatchNormalization": 1,
    # Linear: MAC-based (cost × MAC count, not element count)
    "Gemm": 1,
    "MatMul": 1,
    "Conv": 1,
    # Linear: element-wise
    "Mul": 1,
    "Div": 4,
    "Add": 0,
    "Sub": 0,
    # Reduction
    "ReduceSum": 0,
    "ReduceMean": 1,
    "ReduceMax": 10,
    "ReduceMin": 10,
    "ReduceProd": 1,
    # Pooling
    "AveragePool": 1,
    "GlobalAveragePool": 1,
    # Shape/indexing (free)
    "Flatten": 0,
    "Reshape": 0,
    "Transpose": 0,
    "Squeeze": 0,
    "Unsqueeze": 0,
    "Concat": 0,
    "Split": 0,
    "Gather": 2,
    "GatherElements": 2,
    "Pad": 0,
    "Slice": 0,
    "Tile": 0,
    "Expand": 0,
    "Shape": 0,
    "Cast": 0,
    # Conditional
    "Where": 6,
    # Dropout (identity at inference)
    "Dropout": 0,
    # Recurrent (hook-based fallback; after ONNX export these are decomposed)
    "LSTM": 200,
    "GRU": 120,
    "RNN": 30,
    # Upsampling
    "Resize": 2,
    "Upsample": 2,
    # Pow: computed dynamically based on exponent value
    "Pow": 1,  # placeholder; see _compute_pow_cost_with_value
}

UNKNOWN_OP_COST = 8  # conservative fallback for ops not in the table

# Fixed per-node overhead (rows EZKL allocates per op regardless of MAC count).
# Calibrated empirically: without this, small DWS-conv models are underestimated
# by up to 2.66× because EZKL's circuit overhead per node dominates small kernels.
NODE_OVERHEAD: int = 4096


def snap_to_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= x, mirroring EZKL's logrows ceiling.

    EZKL sets logrows = ceil(log2(required_rows)), so circuit size = 2^logrows.
    The raw proxy is continuous; snapping it makes calibration comparisons fair.
    """
    if x <= 0:
        return 1
    if x & (x - 1) == 0:
        return x  # already a power of 2
    return 1 << math.ceil(math.log2(x))

# ---------------------------------------------------------------------------
# Non-linear op set (used for nonlinear_op_count and depth tracking)
# ---------------------------------------------------------------------------

NON_LINEAR_OPS: set[str] = {
    "Relu",
    "LeakyRelu",
    "Clip",
    "Abs",
    "Sign",
    "HardSigmoid",
    "HardSwish",
    "MaxPool",
    "ArgMax",
    "ArgMin",
    "Sigmoid",
    "Tanh",
    "Exp",
    "Log",
    "Sqrt",
    "Erf",
    "Softmax",
    "LogSoftmax",
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "MeanVarianceNormalization",
    "LpNormalization",
    "LSTM",
    "GRU",
    "RNN",
    "Resize",
    "Upsample",
}

# Multiplicative depth costs for HE systems (ciphertext-ciphertext mults only)
HE_DEPTH_COSTS: dict[str, int] = {
    "Sigmoid": 3,
    "Tanh": 3,
    "Erf": 4,
    "Softmax": 4,
    "LogSoftmax": 4,
}

# Linear ops that count toward linear_op_count (not nonlinear, not free)
_LINEAR_OPS: set[str] = {
    "Add", "Sub", "Mul", "Div", "Where", "Gather", "GatherElements",
    "BatchNormalization", "Flatten", "Reshape", "Transpose",
    "Squeeze", "Unsqueeze", "Concat", "Split", "Pad", "Slice",
    "Tile", "Expand", "Shape", "Cast", "Dropout",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "AveragePool", "GlobalAveragePool", "Gemm", "MatMul", "Conv", "Pow",
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ZKProxyResult:
    constraint_estimate: int
    constraint_estimate_snapped: int  # snap_to_power_of_2(constraint_estimate)
    nonlinear_op_count: int
    linear_op_count: int
    multiplicative_depth: Optional[int]  # None if model contains ReLU (HE-incompatible)
    relu_depth_incompatible: bool
    nonlinear_breakdown: dict[str, int]  # op_name -> element count
    param_count: int
    flops: int
    proxy_method: str  # "onnx" or "hooks"
    ood_warning: Optional[dict] = None  # populated by detect_proxy_ood if OOD patterns found


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shape_elements(shape: list) -> Optional[int]:
    """Return the product of shape dims, or None if any dim is dynamic/None."""
    if not shape:
        return None
    if any(s is None or s == 0 for s in shape):
        return None
    return math.prod(int(s) for s in shape)


def _read_onnx_initializer_scalar(init) -> Optional[float]:
    """Try to read a scalar value from an ONNX TensorProto initializer."""
    # Check typed data fields first
    if init.float_data:
        return float(init.float_data[0])
    if init.double_data:
        return float(init.double_data[0])
    if init.int32_data:
        return float(init.int32_data[0])
    if init.int64_data:
        return float(init.int64_data[0])
    # Fall back to raw_data
    if init.raw_data:
        from onnx import TensorProto
        dt = init.data_type
        try:
            if dt == TensorProto.FLOAT:
                return float(struct.unpack_from("<f", init.raw_data)[0])
            elif dt == TensorProto.DOUBLE:
                return float(struct.unpack_from("<d", init.raw_data)[0])
            elif dt == TensorProto.INT32:
                return float(struct.unpack_from("<i", init.raw_data)[0])
            elif dt == TensorProto.INT64:
                return float(struct.unpack_from("<q", init.raw_data)[0])
        except (struct.error, Exception):
            pass
    return None


def _compute_mac_cost(node, shape_map: dict) -> int:
    """Compute MAC count for Gemm, MatMul, or Conv nodes.

    Returns 0 (with a warning) if shapes are unavailable.
    """
    op_type = node.op_type

    if op_type == "Gemm":
        if len(node.input) < 2:
            warnings.warn("Gemm node has fewer than 2 inputs; skipping MAC cost")
            return 0
        a_name, b_name = node.input[0], node.input[1]
        a_shape = shape_map.get(a_name)
        b_shape = shape_map.get(b_name)
        if a_shape is None or b_shape is None:
            warnings.warn(
                f"Gemm: shapes unavailable for inputs '{a_name}', '{b_name}'; skipping cost"
            )
            return 0
        if len(a_shape) < 2 or len(b_shape) < 2:
            warnings.warn(
                f"Gemm: unexpected shape dimensions {a_shape}, {b_shape}; skipping cost"
            )
            return 0
        # Parse transA / transB attributes
        trans_a = False
        trans_b = False
        for attr in node.attribute:
            if attr.name == "transA":
                trans_a = bool(attr.i)
            elif attr.name == "transB":
                trans_b = bool(attr.i)
        M = a_shape[-1] if trans_a else a_shape[-2]
        K = a_shape[-2] if trans_a else a_shape[-1]
        N = b_shape[-2] if trans_b else b_shape[-1]
        if any(v is None for v in (M, K, N)):
            warnings.warn(f"Gemm: dynamic dimensions M={M} K={K} N={N}; skipping cost")
            return 0
        return int(M) * int(N) * int(K)

    elif op_type == "MatMul":
        if len(node.input) < 2:
            warnings.warn("MatMul node has fewer than 2 inputs; skipping MAC cost")
            return 0
        a_name, b_name = node.input[0], node.input[1]
        a_shape = shape_map.get(a_name)
        b_shape = shape_map.get(b_name)
        if a_shape is None or b_shape is None:
            warnings.warn("MatMul: shapes unavailable; skipping cost")
            return 0
        if len(a_shape) < 2 or len(b_shape) < 2:
            warnings.warn(
                f"MatMul: unexpected shape {a_shape}, {b_shape}; skipping cost"
            )
            return 0
        # A: (..., M, K), B: (..., K, N)
        M = a_shape[-2]
        K = a_shape[-1]
        N = b_shape[-1]
        if any(v is None for v in (M, K, N)):
            warnings.warn("MatMul: dynamic inner dimensions; skipping cost")
            return 0
        batch_dims = a_shape[:-2]
        if any(d is None for d in batch_dims):
            warnings.warn("MatMul: dynamic batch dimensions; skipping cost")
            return 0
        batch = math.prod(int(d) for d in batch_dims) if batch_dims else 1
        return batch * int(M) * int(N) * int(K)

    elif op_type == "Conv":
        if len(node.input) < 2:
            warnings.warn("Conv node has fewer than 2 inputs; skipping MAC cost")
            return 0
        input_name, weight_name = node.input[0], node.input[1]
        input_shape = shape_map.get(input_name)
        weight_shape = shape_map.get(weight_name)
        if input_shape is None or weight_shape is None:
            warnings.warn("Conv: shapes unavailable; skipping cost")
            return 0
        if len(input_shape) < 3 or len(weight_shape) < 3:
            warnings.warn(
                f"Conv: unexpected shape {input_shape}, {weight_shape}; skipping cost"
            )
            return 0
        # Read groups attribute
        for attr in node.attribute:
            if attr.name == "group":
                break
        # Spatial output dims from output tensor shape
        out_spatial: Optional[int] = None
        if node.output:
            out_name = node.output[0]
            out_shape = shape_map.get(out_name)
            if out_shape is not None and len(out_shape) >= 3:
                out_spatial_dims = out_shape[2:]
                if all(d is not None for d in out_spatial_dims):
                    out_spatial = math.prod(int(d) for d in out_spatial_dims)
        if out_spatial is None:
            warnings.warn("Conv: output spatial shape unavailable; skipping cost")
            return 0
        C_out = weight_shape[0]
        C_in_per_group = weight_shape[1]
        kernel_spatial_dims = weight_shape[2:]
        if any(v is None for v in (C_out, C_in_per_group, *kernel_spatial_dims)):
            warnings.warn("Conv: dynamic weight dimensions; skipping cost")
            return 0
        kernel_size = math.prod(int(k) for k in kernel_spatial_dims)
        # MACs = C_out × (C_in/group) × kernel_size × H_out × W_out
        macs = int(C_out) * int(C_in_per_group) * kernel_size * out_spatial
        return macs

    return 0


def _compute_pow_cost_with_value(
    exp_value: Optional[float], out_elements: int
) -> int:
    """Compute Pow cost given a resolved exponent value.

    Integer exponent n: cost = (n - 1) × out_elements (n-1 multiplications).
    Non-integer or unknown: cost = 16 × out_elements (treat as transcendental).
    """
    if out_elements == 0:
        return 0
    if exp_value is not None:
        try:
            exp_int = int(exp_value)
            if float(exp_int) == float(exp_value) and exp_int >= 1:
                return max(0, exp_int - 1) * out_elements
        except (ValueError, OverflowError):
            pass
    # Non-integer or unknown exponent: treat as transcendental
    return 16 * out_elements


def _count_params(model: nn.Module) -> int:
    """Count total parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def _estimate_flops(model: nn.Module, sample_input: torch.Tensor) -> int:
    """Estimate FLOPs using forward hooks on Conv and Linear layers.

    Counts MACs (multiply-accumulate operations) and multiplies by 2 for FLOPs.
    """
    flop_count = [0]
    hooks = []

    def _linear_hook(module: nn.Linear, inp, out):
        if inp and inp[0] is not None:
            batch = inp[0].shape[0] if inp[0].dim() > 1 else 1
            macs = batch * module.out_features * module.in_features
            flop_count[0] += 2 * macs

    def _conv2d_hook(module: nn.Conv2d, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            B, C_out, H_out, W_out = out.shape
            C_in_per_group = module.in_channels // module.groups
            kH, kW = (
                module.kernel_size
                if isinstance(module.kernel_size, tuple)
                else (module.kernel_size, module.kernel_size)
            )
            macs = B * C_out * H_out * W_out * C_in_per_group * kH * kW
            flop_count[0] += 2 * macs

    def _conv1d_hook(module: nn.Conv1d, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            B, C_out, L_out = out.shape
            C_in_per_group = module.in_channels // module.groups
            kL = (
                module.kernel_size
                if isinstance(module.kernel_size, int)
                else module.kernel_size[0]
            )
            macs = B * C_out * L_out * C_in_per_group * kL
            flop_count[0] += 2 * macs

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_linear_hook))
        elif isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(_conv2d_hook))
        elif isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(_conv1d_hook))

    try:
        model.eval()
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in hooks:
            h.remove()

    return flop_count[0]


# ---------------------------------------------------------------------------
# ONNX-based proxy (primary method)
# ---------------------------------------------------------------------------


def _export_to_onnx_proto(model: nn.Module, sample_input: torch.Tensor):
    """Export model to ONNX and return a shape-inferred ModelProto.

    Internal helper shared by _compute_proxy_onnx and compute_proxy (OOD check).
    Raises on any export or load failure.
    """
    import onnx
    import onnx.shape_inference

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        with _suppress_onnx_noise():
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
        model_proto = onnx.load(onnx_path)
        model_proto = onnx.shape_inference.infer_shapes(model_proto)
    finally:
        try:
            os.unlink(onnx_path)
        except OSError:
            pass
    return model_proto


def _compute_proxy_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    _model_proto=None,
) -> ZKProxyResult:
    """Compute ZK proxy metrics by analyzing the ONNX graph.

    Args:
        model: The PyTorch model in eval mode.
        sample_input: A representative input tensor (batch_size=1).
        _model_proto: Optional pre-loaded ONNX ModelProto (with shape inference already
            run). When provided, ONNX export is skipped. Internal use only.
    """
    if _model_proto is not None:
        model_proto = _model_proto
    else:
        model_proto = _export_to_onnx_proto(model, sample_input)

    # Build shape map: value_name -> shape (list where None = dynamic dim)
    shape_map: dict[str, list] = {}
    for vi in (
        list(model_proto.graph.input)
        + list(model_proto.graph.output)
        + list(model_proto.graph.value_info)
    ):
        shape: list = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    shape.append(None)
        shape_map[vi.name] = shape

    # Initializers (weight tensors / constants)
    initializer_set: set[str] = set()
    # Scalar initializer values for Pow exponent resolution
    initializer_scalars: dict[str, float] = {}
    for init in model_proto.graph.initializer:
        shape_map[init.name] = list(init.dims)
        initializer_set.add(init.name)
        # Try to read scalar value (for Pow cost)
        if not init.dims or init.dims == [1]:
            val = _read_onnx_initializer_scalar(init)
            if val is not None:
                initializer_scalars[init.name] = val

    constraint_estimate = 0
    nonlinear_op_count = 0
    linear_op_count = 0
    nonlinear_breakdown: dict[str, int] = {}
    has_relu = False

    # Multiplicative depth tracking (ONNX guarantees topological node order)
    depth_map: dict[str, int] = {}
    for inp in model_proto.graph.input:
        depth_map[inp.name] = 0
    for init in model_proto.graph.initializer:
        depth_map[init.name] = 0  # constants are depth 0

    for node in model_proto.graph.node:
        op_type = node.op_type

        # Compute max element count across all outputs (skip nodes with no shape info)
        out_elements = 0
        for out_name in node.output:
            if out_name and out_name in shape_map:
                elems = _shape_elements(shape_map[out_name])
                if elems is not None:
                    out_elements = max(out_elements, elems)

        # Compute cost for this node
        cost = 0
        if op_type in ("Gemm", "MatMul", "Conv"):
            macs = _compute_mac_cost(node, shape_map)
            cost = OP_COSTS[op_type] * macs + NODE_OVERHEAD
            linear_op_count += 1
        elif op_type == "Pow":
            # Resolve exponent value from initializers
            exp_value: Optional[float] = None
            if len(node.input) >= 2:
                exp_name = node.input[1]
                if exp_name in initializer_scalars:
                    exp_value = initializer_scalars[exp_name]
                elif exp_name in initializer_set:
                    # Initializer exists but wasn't read as scalar above — try again
                    for init in model_proto.graph.initializer:
                        if init.name == exp_name:
                            exp_value = _read_onnx_initializer_scalar(init)
                            break
            cost = _compute_pow_cost_with_value(exp_value, out_elements)
            linear_op_count += 1
        elif op_type in OP_COSTS:
            weight = OP_COSTS[op_type]
            cost = weight * out_elements
        else:
            warnings.warn(
                f"Unknown ONNX op '{op_type}'; using UNKNOWN_OP_COST={UNKNOWN_OP_COST}"
            )
            cost = UNKNOWN_OP_COST * out_elements

        constraint_estimate += cost

        # Track nonlinear / linear op counts
        if op_type in NON_LINEAR_OPS:
            nonlinear_op_count += 1
            nonlinear_breakdown[op_type] = (
                nonlinear_breakdown.get(op_type, 0) + out_elements
            )
            if op_type == "Relu":
                has_relu = True
        elif op_type in _LINEAR_OPS:
            # Already counted above for Gemm/MatMul/Conv/Pow; count others here
            if op_type not in ("Gemm", "MatMul", "Conv", "Pow"):
                linear_op_count += 1
        # else: unknown op — counted neither; warning was already issued above

        # Compute multiplicative depth for this node's outputs
        input_depth = max(
            (depth_map.get(inp_name, 0) for inp_name in node.input if inp_name),
            default=0,
        )

        node_depth_cost = 0
        if op_type == "Relu":
            # HE-incompatible: flag but do not add depth
            node_depth_cost = 0
        elif op_type == "Mul":
            # Ciphertext × ciphertext only (both inputs are non-constant)
            in0 = node.input[0] if len(node.input) > 0 else ""
            in1 = node.input[1] if len(node.input) > 1 else ""
            in0_const = in0 in initializer_set
            in1_const = in1 in initializer_set
            if not in0_const and not in1_const:
                node_depth_cost = 1
        elif op_type in HE_DEPTH_COSTS:
            node_depth_cost = HE_DEPTH_COSTS[op_type]

        node_out_depth = input_depth + node_depth_cost
        for out_name in node.output:
            if out_name:
                depth_map[out_name] = node_out_depth

    # Final multiplicative depth = max depth of graph outputs
    graph_output_depths = [depth_map.get(o.name, 0) for o in model_proto.graph.output]
    mult_depth = max(graph_output_depths) if graph_output_depths else 0

    param_count = _count_params(model)
    flops = _estimate_flops(model, sample_input)

    return ZKProxyResult(
        constraint_estimate=constraint_estimate,
        constraint_estimate_snapped=snap_to_power_of_2(constraint_estimate),
        nonlinear_op_count=nonlinear_op_count,
        linear_op_count=linear_op_count,
        multiplicative_depth=None if has_relu else mult_depth,
        relu_depth_incompatible=has_relu,
        nonlinear_breakdown=nonlinear_breakdown,
        param_count=param_count,
        flops=flops,
        proxy_method="onnx",
    )


# ---------------------------------------------------------------------------
# Hook-based fallback proxy
# ---------------------------------------------------------------------------

# Module class → ONNX op name mapping
MODULE_TO_OP: dict = {
    nn.ReLU: "Relu",
    nn.LeakyReLU: "LeakyRelu",
    nn.Sigmoid: "Sigmoid",
    nn.Tanh: "Tanh",
    nn.GELU: "Erf",  # GELU ≈ Erf-based in opset 17; uses Erf cost
    nn.Softmax: "Softmax",
    nn.LogSoftmax: "LogSoftmax",
    nn.LayerNorm: "LayerNormalization",
    nn.BatchNorm1d: "BatchNormalization",
    nn.BatchNorm2d: "BatchNormalization",
    nn.Linear: "Gemm",
    nn.Conv2d: "Conv",
    nn.Conv1d: "Conv",
    nn.MaxPool2d: "MaxPool",
    nn.AvgPool2d: "AveragePool",
    nn.AdaptiveAvgPool2d: "GlobalAveragePool",
    nn.Dropout: "Dropout",
    nn.Dropout2d: "Dropout",
}


def _compute_proxy_hooks(model: nn.Module, sample_input: torch.Tensor) -> ZKProxyResult:
    """Fallback hook-based proxy when ONNX export fails."""
    constraint_total = [0]
    nonlinear_count = [0]
    linear_count = [0]
    nonlinear_breakdown: dict[str, int] = {}
    has_relu = [False]
    flop_total = [0]
    hooks = []

    def _make_hook(mod: nn.Module):
        op_type = MODULE_TO_OP[type(mod)]

        def hook(module, inp, out):
            out_elements = out.numel() if isinstance(out, torch.Tensor) else 0

            # Compute cost
            cost = 0
            if op_type in ("Gemm", "MatMul", "Conv"):
                if isinstance(module, nn.Linear):
                    if inp and inp[0] is not None:
                        batch = inp[0].shape[0] if inp[0].dim() > 1 else 1
                        macs = batch * module.out_features * module.in_features
                        cost = OP_COSTS[op_type] * macs
                        flop_total[0] += 2 * macs
                elif isinstance(module, nn.Conv2d):
                    if isinstance(out, torch.Tensor) and out.dim() == 4:
                        B, C_out, H_out, W_out = out.shape
                        C_in_per_group = module.in_channels // module.groups
                        kH, kW = (
                            module.kernel_size
                            if isinstance(module.kernel_size, tuple)
                            else (module.kernel_size, module.kernel_size)
                        )
                        macs = B * C_out * H_out * W_out * C_in_per_group * kH * kW
                        cost = OP_COSTS[op_type] * macs
                        flop_total[0] += 2 * macs
                elif isinstance(module, nn.Conv1d):
                    if isinstance(out, torch.Tensor) and out.dim() == 3:
                        B, C_out, L_out = out.shape
                        C_in_per_group = module.in_channels // module.groups
                        kL = (
                            module.kernel_size
                            if isinstance(module.kernel_size, int)
                            else module.kernel_size[0]
                        )
                        macs = B * C_out * L_out * C_in_per_group * kL
                        cost = OP_COSTS[op_type] * macs
                        flop_total[0] += 2 * macs
            else:
                weight = OP_COSTS.get(op_type, UNKNOWN_OP_COST)
                cost = weight * out_elements

            constraint_total[0] += cost

            # Track nonlinear / linear
            if op_type in NON_LINEAR_OPS:
                nonlinear_count[0] += 1
                nonlinear_breakdown[op_type] = (
                    nonlinear_breakdown.get(op_type, 0) + out_elements
                )
                if op_type == "Relu":
                    has_relu[0] = True
            else:
                linear_count[0] += 1

        return hook

    for mod in model.modules():
        if type(mod) in MODULE_TO_OP:
            hooks.append(mod.register_forward_hook(_make_hook(mod)))

    try:
        model.eval()
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in hooks:
            h.remove()

    param_count = _count_params(model)

    return ZKProxyResult(
        constraint_estimate=constraint_total[0],
        constraint_estimate_snapped=snap_to_power_of_2(constraint_total[0]),
        nonlinear_op_count=nonlinear_count[0],
        linear_op_count=linear_count[0],
        multiplicative_depth=None if has_relu[0] else 0,
        relu_depth_incompatible=has_relu[0],
        nonlinear_breakdown=nonlinear_breakdown,
        param_count=param_count,
        flops=flop_total[0],
        proxy_method="hooks",
    )


# ---------------------------------------------------------------------------
# OOD detection
# ---------------------------------------------------------------------------

# In-distribution kernel sizes for Conv nodes (h×w pairs)
_ID_CONV_KERNELS: frozenset[tuple[int, int]] = frozenset({(1, 1), (3, 3)})

# In-distribution norm ops (only BatchNorm is calibrated)
_OOD_NORM_OPS: frozenset[str] = frozenset({
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "MeanVarianceNormalization",
    "LpNormalization",
})

# In-distribution activations in hidden layers (anything else is OOD)
# Sigmoid, Tanh, Erf are valid *output* activations in some models but
# were not present in hidden layers of the 17-model calibration set.
_OOD_ACTIVATION_OPS: frozenset[str] = frozenset({"Sigmoid", "Tanh", "Erf"})

# Maximum ONNX node count considered in-distribution
_MAX_ID_NODE_COUNT: int = 200

# Confidence penalty applied per OOD flag (multiplicative, stacks multiplicatively)
_PENALTY_PER_FLAG: float = 1.5


def detect_proxy_ood(onnx_model_proto) -> dict:
    """Detect whether an ONNX model has op patterns outside the 17-model calibration set.

    The ZK proxy (constraint_estimate) was calibrated against 9 baseline models and
    up to 17 models including NAS candidates — all using standard ReLU/MLP/CNN patterns
    on CIFAR-10 with 3×3 or 1×1 Conv kernels, groups=1 or groups=in_channels (DWS),
    BatchNorm2d, and common pooling/reshape ops.

    Args:
        onnx_model_proto: An ``onnx.ModelProto`` with shape inference already run.

    Returns:
        dict with keys:
            - ``is_ood`` (bool): True if any flag is raised.
            - ``flags`` (list[str]): Specific OOD patterns detected.
            - ``confidence_penalty`` (float): Multiplicative uncertainty factor.
              1.0 = fully in-distribution; higher values indicate the proxy estimate
              should be treated more conservatively (e.g. 2.0 → double the estimate).
    """
    flags: list[str] = []

    graph = onnx_model_proto.graph

    # Build a set of initializer names (weight tensors / constants)
    initializer_names: set[str] = {init.name for init in graph.initializer}

    # Build shape map for inspecting weight tensor shapes
    shape_map: dict[str, list] = {}
    for vi in (
        list(graph.input)
        + list(graph.output)
        + list(graph.value_info)
    ):
        shape: list = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.HasField("dim_value") else None)
        shape_map[vi.name] = shape
    for init in graph.initializer:
        shape_map[init.name] = list(init.dims)

    # Track whether there are any non-output Relu/nonlinear nodes
    # (we only flag Sigmoid/Tanh/Erf when they appear in hidden positions,
    # but since we cannot easily determine "hidden vs output" from the graph,
    # we flag them unconditionally as they were absent from the calibration set)
    total_nodes = len(graph.node)

    # Count MatMul nodes that come from attention (not from Linear/Gemm)
    # A Gemm node represents nn.Linear; a standalone MatMul with non-initializer
    # inputs on both sides is likely an attention operation.
    for node in graph.node:
        op_type = node.op_type

        # --- Conv checks ---
        if op_type == "Conv":
            weight_name = node.input[1] if len(node.input) >= 2 else None
            weight_shape = shape_map.get(weight_name, []) if weight_name else []

            # Check kernel size (spatial dims start at index 2)
            if len(weight_shape) >= 4:
                kH, kW = weight_shape[2], weight_shape[3]
                if kH is not None and kW is not None:
                    if (int(kH), int(kW)) not in _ID_CONV_KERNELS:
                        flags.append(
                            f"non_standard_kernel_{kH}x{kW}"
                        )

            # Check dilation attribute
            for attr in node.attribute:
                if attr.name == "dilations":
                    dilations = list(attr.ints)
                    if any(d != 1 for d in dilations):
                        flags.append(
                            f"dilated_conv_dilation_{dilations}"
                        )
                    break

            # Check groups attribute vs in_channels
            groups = 1
            for attr in node.attribute:
                if attr.name == "group":
                    groups = int(attr.i)
                    break

            if groups != 1:
                # Determine in_channels from weight shape: weight is [C_out, C_in/group, kH, kW]
                # Total in_channels = C_in_per_group * groups
                if len(weight_shape) >= 2 and weight_shape[1] is not None:
                    c_in_per_group = int(weight_shape[1])
                    c_in_total = c_in_per_group * groups
                    is_depthwise = (groups == c_in_total)
                    if not is_depthwise:
                        # Arbitrary grouped conv — not in calibration set
                        flags.append(
                            f"grouped_conv_unusual_group_size_{groups}"
                        )
                else:
                    # Cannot determine if depthwise — flag as potentially OOD
                    if groups > 1:
                        flags.append(
                            f"grouped_conv_unknown_channels_group_{groups}"
                        )

        # --- OOD norm ops ---
        elif op_type in _OOD_NORM_OPS:
            flags.append(f"ood_norm_op_{op_type.lower()}")

        # --- OOD activations ---
        elif op_type in _OOD_ACTIVATION_OPS:
            flags.append(f"ood_activation_{op_type.lower()}")

        # --- Attention-like MatMul ---
        # A MatMul where both inputs are non-initializer tensors (not from nn.Linear)
        # is a strong signal for attention / transformer ops.
        elif op_type == "MatMul":
            if len(node.input) >= 2:
                in0 = node.input[0]
                in1 = node.input[1]
                in0_is_weight = in0 in initializer_names
                in1_is_weight = in1 in initializer_names
                if not in0_is_weight and not in1_is_weight:
                    flags.append("attention_like_matmul")

    # --- Depth check (total node count) ---
    if total_nodes > _MAX_ID_NODE_COUNT:
        flags.append(f"very_deep_net_{total_nodes}_nodes")

    # Deduplicate flags while preserving insertion order
    seen: set[str] = set()
    unique_flags: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            unique_flags.append(f)

    is_ood = len(unique_flags) > 0

    # Confidence penalty: starts at 1.0, multiplied by _PENALTY_PER_FLAG for each
    # distinct flag class (not each individual flag instance — group by prefix).
    # This avoids over-penalising models with many Conv nodes of the same OOD type.
    flag_classes: set[str] = set()
    for f in unique_flags:
        # Extract the flag "class" by trimming trailing numeric suffix
        # e.g. "non_standard_kernel_5x5" and "non_standard_kernel_7x7" → same class
        parts = f.split("_")
        # Drop trailing numeric/dimension tokens to normalise flag class
        while parts and (parts[-1].isdigit() or "x" in parts[-1] or parts[-1].startswith("[")):
            parts = parts[:-1]
        flag_class = "_".join(parts) if parts else f
        flag_classes.add(flag_class)

    confidence_penalty = _PENALTY_PER_FLAG ** len(flag_classes) if flag_classes else 1.0

    return {
        "is_ood": is_ood,
        "flags": unique_flags,
        "confidence_penalty": confidence_penalty,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_proxy(model: nn.Module, sample_input: torch.Tensor) -> ZKProxyResult:
    """Compute ZK proxy metrics. Tries ONNX first, falls back to hooks.

    When the ONNX path succeeds, also runs :func:`detect_proxy_ood` and
    populates ``result.ood_warning`` if any out-of-distribution patterns are found.
    """
    model.eval()
    with torch.no_grad():
        model_proto = None
        try:
            model_proto = _export_to_onnx_proto(model, sample_input)
            result = _compute_proxy_onnx(model, sample_input, _model_proto=model_proto)
        except Exception as e:
            warnings.warn(
                f"ONNX proxy failed ({e}), falling back to hook-based proxy"
            )
            result = _compute_proxy_hooks(model, sample_input)

    # Run OOD check when we successfully obtained an ONNX proto
    if model_proto is not None and result.proxy_method == "onnx":
        try:
            ood_result = detect_proxy_ood(model_proto)
            result.ood_warning = ood_result if ood_result["is_ood"] else None
        except Exception as _e:
            warnings.warn(f"OOD check failed ({_e}); skipping")

    return result


# ---------------------------------------------------------------------------
# Model hash
# ---------------------------------------------------------------------------


def model_hash(onnx_path: str) -> str:
    """SHA256 of the ONNX graph structure (architecture only, not weights)."""
    import onnx

    model = onnx.load(onnx_path)
    stripped = onnx.ModelProto()
    stripped.CopyFrom(model)
    del stripped.graph.initializer[:]
    buf = io.BytesIO()
    buf.write(stripped.SerializeToString())
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]
