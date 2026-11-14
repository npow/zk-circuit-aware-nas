"""Per-backend ZK proxy cost estimation.

Each ZK backend has a fundamentally different cost structure:

- EZKL/Halo2 (``ezkl``):  circuit size is logrows-dominated (snap-to-power-of-2).
  Proxy = snap_to_power_of_2(MAC_count + NODE_OVERHEAD × n_linear_nodes).
  Unit: ``logrows`` (the exponent, not the raw row count).

- DeepProve/GKR (``deepprove``): no SRS, no logrows ceiling.  Dominant cost
  is the LogUp lookup argument over ReLU requantization tables.  Proxy =
  relu_activation_count (total ReLU output elements) + linear_mac_count.
  Unit: ``activation_count``.

- Expander/GKR (``expander``): static layered circuit.  Dominant cost is
  log2 of the widest layer's gate count, which sets the degree of the
  multilinear polynomial the prover must commit to.
  Unit: ``log2_gates``.

All three functions accept an **ONNX ModelProto with shape inference already
run** (the same ``model_proto`` that ``_export_to_onnx_proto`` returns).

Public API
----------
    compute_backend_proxy(model_proto, backend: str) -> dict

    Keys in the returned dict:
        backend          str   – one of "ezkl", "deepprove", "expander"
        proxy_value      int   – headline cost prediction (see unit below)
        capacity_metric  str   – "logrows", "activation_count", or "log2_gates"
        components       dict  – breakdown of contributing terms
"""

from __future__ import annotations

import math

# Re-use the MAC-counting helpers from the existing EZKL proxy.
from .zk_proxy import (
    NODE_OVERHEAD,
    _compute_mac_cost,
    _shape_elements,
    snap_to_power_of_2,
)

# ---------------------------------------------------------------------------
# Supported backends
# ---------------------------------------------------------------------------

BACKENDS: frozenset[str] = frozenset({"ezkl", "deepprove", "expander"})

# Ops that produce ReLU activations (dominant cost for DeepProve).
# We count the total number of output *elements* across all such nodes.
_RELU_OPS: frozenset[str] = frozenset({"Relu", "LeakyRelu", "Clip"})

# Linear ops whose MACs we count for the DeepProve proxy.
_LINEAR_MAC_OPS: frozenset[str] = frozenset({"Gemm", "MatMul", "Conv"})


# ---------------------------------------------------------------------------
# Internal per-backend helpers
# ---------------------------------------------------------------------------


def _proxy_ezkl(model_proto) -> dict:
    """EZKL/Halo2 proxy: snap_to_power_of_2(MACs + NODE_OVERHEAD × n_linear).

    Returns a dict with keys:
        proxy_value      int   – 2^logrows  (the snapped row count)
        capacity_metric  str   – "logrows"
        components       dict  – {mac_count, n_linear_nodes, raw_estimate,
                                  snapped_estimate, logrows}
    """
    graph = model_proto.graph

    # Build shape map (value_name → list of dim sizes; None = dynamic).
    shape_map: dict[str, list] = {}
    for vi in (
        list(graph.input) + list(graph.output) + list(graph.value_info)
    ):
        shape: list = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.HasField("dim_value") else None)
        shape_map[vi.name] = shape
    for init in graph.initializer:
        shape_map[init.name] = list(init.dims)

    mac_count = 0
    n_linear_nodes = 0

    for node in graph.node:
        if node.op_type in _LINEAR_MAC_OPS:
            mac_count += _compute_mac_cost(node, shape_map)
            n_linear_nodes += 1

    raw_estimate = mac_count + NODE_OVERHEAD * n_linear_nodes
    snapped = snap_to_power_of_2(raw_estimate)
    logrows = int(math.ceil(math.log2(snapped))) if snapped > 1 else 0

    return {
        "proxy_value": snapped,
        "capacity_metric": "logrows",
        "components": {
            "mac_count": mac_count,
            "n_linear_nodes": n_linear_nodes,
            "node_overhead_per_node": NODE_OVERHEAD,
            "raw_estimate": raw_estimate,
            "snapped_estimate": snapped,
            "logrows": logrows,
        },
    }


def _proxy_deepprove(model_proto) -> dict:
    """DeepProve/GKR proxy: relu_activation_count + linear_mac_count.

    DeepProve's dominant cost is the LogUp lookup argument over the ReLU
    requantization table (256-entry 8-bit table).  Each ReLU output element
    generates one lookup query.  Linear MAC count is a secondary term (O(n log n)
    via GKR sumcheck), included in the headline value as a combined metric.

    Only ReLU-type activations are supported by DeepProve; Sigmoid/Tanh/etc.
    are unsupported (the model cannot be proved).  We count only Relu/LeakyRelu
    output elements here.

    Returns a dict with keys:
        proxy_value      int   – relu_activation_count + linear_mac_count
        capacity_metric  str   – "activation_count"
        components       dict  – {relu_activation_count, linear_mac_count,
                                  relu_op_breakdown, n_linear_nodes}
    """
    graph = model_proto.graph

    shape_map: dict[str, list] = {}
    for vi in (
        list(graph.input) + list(graph.output) + list(graph.value_info)
    ):
        shape: list = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.HasField("dim_value") else None)
        shape_map[vi.name] = shape
    for init in graph.initializer:
        shape_map[init.name] = list(init.dims)

    relu_activation_count = 0
    relu_op_breakdown: dict[str, int] = {}
    linear_mac_count = 0
    n_linear_nodes = 0

    for node in graph.node:
        op_type = node.op_type

        if op_type in _RELU_OPS:
            # Count total output elements across all outputs of this node.
            out_elements = 0
            for out_name in node.output:
                if out_name and out_name in shape_map:
                    elems = _shape_elements(shape_map[out_name])
                    if elems is not None:
                        out_elements = max(out_elements, elems)
            relu_activation_count += out_elements
            relu_op_breakdown[op_type] = (
                relu_op_breakdown.get(op_type, 0) + out_elements
            )

        elif op_type in _LINEAR_MAC_OPS:
            linear_mac_count += _compute_mac_cost(node, shape_map)
            n_linear_nodes += 1

    proxy_value = relu_activation_count + linear_mac_count

    return {
        "proxy_value": proxy_value,
        "capacity_metric": "activation_count",
        "components": {
            "relu_activation_count": relu_activation_count,
            "linear_mac_count": linear_mac_count,
            "relu_op_breakdown": relu_op_breakdown,
            "n_linear_nodes": n_linear_nodes,
        },
    }


def _proxy_expander(model_proto) -> dict:
    """Expander/GKR proxy: log2 of widest linear-layer gate count.

    Expander uses a layered GKR circuit where the prover works over multilinear
    polynomials of degree proportional to 2^log2_W for the widest layer W.
    The dominant cost is the widest Gemm/MatMul layer: gates = in_features × out_features.
    Conv is not supported by expander-mlp-gen (linear-only); Conv nodes are
    flagged in components["unsupported_ops"].

    Returns a dict with keys:
        proxy_value      int   – log2_gates (the GKR "logrows" analogue)
        capacity_metric  str   – "log2_gates"
        components       dict  – {log2_gates, max_gates, layer_gate_counts,
                                  n_linear_layers, unsupported_ops}
    """
    graph = model_proto.graph

    shape_map: dict[str, list] = {}
    for vi in (
        list(graph.input) + list(graph.output) + list(graph.value_info)
    ):
        shape: list = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.HasField("dim_value") else None)
        shape_map[vi.name] = shape
    for init in graph.initializer:
        shape_map[init.name] = list(init.dims)

    layer_gate_counts: list[int] = []
    unsupported_ops: list[str] = []

    for node in graph.node:
        op_type = node.op_type

        if op_type in ("Gemm", "MatMul"):
            macs = _compute_mac_cost(node, shape_map)
            if macs > 0:
                layer_gate_counts.append(macs)
        elif op_type == "Conv":
            # Conv is not supported by expander-mlp-gen; flag it.
            unsupported_ops.append(op_type)
            # Still compute gate count for informational purposes.
            macs = _compute_mac_cost(node, shape_map)
            if macs > 0:
                layer_gate_counts.append(macs)

    max_gates = max(layer_gate_counts) if layer_gate_counts else 1
    log2_gates = max(1, math.ceil(math.log2(max_gates)))

    return {
        "proxy_value": log2_gates,
        "capacity_metric": "log2_gates",
        "components": {
            "log2_gates": log2_gates,
            "max_gates": max_gates,
            "layer_gate_counts": layer_gate_counts,
            "n_linear_layers": len(layer_gate_counts),
            "unsupported_ops": unsupported_ops,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_backend_proxy(model_proto, backend: str) -> dict:
    """Compute a backend-specific ZK cost estimate from an ONNX ModelProto.

    Parameters
    ----------
    model_proto:
        An ``onnx.ModelProto`` with shape inference already run.  Use
        ``onnx.shape_inference.infer_shapes(onnx.load(path))`` or the
        ``_export_to_onnx_proto`` helper in ``zk_proxy`` to obtain one.
    backend:
        One of ``"ezkl"``, ``"deepprove"``, or ``"expander"``.

    Returns
    -------
    dict with keys:
        ``backend``          str   – the backend name
        ``proxy_value``      int   – headline cost prediction
        ``capacity_metric``  str   – unit of ``proxy_value``
        ``components``       dict  – detailed breakdown

    Raises
    ------
    ValueError
        If ``backend`` is not one of the supported values.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Supported backends: {sorted(BACKENDS)}"
        )

    if backend == "ezkl":
        result = _proxy_ezkl(model_proto)
    elif backend == "deepprove":
        result = _proxy_deepprove(model_proto)
    elif backend == "expander":
        result = _proxy_expander(model_proto)
    else:  # unreachable given the guard above, but satisfies type checkers
        raise ValueError(f"Backend {backend!r} not implemented")

    return {
        "backend": backend,
        "proxy_value": int(result["proxy_value"]),
        "capacity_metric": result["capacity_metric"],
        "components": result["components"],
    }


def compute_all_backends(model_proto) -> dict[str, dict]:
    """Compute proxy estimates for all three backends in one call.

    Returns a dict keyed by backend name, each value is the output of
    ``compute_backend_proxy``.
    """
    return {
        backend: compute_backend_proxy(model_proto, backend)
        for backend in sorted(BACKENDS)
    }
