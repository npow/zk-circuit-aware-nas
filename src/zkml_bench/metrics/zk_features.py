"""Per-architecture ONNX-graph features used to fit and evaluate the ZK cost model.

For each node we record, per op type: node count, total output elements, and (for
Conv/Gemm/MatMul) total multiply-accumulates computed exactly as zk_proxy does.  The
hand-set proxy is  sum_op OP_COSTS[op] * (macs or elements) + NODE_OVERHEAD * n_matmul_nodes;
fitting replaces OP_COSTS and NODE_OVERHEAD by coefficients learned on a fit split.
"""
from __future__ import annotations

from typing import Any

import torch

from .zk_proxy import (OP_COSTS, _compute_mac_cost, _export_to_onnx_proto, _shape_elements,
                       _read_onnx_initializer_scalar)


def extract_features(model: torch.nn.Module, sample: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        proto = _export_to_onnx_proto(model, sample)
    shape_map: dict[str, list] = {}
    for vi in list(proto.graph.input) + list(proto.graph.output) + list(proto.graph.value_info):
        shape = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.HasField("dim_value") else None)
        shape_map[vi.name] = shape
    for init in proto.graph.initializer:
        shape_map[init.name] = list(init.dims)
    feats: dict[str, dict[str, float]] = {}
    for node in proto.graph.node:
        op = node.op_type
        out_el = 0
        for o in node.output:
            if o and o in shape_map:
                e = _shape_elements(shape_map[o])
                if e is not None:
                    out_el = max(out_el, e)
        f = feats.setdefault(op, {"count": 0, "elements": 0, "macs": 0})
        f["count"] += 1
        f["elements"] += out_el
        if op in ("Gemm", "MatMul", "Conv"):
            f["macs"] += _compute_mac_cost(node, shape_map)
            # depthwise convs have group == in_channels
            if op == "Conv":
                grp = next((a.i for a in node.attribute if a.name == "group"), 1)
                if grp > 1:
                    f.setdefault("depthwise_count", 0)
                    f["depthwise_count"] += 1
                    f.setdefault("depthwise_macs", 0)
                    f["depthwise_macs"] += _compute_mac_cost(node, shape_map)
    return feats
