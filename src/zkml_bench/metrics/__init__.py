"""Metrics collection and aggregation for zkml-bench."""
from .accuracy import compute_accuracy, AccuracyResult
from .zk_proxy import compute_proxy, ZKProxyResult, model_hash, OP_COSTS, UNKNOWN_OP_COST, detect_proxy_ood
from .zk_proxy_multi import compute_backend_proxy, compute_all_backends, BACKENDS
from .pareto import compute_pareto, ParetoResult, ModelPoint

__all__ = [
    "compute_accuracy",
    "AccuracyResult",
    "compute_proxy",
    "ZKProxyResult",
    "model_hash",
    "OP_COSTS",
    "UNKNOWN_OP_COST",
    "detect_proxy_ood",
    "compute_backend_proxy",
    "compute_all_backends",
    "BACKENDS",
    "compute_pareto",
    "ParetoResult",
    "ModelPoint",
]
