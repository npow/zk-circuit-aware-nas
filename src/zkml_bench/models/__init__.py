"""Models package for zkml-bench.

Importing this package registers all built-in baseline and ZK-friendly models
into the global model registry.
"""

from .registry import register_model, get_model, list_models
from . import baselines  # noqa: F401 - register built-in models
from . import zk_friendly  # noqa: F401 - register ZK-friendly models
from . import transformers  # noqa: F401 - register transformer family
from . import moe  # noqa: F401 - register MoE models
from . import attention  # noqa: F401 - register attention variants + wide/huge/quant models
from . import poly_transformers  # noqa: F401 - register ZK-friendly polynomial transformers
from . import iso_flop_pairs  # noqa: F401 - register iso-FLOP architectural pairs

__all__ = ["register_model", "get_model", "list_models"]
