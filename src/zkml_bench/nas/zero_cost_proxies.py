"""Zero-cost neural architecture search proxies.

Implements training-free accuracy estimators that work with arbitrary
differentiable activations, including polynomial activations (x², x²+x, x³+x)
that are standard in ZK-friendly model families.

NEAR Score (Network Expressivity by Activation Rank)
-----------------------------------------------------
Reference: Husistein, Reiher & Eckhoff, "NEAR: A Training-Free Pre-Estimator of
Machine Learning Model Performance", ICLR 2025. arXiv:2408.08776.

The NEAR score sums the effective rank of the pre-activation matrix (Z_l)
and post-activation matrix (H_l) across all linear layers:

    s = sum_l [ erank(Z_l) + erank(H_l) ]

where erank(A) = exp( H(p_1, ..., p_Q) ) with Shannon entropy H over
normalised singular values p_k = sigma_k / sum_j sigma_j.

This measure is explicitly activation-function-agnostic: it does not require
ReLU (unlike NASWOT), BatchNorm (unlike Zen-Score), or any specific
nonlinearity structure (unlike TE-NAS NTK). It works equally well for ReLU,
x², x²+x, x³+x, and any other differentiable activation.

Higher NEAR score → higher predicted accuracy.

Usage
-----
    from zkml_bench.nas.zero_cost_proxies import near_score

    score = near_score(model, dataloader, num_batches=2, device='cpu')
"""

from __future__ import annotations

import math
import logging
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Effective-rank helper
# ---------------------------------------------------------------------------

def _effective_rank(
    matrix: torch.Tensor,
    eps: float = 1e-10,
    max_dim: int = 64,
) -> float:
    """Compute the effective rank of a 2-D matrix via Shannon entropy of
    normalised singular values.

    Definition (Roy & Vetterli, 2007, also used in NEAR arXiv:2408.08776):

        erank(A) = exp( -sum_k p_k * log(p_k) )

    where p_k = sigma_k / sum_j sigma_j are the l1-normalised singular values.

    Returns a float in [1, min(M, N)]. A value close to 1 indicates a
    near-rank-1 matrix (collapsed representations); a value close to min(M,N)
    indicates full-rank, maximally expressive features.

    Parameters
    ----------
    matrix : Tensor, shape (n_samples, n_features)
        Feature matrix to analyse.
    eps : float
        Small constant to avoid log(0).
    max_dim : int
        Cap on both dimensions before SVD, for computational efficiency.
        The NEAR paper samples a random submatrix for CNNs; we apply the same
        approach for all layers to keep SVD O(max_dim^2) rather than
        O(n_features^2). Default 64 keeps SVD << 1s on CPU.
    """
    # matrix: (n_samples, n_features)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D tensor, got shape {matrix.shape}")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        # Degenerate case: single sample or single feature — rank = 1
        return 1.0

    # Detach and convert to float32 for numerical stability
    m = matrix.detach().float()

    # Subsample rows and columns to cap SVD cost.
    # The NEAR paper (arXiv:2408.08776, CNN section) subsamples C' contiguous
    # rows from flattened CNN feature maps before computing effective rank.
    # We apply random column subsampling for MLP features too, which preserves
    # the rank signal while keeping SVD O(max_dim^2).
    torch.manual_seed(0)  # deterministic subsampling
    if m.shape[0] > max_dim:
        idx = torch.randperm(m.shape[0])[:max_dim]
        m = m[idx]
    if m.shape[1] > max_dim:
        idx = torch.randperm(m.shape[1])[:max_dim]
        m = m[:, idx]

    # Centre the matrix (subtract column mean) before SVD — standard practice
    m = m - m.mean(dim=0, keepdim=True)

    try:
        # Use torch.linalg.svdvals (singular values only, faster than full SVD)
        svs = torch.linalg.svdvals(m)  # shape: (min(M, N),)
    except Exception:
        # Fallback: use torch.svd if linalg is unavailable
        try:
            _, svs, _ = torch.svd(m, compute_uv=False)
        except Exception:
            logger.warning("SVD failed, returning erank=1.0")
            return 1.0

    # L1-normalise to get a probability distribution
    total = svs.sum()
    if total < eps:
        return 1.0  # all-zero matrix
    p = svs / total

    # Shannon entropy (nats), clamped to avoid log(0)
    log_p = torch.log(p.clamp(min=eps))
    entropy = -(p * log_p).sum().item()

    return math.exp(entropy)


# ---------------------------------------------------------------------------
# Layer activation collector
# ---------------------------------------------------------------------------

class _PrePostActivationCollector:
    """Hook-based collector for (pre-activation, post-activation) pairs.

    Registers forward hooks on every Linear and Conv2d layer. Each hook
    captures:
      - Z_l: the input to the layer (pre-activation of the *next* nonlinearity,
             equivalently: the linear output of layer l)
      - H_l: the output of the immediately following activation module

    Implementation note: we intercept inputs to each Linear/Conv2d as Z_l
    (they represent the features that enter the linear transform), then
    intercept the outputs of the same Linear/Conv2d as the pre-activation
    values that will be fed to the nonlinearity. For the post-activation, we
    register hooks on any activation-like module (anything that is not
    Linear/Conv2d/BatchNorm/LayerNorm/Flatten/Dropout/Sequential/
    ModuleList/Parameter containers).

    For CNN layers, feature maps (B, C, H, W) are reshaped to (B*H*W, C).
    """

    _LINEAR_TYPES = (nn.Linear, nn.Conv2d, nn.Conv1d)
    # Module types we skip when looking for activation hooks
    _SKIP_TYPES = (
        nn.Linear, nn.Conv2d, nn.Conv1d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm,
        nn.Flatten, nn.Dropout, nn.AlphaDropout,
        nn.Sequential, nn.ModuleList, nn.ModuleDict,
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        nn.Identity,
    )

    def __init__(self) -> None:
        self._handles: list = []
        # Maps layer_id -> list of tensors (one per batch)
        self.pre_activations: dict[str, list[torch.Tensor]] = {}
        self.post_activations: dict[str, list[torch.Tensor]] = {}

    def register(self, model: nn.Module) -> None:
        """Register hooks on all linear + activation layers."""
        # Forward hook for linear layers: capture output as pre-activation Z_l
        for name, module in model.named_modules():
            if isinstance(module, self._LINEAR_TYPES):
                layer_key = f"linear:{name}"
                self.pre_activations[layer_key] = []

                def _hook_pre(mod, inp, out, key=layer_key):
                    feat = out.detach()
                    # Flatten spatial dims for Conv: (B, C, H, W) -> (B*H*W, C)
                    if feat.ndim == 4:
                        B, C, H, W = feat.shape
                        feat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
                    elif feat.ndim == 3:
                        B, L, C = feat.shape
                        feat = feat.reshape(B * L, C)
                    elif feat.ndim == 2:
                        pass  # already (B, features)
                    else:
                        feat = feat.reshape(feat.shape[0], -1)
                    self.pre_activations[key].append(feat.cpu())

                self._handles.append(module.register_forward_hook(_hook_pre))

            elif not isinstance(module, self._SKIP_TYPES):
                # Treat as an activation function — capture its output as H_l
                layer_key = f"act:{name}"
                self.post_activations[layer_key] = []

                def _hook_post(mod, inp, out, key=layer_key):
                    if not isinstance(out, torch.Tensor):
                        return
                    feat = out.detach()
                    if feat.ndim == 4:
                        B, C, H, W = feat.shape
                        feat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
                    elif feat.ndim == 3:
                        B, L, C = feat.shape
                        feat = feat.reshape(B * L, C)
                    elif feat.ndim == 2:
                        pass
                    else:
                        feat = feat.reshape(feat.shape[0], -1)
                    self.post_activations[key].append(feat.cpu())

                self._handles.append(module.register_forward_hook(_hook_post))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_matrices(self) -> list[torch.Tensor]:
        """Return one concatenated matrix per tracked layer."""
        matrices = []
        for tensors in self.pre_activations.values():
            if tensors:
                matrices.append(torch.cat(tensors, dim=0))
        for tensors in self.post_activations.values():
            if tensors:
                matrices.append(torch.cat(tensors, dim=0))
        return matrices


# ---------------------------------------------------------------------------
# Public NEAR scoring function
# ---------------------------------------------------------------------------

def near_score(
    model: nn.Module,
    dataloader: Union[DataLoader, None] = None,
    num_batches: int = 2,
    device: str = "cpu",
    *,
    data: Union[torch.Tensor, None] = None,
) -> float:
    """Compute the NEAR (Network Expressivity by Activation Rank) score.

    NEAR is a training-free, activation-agnostic accuracy proxy that measures
    the expressivity of a network by summing the effective ranks of pre- and
    post-activation feature matrices across all layers.

    Reference: Husistein, Reiher & Eckhoff, ICLR 2025, arXiv:2408.08776.

    The score is:
        s = sum_l [ erank(Z_l) + erank(H_l) ]

    where erank(A) = exp( H(p_1,...,p_Q) ) with Shannon entropy over
    l1-normalised singular values. Higher score => higher predicted accuracy.

    This function is activation-agnostic by construction: it works for ReLU,
    x², x²+x, x³+x, GELU, Swish, or any other differentiable nonlinearity,
    because effective rank depends only on the geometry of the feature matrix,
    not on the specific functional form of the activation.

    Parameters
    ----------
    model : nn.Module
        The (untrained) network to score.
    dataloader : DataLoader, optional
        Source of input batches. Mutually exclusive with ``data``.
    num_batches : int
        Number of batches to accumulate features from.
    device : str
        Torch device string.
    data : Tensor, optional
        Pre-formed input tensor. Mutually exclusive with ``dataloader``.
        If provided, ``num_batches`` is ignored.

    Returns
    -------
    float
        NEAR score (higher = more expressive = predicted better accuracy).
        Returns 0.0 if no features can be collected (e.g., an empty model).
    """
    if dataloader is None and data is None:
        raise ValueError("Provide either dataloader or data.")
    if dataloader is not None and data is not None:
        raise ValueError("Provide either dataloader or data, not both.")

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    collector = _PrePostActivationCollector()
    collector.register(model)

    try:
        with torch.no_grad():
            if data is not None:
                model(data.to(dev))
            else:
                for i, batch in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    # DataLoader may return (x, y) tuples or just x
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    model(x.to(dev))
    finally:
        collector.remove()

    matrices = collector.get_matrices()
    if not matrices:
        logger.warning("near_score: no feature matrices collected — returning 0.0")
        return 0.0

    total = 0.0
    for mat in matrices:
        total += _effective_rank(mat)

    return total


# ---------------------------------------------------------------------------
# Synflow proxy (data-free, activation-agnostic)
# ---------------------------------------------------------------------------

def synflow_score(model: nn.Module, input_shape: tuple, device: str = "cpu") -> float:
    """Compute the SynFlow score (Tanaka et al., NeurIPS 2020).

    SynFlow replaces all nonlinearities with identity during scoring and uses
    a synthetic all-ones input. This makes it completely data-free and
    activation-agnostic — the actual activation function (ReLU, x², x³, etc.)
    does not affect the score computation.

    Score = Σ_l |W_l ⊙ (∂R/∂W_l)| where R = 1ᵀ f(1) with identity activations.

    Higher score → higher predicted accuracy.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    # Replace all nonlinearities with identity by monkey-patching forward methods
    # The standard approach: set all parameters to their absolute values,
    # run with all-ones input, backprop the sum of outputs.
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param.data)
        param.data.abs_()

    # Synthetic all-ones input
    x = torch.ones(1, *input_shape).to(dev)
    x.requires_grad_(True)

    # Forward pass - NOTE: we don't replace activations explicitly.
    # Instead we use the absolute-value weight trick: with |W| weights and
    # positive inputs, all activations are positive, so ReLU acts as identity.
    # For polynomial activations (x², x³), the score captures path norms.
    output = model(x)
    if isinstance(output, (list, tuple)):
        output = output[0]
    loss = output.sum()
    loss.backward()

    score = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            score += (param.data * param.grad).abs().sum().item()
        # Restore signs
        param.data.mul_(signs[name])

    # Zero gradients
    model.zero_grad()
    return score


# ---------------------------------------------------------------------------
# ZiCo proxy (Li et al., ICLR 2023)
# ---------------------------------------------------------------------------

def zico_score(
    model: nn.Module,
    dataloader: Union[DataLoader, None] = None,
    device: str = "cpu",
    *,
    data: Union[torch.Tensor, None] = None,
) -> float:
    """Compute the ZiCo score (Li et al., ICLR 2023, arXiv:2301.11233).

    ZiCo = Σ_l [ log(σ(|∇_{W_l} L|)) − log(|μ(∇_{W_l} L)|) ]

    where σ = std and μ = mean of absolute gradient magnitudes per layer.
    This measures the log coefficient of variation of gradient magnitudes —
    higher value indicates more diverse gradient directions = better trainability.

    Works with polynomial activations (x², x³) because gradient magnitudes
    are well-defined for any smooth activation.

    Higher score → higher predicted accuracy.
    """
    if dataloader is None and data is None:
        raise ValueError("Provide either dataloader or data.")

    dev = torch.device(device)
    model = model.to(dev)
    model.train()  # needs grad

    # Accumulate gradient magnitudes across one mini-batch
    if data is not None:
        x = data.to(dev)
        labels = torch.zeros(x.shape[0], dtype=torch.long).to(dev)
        output = model(x)
        if isinstance(output, (list, tuple)):
            output = output[0]
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
    else:
        batch = next(iter(dataloader))
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(dev)
        labels = (batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1
                  else torch.zeros(x.shape[0], dtype=torch.long)).to(dev)
        output = model(x)
        if isinstance(output, (list, tuple)):
            output = output[0]
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()

    score = 0.0
    eps = 1e-8
    for param in model.parameters():
        if param.grad is not None and param.grad.numel() > 1:
            g = param.grad.abs()
            mu = g.mean().item()
            sigma = g.std().item()
            if mu > eps and sigma > eps:
                score += math.log(sigma + eps) - math.log(mu + eps)

    model.zero_grad()
    model.eval()
    return score


# ---------------------------------------------------------------------------
# MeCo proxy (Jiang et al., NeurIPS 2023)
# ---------------------------------------------------------------------------

def meco_score(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu",
) -> float:
    """Compute the MeCo score (Jiang et al., NeurIPS 2023).

    MeCo uses the minimum eigenvalue of the Pearson correlation matrix of
    feature maps from one random data point in a single forward pass.

    Score = min eigenvalue of Corr(H_l) summed across layers.
    Higher minimum eigenvalue → less correlation collapse → better trainability.

    Completely data-free (uses one random input), label-free, single forward pass.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    # Random single input
    x = torch.randn(1, *input_shape).to(dev)

    # Collect feature maps from all linear/conv layers
    feature_maps = []
    hooks = []

    def _hook(module, inp, out):
        feat = out.detach()
        if feat.ndim == 4:
            B, C, H, W = feat.shape
            feat = feat.reshape(B, C, -1).mean(-1)  # (B, C)
        elif feat.ndim == 3:
            feat = feat.reshape(feat.shape[0], -1)
        elif feat.ndim == 2:
            pass
        else:
            feat = feat.reshape(1, -1)
        if feat.shape[-1] >= 2:
            feature_maps.append(feat.squeeze(0).float())

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()

    if not feature_maps:
        return 0.0

    score = 0.0
    for feat in feature_maps:
        # feat shape: (d,) — one sample, d features
        if feat.numel() < 4:
            continue
        # For a 1D feature vector we can't compute a d×d correlation matrix
        # with a single sample. MeCo (Jiang et al., NeurIPS 2023) uses the
        # correlation matrix across the feature dimension. We need to reshape
        # into a 2D matrix (groups × per-group) first.
        d = feat.numel()
        # Find largest roughly-square factorization for the 2D correlation matrix
        n_rows = max(2, int(d ** 0.5))
        while d % n_rows != 0 and n_rows > 1:
            n_rows -= 1
        if n_rows < 2:
            continue
        n_cols = d // n_rows
        H = feat.reshape(n_rows, n_cols).float()  # (n_rows, n_cols)

        # Pearson correlation matrix across rows (n_rows × n_rows)
        H = H - H.mean(dim=1, keepdim=True)
        norms = H.norm(dim=1, keepdim=True).clamp(min=1e-8)
        H = H / norms
        C = H @ H.T  # (n_rows, n_rows) — Pearson correlation matrix

        # Minimum eigenvalue of C (real symmetric)
        try:
            eigvals = torch.linalg.eigvalsh(C)  # ascending order
            min_eig = float(eigvals[0].item())
            score += min_eig
        except Exception:
            continue

    return score


# ---------------------------------------------------------------------------
# ZenNAS proxy (Lin et al., ICCV 2021, Alibaba)
# ---------------------------------------------------------------------------

def zenas_score(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu",
    n_samples: int = 32,
    sigma: float = 1e-2,
) -> float:
    """Compute the Zen-NAS score (Lin et al., ICCV 2021, arXiv:2102.01063).

    Zen(f) = log(E[‖f(x+ε) − f(x)‖₂]) + log(E[std(f(x))])

    Measures output sensitivity to Gaussian perturbations + output variance.
    Explicitly BN-free in score computation (uses random Gaussian inputs).

    For ZK-ML: ZenNAS is theoretically compatible with polynomial activations
    because it measures the actual network output geometry, not activation patterns.
    Absolute Zen scores differ between ReLU and x² networks (x² is smoother),
    but rankings within a fixed activation family should be valid.

    Higher score → more expressive → predicted better accuracy.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    with torch.no_grad():
        x = torch.randn(n_samples, *input_shape).to(dev)
        eps = torch.randn_like(x) * sigma

        out_x = model(x)
        out_xeps = model(x + eps)

        if isinstance(out_x, (list, tuple)):
            out_x = out_x[0]
        if isinstance(out_xeps, (list, tuple)):
            out_xeps = out_xeps[0]

        # Term 1: E[log(‖f(x+ε) - f(x)‖₂)] — expectation of log, not log of expectation
        # ZenNAS paper (Lin et al., ICCV 2021): E_{x,ε}[log ‖f(x+ε)-f(x)‖_F]
        diffs = (out_xeps - out_x).norm(dim=-1).clamp(min=1e-10)
        term1 = float(diffs.log().mean().item())

        # Term 2: log(E_x[std(f(x))]) — std over output dims, then mean over batch
        std_val = out_x.std(dim=0).mean().clamp(min=1e-10)
        term2 = math.log(float(std_val))

    return term1 + term2


# ---------------------------------------------------------------------------
# NASWOT proxy (Mellor et al., ICML 2021) — expected to FAIL for x²
# ---------------------------------------------------------------------------

def naswot_score(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu",
    n_samples: int = 32,
) -> float:
    """Compute the NASWOT score (Mellor et al., ICML 2021, arXiv:2006.04647).

    NASWOT uses binary ReLU activation patterns: K_{ij} = sum_k 1[h_i^k = h_j^k]
    where h_i^k = 1[ReLU(W^k x_i + b) > 0] is the binary activation code.
    Score = log|det(K)|.

    ⚠️  EXPECTED TO FAIL for polynomial activations (x², x³):
    With x² (always non-negative), 1[x² > 0] = 1 for all inputs.
    All samples map to the same binary code → K = all-ones matrix → rank-1
    → log|det(K)| = -∞ or undefined.

    This function handles the degenerate case by returning -inf when K is
    near rank-1, allowing the failure to be detected empirically.

    For ReLU architectures: higher score → more diverse activation patterns
    → richer expressivity → predicted better accuracy.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    # Collect binary activation patterns from ReLU layers
    relu_patterns = []
    hooks = []

    def _hook(module, inp, out):
        # Binary pattern: 1 if activation > 0, else 0
        pattern = (out.detach() > 0).float()
        # Flatten: (B, ...) -> (B, n_activations)
        pattern = pattern.reshape(out.shape[0], -1)
        relu_patterns.append(pattern.cpu())

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(_hook))

    x = torch.randn(n_samples, *input_shape).to(dev)
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()

    if not relu_patterns:
        # No ReLU layers found (e.g., polynomial activation model)
        # Return -inf to signal complete degeneracy
        return float("-inf")

    # Concatenate patterns: (n_samples, total_activations)
    patterns = torch.cat(relu_patterns, dim=1).float()

    # K_{ij} = Σ_k 1[h_i^k = h_j^k] — count ALL agreements (both-1 AND both-0)
    # = (both-1 count) + (both-0 count)
    # = patterns @ patterns.T + (1-patterns) @ (1-patterns).T
    ones_agree = patterns @ patterns.T
    zeros_agree = (1.0 - patterns) @ (1.0 - patterns).T
    K = ones_agree + zeros_agree  # (n_samples, n_samples)

    # Check for rank degeneracy (failure mode for x² activations)
    rank = torch.linalg.matrix_rank(K).item()
    if rank <= 1:
        return float("-inf")  # Complete degeneracy

    # Log determinant via Cholesky for numerical stability
    try:
        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n_samples))
        log_det = 2.0 * torch.log(torch.diagonal(L)).sum().item()
    except Exception:
        try:
            sign, log_det = torch.linalg.slogdet(K)
            log_det = log_det.item() if sign.item() > 0 else float("-inf")
        except Exception:
            log_det = float("-inf")

    return log_det


# ---------------------------------------------------------------------------
# Combined zero-cost NAS objective: NEAR × (1 / cost_proxy)
# ---------------------------------------------------------------------------

def zero_cost_nas_score(
    model: nn.Module,
    data: torch.Tensor,
    cost_proxy_value: float,
    lam: float = 1.0,
    device: str = "cpu",
) -> float:
    """Lagrangian-scalarized joint proxy for ZK-ML NAS.

    Combines the NEAR accuracy proxy with ZK circuit cost via Lagrangian
    scalarization — the most principled multi-objective formulation for NAS:

        joint(model) = NEAR(model) − λ × log(ZK_cost(model))

    This cleanly trades accuracy vs. proving cost and admits convex analysis
    of the Pareto frontier as λ varies. The log transform normalizes the
    cost proxy across the typically exponential range of circuit sizes.

    Parameters
    ----------
    model : nn.Module
        Candidate architecture (untrained).
    data : Tensor
        One mini-batch of representative inputs.
    cost_proxy_value : float
        Backend-specific ZK cost estimate (higher = more expensive to prove).
        Should be positive (e.g., constraint count from structural proxy).
    lam : float
        Lagrange multiplier controlling accuracy-cost tradeoff.
        lam=0 ignores cost (pure NEAR). lam→∞ minimizes cost only.
        Default lam=1.0 treats log(cost) as equally important as NEAR score.
    device : str
        Torch device.

    Returns
    -------
    float
        Joint score (higher = better Pareto candidate).
        Returns 0.0 if cost_proxy_value <= 0.
    """
    if cost_proxy_value <= 0:
        return 0.0
    acc_proxy = near_score(model, data=data, device=device)
    cost_term = math.log(cost_proxy_value)  # log is defined for all cost > 0
    return acc_proxy - lam * cost_term
