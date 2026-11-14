"""Backend-aware ZK-NAS agent loop.

Extends the single-backend NAS (agent_loop.py) with an explicit target-backend
cost model.  The key contribution: the same LLM-driven search loop, but the
system prompt tells the agent which ZK prover it is optimising for and what
that prover penalises.  This causes the agent to propose structurally different
architectures depending on the target backend:

- EZKL / Halo2 (``ezkl``):  cost snaps to powers of 2; depth is approximately
  free at iso-FLOPs (both architectures land in the same logrow bucket).
  Agent can use deep-narrow designs freely.

- DeepProve / GKR (``deepprove``): each ReLU output element and each linear MAC
  adds to the per-sample sumcheck cost.  Depth multiplies ReLU counts —
  wide-shallow architectures with fewer total ReLU activations win.

- Expander / GKR (``expander``): cost is log2 of the widest layer gate count.
  Agent should minimise the maximum layer width, not total ops.

The ``BackendAwareNAS`` class is a drop-in replacement for ``run_nas``; it adds
a ``target_backend`` parameter and selects the proxy + system-prompt accordingly.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import anthropic

from ..metrics.pareto import ModelPoint, compute_pareto
from ..metrics.zk_proxy_multi import compute_backend_proxy
from .evaluate import EvalResult, evaluate_candidate

logger = logging.getLogger(__name__)

_MODEL_ID = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Backend-specific cost model descriptions (injected into system prompt)
# ---------------------------------------------------------------------------

_BACKEND_COST_MODELS: dict[str, str] = {
    "ezkl": """\
## Target Backend: EZKL (Halo2 / KZG)

**Cost metric**: `logrows` — the base-2 exponent of total circuit row count.
Circuit size is *snapped to the next power of two*, so two architectures with
matched MACs land in the same logrows bucket even if one is twice as deep.

**Implications for architecture search**:
- Depth is approximately cost-neutral at iso-FLOPs on this backend.
- Deep-narrow and wide-shallow models of equal total MACs cost the same to prove.
- The dominant cost driver is the **total MAC count** (linear ops) and nonlinear
  activation count.
- Polynomial activations (x², x²+x) are cheaper than ReLU on Halo2 because they
  avoid the lookup-table argument.
- You are free to use 4-8 layer MLPs or deep CNNs — depth does not hurt you here.
- Minimise total parameter count and nonlinear ops. Deep, moderately narrow
  networks are fine.

**Optimisation target**: minimise `2^logrows` (equivalently, minimise total MACs
+ activation count, rounded to the nearest power of 2).
""",

    "deepprove": """\
## Target Backend: DeepProve (GKR / LogUp)

**Cost metric**: `activation_count` = total ReLU output elements + linear MAC count.
DeepProve's bottleneck is the LogUp lookup argument over the 8-bit ReLU
requantisation table: each ReLU output element generates one lookup query, and
the sumcheck cost accumulates **per layer** through the network.

**Critical finding (measured)**:
- A 2-layer × 512-unit MLP proves in 0.72 s on DeepProve.
- A 6-layer × 372-unit MLP with identical FLOPs proves in 0.98 s — **1.36× slower**.
- Depth multiplies the ReLU activation count even at iso-FLOPs.
- At larger scale (12.6M FLOPs), the depth penalty is 1.18× (2.41 s vs 2.85 s).

**Implications for architecture search** (THE REVERSAL vs Halo2):
- **Prefer wide-shallow architectures.** A 1-layer × 2048 MLP proves faster than
  a 10-layer × 684 MLP with identical FLOPs on DeepProve.
- Minimise the **number of layers** even if it means increasing width.
- Minimise total ReLU activation count (width × number of ReLU layers).
- Use ReLU activations — polynomial activations (Mul nodes) are NOT supported by
  DeepProve and will cause proving to fail.
- Avoid BatchNorm (not supported by DeepProve; use LayerNorm or no normalisation).
- Avoid AvgPool (not supported; MaxPool is OK).
- Avoid residual adds (not supported by DeepProve).
- 1-2 layer MLPs or very shallow CNNs (≤2 conv layers) are ideal.
- Target: 1-2 hidden layers, width 512-2048, ReLU activations only.

**Optimisation target**: minimise `relu_activation_count + linear_mac_count`.
To minimise this: use as few layers as possible, each as wide as needed for
accuracy, with no unnecessary nonlinearities.
""",

    "expander": """\
## Target Backend: Expander (GKR / ECC layered circuit)

**Cost metric**: `log2_gates` — log2 of the widest single-layer gate count.
Expander uses a static layered GKR circuit; the prover commits to a multilinear
polynomial of degree proportional to the widest layer.

**Implications for architecture search**:
- Minimise the **width of the widest layer** (the bottleneck), not total ops.
- A 4-layer × 256-unit MLP costs the same as a 2-layer × 256-unit MLP (width=256 dominates).
- But a 2-layer × 1024-unit MLP costs more (log2(1024²) > log2(256²)).
- Keep all layers at moderate, equal widths. Avoid large bottleneck expansions.
- Conv is NOT supported by Expander's linear-circuit generator — use MLP only.
- Polynomial activations (x²) ARE supported and preferred over ReLU.
- Residual connections: not supported in linear-MLP mode.
- Target: 3-6 layers, uniform width 256-512, polynomial activations.

**Optimisation target**: minimise `log2(max_layer_gates)`.
Design all layers to have roughly equal width; avoid any single very large layer.
""",
}

_PROGRAM_MD = (Path(__file__).parent / "program.md").read_text()

_DATASET_CONTEXT: dict[str, str] = {
    "mnist": """\
## Dataset: MNIST
- 10 classes, input (1, 28, 28), 60K train / 10K test
- Accuracy range: MLPs get 95-98%, small CNNs get 98-99.5%
- Constraint range: an MLP baseline runs ~242K constraints at 97.7% accuracy
- **Target**: find architectures that exceed 98.5% accuracy at under 150K constraints,
  or match 97%+ accuracy at under 80K constraints
- MNIST is easy — don't over-engineer. Given the backend constraints (see above),
  favour architectures compatible with the target backend.""",

    "cifar10": """\
## Dataset: CIFAR-10
- 10 classes, input (3, 32, 32), 50K train / 10K test
- **Target**: 88-93% accuracy at minimal backend-specific cost
- Key challenge: respect backend op-coverage constraints (see above).""",
}


def _build_system_prompt(dataset_name: str, target_backend: str) -> str:
    backend_section = _BACKEND_COST_MODELS.get(
        target_backend,
        f"## Target Backend: {target_backend}\nNo specific cost model registered.\n",
    )
    dataset_section = _DATASET_CONTEXT.get(dataset_name, "")
    # Inject backend cost model BEFORE the generic program, so it overrides defaults
    return backend_section + "\n\n" + _PROGRAM_MD + (
        "\n\n" + dataset_section if dataset_section else ""
    )


# ---------------------------------------------------------------------------
# History record (identical schema to agent_loop.IterationRecord so logs are
# interchangeable, but adds a `backend` field)
# ---------------------------------------------------------------------------

@dataclass
class BackendIterationRecord:
    iteration: int
    backend: str
    candidate_name: str
    reasoning: str
    code: str
    top1: float
    top5: float
    constraints: int          # backend-specific proxy value
    proxy_value_ezkl: int     # always computed for comparison
    proxy_value_deepprove: int
    proxy_value_expander: int
    nonlinear_ops: int
    mult_depth: int
    n_params: int
    train_time_s: float
    proxy_method: str
    error: Optional[str]
    accepted: bool
    hypervolume_before: float
    hypervolume_after: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def failed(self) -> bool:
        return self.error is not None


# ---------------------------------------------------------------------------
# Pareto tracking (keyed on backend-specific constraint proxy)
# ---------------------------------------------------------------------------

def _hypervolume(history: list[BackendIterationRecord]) -> float:
    points = _to_model_points(history)
    if not points:
        return 0.0
    result = compute_pareto(points, cost_metric="constraints")
    return result.hypervolume


def _to_model_points(history: list[BackendIterationRecord]) -> list[ModelPoint]:
    return [
        ModelPoint(
            name=r.candidate_name,
            accuracy_mean=r.top1,
            accuracy_std=None,
            cost=float(r.constraints),
        )
        for r in history
        if not r.failed and r.top1 > 0
    ]


def _frontier_summary(history: list[BackendIterationRecord], backend: str) -> str:
    points = _to_model_points(history)
    if not points:
        return "No successful evaluations yet."
    result = compute_pareto(points, cost_metric="constraints")
    lines = [f"Current Pareto frontier (backend={backend}):"]
    for p in sorted(result.frontier, key=lambda x: -x.accuracy_mean):
        lines.append(
            f"  {p.name}: top1={p.accuracy_mean:.2f}%  "
            f"{backend}_proxy={int(p.cost):,}"
        )
    lines.append(f"Hypervolume: {result.hypervolume:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_user_prompt(
    iteration: int,
    max_iterations: int,
    dataset_name: str,
    epochs: int,
    target_backend: str,
    history: list[BackendIterationRecord],
) -> str:
    parts = [
        f"## Iteration {iteration}/{max_iterations}",
        f"Dataset: {dataset_name} | Backend: {target_backend} | Training budget: {epochs} epochs",
        "",
    ]

    if not history:
        # First iteration: give a concrete backend-specific hint
        if target_backend == "deepprove":
            parts += [
                "This is the first iteration. No history yet.",
                "",
                "For DeepProve (GKR), start with a **very shallow MLP**: 1-2 hidden layers,",
                "width 512-1024, ReLU activations only (NO polynomial activations — they are",
                "not supported by DeepProve). No BatchNorm, no residual connections.",
                "A 2-layer MLP with ReLU is the right baseline for this backend.",
            ]
        elif target_backend == "expander":
            parts += [
                "This is the first iteration. No history yet.",
                "",
                "For Expander (GKR), start with a **moderate-width MLP**: 3-4 hidden layers,",
                "uniform width ~256, polynomial activations (x² or x²+x). No Conv layers.",
                "Keep all layer widths roughly equal to minimise the widest-layer gate count.",
            ]
        else:  # ezkl
            parts += [
                "This is the first iteration. No history yet.",
                "",
                "For EZKL (Halo2), depth is approximately free at iso-FLOPs.",
                "Try a small depthwise-separable CNN or a 4-6 layer MLP with polynomial",
                "activations. The logrows bucket is determined by total MACs, not depth.",
            ]
    else:
        parts.append(_frontier_summary(history, target_backend))
        parts.append("")
        parts.append(f"## All experiments ({len(history)} total)")
        parts.append("")
        for r in history[-20:]:
            status = "✓ ACCEPTED" if r.accepted else ("✗ FAILED" if r.failed else "~ rejected")
            parts.append(
                f"[{r.iteration}] {r.candidate_name} {status} | "
                f"top1={r.top1:.2f}% {target_backend}_proxy={r.constraints:,} "
                f"params={r.n_params:,} nonlinear={r.nonlinear_ops} "
                f"mult_depth={r.mult_depth}"
            )
            if r.reasoning:
                parts.append(f"    Reasoning: {r.reasoning}")
            if r.error:
                parts.append(f"    Error: {r.error[:200]}")
        parts.append("")
        parts.append("## Your task")
        parts.append(
            f"Propose a new model architecture optimised for {target_backend.upper()}.\n"
            "Think step by step about:\n"
            f"1. What worked for {target_backend}? What failed?\n"
            f"2. How does the {target_backend} cost model penalise depth vs width?\n"
            "3. What architectural change will reduce the backend-specific cost?\n"
            "\nThen output your candidate."
        )

    parts += [
        "",
        "## Output format",
        "Respond with:",
        '1. CANDIDATE_NAME = "your_name_here"  (snake_case, no spaces)',
        "2. USE_POLY_INIT = True/False",
        '3. REASONING = "one sentence hypothesis"',
        "4. A ```python code block with `get_model(num_classes, input_shape) -> nn.Module`",
        "",
        "Do not include training code. Only the model definition and the three variables.",
    ]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Code extraction (same helpers as agent_loop.py)
# ---------------------------------------------------------------------------

def _extract_code(response_text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = response_text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if "def get_model" in line or "CANDIDATE_NAME" in line:
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()
    raise ValueError("No Python code block found in agent response.")


def _extract_metadata(response_text: str, code: str) -> tuple[str, bool, str]:
    combined = response_text + "\n" + code
    name_match = re.search(r'CANDIDATE_NAME\s*=\s*["\']([^"\']+)["\']', combined)
    candidate_name = name_match.group(1) if name_match else "unnamed"
    poly_match = re.search(r'USE_POLY_INIT\s*=\s*(True|False)', combined)
    use_poly_init = poly_match.group(1) == "True" if poly_match else False
    reason_match = re.search(r'REASONING\s*=\s*["\']([^"\']+)["\']', combined)
    reasoning = reason_match.group(1) if reason_match else ""
    return candidate_name, use_poly_init, reasoning


# ---------------------------------------------------------------------------
# Multi-backend proxy computation from an already-evaluated candidate
# ---------------------------------------------------------------------------

def _compute_all_proxies(code: str, dataset_name: str) -> dict[str, int]:
    """Return {backend: proxy_value} for all three backends, or zeros on failure."""
    import torch
    import torch.nn as nn
    from ..metrics.zk_proxy import _export_to_onnx_proto

    namespace: dict = {"torch": torch, "nn": nn}
    try:
        exec(compile(code, "<candidate>", "exec"), namespace)  # noqa: S102
    except Exception:
        return {"ezkl": 0, "deepprove": 0, "expander": 0}

    get_model_fn = namespace.get("get_model")
    if not callable(get_model_fn):
        return {"ezkl": 0, "deepprove": 0, "expander": 0}

    from ..datasets import get_dataset
    try:
        ds = get_dataset(dataset_name)
        sample = ds.sample_input(1)
        model = get_model_fn(ds.num_classes, ds.input_shape)
        model.eval()
        proto = _export_to_onnx_proto(model, sample)
    except Exception:
        return {"ezkl": 0, "deepprove": 0, "expander": 0}

    result = {}
    for backend in ("ezkl", "deepprove", "expander"):
        try:
            r = compute_backend_proxy(proto, backend)
            result[backend] = r["proxy_value"]
        except Exception:
            result[backend] = 0
    return result


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_log(history: list[BackendIterationRecord], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in history], f, indent=2)


def _write_summary(
    history: list[BackendIterationRecord],
    out_path: Path,
    backend: str,
) -> None:
    lines = [f"# Backend-Aware ZK-NAS Results (backend={backend})\n"]
    lines.append(_frontier_summary(history, backend))
    lines.append("\n## All Iterations\n")
    lines.append(
        "| # | Name | top1 | ezkl_proxy | dp_proxy | exp_proxy | accepted |"
    )
    lines.append("|---|------|------|-----------|---------|----------|----------|")
    for r in history:
        status = "✓" if r.accepted else ("✗ fail" if r.failed else "~")
        lines.append(
            f"| {r.iteration} | {r.candidate_name} | {r.top1:.2f}% "
            f"| {r.proxy_value_ezkl:,} | {r.proxy_value_deepprove:,} "
            f"| {r.proxy_value_expander:,} | {status} |"
        )
    (out_path / "nas_summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

class BackendAwareNAS:
    """Backend-aware ZK-NAS: same LLM loop as agent_loop.run_nas, but the
    system prompt and cost function are conditioned on ``target_backend``.

    Zero-cost mode
    --------------
    When ``use_zero_cost_proxy=True``, training is skipped entirely.  Each
    candidate is scored by the combined objective:

        score = NEAR(model, data) / cost_proxy_value

    where NEAR (arXiv:2408.08776, ICLR 2025) estimates accuracy from the
    effective rank of pre/post-activation feature matrices, and
    ``cost_proxy_value`` is the backend-specific ZK cost estimate.  Both are
    training-free; together they form the first training-free + proving-free
    NAS objective for ZKML.
    """

    def __init__(
        self,
        target_backend: str = "ezkl",
        dataset_name: str = "mnist",
        max_iterations: int = 20,
        epochs_per_candidate: int = 10,
        batch_size: int = 512,
        lr: float = 0.1,
        device_str: str = "auto",
        seed: int = 42,
        data_dir: str = "~/.zkml_bench/data",
        output_dir: str = "./nas_results_backend",
        api_key: Optional[str] = None,
        use_zero_cost_proxy: bool = False,
    ):
        from ..metrics.zk_proxy_multi import BACKENDS

        if target_backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend {target_backend!r}. "
                f"Supported: {sorted(BACKENDS)}"
            )
        self.target_backend = target_backend
        self.dataset_name = dataset_name
        self.max_iterations = max_iterations
        self.epochs_per_candidate = epochs_per_candidate
        self.batch_size = batch_size
        self.lr = lr
        self.device_str = device_str
        self.seed = seed
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.use_zero_cost_proxy = use_zero_cost_proxy

    # ------------------------------------------------------------------

    def _make_client(self) -> anthropic.Anthropic:
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        effective_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY") or (
            "sk-dummy" if base_url else None
        )
        if base_url:
            return anthropic.Anthropic(api_key=effective_key, base_url=base_url)
        return anthropic.Anthropic(api_key=effective_key)

    # ------------------------------------------------------------------

    def _evaluate_zero_cost(
        self,
        code: str,
        candidate_name: str,
    ) -> tuple[float, float, int, int, int, int, str, Optional[str]]:
        """Evaluate a candidate using the zero-cost NEAR proxy (no training).

        Returns
        -------
        (near_score, combined_score, constraints, nonlinear_ops, mult_depth,
         n_params, proxy_method, error)
        """
        import torch
        import torch.nn as nn

        from ..datasets import get_dataset
        from ..metrics.zk_proxy import _suppress_onnx_noise, compute_proxy
        from ..metrics.zk_proxy_multi import compute_backend_proxy
        from ..metrics.zk_proxy import _export_to_onnx_proto
        from .zero_cost_proxies import near_score as _near_score

        namespace: dict = {"torch": torch, "nn": nn}
        try:
            exec(compile(code, "<candidate>", "exec"), namespace)  # noqa: S102
        except Exception as e:
            return 0.0, 0.0, 0, 0, 0, 0, "none", f"Code exec failed: {e}"

        get_model_fn = namespace.get("get_model")
        if not callable(get_model_fn):
            return 0.0, 0.0, 0, 0, 0, 0, "none", "Code must define get_model"

        try:
            dataset_obj = get_dataset(self.dataset_name, data_dir=self.data_dir)
            sample = dataset_obj.sample_input(batch_size=32)  # >1 sample for rank
            model = get_model_fn(dataset_obj.num_classes, dataset_obj.input_shape)
            model.eval()
        except Exception as e:
            return 0.0, 0.0, 0, 0, 0, 0, "none", f"Model instantiation failed: {e}"

        # Compute NEAR score
        try:
            nr = _near_score(model, data=sample, device="cpu")
        except Exception as e:
            logger.warning("NEAR scoring failed: %s", e)
            nr = 0.0

        # Compute ZK proxy (architecture-only, no training)
        try:
            with _suppress_onnx_noise():
                proxy = compute_proxy(model, dataset_obj.sample_input(1))
            constraints = proxy.constraint_estimate
            nonlinear_ops = proxy.nonlinear_op_count
            mult_depth = (
                proxy.multiplicative_depth if proxy.multiplicative_depth is not None else 0
            )
            n_params = proxy.param_count
            proxy_method = proxy.proxy_method
        except Exception as e:
            return nr, 0.0, 0, 0, 0, 0, "none", f"ZK proxy failed: {e}"

        # Backend-specific cost for combined score
        try:
            proto = _export_to_onnx_proto(model, dataset_obj.sample_input(1))
            backend_result = compute_backend_proxy(proto, self.target_backend)
            backend_cost = max(1, backend_result["proxy_value"])
        except Exception:
            backend_cost = max(1, constraints)

        combined = nr / backend_cost if backend_cost > 0 else 0.0
        return nr, combined, constraints, nonlinear_ops, mult_depth, n_params, proxy_method, None

    # ------------------------------------------------------------------

    def run(self) -> list[BackendIterationRecord]:
        """Run the backend-aware NAS loop and return the full history."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stdout,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "nas_log.json"
        candidates_dir = self.output_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)

        history: list[BackendIterationRecord] = []

        # Resume from existing log if present
        if log_path.exists():
            with open(log_path) as f:
                raw = json.load(f)
            history = [BackendIterationRecord(**r) for r in raw]
            logger.info("Resumed from %d previous iterations", len(history))

        client = self._make_client()
        system_prompt = _build_system_prompt(self.dataset_name, self.target_backend)

        mode_label = "ZERO-COST (NEAR proxy)" if self.use_zero_cost_proxy else (
            f"{self.epochs_per_candidate} epochs each"
        )
        print(f"\n{'='*60}")
        print(
            f"Backend-Aware ZK-NAS: {self.dataset_name.upper()} | "
            f"backend={self.target_backend} | "
            f"{self.max_iterations} iterations | "
            f"{mode_label}"
        )
        print(f"Output: {self.output_dir.resolve()}")
        print(f"{'='*60}\n")

        agent_iters_done = sum(1 for r in history if r.iteration > 0)
        for iteration in range(agent_iters_done + 1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{self.max_iterations} (backend={self.target_backend}) ---")
            hv_before = _hypervolume(history)

            user_prompt = _build_user_prompt(
                iteration=iteration,
                max_iterations=self.max_iterations,
                dataset_name=self.dataset_name,
                epochs=self.epochs_per_candidate,
                target_backend=self.target_backend,
                history=history,
            )

            print("Querying agent...", flush=True)
            response_text = None
            for _attempt in range(5):
                try:
                    message = client.messages.create(
                        model=_MODEL_ID,
                        max_tokens=2048,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    response_text = message.content[0].text
                    break
                except Exception as e:
                    wait = 2 ** _attempt * 10
                    logger.warning(
                        "API call failed (attempt %d/5): %s — retrying in %ds",
                        _attempt + 1, e, wait,
                    )
                    print(f"API error (attempt {_attempt + 1}/5): {e} — retrying in {wait}s")
                    time.sleep(wait)

            if response_text is None:
                logger.error("API call failed after 5 attempts, stopping.")
                break

            # Extract code + metadata
            try:
                code = _extract_code(response_text)
            except ValueError as e:
                print(f"Could not extract code: {e}")
                rec = BackendIterationRecord(
                    iteration=iteration,
                    backend=self.target_backend,
                    candidate_name="parse_error",
                    reasoning="",
                    code=response_text[:2000],
                    top1=0.0, top5=0.0, constraints=0,
                    proxy_value_ezkl=0, proxy_value_deepprove=0,
                    proxy_value_expander=0,
                    nonlinear_ops=0, mult_depth=0, n_params=0,
                    train_time_s=0.0, proxy_method="none",
                    error=f"Code extraction failed: {e}",
                    accepted=False,
                    hypervolume_before=hv_before,
                    hypervolume_after=hv_before,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                history.append(rec)
                _save_log(history, log_path)
                continue

            candidate_name, use_poly_init, reasoning = _extract_metadata(
                response_text, code
            )

            full_code = (
                f"CANDIDATE_NAME = {candidate_name!r}\n"
                f"USE_POLY_INIT = {use_poly_init!r}\n"
                f"REASONING = {reasoning!r}\n\n"
            ) + code

            code_file = candidates_dir / f"iter_{iteration:03d}_{candidate_name}.py"
            code_file.write_text(full_code)

            print(f"Candidate: {candidate_name} (poly_init={use_poly_init})")
            print(f"Reasoning: {reasoning}")

            if self.use_zero_cost_proxy:
                # --------------------------------------------------------
                # Zero-cost evaluation: NEAR × (1/cost), no training
                # --------------------------------------------------------
                print("Scoring with NEAR zero-cost proxy (no training)...", flush=True)
                t0 = time.perf_counter()
                (
                    nr, combined_score, constraints, nonlinear_ops,
                    mult_depth, n_params, proxy_method, zc_error,
                ) = self._evaluate_zero_cost(full_code, candidate_name)
                eval_time = time.perf_counter() - t0

                all_proxies = _compute_all_proxies(full_code, self.dataset_name)
                backend_constraint = all_proxies.get(self.target_backend, constraints) or constraints

                failed = zc_error is not None
                # In zero-cost mode, "top1" is the NEAR score (for Pareto tracking
                # we use it as a proxy for accuracy; combined_score is the objective)
                top1_proxy = nr  # not a true accuracy — clearly labelled below
                top5_proxy = nr

                hv_after = hv_before
                accepted = False
                if not failed and combined_score > 0 and backend_constraint > 0:
                    history_with_new = history + [BackendIterationRecord(
                        iteration=iteration,
                        backend=self.target_backend,
                        candidate_name=candidate_name,
                        reasoning=reasoning,
                        code=full_code,
                        top1=top1_proxy, top5=top5_proxy,
                        constraints=backend_constraint,
                        proxy_value_ezkl=all_proxies.get("ezkl", 0),
                        proxy_value_deepprove=all_proxies.get("deepprove", 0),
                        proxy_value_expander=all_proxies.get("expander", 0),
                        nonlinear_ops=nonlinear_ops,
                        mult_depth=mult_depth,
                        n_params=n_params,
                        train_time_s=eval_time,
                        proxy_method=f"near:{proxy_method}",
                        error=None,
                        accepted=True,
                        hypervolume_before=hv_before,
                        hypervolume_after=hv_before,
                        timestamp="",
                    )]
                    hv_after = _hypervolume(history_with_new)
                    accepted = hv_after > hv_before

                rec = BackendIterationRecord(
                    iteration=iteration,
                    backend=self.target_backend,
                    candidate_name=candidate_name,
                    reasoning=reasoning,
                    code=full_code,
                    top1=top1_proxy, top5=top5_proxy,
                    constraints=backend_constraint,
                    proxy_value_ezkl=all_proxies.get("ezkl", 0),
                    proxy_value_deepprove=all_proxies.get("deepprove", 0),
                    proxy_value_expander=all_proxies.get("expander", 0),
                    nonlinear_ops=nonlinear_ops,
                    mult_depth=mult_depth,
                    n_params=n_params,
                    train_time_s=eval_time,
                    proxy_method=f"near:{proxy_method}",
                    error=zc_error,
                    accepted=accepted,
                    hypervolume_before=hv_before,
                    hypervolume_after=hv_after,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                history.append(rec)

                if failed:
                    print(f"FAILED — {zc_error}")
                else:
                    print(
                        f"{candidate_name}: NEAR={nr:.2f}  "
                        f"combined={combined_score:.6f}  "
                        f"{self.target_backend}_cost={backend_constraint:,}  "
                        f"params={n_params:,}  eval_time={eval_time:.2f}s"
                    )
                    if accepted:
                        print(f"ACCEPTED — hypervolume {hv_before:.4f} -> {hv_after:.4f}")
                    else:
                        print(f"rejected (no Pareto improvement; hv={hv_after:.4f})")

            else:
                # --------------------------------------------------------
                # Training-based evaluation (original path)
                # --------------------------------------------------------
                print("Evaluating...", flush=True)

                result: EvalResult = evaluate_candidate(
                    code=full_code,
                    dataset_name=self.dataset_name,
                    epochs=self.epochs_per_candidate,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    device_str=self.device_str,
                    seed=self.seed,
                    data_dir=self.data_dir,
                )

                # Compute all three backend proxies for comparison
                all_proxies = _compute_all_proxies(full_code, self.dataset_name)

                # Backend-specific constraint value for Pareto tracking
                backend_constraint = result.constraints  # default (EZKL proxy)
                if not result.failed and all_proxies[self.target_backend] > 0:
                    backend_constraint = all_proxies[self.target_backend]
                elif result.failed:
                    backend_constraint = 0

                # Pareto improvement check using backend-specific cost
                hv_after = hv_before
                accepted = False
                if not result.failed:
                    history_with_new = history + [BackendIterationRecord(
                        iteration=iteration,
                        backend=self.target_backend,
                        candidate_name=result.candidate_name,
                        reasoning=reasoning,
                        code=full_code,
                        top1=result.top1, top5=result.top5,
                        constraints=backend_constraint,
                        proxy_value_ezkl=all_proxies.get("ezkl", 0),
                        proxy_value_deepprove=all_proxies.get("deepprove", 0),
                        proxy_value_expander=all_proxies.get("expander", 0),
                        nonlinear_ops=result.nonlinear_ops,
                        mult_depth=result.mult_depth,
                        n_params=result.n_params,
                        train_time_s=result.train_time_s,
                        proxy_method=result.proxy_method,
                        error=None,
                        accepted=True,
                        hypervolume_before=hv_before,
                        hypervolume_after=hv_before,
                        timestamp="",
                    )]
                    hv_after = _hypervolume(history_with_new)
                    accepted = hv_after > hv_before

                rec = BackendIterationRecord(
                    iteration=iteration,
                    backend=self.target_backend,
                    candidate_name=result.candidate_name,
                    reasoning=reasoning,
                    code=full_code,
                    top1=result.top1, top5=result.top5,
                    constraints=backend_constraint,
                    proxy_value_ezkl=all_proxies.get("ezkl", 0),
                    proxy_value_deepprove=all_proxies.get("deepprove", 0),
                    proxy_value_expander=all_proxies.get("expander", 0),
                    nonlinear_ops=result.nonlinear_ops,
                    mult_depth=result.mult_depth,
                    n_params=result.n_params,
                    train_time_s=result.train_time_s,
                    proxy_method=result.proxy_method,
                    error=result.error,
                    accepted=accepted,
                    hypervolume_before=hv_before,
                    hypervolume_after=hv_after,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                history.append(rec)

                print(result.summary())
                if accepted:
                    print(f"ACCEPTED — hypervolume {hv_before:.4f} -> {hv_after:.4f}")
                elif result.failed:
                    print(f"FAILED — {result.error}")
                else:
                    print(f"rejected (no Pareto improvement; hv={hv_after:.4f})")

                # Print cross-backend proxy comparison
                print(
                    f"  Cross-backend proxies: "
                    f"ezkl={all_proxies.get('ezkl', 0):,}  "
                    f"deepprove={all_proxies.get('deepprove', 0):,}  "
                    f"expander={all_proxies.get('expander', 0):,}"
                )

            _save_log(history, log_path)

        # Final summary
        print(f"\n{'='*60}")
        print(f"Backend-Aware ZK-NAS Complete (backend={self.target_backend})")
        print(_frontier_summary(history, self.target_backend))
        accepted_count = sum(1 for r in history if r.accepted)
        print(f"Accepted: {accepted_count}/{len(history)} iterations improved Pareto hypervolume")
        print(f"Results saved to: {self.output_dir.resolve()}")
        print(f"{'='*60}\n")

        _write_summary(history, self.output_dir, self.target_backend)
        return history


def run_backend_aware_nas(
    target_backend: str = "ezkl",
    dataset_name: str = "mnist",
    max_iterations: int = 20,
    epochs_per_candidate: int = 10,
    batch_size: int = 512,
    lr: float = 0.1,
    device_str: str = "auto",
    seed: int = 42,
    data_dir: str = "~/.zkml_bench/data",
    output_dir: str = "./nas_results_backend",
    api_key: Optional[str] = None,
    use_zero_cost_proxy: bool = False,
) -> list[BackendIterationRecord]:
    """Functional entry point for the backend-aware NAS loop."""
    nas = BackendAwareNAS(
        target_backend=target_backend,
        dataset_name=dataset_name,
        max_iterations=max_iterations,
        epochs_per_candidate=epochs_per_candidate,
        batch_size=batch_size,
        lr=lr,
        device_str=device_str,
        seed=seed,
        data_dir=data_dir,
        output_dir=output_dir,
        api_key=api_key,
        use_zero_cost_proxy=use_zero_cost_proxy,
    )
    return nas.run()
