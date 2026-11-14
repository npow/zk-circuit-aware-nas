"""Autonomous ZK-NAS agent loop.

Analogous to karpathy/autoresearch: an LLM iteratively proposes neural
architectures, we evaluate them on a fixed training budget, and the agent
steers toward the Pareto frontier of accuracy vs. ZK constraint cost.
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
from .evaluate import EvalResult, evaluate_candidate

logger = logging.getLogger(__name__)

_PROGRAM_MD = (Path(__file__).parent / "program.md").read_text()

_MODEL_ID = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# History record
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    iteration: int
    candidate_name: str
    reasoning: str
    code: str
    top1: float
    top5: float
    constraints: int
    nonlinear_ops: int
    mult_depth: int
    n_params: int
    train_time_s: float
    proxy_method: str
    error: Optional[str]
    accepted: bool          # True if it improved the Pareto hypervolume
    hypervolume_before: float
    hypervolume_after: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def failed(self) -> bool:
        return self.error is not None


# ---------------------------------------------------------------------------
# Pareto tracker
# ---------------------------------------------------------------------------

def _hypervolume(history: list[IterationRecord]) -> float:
    points = _to_model_points(history)
    if not points:
        return 0.0
    result = compute_pareto(points, cost_metric="constraints")
    return result.hypervolume


def _to_model_points(history: list[IterationRecord]) -> list[ModelPoint]:
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


def _frontier_summary(history: list[IterationRecord]) -> str:
    points = _to_model_points(history)
    if not points:
        return "No successful evaluations yet."
    result = compute_pareto(points, cost_metric="constraints")
    lines = ["Current Pareto frontier:"]
    for p in sorted(result.frontier, key=lambda x: -x.accuracy_mean):
        lines.append(f"  {p.name}: top1={p.accuracy_mean:.2f}% constraints={int(p.cost):,}")
    lines.append(f"Hypervolume: {result.hypervolume:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_DATASET_CONTEXT: dict[str, str] = {
    "mnist": """\
## Dataset: MNIST
- 10 classes, input (1, 28, 28), 60K train / 10K test
- Accuracy range: MLPs get 95-98%, small CNNs get 98-99.5%
- Constraint range: an MLP baseline runs ~242K constraints at 97.7% accuracy
- **Target**: find architectures that exceed 98.5% accuracy at under 150K constraints,
  or match 97%+ accuracy at under 80K constraints
- MNIST is easy — don't over-engineer. Small depthwise CNNs with poly activations
  and GlobalAveragePool dominate this dataset.""",

    "cifar10": """\
## Dataset: CIFAR-10
- 10 classes, input (3, 32, 32), 50K train / 10K test
- With the modern training recipe (200 epochs, Mixup+CutMix, SWA, TTA, TrivialAugment):
  - ResNet-20 (~41M constraints) would reach ~93-94% with modern training
  - LeNet-5 (~666K constraints) would reach ~79-81%
  - A well-designed DWS-CNN at 3-10M constraints should reach 85-92%
- **The gap to fill**: nothing between ~666K and ~41M on the Pareto frontier.
  Target: 88-93% accuracy at 3-15M constraints — this would be a major contribution.
- Key challenge: poly activations hurt CIFAR-10 accuracy more than MNIST.
  Compensate with BatchNorm + skip connections + wider channels (regularization is strong).
- Depthwise separable CNNs with ReLU or GELU at 128-256 channels are the most promising.
- Don't be conservative with model size — the training recipe handles overfitting.""",

    "cifar100": """\
## Dataset: CIFAR-100
- 100 classes, input (3, 32, 32), 50K train / 10K test
- With modern training recipe (200 epochs, Mixup+CutMix, SWA, TTA):
  - ResNet-20 (~41M constraints) would reach ~70-72%
  - LeNet-5 (~666K constraints) would reach ~44-47%
  - Target: 55-68% accuracy at 3-15M constraints
- More classes = need more model capacity. Use at least 128-256 channels.
- The final Linear(C, 100) is expensive in ZK — minimize C via GlobalAveragePool.
- BatchNorm + skip connections + DropPath are essential for convergence.
- Aim for 4-6 conv stages with progressive channel widening (32→64→128→256).""",
}


def _build_system_prompt(dataset_name: str) -> str:
    context = _DATASET_CONTEXT.get(dataset_name, "")
    return _PROGRAM_MD + ("\n\n" + context if context else "")


def _build_user_prompt(
    iteration: int,
    max_iterations: int,
    dataset_name: str,
    epochs: int,
    history: list[IterationRecord],
) -> str:
    parts = [
        f"## Iteration {iteration}/{max_iterations}",
        f"Dataset: {dataset_name} | Training budget: {epochs} epochs per candidate",
        "",
    ]

    if not history:
        parts += [
            "This is the first iteration. No history yet.",
            "",
            "Skip the MLP — go straight to a small CNN. On this dataset, even a tiny",
            "depthwise separable CNN with polynomial activations will outperform any MLP",
            "on both accuracy and ZK cost. Propose a CNN baseline you can improve on.",
        ]
    else:
        parts.append(_frontier_summary(history))
        parts.append("")
        parts.append(f"## All experiments ({len(history)} total)")
        parts.append("")
        for r in history[-20:]:  # show last 20 to fit context
            status = "✓ ACCEPTED" if r.accepted else ("✗ FAILED" if r.failed else "~ rejected")
            parts.append(
                f"[{r.iteration}] {r.candidate_name} {status} | "
                f"top1={r.top1:.2f}% constraints={r.constraints:,} "
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
            "Propose a new model architecture. Think step by step about:\n"
            "1. What worked well in previous iterations?\n"
            "2. What tradeoffs exist between accuracy and ZK cost?\n"
            "3. What specific change might improve the Pareto frontier?\n"
            "\nThen output your candidate."
        )

    parts += [
        "",
        "## Output format",
        "Respond with:",
        "1. CANDIDATE_NAME = \"your_name_here\"  (snake_case, no spaces)",
        "2. USE_POLY_INIT = True/False",
        "3. REASONING = \"one sentence hypothesis\"",
        "4. A ```python code block with `get_model(num_classes, input_shape) -> nn.Module`",
        "",
        "Do not include training code. Only the model definition and the three variables.",
    ]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _extract_code(response_text: str) -> str:
    """Extract the first Python code block from the response."""
    # Match ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: look for def get_model
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
    """Extract CANDIDATE_NAME, USE_POLY_INIT, REASONING from response text or code."""
    combined = response_text + "\n" + code

    name_match = re.search(r'CANDIDATE_NAME\s*=\s*["\']([^"\']+)["\']', combined)
    candidate_name = name_match.group(1) if name_match else "unnamed"

    poly_match = re.search(r'USE_POLY_INIT\s*=\s*(True|False)', combined)
    use_poly_init = poly_match.group(1) == "True" if poly_match else False

    reason_match = re.search(r'REASONING\s*=\s*["\']([^"\']+)["\']', combined)
    reasoning = reason_match.group(1) if reason_match else ""

    return candidate_name, use_poly_init, reasoning


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _load_seed_records(seed_results_path: str, dataset_name: str) -> list[IterationRecord]:
    """Load benchmark results as iteration-0 seed records for the given dataset."""
    with open(seed_results_path) as f:
        data = json.load(f)

    seeds = []
    for r in data.get("results", []):
        if r.get("dataset") != dataset_name:
            continue
        if r.get("training", {}).get("training_failed"):
            continue
        top1 = r.get("accuracy_top1_mean", 0.0)
        if top1 <= 10.0:  # skip models stuck at random chance
            continue
        proxy = r.get("zk_proxy", {})
        seeds.append(IterationRecord(
            iteration=0,
            candidate_name=r["model_name"],
            reasoning="benchmark baseline",
            code="",
            top1=top1,
            top5=r.get("accuracy_top5_mean", 0.0),
            constraints=proxy.get("constraint_estimate", 0),
            nonlinear_ops=proxy.get("nonlinear_op_count", 0),
            mult_depth=proxy.get("multiplicative_depth") or 0,
            n_params=proxy.get("param_count", 0),
            train_time_s=r.get("training", {}).get("train_time_s", 0.0),
            proxy_method=proxy.get("proxy_method", "onnx"),
            error=None,
            accepted=True,
            hypervolume_before=0.0,
            hypervolume_after=0.0,
            timestamp=data.get("timestamp", ""),
        ))

    logger.info("Loaded %d seed baselines from %s", len(seeds), seed_results_path)
    return seeds


def run_nas(
    dataset_name: str = "mnist",
    max_iterations: int = 20,
    epochs_per_candidate: int = 10,
    batch_size: int = 512,
    lr: float = 0.1,
    device_str: str = "auto",
    seed: int = 42,
    data_dir: str = "~/.zkml_bench/data",
    output_dir: str = "./nas_results",
    api_key: Optional[str] = None,
    seed_results: Optional[str] = None,
) -> list[IterationRecord]:
    """Run the ZK-NAS autonomous agent loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "nas_log.json"
    candidates_dir = out_path / "candidates"
    candidates_dir.mkdir(exist_ok=True)

    # Support Netflix internal proxy (ANTHROPIC_BASE_URL set but no ANTHROPIC_API_KEY)
    # Also support Netflix model gateway (MGP) via ANTHROPIC_BASE_URL + dummy key
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    effective_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or (
        "sk-dummy" if base_url else None
    )
    if base_url:
        client = anthropic.Anthropic(api_key=effective_key, base_url=base_url)
    else:
        client = anthropic.Anthropic(api_key=effective_key)

    history: list[IterationRecord] = []

    # Resume from existing log if present
    if log_path.exists():
        with open(log_path) as f:
            raw = json.load(f)
        history = [IterationRecord(**r) for r in raw]
        logger.info("Resumed from %d previous iterations", len(history))
    elif seed_results:
        # Seed history with benchmark baselines (only on fresh runs, not resumes)
        history = _load_seed_records(seed_results, dataset_name)
        if history:
            _save_log(history, log_path)

    print(f"\n{'='*60}")
    print(f"ZK-NAS: {dataset_name.upper()} | {max_iterations} iterations | {epochs_per_candidate} epochs each")
    print(f"Output: {out_path.resolve()}")
    print(f"{'='*60}\n")

    system_prompt = _build_system_prompt(dataset_name)

    agent_iters_done = sum(1 for r in history if r.iteration > 0)
    for iteration in range(agent_iters_done + 1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        hv_before = _hypervolume(history)

        # --- Ask the agent ---
        user_prompt = _build_user_prompt(
            iteration=iteration,
            max_iterations=max_iterations,
            dataset_name=dataset_name,
            epochs=epochs_per_candidate,
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
                logger.warning("API call failed (attempt %d/5): %s — retrying in %ds", _attempt + 1, e, wait)
                print(f"API error (attempt {_attempt + 1}/5): {e} — retrying in {wait}s")
                time.sleep(wait)
        if response_text is None:
            logger.error("API call failed after 5 attempts, stopping.")
            break

        # --- Extract code + metadata ---
        try:
            code = _extract_code(response_text)
        except ValueError as e:
            print(f"Could not extract code: {e}")
            print("Agent response (first 500 chars):", response_text[:500])
            # Record as failed iteration and continue
            rec = IterationRecord(
                iteration=iteration,
                candidate_name="parse_error",
                reasoning="",
                code=response_text[:2000],
                top1=0.0, top5=0.0, constraints=0,
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

        candidate_name, use_poly_init, reasoning = _extract_metadata(response_text, code)

        # Prepend metadata vars to code so evaluate_candidate can find them
        full_code = (
            f"CANDIDATE_NAME = {candidate_name!r}\n"
            f"USE_POLY_INIT = {use_poly_init!r}\n"
            f"REASONING = {reasoning!r}\n\n"
        ) + code

        # Save candidate code
        code_file = candidates_dir / f"iter_{iteration:03d}_{candidate_name}.py"
        code_file.write_text(full_code)

        print(f"Candidate: {candidate_name} (poly_init={use_poly_init})")
        print(f"Reasoning: {reasoning}")
        print("Evaluating...", flush=True)

        # --- Evaluate ---
        result: EvalResult = evaluate_candidate(
            code=full_code,
            dataset_name=dataset_name,
            epochs=epochs_per_candidate,
            batch_size=batch_size,
            lr=lr,
            device_str=device_str,
            seed=seed,
            data_dir=data_dir,
        )

        # --- Check Pareto improvement ---
        hv_after = hv_before
        accepted = False
        if not result.failed:
            history_with_new = history + [IterationRecord(
                iteration=iteration, candidate_name=result.candidate_name,
                reasoning=reasoning, code=full_code,
                top1=result.top1, top5=result.top5, constraints=result.constraints,
                nonlinear_ops=result.nonlinear_ops, mult_depth=result.mult_depth,
                n_params=result.n_params, train_time_s=result.train_time_s,
                proxy_method=result.proxy_method, error=None,
                accepted=True,  # placeholder
                hypervolume_before=hv_before, hypervolume_after=hv_before,
                timestamp="",
            )]
            hv_after = _hypervolume(history_with_new)
            accepted = hv_after > hv_before

        # --- Record ---
        rec = IterationRecord(
            iteration=iteration,
            candidate_name=result.candidate_name,
            reasoning=reasoning,
            code=full_code,
            top1=result.top1,
            top5=result.top5,
            constraints=result.constraints,
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

        # --- Print result ---
        print(result.summary())
        if accepted:
            print(f"✓ ACCEPTED — hypervolume {hv_before:.4f} → {hv_after:.4f}")
        elif result.failed:
            print(f"✗ FAILED — {result.error}")
        else:
            print(f"~ rejected (no Pareto improvement; hv={hv_after:.4f})")

        # --- Save ---
        _save_log(history, log_path)

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("ZK-NAS Complete")
    print(_frontier_summary(history))
    accepted_count = sum(1 for r in history if r.accepted)
    print(f"Accepted: {accepted_count}/{len(history)} iterations improved Pareto hypervolume")
    print(f"Results saved to: {out_path.resolve()}")
    print(f"{'='*60}\n")

    _write_summary(history, out_path)
    return history


def _save_log(history: list[IterationRecord], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in history], f, indent=2)


def _write_summary(history: list[IterationRecord], out_path: Path) -> None:
    """Write a markdown summary of all iterations."""
    lines = ["# ZK-NAS Results\n"]
    lines.append(_frontier_summary(history))
    lines.append("\n## All Iterations\n")
    lines.append("| # | Name | top1 | constraints | params | accepted |")
    lines.append("|---|------|------|-------------|--------|----------|")
    for r in history:
        status = "✓" if r.accepted else ("✗ fail" if r.failed else "~")
        lines.append(
            f"| {r.iteration} | {r.candidate_name} | {r.top1:.2f}% "
            f"| {r.constraints:,} | {r.n_params:,} | {status} |"
        )
    (out_path / "nas_summary.md").write_text("\n".join(lines))
