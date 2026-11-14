"""GPU validation experiment for the paper's CIFAR accuracy claims.

This runner is intentionally separate from ``generate_sweep.py``.  It keeps
the original training recipe, but adds the publication-critical controls that
the sweep runner lacks: explicit seeds, fresh 30- and 100-epoch schedules,
matched activation comparisons, learning curves, divergence reporting, and
incremental Hugging Face artifact uploads.

Examples
--------
Smoke test::

    python gpu_validation_experiment.py --suite smoke --epoch-budgets 1 \
        --seeds 42 --output-dir /tmp/zknas-smoke

Publication validation::

    python gpu_validation_experiment.py --suite publication_core \
        --epoch-budgets 30 100 --seeds 42 43 44 \
        --output-dir /tmp/zknas-validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Required for deterministic CUDA matrix multiplication.  Set before torch is
# imported so CUDA observes it when the context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "src/zkml_bench/nas")

from generate_sweep import build_model, make_sweep_configs
# Import the standalone proxy module directly. Importing ``zkml_bench.nas``
# eagerly loads the optional Anthropic-driven search loop, which is unrelated
# to this fixed experiment and should not be a GPU-job dependency.
from zero_cost_proxies import near_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("gpu_validation")


ACTIVATIONS = ("relu", "x2", "x3px", "x2px")

# These are the exact architectures behind the central accuracy/proving table.
# They run first, so even a preempted job yields the highest-value evidence.
PAPER_CLAIMS = [
    "mlp_x2px_none_128x64x32",
    "mlp_x2px_layernorm_128x64x32",
    "dws_relu_none_16_32_64",
    "dws_relu_none_32_64_128",
    "dws_relu_none_32_64_128_256",
    "cnn_relu_none_16_32_64",
    "cnn_relu_none_16_32_64_128",
    "cnn_relu_none_64_128_256",
]


def publication_core_names() -> list[str]:
    """Return claimed models plus activation-matched controls (26 total)."""
    matched_groups = [
        ("mlp", "none", "128x64x32"),
        ("mlp", "layernorm", "128x64x32"),
        ("dws", "none", "16_32_64"),
        ("dws", "none", "32_64_128_256"),
        ("cnn", "none", "16_32_64"),
        ("cnn", "none", "16_32_64_128"),
    ]
    names = list(PAPER_CLAIMS)
    for family, norm, shape in matched_groups:
        for activation in ACTIVATIONS:
            name = f"{family}_{activation}_{norm}_{shape}"
            if name not in names:
                names.append(name)
    return names


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataset_transforms(dataset: str) -> tuple[T.Compose, T.Compose]:
    """Match the transforms used by ``generate_sweep.py`` exactly."""
    if dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return train_transform, test_transform


def load_datasets(dataset: str, data_dir: Path) -> tuple[Any, Any]:
    train_transform, test_transform = dataset_transforms(dataset)
    dataset_type = (
        torchvision.datasets.CIFAR10
        if dataset == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    mirror_env = "CIFAR10_MIRROR_URL" if dataset == "cifar10" else "CIFAR100_MIRROR_URL"
    mirror_url = os.environ.get(mirror_env)
    if mirror_url:
        # torchvision still verifies the archive against the canonical MD5,
        # so a mirror cannot silently change the dataset bytes.
        dataset_type.url = mirror_url
        log.info("Using MD5-verified %s mirror: %s", dataset, mirror_url)
    train_dataset = dataset_type(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    test_dataset = dataset_type(
        root=str(data_dir), train=False, download=True, transform=test_transform
    )
    return train_dataset, test_dataset


def make_loaders(
    train_dataset: Any,
    test_dataset: Any,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn if num_workers else None,
    }
    if num_workers:
        # Keep workers alive across epochs. Re-spawning them on every CIFAR
        # epoch is a material fraction of runtime for these compact models.
        common.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(256, batch_size),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader


def resolve_configs(dataset: str, suite: str, explicit_names: list[str]) -> list[dict]:
    num_classes = 10 if dataset == "cifar10" else 100
    all_configs = make_sweep_configs(num_classes=num_classes)
    by_name = {cfg["name"]: cfg for cfg in all_configs}

    if explicit_names:
        names = explicit_names
    elif suite == "smoke":
        names = ["dws_relu_none_16_32_64"]
    else:
        names = publication_core_names()

    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown architecture names: {missing}")

    configs = []
    for name in names:
        cfg = dict(by_name[name])
        cfg["dataset"] = dataset
        configs.append(cfg)
    return configs


def training_recipe(model: nn.Module) -> dict[str, Any]:
    module_names = {type(module).__name__ for module in model.modules()}
    is_poly = bool(module_names & {"Square", "CubicPlusLinear", "SquarePlusLinear"})
    has_norm = any(
        isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))
        for module in model.modules()
    )
    safe_polynomial_recipe = is_poly and not has_norm
    return {
        "optimizer": "SGD",
        "lr": 0.01 if safe_polynomial_recipe else 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "scheduler": "CosineAnnealingLR",
        "gradient_clip_norm": 1.0 if safe_polynomial_recipe else None,
        "safe_polynomial_recipe": safe_polynomial_recipe,
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits during evaluation")
        loss_sum += criterion(logits, targets).item()
        k = min(5, logits.shape[1])
        predictions = logits.topk(k, dim=1).indices
        correct1 += (predictions[:, 0] == targets).sum().item()
        correct5 += (predictions == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.shape[0]
    return {
        "loss": loss_sum / total,
        "top1": 100.0 * correct1 / total,
        "top5": 100.0 * correct5 / total,
    }


def checkpoint_epochs(epoch_budget: int) -> set[int]:
    requested = {1, 5, 10, 15, 20, 30, 50, 75, 100, epoch_budget}
    return {epoch for epoch in requested if epoch <= epoch_budget}


def train_run(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epoch_budget: int,
) -> dict[str, Any]:
    recipe = training_recipe(model)
    model.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=recipe["lr"],
        momentum=recipe["momentum"],
        weight_decay=recipe["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_budget)
    criterion = nn.CrossEntropyLoss()
    curve = []
    checkpoints = checkpoint_epochs(epoch_budget)
    started = time.perf_counter()
    divergence_reason = None
    epochs_completed = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epoch_budget + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                divergence_reason = f"non-finite training loss at epoch {epoch}"
                break
            loss.backward()
            if recipe["gradient_clip_norm"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip_norm"])
            optimizer.step()
            train_loss_sum += loss.item() * targets.shape[0]
            train_examples += targets.shape[0]

        if divergence_reason is not None:
            break

        scheduler.step()
        epochs_completed = epoch
        if epoch in checkpoints:
            metrics = evaluate(model, test_loader, device)
            point = {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_examples,
                "lr": optimizer.param_groups[0]["lr"],
                **metrics,
            }
            curve.append(point)
            log.info(
                "epoch=%d/%d train_loss=%.4f test_top1=%.2f test_top5=%.2f",
                epoch,
                epoch_budget,
                point["train_loss"],
                point["top1"],
                point["top5"],
            )

    runtime_seconds = time.perf_counter() - started
    final_metrics = curve[-1] if curve else None
    return {
        "status": "diverged" if divergence_reason else "complete",
        "divergence_reason": divergence_reason,
        "epochs_completed": epochs_completed,
        "runtime_seconds": runtime_seconds,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "recipe": recipe,
        "learning_curve": curve,
        "final": final_metrics,
        "best_top1": max((point["top1"] for point in curve), default=None),
    }


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    cuda_metadata: dict[str, Any] | None = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        cuda_metadata = {
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
            "total_memory_bytes": props.total_memory,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda": cuda_metadata,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class HubArtifact:
    """Incrementally mirror one result file to a private HF dataset repo."""

    def __init__(self, repo_id: str | None, path_in_repo: str, local_path: Path):
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo.strip("/")
        self.local_path = local_path
        self.api = None
        if repo_id:
            from huggingface_hub import HfApi

            self.api = HfApi()
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )

    def restore(self) -> bool:
        if not self.repo_id:
            return False
        from huggingface_hub import hf_hub_download

        try:
            cached = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=self.path_in_repo,
            )
        except Exception as exc:
            log.info("No prior Hub artifact to resume: %s", exc)
            return False
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_bytes(Path(cached).read_bytes())
        log.info("Restored prior results from %s/%s", self.repo_id, self.path_in_repo)
        return True

    def upload(self, message: str) -> None:
        if not self.repo_id or self.api is None:
            return
        self.api.upload_file(
            path_or_fileobj=str(self.local_path),
            path_in_repo=self.path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=message,
        )
        log.info("Uploaded artifact to hf://datasets/%s/%s", self.repo_id, self.path_in_repo)

    def upload_support_file(self, source: Path, target: str) -> None:
        if not self.repo_id or self.api is None or not source.exists():
            return
        self.api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=target.strip("/"),
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Record experiment code: {source.name}",
        )


def run_key(name: str, seed: int, epoch_budget: int) -> str:
    return f"{name}::seed={seed}::epochs={epoch_budget}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--suite", choices=["smoke", "publication_core"], default="publication_core")
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--epoch-budgets", nargs="+", type=int, default=[30, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/cifar-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gpu_validation"))
    parser.add_argument("--hf-repo", default=os.environ.get("HF_RESULTS_REPO"))
    parser.add_argument("--upload-prefix", default="cifar10-publication-core")
    parser.add_argument("--no-resume-from-hub", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()

    if any(budget <= 0 for budget in args.epoch_budgets):
        parser.error("all --epoch-budgets must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    configs = resolve_configs(args.dataset, args.suite, args.names)
    if args.list_configs:
        for cfg in configs:
            model = build_model(cfg)
            print(f"{cfg['name']}\t{sum(p.numel() for p in model.parameters())}")
        return 0

    result_path = args.output_dir / "results.json"
    path_in_repo = f"{args.upload_prefix.strip('/')}/results.json"
    hub = HubArtifact(args.hf_repo, path_in_repo, result_path)
    if not result_path.exists() and not args.no_resume_from_hub:
        hub.restore()

    if result_path.exists():
        results = json.loads(result_path.read_text())
    else:
        results = {
            "schema_version": 1,
            "experiment": "zk-circuit-aware-nas-publication-validation",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "source_git_sha": os.environ.get("SOURCE_GIT_SHA"),
            "hf_job_id": os.environ.get("HF_JOB_ID") or os.environ.get("JOB_ID"),
            "dataset": args.dataset,
            "suite": args.suite,
            "seeds": args.seeds,
            "epoch_budgets": args.epoch_budgets,
            "batch_size": args.batch_size,
            "config_names": [cfg["name"] for cfg in configs],
            "protocol": {
                "augmentation": "RandomCrop(32,padding=4), RandomHorizontalFlip",
                "dataset_archive_url": (
                    os.environ.get("CIFAR10_MIRROR_URL")
                    if args.dataset == "cifar10"
                    else os.environ.get("CIFAR100_MIRROR_URL")
                ),
                "dataset_archive_integrity": "torchvision canonical MD5",
                "optimizer": "SGD, momentum=0.9, weight_decay=5e-4",
                "schedule": "fresh CosineAnnealingLR schedule for each epoch budget",
                "polynomial_without_norm": "lr=0.01, gradient clip=1.0",
                "all_other_models": "lr=0.1, no gradient clipping",
                "deterministic_algorithms": True,
            },
            "hardware": hardware_metadata(device),
            "runs": [],
        }

    # Refresh execution-specific metadata on every resume. A replacement GPU
    # job may have different hardware or worker settings from the first attempt.
    current_hardware = hardware_metadata(device)
    runner_path = Path(__file__).resolve()
    attempt = {
        "started_at": utc_now(),
        "hf_job_id": os.environ.get("HF_JOB_ID") or os.environ.get("JOB_ID"),
        "hardware": current_hardware,
        "num_workers": args.num_workers,
        "runner_sha256": sha256_file(runner_path),
    }
    results["hardware"] = current_hardware
    results["runner_sha256"] = attempt["runner_sha256"]
    results.setdefault("execution_attempts", []).append(attempt)

    expected = len(configs) * len(args.seeds) * len(args.epoch_budgets)
    completed_keys = {
        run["run_key"]
        for run in results.get("runs", [])
        if run.get("status") in {"complete", "diverged", "error"}
    }
    log.info(
        "Starting %d expected runs (%d already recorded), device=%s, artifact=%s",
        expected,
        len(completed_keys),
        device,
        path_in_repo if args.hf_repo else result_path,
    )

    atomic_write_json(result_path, results)
    artifact_prefix = str(Path(path_in_repo).parent)
    hub.upload_support_file(runner_path, f"{artifact_prefix}/gpu_validation_experiment.py")
    hub.upload_support_file(Path("hf_gpu_entrypoint.sh"), f"{artifact_prefix}/hf_gpu_entrypoint.sh")
    hub.upload("Initialize GPU validation artifact")
    train_dataset, test_dataset = load_datasets(args.dataset, args.data_dir)

    # Budget-first ordering completes the direct 30-epoch replications before
    # spending time on 100-epoch extensions.
    for epoch_budget in args.epoch_budgets:
        for cfg in configs:
            for seed in args.seeds:
                key = run_key(cfg["name"], seed, epoch_budget)
                if key in completed_keys:
                    log.info("Skipping completed run %s", key)
                    continue

                log.info("RUN %s", key)
                record: dict[str, Any] = {
                    "run_key": key,
                    "name": cfg["name"],
                    "config": cfg,
                    "seed": seed,
                    "epoch_budget": epoch_budget,
                    "started_at": utc_now(),
                }
                try:
                    set_seed(seed)
                    model = build_model(cfg)
                    record["n_params"] = sum(p.numel() for p in model.parameters())

                    # NEAR is evaluated on deterministic synthetic data so that
                    # score differences isolate architecture and initialization.
                    proxy_generator = torch.Generator().manual_seed(seed)
                    proxy_data = torch.randn(16, 3, 32, 32, generator=proxy_generator)
                    try:
                        record["near_score"] = near_score(
                            model, data=proxy_data, device=str(device)
                        )
                    except Exception as exc:
                        record["near_score"] = None
                        record["near_error"] = f"{type(exc).__name__}: {exc}"
                        log.warning("NEAR failed for %s: %s", key, exc)

                    # near_score deliberately seeds its sub-sampling. Restore the
                    # run seed before constructing the shuffled DataLoader.
                    set_seed(seed)
                    train_loader, test_loader = make_loaders(
                        train_dataset,
                        test_dataset,
                        seed,
                        args.batch_size,
                        args.num_workers,
                    )
                    record.update(
                        train_run(
                            model,
                            train_loader,
                            test_loader,
                            device,
                            epoch_budget,
                        )
                    )
                except Exception as exc:
                    record.update(
                        {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    log.exception("Run failed: %s", key)
                finally:
                    record["finished_at"] = utc_now()
                    results.setdefault("runs", []).append(record)
                    results["updated_at"] = utc_now()
                    results["completed_runs"] = len(results["runs"])
                    results["expected_runs"] = expected
                    atomic_write_json(result_path, results)
                    hub.upload(f"Record {key}")
                    completed_keys.add(key)
                    if "model" in locals():
                        del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    statuses: dict[str, int] = {}
    for run in results["runs"]:
        status = run.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    results["finished_at"] = utc_now()
    results["status_counts"] = statuses
    atomic_write_json(result_path, results)
    hub.upload("Complete GPU validation experiment")
    log.info("Experiment complete: %s", statuses)
    return 0 if statuses.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
