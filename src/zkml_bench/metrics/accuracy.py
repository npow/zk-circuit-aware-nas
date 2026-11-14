import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass
class AccuracyResult:
    top1: float  # percentage 0-100
    top5: float  # percentage 0-100


def compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: tuple[int, ...] = (1, 5),
    tta: bool = False,
) -> AccuracyResult:
    """Compute top-1 and top-5 accuracy.

    If tta=True, averages logits over the original image and its horizontal flip
    (test-time augmentation), giving a free ~0.3-0.5pp improvement.
    """
    model.eval()
    correct = {k: 0 for k in top_k}
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if tta:
                outputs = (model(inputs) + model(inputs.flip(-1))) / 2
            else:
                outputs = model(inputs)
            batch_size = targets.size(0)
            total += batch_size
            maxk = max(top_k)
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()  # (maxk, B)
            correct_mask = pred.eq(targets.view(1, -1).expand_as(pred))
            for k in top_k:
                correct[k] += correct_mask[:k].any(dim=0).sum().item()
    return AccuracyResult(
        top1=100.0 * correct[1] / total,
        top5=100.0 * correct[5] / total if 5 in correct else 0.0,
    )
