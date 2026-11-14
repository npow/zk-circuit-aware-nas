"""ImageNet-16-120 dataset loader (NASBench-201 format)."""
import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageNet16(Dataset):
    """ImageNet-16-120: 16×16 images, 120 classes (NASBench-201 format)."""

    TRAIN_BATCHES = [f"train_data_batch_{i}" for i in range(1, 11)]

    def __init__(self, root: str, train: bool, transform=None, num_classes: int = 120):
        self.transform = transform
        self.num_classes = num_classes
        data, labels = [], []

        if train:
            for fname in self.TRAIN_BATCHES:
                path = os.path.join(root, fname)
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    batch = pickle.load(f, encoding="bytes")
                raw = batch["data"]
                lbs = batch["labels"]
                # Filter to first num_classes classes
                mask = [i for i, l in enumerate(lbs) if 1 <= l <= num_classes]
                data.append(raw[mask])
                labels.extend([lbs[i] - 1 for i in mask])  # 0-indexed
        else:
            path = os.path.join(root, "val_data")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    batch = pickle.load(f, encoding="bytes")
                raw = batch["data"]
                lbs = batch["labels"]
                mask = [i for i, l in enumerate(lbs) if 1 <= l <= num_classes]
                data.append(raw[mask])
                labels.extend([lbs[i] - 1 for i in mask])

        if data:
            self.data = np.vstack(data)  # (N, 3*16*16)
            self.targets = labels
        else:
            self.data = np.zeros((0, 3*16*16), dtype=np.uint8)
            self.targets = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 16, 16).transpose(1, 2, 0)  # HWC
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Normalize with ImageNet-ish mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]
