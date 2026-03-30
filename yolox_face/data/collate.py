import numpy as np
import torch


def collate_face(batch):
    images = torch.from_numpy(np.stack([b["image"] for b in batch], axis=0)).float()
    boxes = [torch.from_numpy(b["boxes"]).float() for b in batch]
    labels = [torch.from_numpy(b["labels"]).long() for b in batch]
    landmarks = [torch.from_numpy(b["landmarks"]).float() for b in batch]
    has_det = torch.tensor([bool(b["has_det"]) for b in batch], dtype=torch.bool)
    has_lmk = torch.tensor([bool(b["has_lmk"]) for b in batch], dtype=torch.bool)
    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "landmarks": landmarks,
        "has_det": has_det,
        "has_lmk": has_lmk,
        "image_paths": [b["image_path"] for b in batch],
    }