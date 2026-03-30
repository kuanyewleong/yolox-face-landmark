from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset


@dataclass
class FaceSample:
    image_path: str
    boxes: np.ndarray
    labels: np.ndarray
    landmarks: np.ndarray
    has_det: bool
    has_lmk: bool


class WiderFaceDataset(Dataset):
    def __init__(self, root: str, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        ann_file = self.root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        img_dir = self.root / f"WIDER_{split}" / "images"
        self.samples = self._parse_wider(ann_file, img_dir)

    @staticmethod
    def _parse_wider(ann_file: Path, img_dir: Path):
        samples = []
        with open(ann_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        i = 0
        while i < len(lines):
            rel_path = lines[i]
            i += 1
            num_boxes = int(lines[i])
            i += 1
            boxes = []
            for _ in range(num_boxes):
                vals = [int(v) for v in lines[i].split()]
                i += 1
                x, y, w, h = vals[:4]
                if w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
            boxes_np = np.asarray(boxes, dtype=np.float32) if len(boxes) else np.zeros((0, 4), dtype=np.float32)
            labels_np = np.zeros((len(boxes_np),), dtype=np.int64)
            landmarks_np = np.full((len(boxes_np), 5, 2), -1.0, dtype=np.float32)
            samples.append(FaceSample(str(img_dir / rel_path), boxes_np, labels_np, landmarks_np, True, False))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = {
            "image": img,
            "boxes": sample.boxes.copy(),
            "labels": sample.labels.copy(),
            "landmarks": sample.landmarks.copy(),
            "has_det": sample.has_det,
            "has_lmk": sample.has_lmk,
            "image_path": sample.image_path,
        }
        return self.transform(out) if self.transform is not None else out