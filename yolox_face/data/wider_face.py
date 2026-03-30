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
    def _looks_like_image_path(s: str) -> bool:
        s = s.strip().lower()
        return s.endswith(".jpg") or s.endswith(".jpeg") or s.endswith(".png")

    @staticmethod
    def _is_int_line(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        try:
            int(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def _parse_bbox_line(s: str):
        """
        WIDER bbox lines normally contain 10 integers:
        x y w h blur expression illumination invalid occlusion pose
        """
        parts = s.strip().split()
        if len(parts) < 4:
            return None

        try:
            vals = [int(v) for v in parts]
        except ValueError:
            return None

        if len(vals) < 4:
            return None

        x, y, w, h = vals[:4]
        return x, y, w, h, vals

    @staticmethod
    def _parse_wider(ann_file: Path, img_dir: Path):
        samples = []

        with open(ann_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        n = len(lines)
        num_recoveries = 0

        while i < n:
            # Resync until we find an image path
            if not WiderFaceDataset._looks_like_image_path(lines[i]):
                # Skip stray bbox/malformed lines until next image path
                num_recoveries += 1
                i += 1
                continue

            rel_path = lines[i]
            i += 1

            # Count line handling
            if i >= n:
                num_boxes = 0
            elif WiderFaceDataset._looks_like_image_path(lines[i]):
                # Missing count line, assume zero boxes
                num_boxes = 0
                num_recoveries += 1
            elif WiderFaceDataset._is_int_line(lines[i]):
                num_boxes = int(lines[i])
                i += 1
            else:
                # If the next line is not an int, but looks like a bbox row, assume malformed count
                maybe_bbox = WiderFaceDataset._parse_bbox_line(lines[i])
                if maybe_bbox is not None:
                    # Treat as one bbox and continue consuming bbox-like rows
                    num_boxes = 1
                    num_recoveries += 1
                else:
                    # Unknown garbage: skip this sample safely
                    num_recoveries += 1
                    continue

            boxes = []
            boxes_read = 0

            while i < n and boxes_read < num_boxes:
                # If we unexpectedly encounter the next image path early, stop this record
                if WiderFaceDataset._looks_like_image_path(lines[i]):
                    num_recoveries += 1
                    break

                parsed = WiderFaceDataset._parse_bbox_line(lines[i])
                if parsed is None:
                    # Skip malformed row and continue trying to consume remaining bbox lines
                    num_recoveries += 1
                    i += 1
                    continue

                x, y, w, h, _ = parsed
                i += 1
                boxes_read += 1

                if w <= 1 or h <= 1:
                    continue

                boxes.append([x, y, x + w, y + h])

            # Extra safeguard:
            # if bbox count was malformed and there are stray bbox rows before the next image path,
            # consume them so the parser fully resynchronizes.
            while i < n and not WiderFaceDataset._looks_like_image_path(lines[i]):
                parsed = WiderFaceDataset._parse_bbox_line(lines[i])
                if parsed is None:
                    break
                # This is a stray bbox row not accounted for by count, skip it.
                num_recoveries += 1
                i += 1

            boxes_np = (
                np.asarray(boxes, dtype=np.float32)
                if len(boxes)
                else np.zeros((0, 4), dtype=np.float32)
            )
            labels_np = np.zeros((len(boxes_np),), dtype=np.int64)
            landmarks_np = np.full((len(boxes_np), 5, 2), -1.0, dtype=np.float32)

            samples.append(
                FaceSample(
                    image_path=str(img_dir / rel_path),
                    boxes=boxes_np,
                    labels=labels_np,
                    landmarks=landmarks_np,
                    has_det=True,
                    has_lmk=False,
                )
            )

        print(f"[WiderFaceDataset] Parsed {len(samples)} samples with {num_recoveries} recovery events.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample.image_path)
        if img is None:
            raise FileNotFoundError(sample.image_path)
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