from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
from .wider_face import FaceSample


class WFLWDataset(Dataset):
    def __init__(self, root: str, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        if split == "train":
            ann_file = self.root / "WFLW_annotations" / "list_98pt_rect_attr_train_test" / "list_98pt_rect_attr_train.txt"
        else:
            ann_file = self.root / "WFLW_annotations" / "list_98pt_rect_attr_train_test" / "list_98pt_rect_attr_test.txt"
        self.samples = self._parse_wflw(ann_file)

    @staticmethod
    def _wflw98_to_5pts(pts98: np.ndarray) -> np.ndarray:
        left_eye = pts98[60:68].mean(axis=0)
        right_eye = pts98[68:76].mean(axis=0)
        nose = pts98[54]
        mouth_left = pts98[76]
        mouth_right = pts98[82]
        return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0).astype(np.float32)

    def _parse_wflw(self, ann_file: Path):
        samples = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                coords = np.asarray(parts[:196], dtype=np.float32).reshape(98, 2)
                x1, y1, x2, y2 = map(float, parts[196:200])
                rel_path = parts[-1]
                lm5 = self._wflw98_to_5pts(coords)
                samples.append(FaceSample(
                    image_path=str(self.root / rel_path),
                    boxes=np.asarray([[x1, y1, x2, y2]], dtype=np.float32),
                    labels=np.asarray([0], dtype=np.int64),
                    landmarks=lm5[None, ...],
                    has_det=True,
                    has_lmk=True,
                ))
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