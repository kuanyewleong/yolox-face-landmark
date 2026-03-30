import cv2

class FaceTrainTransform:
    def __init__(self, input_size=(640, 640), hsv_prob=1.0, flip_prob=0.5):
        self.input_size = input_size
        self.hsv_prob = hsv_prob
        self.flip_prob = flip_prob

    @staticmethod
    def _augment_hsv(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        r = np.random.uniform(-1, 1, 3) * np.array([5, 30, 30], dtype=np.float32)
        hsv[..., 0] = np.clip(hsv[..., 0] + r[0], 0, 179)
        hsv[..., 1] = np.clip(hsv[..., 1] + r[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + r[2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def __call__(self, sample):
        img = sample["image"]
        boxes = sample["boxes"]
        labels = sample["labels"]
        landmarks = sample["landmarks"]

        h0, w0 = img.shape[:2]
        th, tw = self.input_size
        scale = min(tw / w0, th / h0)
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        dx = (tw - nw) // 2
        dy = (th - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        img = canvas

        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dy

        if landmarks.size > 0:
            landmarks = landmarks.copy()
            valid = landmarks[..., 0] >= 0
            landmarks[..., 0][valid] = landmarks[..., 0][valid] * scale + dx
            valid = landmarks[..., 1] >= 0
            landmarks[..., 1][valid] = landmarks[..., 1][valid] * scale + dy

        if random.random() < self.hsv_prob:
            img = self._augment_hsv(img)

        if random.random() < self.flip_prob:
            img = img[:, ::-1, :].copy()
            if len(boxes) > 0:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = tw - x2
                boxes[:, 2] = tw - x1
            if landmarks.size > 0:
                valid = landmarks[..., 0] >= 0
                landmarks[..., 0][valid] = tw - landmarks[..., 0][valid]
                landmarks = landmarks[:, [1, 0, 2, 4, 3], :]

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return {
            "image": img,
            "boxes": boxes.astype(np.float32),
            "labels": labels.astype(np.int64),
            "landmarks": landmarks.astype(np.float32),
            "has_det": sample["has_det"],
            "has_lmk": sample["has_lmk"],
            "image_path": sample["image_path"],
        }


class FaceValTransform(FaceTrainTransform):
    def __init__(self, input_size=(640, 640)):
        super().__init__(input_size=input_size, hsv_prob=0.0, flip_prob=0.0)