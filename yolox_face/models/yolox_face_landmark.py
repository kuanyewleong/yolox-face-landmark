import torch.nn as nn
from .head import YOLOXFaceLandmarkHead
from .neck import YOLOPAFPN


class YOLOXFaceLandmark(nn.Module):
    def __init__(self, depth=1.0, width=1.0, num_classes=1, depthwise=False):
        super().__init__()
        self.backbone = YOLOPAFPN(depth=depth, width=width, depthwise=depthwise)
        self.head = YOLOXFaceLandmarkHead(num_classes=num_classes, width=width, depthwise=depthwise)
        self.head.initialize_biases(1e-2)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)