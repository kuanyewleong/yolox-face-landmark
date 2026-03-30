import math
import torch.nn as nn
from .blocks import BaseConv, DWConv


class YOLOXFaceLandmarkHead(nn.Module):
    def __init__(self, num_classes=1, width=1.0, in_channels=(256, 512, 1024), depthwise=False):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.n_anchors = 1
        self.num_classes = num_classes
        self.strides = [8, 16, 32]
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.lmk_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.lmk_preds = nn.ModuleList()

        for c in in_channels:
            ch = int(c * width)
            mid = int(256 * width)
            self.stems.append(BaseConv(ch, mid, 1, 1))
            self.cls_convs.append(nn.Sequential(Conv(mid, mid, 3, 1), Conv(mid, mid, 3, 1)))
            self.reg_convs.append(nn.Sequential(Conv(mid, mid, 3, 1), Conv(mid, mid, 3, 1)))
            self.lmk_convs.append(nn.Sequential(Conv(mid, mid, 3, 1), Conv(mid, mid, 3, 1)))
            self.cls_preds.append(nn.Conv2d(mid, self.n_anchors * num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(mid, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(mid, 1, 1, 1, 0))
            self.lmk_preds.append(nn.Conv2d(mid, 10, 1, 1, 0))

    def initialize_biases(self, prior_prob=1e-2):
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(bias_value)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(bias_value)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feats):
        outputs = []
        for k, x in enumerate(feats):
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            reg_feat = self.reg_convs[k](x)
            lmk_feat = self.lmk_convs[k](x)
            outputs.append({
                "cls": self.cls_preds[k](cls_feat),
                "reg": self.reg_preds[k](reg_feat),
                "obj": self.obj_preds[k](reg_feat),
                "lmk": self.lmk_preds[k](lmk_feat),
                "stride": self.strides[k],
            })
        return outputs