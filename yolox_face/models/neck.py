import torch
import torch.nn as nn
from .backbone import CSPDarknet
from .blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=(256, 512, 1024), depthwise=False):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise)
        Conv = DWConv if depthwise else BaseConv
        c3, c4, c5 = [int(c * width) for c in in_features]
        base_depth = max(round(depth * 3), 1)
        self.lateral_conv0 = BaseConv(c5, c4, 1, 1)
        self.C3_p4 = CSPLayer(2 * c4, c4, n=base_depth, shortcut=False, depthwise=depthwise)
        self.reduce_conv1 = BaseConv(c4, c3, 1, 1)
        self.C3_p3 = CSPLayer(2 * c3, c3, n=base_depth, shortcut=False, depthwise=depthwise)
        self.bu_conv2 = Conv(c3, c3, 3, 2)
        self.C3_n3 = CSPLayer(2 * c3, c4, n=base_depth, shortcut=False, depthwise=depthwise)
        self.bu_conv1 = Conv(c4, c4, 3, 2)
        self.C3_n4 = CSPLayer(2 * c4, c5, n=base_depth, shortcut=False, depthwise=depthwise)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        feat3, feat4, feat5 = self.backbone(x)
        fpn_out0 = self.lateral_conv0(feat5)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, feat4], dim=1)
        f_out0 = self.C3_p4(f_out0)
        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, feat3], dim=1)
        pan_out2 = self.C3_p3(f_out1)
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)
        pan_out1 = self.C3_n3(p_out1)
        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)
        pan_out0 = self.C3_n4(p_out0)
        return pan_out2, pan_out1, pan_out0