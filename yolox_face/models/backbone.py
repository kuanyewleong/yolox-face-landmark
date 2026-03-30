import torch.nn as nn
from .blocks import BaseConv, CSPLayer, DWConv, Focus, SPPBottleneck


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul=1.0, wid_mul=1.0, depthwise=False):
        super().__init__()
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        Conv = DWConv if depthwise else BaseConv

        self.stem = Focus(3, base_channels, 3)
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise),
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise),
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise),
        )
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPPBottleneck(base_channels * 16, base_channels * 16),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        feat3 = self.dark3(x)
        feat4 = self.dark4(feat3)
        feat5 = self.dark5(feat4)
        return feat3, feat4, feat5