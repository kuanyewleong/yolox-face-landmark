import torch
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, act=True):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize, stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False):
        super().__init__()
        hidden = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = Conv(hidden, out_channels, 3, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden, 1, 1)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut, 1.0, depthwise) for _ in range(n)])

    def forward(self, x):
        x1 = self.m(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat([x1, x2], dim=1))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, 1)

    def forward(self, x):
        patch_tl = x[..., ::2, ::2]
        patch_tr = x[..., ::2, 1::2]
        patch_bl = x[..., 1::2, ::2]
        patch_br = x[..., 1::2, 1::2]
        x = torch.cat([patch_tl, patch_bl, patch_tr, patch_br], dim=1)
        return self.conv(x)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=(5, 9, 13)):
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernels])
        self.conv2 = BaseConv(hidden * (len(kernels) + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        return self.conv2(x)