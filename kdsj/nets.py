"""识别网络（PyTorch）。三种结构都有 64 维的 fc1 特征层，供 LSSVM 使用。

- cnn        : 3 层卷积（通道 16/32/64），原始方案
- resnet     : 3 个残差块（通道 16/32/64），默认方案
- eca_resnet : resnet + ECA 通道注意力（Wang Q. et al., CVPR 2020，本项目自行实现）

输入：N x 3 x 64 x 64（jet 颜色映射后的时频图，已按训练集均值/标准差标准化）。
"""
import torch
import torch.nn as nn


class ECA(nn.Module):
    """全局平均池化 -> 跨通道 1D 卷积（核长 3）-> sigmoid -> 按通道加权。"""

    def __init__(self, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        w = torch.sigmoid(self.conv(x.mean((2, 3)).unsqueeze(1))).transpose(1, 2)
        return x * w.unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride, eca=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
            ECA() if eca else nn.Identity())
        self.short = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.short(x))


class Net(nn.Module):
    def __init__(self, backbone, n_classes, feat_in=64):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_in, 64))
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(64, n_classes))

    def features(self, x):
        return self.fc1(self.backbone(x))

    def forward(self, x):
        return self.head(self.features(x))


def _conv(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]


def build(name, n_classes):
    name = name.lower()
    if name == 'cnn':
        bb = nn.Sequential(*_conv(3, 16), nn.MaxPool2d(2), *_conv(16, 32), nn.MaxPool2d(2), *_conv(32, 64),
                           nn.AdaptiveAvgPool2d(1), nn.Flatten())
    elif name in ('resnet', 'eca_resnet'):
        e = name == 'eca_resnet'
        bb = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                           ResBlock(16, 16, 1, e), ResBlock(16, 32, 2, e), ResBlock(32, 64, 2, e),
                           nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        raise ValueError(f'未知网络：{name}（可选 cnn / resnet / eca_resnet）')
    return Net(bb, n_classes)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
