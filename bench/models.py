"""识别网络（PyTorch）。三种结构共用同一个输出接口：forward 返回 logits，features 返回 fc1 的 64 维特征。

- cnn        : v2 的基线 CNN（与 run_step3_cnn_lssvm.m 中 layers 相同）
- cnn_wide   : 加宽的基线 CNN（通道 32/64/96，参数量与 resnet 相当），对照组
- resnet     : 小型残差网络（3 个 BasicBlock，通道 16/32/64），用于消融
- eca_resnet : resnet + ECA 通道注意力（Wang et al., CVPR 2020），本次升级的主模型

三者的空间分辨率路径相同（64 -> 32 -> 16 -> 全局平均池化），fc1 都是 64 维，
因此 LSSVM 阶段完全不用改。MATLAB 版见 src/build_net.m 与 src/ecaLayer.m。
"""
import torch
import torch.nn as nn


class ECA(nn.Module):
    """ECA 通道注意力：全局平均池化 -> 跨通道 1D 卷积(k=3) -> sigmoid -> 按通道加权。
    通道数 16/32/64 时论文公式给出的 k 都是 3，只多 3 个参数。"""

    def __init__(self, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        w = x.mean((2, 3)).unsqueeze(1)                 # B x 1 x C
        w = torch.sigmoid(self.conv(w)).transpose(1, 2)  # B x C x 1
        return x * w.unsqueeze(-1)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, eca):
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


def _blk(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]


def build(name, n_classes):
    if name == 'cnn':
        bb = nn.Sequential(*_blk(3, 16), nn.MaxPool2d(2), *_blk(16, 32), nn.MaxPool2d(2), *_blk(32, 64),
                           nn.AdaptiveAvgPool2d(1), nn.Flatten())
    elif name == 'cnn_wide':
        # 对照组：把基线 CNN 加宽到和 ResNet 差不多的参数量（约 8.2 万），
        # 用来区分"残差结构的作用"和"参数变多的作用"
        bb = nn.Sequential(*_blk(3, 32), nn.MaxPool2d(2), *_blk(32, 64), nn.MaxPool2d(2), *_blk(64, 96),
                           nn.AdaptiveAvgPool2d(1), nn.Flatten())
        return Net(bb, n_classes, feat_in=96)
    elif name in ('resnet', 'eca_resnet'):
        eca = name == 'eca_resnet'
        bb = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            BasicBlock(16, 16, 1, eca), BasicBlock(16, 32, 2, eca), BasicBlock(32, 64, 2, eca),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        raise ValueError(f'未知模型 {name}')
    return Net(bb, n_classes)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
