"""只用正常声音训练的异常检测（每台设备单独建模）。

现场通常录不到故障声音，但正常运行的声音很容易录到。这里的检测器只用正常样本训练，
输出"异常分数"（越大越可能异常），再用一批没参与训练的正常样本定告警阈值。

检测器（输入都是 64x64 uint8 时频图，颜色索引 0..255）：
  GaussianDetector  每个频带在时间上的均值和标准差（128 维）-> 多元高斯，分数 = 马氏距离
  AEDetector        自编码器重构连续 5 帧的频谱（与 MIMII 官方基线思路相同），分数 = 平均重构误差
  IDClassifierDetector
                    同一类型的多台设备的正常声音一起训练 ResNet 去区分"这是哪一台"，
                    分数 = -log p(真实编号)。设备出故障后声音特征变了，网络就不确定它是哪一台。
                    （自监督方法，参考 DCASE 2020 Task 2 中的设备编号分类方法；不需要任何异常样本）
                    同时提供 'idmaha'：在该网络的 fc1 特征上对每台设备做马氏距离。

阈值：calibrate(正常样本, q) 取正常样本分数的 q 分位数，例如 q=0.95 时目标误报率约 5%。
"""
import copy
import numpy as np
import torch
import torch.nn as nn


def _values(X):
    return np.asarray(X, np.float32) / 255.0          # 颜色索引 -> [0,1]（对数幅值）


def band_stats(X):
    """N x 64 x 64（频带 x 时间）-> N x 128：每个频带在时间上的均值和标准差。"""
    V = _values(X)
    return np.concatenate([V.mean(2), V.std(2)], 1)


class _Maha:
    def fit(self, F, shrink=0.05):
        F = np.asarray(F, np.float64)
        self.mu = F.mean(0)
        C = np.cov(F - self.mu, rowvar=False)
        d = C.shape[0]
        C = (1 - shrink) * C + shrink * np.trace(C) / d * np.eye(d)   # 收缩估计，样本少时也稳定
        self.P = np.linalg.inv(C)
        return self

    def score(self, F):
        D = np.asarray(F, np.float64) - self.mu
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', D, self.P, D), 0))


class _Base:
    threshold = None

    def calibrate(self, X_normal, q=0.95):
        self.threshold = float(np.quantile(self.score(X_normal), q))
        return self.threshold

    def predict(self, X):
        """返回 (是否异常, 分数)。"""
        s = self.score(X)
        return s > self.threshold, s


class GaussianDetector(_Base):
    kind = 'gauss'

    def fit(self, X, **_):
        self.m = _Maha().fit(band_stats(X))
        return self

    def score(self, X):
        return self.m.score(band_stats(X))

    def state(self):
        return {'kind': self.kind, 'mu': self.m.mu, 'P': self.m.P, 'threshold': self.threshold}


class _AE(nn.Module):
    def __init__(self, d, h=128, z=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU(),
                                 nn.Linear(z, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.net(x)


def _context(X, ctx):
    """N x 64 x 64 -> (N x (64-ctx+1)) x (ctx*64)：连续 ctx 帧拼成一个向量。"""
    V = _values(X).transpose(0, 2, 1)                # N x 时间 x 频带
    T = V.shape[1] - ctx + 1
    W = np.stack([V[:, t:t + ctx].reshape(len(V), -1) for t in range(T)], 1)
    return W                                          # N x T x (ctx*64)


class AEDetector(_Base):
    kind = 'ae'

    def __init__(self, ctx=5, epochs=50, batch=512, lr=1e-3, device='cpu'):
        self.ctx, self.epochs, self.batch, self.lr, self.device = ctx, epochs, batch, lr, torch.device(device)

    def fit(self, X, seed=0, **_):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        W = _context(X, self.ctx).reshape(-1, self.ctx * 64)
        self.mu, self.sd = W.mean(0), W.std(0) + 1e-6
        Z = torch.from_numpy((W - self.mu) / self.sd).to(self.device)
        self.net = _AE(Z.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), self.lr)
        for _ in range(self.epochs):
            perm = torch.from_numpy(rng.permutation(len(Z))).to(self.device)
            for k in range(0, len(Z), self.batch):
                b = Z[perm[k:k + self.batch]]
                opt.zero_grad()
                ((self.net(b) - b) ** 2).mean().backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, X):
        W = _context(X, self.ctx)
        n, T, d = W.shape
        Z = torch.from_numpy(((W.reshape(-1, d) - self.mu) / self.sd).astype(np.float32)).to(self.device)
        err = torch.cat([((self.net(Z[k:k + 8192]) - Z[k:k + 8192]) ** 2).mean(1) for k in range(0, len(Z), 8192)])
        return err.cpu().numpy().reshape(n, T).mean(1)

    def state(self):
        return {'kind': self.kind, 'ctx': self.ctx, 'mu': self.mu, 'sd': self.sd,
                'net': self.net.state_dict(), 'threshold': self.threshold}


class IDClassifierDetector:
    """同一类型多台设备共用一个网络。fit 时传入所有设备的正常样本和设备编号。

    score(X, ids, method='idclf' | 'idmaha')；calibrate / predict 按设备编号分别定阈值。
    """
    kind = 'idclf'

    def __init__(self, model='resnet', epochs=40, patience=6, batch=64, lr=1e-3, device='cpu'):
        self.model, self.epochs, self.patience, self.batch, self.lr = model, epochs, patience, batch, lr
        self.device = torch.device(device)
        self.thresholds = {}

    def _t(self, X):
        from .tfr import JET
        return ((torch.from_numpy(JET[np.asarray(X)]).permute(0, 3, 1, 2) - self.m) / self.s)

    def fit(self, X_tr, id_tr, X_va, id_va, seed=0):
        from .nets import build
        from .tfr import JET
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.ids = sorted(set(np.asarray(id_tr).tolist()))
        lab = {v: i for i, v in enumerate(self.ids)}
        T = torch.from_numpy(JET[np.asarray(X_tr)]).permute(0, 3, 1, 2)
        self.m, self.s = T.mean((0, 2, 3), keepdim=True), T.std((0, 2, 3), keepdim=True) + 1e-6
        Xt = ((T - self.m) / self.s).to(self.device)
        Xv = self._t(X_va).to(self.device)
        yt = torch.tensor([lab[v] for v in id_tr], device=self.device)
        yv = torch.tensor([lab[v] for v in id_va], device=self.device)
        net = build(self.model, len(self.ids)).to(self.device)
        opt = torch.optim.Adam(net.parameters(), self.lr)
        lossf = nn.CrossEntropyLoss()
        best, bad, state = np.inf, 0, None
        for ep in range(self.epochs):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(yt))).to(self.device)
            for k in range(0, len(perm), self.batch):
                b = perm[k:k + self.batch]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                lossf(net(Xt[b]), yt[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = np.mean([lossf(net(Xv[k:k + 512]), yv[k:k + 512]).item() for k in range(0, len(yv), 512)])
            if vl < best:
                best, bad, state = vl, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(state)
        self.net, self.epochs_run = net.eval(), ep + 1
        F, _ = self._forward(X_tr)                         # 每台设备的正常特征 -> 马氏距离模型
        id_tr = np.asarray(id_tr)
        self.maha = {v: _Maha().fit(F[id_tr == v]) for v in self.ids}
        return self

    @torch.no_grad()
    def _forward(self, X):
        F, P = [], []
        for k in range(0, len(X), 256):
            f = self.net.features(self._t(X[k:k + 256]).to(self.device))
            F.append(f.cpu()); P.append(torch.log_softmax(self.net.head(f), 1).cpu())
        return torch.cat(F).numpy(), torch.cat(P).numpy()

    def score(self, X, ids, method='idclf'):
        ids = np.asarray(ids)
        F, LP = self._forward(X)
        if method == 'idclf':
            return -LP[np.arange(len(ids)), [self.ids.index(v) for v in ids]]
        s = np.zeros(len(ids))
        for v in np.unique(ids):
            s[ids == v] = self.maha[v].score(F[ids == v])
        return s


# ---------------- 评价指标 ----------------
def auc(score_normal, score_abnormal):
    """ROC 曲线下面积（Mann-Whitney 统计量，并列算一半）。"""
    s = np.r_[score_normal, score_abnormal]
    order = s.argsort(kind='mergesort')
    ranks = np.empty(s.size)
    sv = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n0, n1 = len(score_normal), len(score_abnormal)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def pauc(score_normal, score_abnormal, max_fpr=0.1):
    """误报率 ≤ max_fpr 部分的 ROC 面积，归一化到 [0,1]（DCASE 的 pAUC 定义）。"""
    thr = np.sort(score_normal)[::-1]
    k = max(1, int(np.floor(max_fpr * len(thr))))
    tpr = [np.mean(score_abnormal > t) for t in thr[:k]]          # 误报率依次为 0, 1/n, ..., (k-1)/n
    return float(np.mean(tpr))


# ---------------- 部署用：单台设备的异常检测模型 ----------------
class AnomalyModel:
    """一台设备的异常检测模型：检测器 + 时频图设置 + 告警阈值。可保存、加载，供命令行和界面使用。

    method：'gauss'（默认，不需要 GPU，实验中 AUC 最高）| 'ae'
    feature：{'kind': 'mel' | 'set', 'win_s': 窗长（秒）, 'fs': 采样率}
    """
    is_anomaly = True

    def __init__(self, method='gauss', feature=None, q=0.95, device='cpu'):
        self.method, self.q, self.device = method, q, device
        self.feature = dict(feature or {'kind': 'mel', 'win_s': 10.0, 'fs': 16000})
        self.det, self.threshold = None, None

    def fit(self, X_train, X_cal, seed=0):
        """X_train：用于建模的正常时频图；X_cal：另一批正常时频图，用来定阈值（不能与 X_train 重复）。"""
        self.det = GaussianDetector() if self.method == 'gauss' else AEDetector(device=self.device)
        self.det.fit(X_train, seed=seed)
        self.threshold = self.det.calibrate(X_cal, self.q)
        self.n_train, self.n_cal = len(X_train), len(X_cal)
        return self

    def score(self, X):
        return self.det.score(X)

    def save(self, path):
        torch.save({'type': 'anomaly', 'method': self.method, 'q': self.q, 'feature': self.feature,
                    'threshold': self.threshold, 'det': self.det.state(),
                    'n_train': self.n_train, 'n_cal': self.n_cal}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        d = torch.load(path, map_location='cpu', weights_only=False)
        m = cls(d['method'], d['feature'], d['q'], device)
        s = d['det']
        if d['method'] == 'gauss':
            m.det = GaussianDetector()
            m.det.m = _Maha()
            m.det.m.mu, m.det.m.P = s['mu'], s['P']
        else:
            m.det = AEDetector(ctx=s['ctx'], device=device)
            m.det.mu, m.det.sd = s['mu'], s['sd']
            m.det.net = _AE(s['ctx'] * 64).to(m.det.device)
            m.det.net.load_state_dict(s['net'])
            m.det.net.eval()
        m.threshold = m.det.threshold = d['threshold']
        m.n_train, m.n_cal = d.get('n_train', 0), d.get('n_cal', 0)
        return m


def load_model(path, device=None):
    """自动识别模型文件类型：异常检测模型返回 AnomalyModel，分类模型返回 Recognizer。"""
    d = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(d, dict) and d.get('type') == 'anomaly':
        return AnomalyModel.load(path, device or 'cpu')
    from .recognizer import Recognizer
    return Recognizer.load(path, device)
