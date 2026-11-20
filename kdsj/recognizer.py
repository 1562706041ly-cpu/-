"""识别器：时频图 -> 网络（fc1 特征）-> BKA-LSSVM 分类。可训练、保存、加载、预测。

训练流程（测试集完全不参与）：
  1. 用训练集训练网络，验证集损失连续 patience 轮不下降就停止，取验证损失最低时的权重；
  2. 提取训练+验证集的 fc1 特征，用 BKA 在交叉验证错误率上寻优 LS-SVM 的 (gam, sig2)；
  3. 预测时用 LS-SVM 给出类别，用网络 softmax 给出各类概率（界面上显示置信度）。
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from .nets import build, n_params
from .lssvm import BKALSSVM
from .tfr import JET


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def to_tensor(idx):
    """uint8 颜色索引 N x 64 x 64 -> float32 N x 3 x 64 x 64（RGB）。"""
    return torch.from_numpy(JET[np.asarray(idx)]).permute(0, 3, 1, 2).contiguous()


class Recognizer:
    def __init__(self, class_names, model='resnet', device=None, epochs=60, patience=8, batch=32, lr=1e-3,
                 use_lssvm=True, class_weight=None, lssvm_kw=None, alarm_classes=None, feature=None):
        self.class_names = list(class_names)
        # 输入特征的说明，预测新音频时按同样方式生成时频图：kind = 'set' | 'mel'，win_s = 窗长（秒），fs = 采样率
        self.feature = dict(feature or {'kind': 'set', 'win_s': 10.0, 'fs': 16000})
        self.model_name, self.epochs, self.patience, self.batch, self.lr = model, epochs, patience, batch, lr
        self.use_lssvm, self.class_weight, self.lssvm_kw = use_lssvm, class_weight, lssvm_kw or {}
        self.alarm_classes = list(alarm_classes or [c for c in class_names if '异常' in c or 'abnormal' in c.lower()])
        self.device = torch.device(device or default_device())
        self.net, self.lssvm, self.history = None, None, {}

    @property
    def n_classes(self):
        return len(self.class_names)

    # ---------------- 训练 ----------------
    def fit(self, X_tr, y_tr, X_va, y_va, seed=0, log=print):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        Xtr, Xva = to_tensor(X_tr), to_tensor(X_va)
        self.mu = Xtr.mean((0, 2, 3), keepdim=True)
        self.sd = Xtr.std((0, 2, 3), keepdim=True) + 1e-6
        dev = self.device
        Xtr = ((Xtr - self.mu) / self.sd).to(dev)
        Xva = ((Xva - self.mu) / self.sd).to(dev)
        Ytr = torch.as_tensor(np.asarray(y_tr), dtype=torch.long, device=dev)
        Yva = torch.as_tensor(np.asarray(y_va), dtype=torch.long, device=dev)
        w = None
        if self.class_weight == 'balanced':
            cnt = np.bincount(np.asarray(y_tr), minlength=self.n_classes).astype(float)
            w = torch.tensor(cnt.sum() / (self.n_classes * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
        net = build(self.model_name, self.n_classes).to(dev)
        opt = torch.optim.Adam(net.parameters(), self.lr)
        lossf = nn.CrossEntropyLoss(weight=w)
        best, bad, state, t0 = np.inf, 0, None, time.time()
        curve = []
        for ep in range(self.epochs):
            net.train()
            perm = torch.as_tensor(rng.permutation(len(Ytr)), device=dev)
            for k in range(0, len(perm), self.batch):
                b = perm[k:k + self.batch]
                if len(b) < 2:                      # BatchNorm 不能处理只有 1 个样本的批次
                    continue
                opt.zero_grad()
                lossf(net(Xtr[b]), Ytr[b]).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            vl = self._loss(net, Xva, Yva, lossf)
            curve.append(vl)
            if vl < best:
                best, bad, state = vl, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(state)
        self.net = net.eval()
        self.history = {'epochs': ep + 1, 'val_loss': curve, 'train_s': time.time() - t0, 'n_params': n_params(net)}
        log(f'网络训练完成：{ep + 1} 轮，最佳验证损失 {best:.4f}，用时 {time.time() - t0:.0f} s')
        if self.use_lssvm:
            t1 = time.time()
            F = self.features(np.concatenate([X_tr, X_va]))
            y = np.r_[np.asarray(y_tr), np.asarray(y_va)]
            self.lssvm = BKALSSVM(self.n_classes, **self.lssvm_kw).fit(F, y, seed=seed)
            log(f'BKA-LSSVM 完成：gam={self.lssvm.gam:.3g}，sig2={self.lssvm.sig2:.3g}，'
                f'交叉验证错误率 {self.lssvm.cv_error:.4f}，用时 {time.time() - t1:.0f} s')
        return self

    @torch.no_grad()
    def _loss(self, net, X, Y, lossf):
        net.eval()
        tot = 0.0
        for k in range(0, len(Y), 512):
            tot += lossf(net(X[k:k + 512]), Y[k:k + 512]).item() * len(Y[k:k + 512])
        return tot / len(Y)

    # ---------------- 推理 ----------------
    @torch.no_grad()
    def _forward(self, X):
        feats, probs = [], []
        for k in range(0, len(X), 256):
            xb = ((to_tensor(X[k:k + 256]) - self.mu) / self.sd).to(self.device)
            f = self.net.features(xb)
            feats.append(f.cpu())
            probs.append(torch.softmax(self.net.head(f), 1).cpu())
        return torch.cat(feats).numpy().astype(np.float64), torch.cat(probs).numpy()

    def features(self, X):
        return self._forward(X)[0]

    def predict(self, X, head='auto'):
        """返回 (类别下标, 网络 softmax 概率 N x C)。head='net' 只用网络；'auto' 有 LSSVM 时用 LSSVM。"""
        F, P = self._forward(X)
        if head != 'net' and self.lssvm is not None:
            return self.lssvm.predict(F), P
        return P.argmax(1), P

    # ---------------- 保存 / 加载 ----------------
    def save(self, path):
        torch.save({'class_names': self.class_names, 'model': self.model_name, 'state_dict': self.net.state_dict(),
                    'mu': self.mu, 'sd': self.sd, 'lssvm': self.lssvm.state() if self.lssvm else None,
                    'alarm_classes': self.alarm_classes, 'history': self.history, 'feature': self.feature,
                    'train_cfg': {'epochs': self.epochs, 'patience': self.patience, 'batch': self.batch,
                                  'lr': self.lr, 'class_weight': self.class_weight}}, path)

    @classmethod
    def load(cls, path, device=None):
        d = torch.load(path, map_location='cpu', weights_only=False)
        r = cls(d['class_names'], d['model'], device, alarm_classes=d.get('alarm_classes'),
                feature=d.get('feature'), **d.get('train_cfg', {}))
        r.net = build(d['model'], len(d['class_names']))
        r.net.load_state_dict(d['state_dict'])
        r.net = r.net.to(r.device).eval()
        r.mu, r.sd, r.history = d['mu'], d['sd'], d.get('history', {})
        r.lssvm = BKALSSVM.from_state(d['lssvm']) if d['lssvm'] else None
        return r


def metrics(y, p, n_classes):
    """准确率、平衡准确率、宏平均 F1、混淆矩阵（行 = 真实，列 = 预测；预测 -1 计为错误）。"""
    y, p = np.asarray(y), np.asarray(p)
    C = np.zeros((n_classes, n_classes), int)
    for a, b in zip(y, p):
        if 0 <= b < n_classes:
            C[a, b] += 1
    tp = np.diag(C).astype(float)
    rec = tp / np.maximum(np.bincount(y, minlength=n_classes), 1)
    prec = tp / np.maximum(C.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return {'acc': float(np.mean(y == p)), 'bal_acc': float(rec.mean()), 'macro_f1': float(f1.mean()),
            'recall': rec, 'precision': prec, 'confusion': C}
