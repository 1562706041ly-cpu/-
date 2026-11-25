"""最小二乘支持向量机（LS-SVM）一对一多分类，RBF 核，以及 BKA 超参数寻优（只依赖 numpy）。

LS-SVM：Suykens J.A.K., Vandewalle J. Least squares support vector machine classifiers.
Neural Processing Letters, 1999。本项目自行实现，不依赖 LSSVMlab。
超参数 (gam, sig2) 只用训练+验证集内的分层 k 折交叉验证错误率来选，测试集不参与。
LS-SVM 训练是 O(n³)，样本多时按类别抽样（cv_cap / fit_cap），见 BKALSSVM 的说明。
"""
import numpy as np
from .optim import Evaluator, optimize


def rbf(A, B, sig2):
    D = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-np.maximum(D, 0) / sig2)


def _solve_pairs(y, K, gam, n_classes):
    """训练所有一对一二分类器，返回 [(a, b, 样本下标, alpha*y, 偏置), ...]。"""
    out = []
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            idx = np.where((y == a) | (y == b))[0]
            if idx.size == 0:
                continue
            yb = np.where(y[idx] == a, 1.0, -1.0)
            n = idx.size
            A = np.zeros((n + 1, n + 1))
            A[0, 1:] = yb
            A[1:, 0] = yb
            A[1:, 1:] = np.outer(yb, yb) * K[np.ix_(idx, idx)] + np.eye(n) / gam
            try:
                sol = np.linalg.solve(A, np.r_[0.0, np.ones(n)])
                out.append((a, b, idx, sol[1:] * yb, sol[0]))
            except np.linalg.LinAlgError:
                out.append((a, b, idx, None, None))
    return out


def _vote(pairs, Kte, n_classes):
    nte = Kte.shape[0]
    votes = np.zeros((nte, n_classes))
    score = np.zeros((nte, n_classes))
    for a, b, idx, ay, bias in pairs:
        d = np.full(nte, np.nan) if ay is None else Kte[:, idx] @ ay + bias
        votes[:, a] += d > 0
        votes[:, b] += d <= 0
        score[:, a] += d
        score[:, b] -= d
    with np.errstate(invalid='ignore'):
        pred = np.argmax(votes + 1e-6 * np.tanh(score), 1)
    bad = ~np.isfinite(score).all(1)
    pred[bad] = -1                                   # 无法判别：记为 -1，计入错误
    return pred, votes


def decode(z, log10_gam=(-2, 6), log10_sig2=(-2, 4)):
    z = np.clip(np.asarray(z, float), 0, 1)
    return (10 ** (log10_gam[0] + z[0] * (log10_gam[1] - log10_gam[0])),
            10 ** (log10_sig2[0] + z[1] * (log10_sig2[1] - log10_sig2[0])))


def _cap(y, cap, rng):
    """每类最多取 cap 个样本（随机）。"""
    if cap is None:
        return np.arange(y.size)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        keep.append(idx if idx.size <= cap else np.sort(rng.choice(idx, cap, replace=False)))
    return np.concatenate(keep)


class BKALSSVM:
    """用 BKA 在交叉验证错误率上寻优 (gam, sig2) 的 LS-SVM 分类器。标签为 0..n_classes-1。

    cv_cap：交叉验证时每类最多用多少样本（控制寻优耗时）；fit_cap：最终训练时每类最多用多少样本。
    样本少于这两个值时（如齿轮箱数据每类 156 个）不会抽样，与原流程完全相同。
    """

    def __init__(self, n_classes, pop=12, iters=20, folds=5, cv_cap=200, fit_cap=600,
                 log10_gam=(-2, 6), log10_sig2=(-2, 4)):
        self.n_classes, self.pop, self.iters, self.folds = n_classes, pop, iters, folds
        self.cv_cap, self.fit_cap, self.lg, self.ls = cv_cap, fit_cap, log10_gam, log10_sig2

    def fit(self, F, y, seed=0):
        rng = np.random.default_rng(seed)
        F = np.asarray(F, np.float64)
        y = np.asarray(y, int)
        self.mu = F.mean(0)
        self.sd = F.std(0, ddof=1)
        self.sd[self.sd == 0] = 1
        Z = (F - self.mu) / self.sd
        cv = _cap(y, self.cv_cap, rng)
        Zc, yc = Z[cv], y[cv]
        folds = np.zeros(yc.size, int)
        for c in np.unique(yc):
            idx = rng.permutation(np.where(yc == c)[0])
            folds[idx] = np.arange(idx.size) % self.folds
        D = np.maximum((Zc ** 2).sum(1)[:, None] + (Zc ** 2).sum(1)[None, :] - 2 * Zc @ Zc.T, 0)

        def cv_err(z):
            gam, sig2 = decode(z, self.lg, self.ls)
            K = np.exp(-D / sig2)
            wrong = 0
            for k in range(self.folds):
                te, tr = folds == k, folds != k
                pairs = _solve_pairs(yc[tr], K[np.ix_(tr, tr)], gam, self.n_classes)
                p, _ = _vote(pairs, K[np.ix_(te, tr)], self.n_classes)
                wrong += int((p != yc[te]).sum())
            return wrong / yc.size, None

        ev = Evaluator(cv_err, keyfun=lambda z: (round(z[0], 4), round(z[1], 4)))  # 与 MATLAB 版缓存键相同
        res = optimize('bka', ev, 2, self.pop, self.iters, seed=int(rng.integers(1 << 31)))
        self.gam, self.sig2 = decode(res['best_z'], self.lg, self.ls)
        self.cv_error, self.trace, self.n_unique = res['best_fit'], res['trace'], res['n_unique']
        fi = _cap(y, self.fit_cap, rng)
        self.Ztr, self.ytr = Z[fi], y[fi]
        self.pairs = _solve_pairs(self.ytr, rbf(self.Ztr, self.Ztr, self.sig2), self.gam, self.n_classes)
        return self

    def predict(self, F, return_votes=False):
        Zt = (np.asarray(F, np.float64) - self.mu) / self.sd
        pred, votes = _vote(self.pairs, rbf(Zt, self.Ztr, self.sig2), self.n_classes)
        return (pred, votes) if return_votes else pred

    def state(self):
        return {'n_classes': self.n_classes, 'gam': self.gam, 'sig2': self.sig2, 'cv_error': self.cv_error,
                'mu': self.mu, 'sd': self.sd, 'Ztr': self.Ztr, 'ytr': self.ytr,
                'pairs': [(a, b, idx, ay, bias) for a, b, idx, ay, bias in self.pairs]}

    @classmethod
    def from_state(cls, s):
        m = cls(s['n_classes'])
        for k in ('gam', 'sig2', 'cv_error', 'mu', 'sd', 'Ztr', 'ytr', 'pairs'):
            setattr(m, k, s[k])
        return m
