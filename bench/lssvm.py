"""BKA 优化 LS-SVM（numpy 版，逐行对应 MATLAB 的 src/lssvm_stage.m、lssvm_ovo.m、
lssvm_cv_error.m、lssvm_decode.m、stratified_folds.m、bka_optimize.m、EvalCounter.m）。

测试集只在最后预测一次；超参数只用"训练+验证集"内的分层 k 折交叉验证错误率来选。
"""
import numpy as np


def rbf_kernel(A, B, sig2):
    D = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-np.maximum(D, 0) / sig2)


def lssvm_ovo(ytr, Ktr, Kte, gam, n_classes):
    """一对一 LS-SVM。返回预测（1..n_classes，无法判别为 0）和无法判别的样本数。"""
    nte = Kte.shape[0]
    votes = np.zeros((nte, n_classes)); score = np.zeros((nte, n_classes))
    for a in range(1, n_classes):
        for b in range(a + 1, n_classes + 1):
            idx = np.where((ytr == a) | (ytr == b))[0]
            if idx.size == 0:
                continue
            yb = np.where(ytr[idx] == a, 1.0, -1.0)
            n = idx.size
            A = np.zeros((n + 1, n + 1))
            A[0, 1:] = yb; A[1:, 0] = yb
            A[1:, 1:] = np.outer(yb, yb) * Ktr[np.ix_(idx, idx)] + np.eye(n) / gam
            try:
                sol = np.linalg.solve(A, np.r_[0.0, np.ones(n)])
                d = Kte[:, idx] @ (sol[1:] * yb) + sol[0]
            except np.linalg.LinAlgError:
                d = np.full(nte, np.nan)
            votes[:, a - 1] += d > 0; votes[:, b - 1] += d <= 0
            score[:, a - 1] += d; score[:, b - 1] -= d
    with np.errstate(invalid='ignore'):
        pred = np.argmax(votes + 1e-6 * np.tanh(score), 1) + 1
    bad = ~np.isfinite(score).all(1)
    pred[bad] = 0
    return pred, int(bad.sum())


def decode(z, log10_gam=(-2, 6), log10_sig2=(-2, 4)):
    z = np.clip(np.asarray(z, float), 0, 1)
    return (10 ** (log10_gam[0] + z[0] * (log10_gam[1] - log10_gam[0])),
            10 ** (log10_sig2[0] + z[1] * (log10_sig2[1] - log10_sig2[0])))


def stratified_folds(y, k, rng):
    folds = np.zeros(y.size, int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        folds[idx] = np.arange(idx.size) % k + 1
    return folds


def bka_optimize(pop, T, dim, fobj, rng, p=0.9):
    """黑翅鸢优化（Wang et al. 2024），在 [0,1]^dim 中最小化 fobj。修正同 MATLAB 版。"""
    lb, ub = np.zeros(dim), np.ones(dim)
    X = lb + rng.random((pop, dim)) * (ub - lb)
    F = np.array([fobj(x) for x in X])
    ib = F.argmin(); best_fit, best_pos = F[ib], X[ib].copy()
    for t in range(1, T + 1):
        n = 0.05 * np.exp(-2 * (t / T) ** 2)
        for i in range(pop):
            r = rng.random()
            if p < r:
                Xn = X[i] + n * (1 + np.sin(r)) * X[i]
            else:
                Xn = X[i] * (n * (2 * rng.random(dim) - 1) + 1)
            Xn = np.clip(Xn, lb, ub); fn = fobj(Xn)
            if fn < F[i]:
                X[i], F[i] = Xn, fn
            m = 2 * np.sin(r + np.pi / 2)
            s = rng.integers(pop)
            c = np.tan((rng.random(dim) - 0.5) * np.pi)
            Xn = X[i] + c * (X[i] - best_pos) if F[i] < F[s] else X[i] + c * (best_pos - m * X[i])
            Xn = np.clip(Xn, lb, ub); fn = fobj(Xn)
            if fn < F[i]:
                X[i], F[i] = Xn, fn
            if F[i] < best_fit:
                best_fit, best_pos = F[i], X[i].copy()
    return best_fit, best_pos


def lssvm_stage(Fdev, ydev, tests, n_classes, rng, pop=12, iters=20, k=5,
                log10_gam=(-2, 6), log10_sig2=(-2, 4)):
    """tests: {名称: (F, y)}。返回所选超参数、CV 错误率和各测试集上的预测。"""
    mu = Fdev.mean(0); sd = Fdev.std(0, ddof=1); sd[sd == 0] = 1
    Xd = (Fdev - mu) / sd
    folds = stratified_folds(ydev, k, rng)
    D = (Xd ** 2).sum(1)[:, None] + (Xd ** 2).sum(1)[None, :] - 2 * Xd @ Xd.T
    D = np.maximum(D, 0)
    cache = {}

    def cv_err(z):
        key = f'{z[0]:.4f}_{z[1]:.4f}'                    # 与 EvalCounter 的缓存键相同
        if key not in cache:
            gam, sig2 = decode(z, log10_gam, log10_sig2)
            K = np.exp(-D / sig2); wrong = 0
            for f in range(1, k + 1):
                te = folds == f; tr = ~te
                p, _ = lssvm_ovo(ydev[tr], K[np.ix_(tr, tr)], K[np.ix_(te, tr)], gam, n_classes)
                wrong += int((p != ydev[te]).sum())
            cache[key] = wrong / ydev.size
        return cache[key]

    best_fit, best_z = bka_optimize(pop, iters, 2, cv_err, rng)
    gam, sig2 = decode(best_z, log10_gam, log10_sig2)
    Kd = rbf_kernel(Xd, Xd, sig2)
    out = {'gam': gam, 'sig2': sig2, 'cv_error': best_fit, 'n_unique': len(cache), 'pred': {}, 'invalid': 0}
    for name, (Ft, yt) in tests.items():
        p, nbad = lssvm_ovo(ydev, Kd, rbf_kernel((Ft - mu) / sd, Xd, sig2), gam, n_classes)
        out['pred'][name] = p; out['invalid'] += nbad
    return out
