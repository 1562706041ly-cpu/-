"""变分模态分解（VMD）、适应度指标、参数编解码与 VMD 降噪（只依赖 numpy）。

VMD 算法：Dragomiretskiy K., Zosso D. Variational mode decomposition. IEEE TSP, 2014。
本实现只保留当前一次迭代的频谱（原版保存全部迭代），内存约为原版的 1/500，结果相同。
"""
import numpy as np


def vmd(signal, alpha=2000, K=5, tau=0.0, DC=False, init=1, tol=1e-7, max_iter=500, seed=None):
    """返回 (u: K x L 各模态，按中心频率从低到高, omega: 中心频率（周期/采样点）, 迭代次数)。"""
    x = np.asarray(signal, np.float64).ravel()
    L = x.size
    odd = L % 2 == 1
    if odd:
        x = np.r_[x, x[-1]]
    T0 = x.size
    f = np.r_[x[T0 // 2 - 1::-1], x, x[:T0 // 2 - 1:-1]]          # 镜像延拓到 2*T0
    T = f.size
    freqs = np.arange(1, T + 1) / T - 0.5 - 1 / T
    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat[:T // 2] = 0
    half = slice(T // 2, T)
    u_hat = np.zeros((K, T), complex)
    if init == 1:
        omega = 0.5 / K * np.arange(K)
    elif init == 2:
        rng = np.random.default_rng(seed)
        omega = np.sort(np.exp(np.log(1 / T0) + (np.log(0.5) - np.log(1 / T0)) * rng.random(K)))
    else:
        omega = np.zeros(K)
    if DC:
        omega[0] = 0
    lam = np.zeros(T, complex)
    diff, n = tol + np.finfo(float).eps, 0
    fh = freqs[half]
    while diff > tol and n < max_iter - 1:
        u_old = u_hat.copy()
        total = u_hat.sum(0)
        for k in range(K):
            rest = total - u_hat[k]
            u_hat[k] = (f_hat - rest - lam / 2) / (1 + alpha * (freqs - omega[k]) ** 2)
            total = rest + u_hat[k]
            if not (DC and k == 0):
                p = np.abs(u_hat[k, half]) ** 2
                omega[k] = fh @ p / p.sum()
        lam = lam + tau * (total - f_hat)
        n += 1
        d = u_hat - u_old
        diff = abs(np.finfo(float).eps + np.sum(d * d.conj()).real / T)
    full = np.zeros((K, T), complex)
    full[:, half] = u_hat[:, half]
    full[:, T // 2:0:-1] = u_hat[:, half].conj()
    full[:, 0] = full[:, -1].conj()
    u = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=1), axis=1))[:, T // 4:3 * T // 4]
    if odd:
        u = u[:, :L]
    order = np.argsort(omega)
    return u[order], omega[order], n


def analytic_signal(x):
    x = np.asarray(x, np.float64).ravel()
    n = x.size
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(np.fft.fft(x) * h)


def envelope_entropy(x):
    env = np.abs(analytic_signal(x))
    p = env / env.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kurtosis(x):
    x = np.asarray(x, np.float64) - np.mean(x)
    return float(np.mean(x ** 4) / (np.mean(x ** 2) ** 2 + 1e-300))


def permutation_entropy(x, m=3, delay=1):
    from itertools import permutations
    from math import factorial
    x = np.asarray(x, np.float64).ravel()
    n = x.size - (m - 1) * delay
    if n <= 0:
        return 0.0
    emb = np.stack([x[i * delay:i * delay + n] for i in range(m)], 1)
    codes = np.argsort(emb, 1, kind='stable') @ (m ** np.arange(m))
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / n
    return float(-(p * np.log(p)).sum() / np.log(factorial(m)))


def decode(z, alpha_range=(100, 2500), K_range=(3, 10)):
    """归一化变量 z∈[0,1]^2 -> (alpha, K)。适应度和最终分解都只用这个函数，保证参数一致。"""
    z = np.clip(np.asarray(z, float).ravel(), 0, 1)
    alpha = int(round(alpha_range[0] + z[0] * (alpha_range[1] - alpha_range[0])))
    nK = K_range[1] - K_range[0] + 1
    K = K_range[0] + min(int(np.floor(z[1] * nK)), nK - 1)
    return alpha, K


def encode(alpha, K, alpha_range=(100, 2500), K_range=(3, 10)):
    nK = K_range[1] - K_range[0] + 1
    return np.array([(alpha - alpha_range[0]) / (alpha_range[1] - alpha_range[0]), (K - K_range[0] + 0.5) / nK])


METRICS = {'envelope': envelope_entropy, 'permutation': permutation_entropy,
           'neg_kurtosis': lambda u: -kurtosis(u)}


def fitness(z, x, metric='envelope', aggregate='min', alpha_range=(100, 2500), K_range=(3, 10), **vmd_kw):
    """VMD 参数寻优的适应度（越小越好）。返回 (适应度, 信息字典)。"""
    alpha, K = decode(z, alpha_range, K_range)
    u, omega, _ = vmd(x, alpha, K, **vmd_kw)
    vals = np.array([METRICS[metric](ui) for ui in u])
    fit = float(vals.min() if aggregate == 'min' else vals.mean())
    return fit, {'alpha': alpha, 'K': K, 'per_imf': vals, 'omega': omega}


def key(z, alpha_range=(100, 2500), K_range=(3, 10)):
    """缓存键：解码后相同的 (alpha, K) 视为同一个解。"""
    return decode(z, alpha_range, K_range)


def denoise(x, alpha=2000, K=5, keep_ratio=0.5, **vmd_kw):
    """VMD 降噪：分解后只保留与原信号相关系数 ≥ keep_ratio × 最大相关系数的模态，再相加重构。
    返回 (重构信号, 保留的模态编号, 各模态相关系数)。"""
    x = np.asarray(x, np.float64).ravel()
    u, _, _ = vmd(x, alpha, K, **vmd_kw)
    xc = x - x.mean()
    r = np.array([np.dot(ui - ui.mean(), xc) / (np.linalg.norm(ui - ui.mean()) * np.linalg.norm(xc) + 1e-300)
                  for ui in u])
    keep = np.where(r >= keep_ratio * r.max())[0]
    return u[keep].sum(0), keep, r
