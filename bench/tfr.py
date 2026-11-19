"""时频图生成（与 MATLAB 版 src/set_transform.m + src/resize_area.m + src/tfr_to_rgb.m 一致）。

只用 numpy。图像以 uint8 颜色索引（0..255）保存，训练时再用 jet(256) 映射成 RGB，
和 MATLAB 版 tfr_to_rgb 的输出逐像素相同，但文件小 12 倍。
"""
import numpy as np


def set_transform(x, nv=32, cf=6.0):
    """小波同步提取变换（自研实现，对应 src/set_transform.m）。返回 |Te|（na x N，低频在下）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - x.mean()
    N = x.size
    noct = int(np.floor(np.log2(N))) - 1
    na = noct * nv
    scales = 2.0 ** (np.arange(1, na + 1) / nv)
    pos = np.arange(1, N // 2 + 1) * (2 * np.pi / N)
    neg = -pos[: (N - 1) // 2][::-1]
    omega = np.concatenate([[0.0], pos, neg])
    X = np.fft.fft(x)
    psi = np.exp(-((scales[:, None] * omega[None, :] - cf) ** 2) / 2) * (omega > 0)
    W = np.fft.ifft(X[None, :] * psi, axis=1)
    dW = np.fft.ifft(X[None, :] * (1j * omega[None, :] * psi), axis=1)
    ifreq = np.imag(dW / (W + np.finfo(float).eps)) / (2 * np.pi)
    fc = cf / (2 * np.pi * scales[:, None])
    df = fc * (2 ** (1 / nv) - 1)
    absW = np.abs(W)
    mask = (np.abs(ifreq - fc) <= df / 2) & (absW > 1e-8 * absW.max())
    return np.flipud(absW * mask)


def _area_weights(m, n):
    W = np.zeros((m, n))
    s = n / m
    for i in range(m):
        a, b = i * s, (i + 1) * s
        for j in range(int(np.floor(a)), min(int(np.ceil(b)), n)):
            W[i, j] = min(b, j + 1) - max(a, j)
    return W / W.sum(1, keepdims=True)


_WCACHE = {}


def resize_area(A, size=(64, 64)):
    """面积平均缩放（对应 src/resize_area.m）。"""
    h, w = A.shape
    key = (h, w, size)
    if key not in _WCACHE:
        _WCACHE[key] = (_area_weights(size[0], h), _area_weights(size[1], w))
    Wh, Ww = _WCACHE[key]
    return Wh @ A @ Ww.T


def tfr_index(x, size=(64, 64), nv=32, cf=6.0):
    """信号 -> 64x64 uint8 颜色索引（0..255），对应 tfr_to_rgb(...,'area') 中 idx-1。"""
    A = resize_area(set_transform(x, nv, cf), size)
    A = np.log1p(A / (A.max() + np.finfo(float).eps) * 100) / np.log1p(100)
    A = np.clip(A, 0, 1)
    # MATLAB round 为四舍五入（远离 0），numpy.round 为银行家舍入，这里用 floor(x+0.5)
    return np.floor(A * 255 + 0.5).astype(np.uint8)


def jet(m=256):
    """MATLAB jet(m) 的同一公式。"""
    n = int(np.ceil(m / 4))
    u = np.concatenate([np.arange(1, n + 1) / n, np.ones(n - 1), np.arange(n, 0, -1) / n])
    g = int(np.ceil(n / 2)) - (m % 4 == 1) + np.arange(1, u.size + 1)
    r, b = g + n, g - n
    g, r, b = g[g <= m], r[r <= m], b[b >= 1]
    J = np.zeros((m, 3))
    J[r - 1, 0] = u[: r.size]
    J[g - 1, 1] = u[: g.size]
    J[b - 1, 2] = u[u.size - b.size:]
    return J
