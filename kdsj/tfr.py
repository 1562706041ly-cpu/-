"""时频图：同步提取变换（SET）、对数梅尔谱、面积平均缩放、jet 颜色映射（只依赖 numpy）。

所有图都以 64x64 uint8 颜色索引（0..255）表示，用 jet(256) 映射成 RGB 后送入网络。
set_image 与 MATLAB 版 src/set_transform.m + src/tfr_to_rgb.m（'area'）逐像素相同。

参考：Yu G., Yu M., Xu C. Synchroextracting transform. IEEE Trans. Industrial Electronics, 2017.
"""
import numpy as np

EPS = np.finfo(float).eps


def area_weights(m, n):
    """m x n 面积平均矩阵：输出第 i 个像素 = 原坐标区间 [i*n/m, (i+1)*n/m] 上的加权平均。"""
    W = np.zeros((m, n))
    s = n / m
    for i in range(m):
        a, b = i * s, (i + 1) * s
        for j in range(int(np.floor(a)), min(int(np.ceil(b)), n)):
            W[i, j] = min(b, j + 1) - max(a, j)
    return W / W.sum(1, keepdims=True)


_W_CACHE = {}


def resize_area(A, size=(64, 64)):
    h, w = A.shape
    key = (h, w, tuple(size))
    if key not in _W_CACHE:
        _W_CACHE[key] = (area_weights(size[0], h), area_weights(size[1], w))
    Wh, Ww = _W_CACHE[key]
    return Wh @ A @ Ww.T


def _omega(N):
    pos = np.arange(1, N // 2 + 1) * (2 * np.pi / N)
    return np.concatenate([[0.0], pos, -pos[:(N - 1) // 2][::-1]])


def set_matrix(x, nv=32, cf=6.0, n_oct=None):
    """同步提取变换 |Te|（尺度数 x N，低频在下）。n_oct 默认 floor(log2 N) - 1（与 MATLAB 版相同）。"""
    x = np.asarray(x, np.float64).ravel()
    x = x - x.mean()
    N = x.size
    noct = int(np.floor(np.log2(N))) - 1 if n_oct is None else n_oct
    scales = 2.0 ** (np.arange(1, noct * nv + 1) / nv)
    omega = _omega(N)
    X = np.fft.fft(x)
    psi = np.exp(-((scales[:, None] * omega[None, :] - cf) ** 2) / 2) * (omega > 0)
    W = np.fft.ifft(X[None, :] * psi, axis=1)
    dW = np.fft.ifft(X[None, :] * (1j * omega[None, :] * psi), axis=1)
    ifreq = np.imag(dW / (W + EPS)) / (2 * np.pi)
    fc = cf / (2 * np.pi * scales[:, None])
    df = fc * (2 ** (1 / nv) - 1)
    aW = np.abs(W)
    mask = (np.abs(ifreq - fc) <= df / 2) & (aW > 1e-8 * aW.max())
    return np.flipud(aW * mask)


def _compress(A):
    A = np.log1p(A / (A.max() + EPS) * 100) / np.log1p(100)
    return np.floor(np.clip(A, 0, 1) * 255 + 0.5).astype(np.uint8)   # 与 MATLAB round 一致


def set_image(x, size=(64, 64), nv=32, cf=6.0):
    """短信号（如 2048 点）的同步提取变换时频图，精确计算。"""
    return _compress(resize_area(set_matrix(x, nv, cf), size))


def _time_area_mean(v, edges):
    N = v.shape[-1]
    cs = np.concatenate([np.zeros(v.shape[:-1] + (1,)), np.cumsum(v, -1)], -1)
    i = np.minimum(np.floor(edges).astype(int), N - 1)
    f = edges - i
    at = cs[..., i] + f * (cs[..., i + 1] - cs[..., i])
    return np.diff(at, axis=-1) / np.diff(edges)


def set_image_fast(x, size=(64, 64), n_oct=10, nv=32, cf=6.0, over=16, K=6.0):
    """长信号（如 10 s 音频）的同步提取变换时频图，快速近似计算，内存占用小。

    每个尺度的解析 Morlet 小波在频域只占 (0, (cf+K)/s] 一段（之外 ψ < exp(-K²/2) ≈ 1.5e-8），
    只对这段频谱做长度 M ≈ over × 带宽 的 IFFT（相当于时间上降采样）。|W| 与瞬时频率 imag(dW/W)
    不受降采样影响，掩膜和面积平均在降采样后的时间点上计算。
    n_oct 固定为 10（与 2048 点窗口相同），16 kHz 采样时覆盖约 15 Hz–8 kHz。
    在 MIMII 10 s 音频上与精确算法相比：颜色索引平均差 0.2（满量程 255），99% 像素差 ≤ 2。
    """
    x = np.asarray(x, np.float64).ravel()
    x = x - x.mean()
    N = x.size
    na = n_oct * nv
    scales = 2.0 ** (np.arange(1, na + 1) / nv)
    X = np.fft.fft(x)
    edges = np.linspace(0, N, size[1] + 1)
    rows = np.zeros((na, size[1]))
    for j, s in enumerate(scales):
        k1 = min(N // 2, int(np.ceil((cf + K) / s * N / (2 * np.pi))))
        k = np.arange(1, k1 + 1)
        w = k * (2 * np.pi / N)
        Y = X[k] * np.exp(-(s * w - cf) ** 2 / 2)
        bw = k.size
        M = N if over * bw >= N else int(2 ** np.ceil(np.log2(over * bw)))
        Z = np.zeros((2, M), complex)
        Z[0, :bw] = Y
        Z[1, :bw] = Y * (1j * w)
        W, dW = np.fft.ifft(Z, axis=1) * (M / N)
        ifreq = np.imag(dW / (W + EPS)) / (2 * np.pi)
        fc = cf / (2 * np.pi * s)
        df = fc * (2 ** (1 / nv) - 1)
        v = np.abs(W) * (np.abs(ifreq - fc) <= df / 2)
        rows[j] = _time_area_mean(v, edges * (M / N))
    return _compress(area_weights(size[0], na) @ np.flipud(rows))


def mel_filterbank(fs, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = fmax or fs / 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / fs)
    fb = np.zeros((n_mels, bins.size))
    for i in range(n_mels):
        l, c, r = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l), (r - bins) / (r - c)), 0, None)
    return fb


_FB = {}


def logmel_image(x, fs=16000, size=(64, 64), n_fft=1024, hop=512, top_db=80.0):
    """对数梅尔谱图（64 个梅尔带，时间方向面积平均到 64 帧，动态范围 80 dB）。"""
    x = np.asarray(x, np.float64).ravel()
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    n = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    P = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) ** 2
    key = (fs, n_fft, size[0])
    if key not in _FB:
        _FB[key] = mel_filterbank(fs, n_fft, size[0])
    M = (_FB[key] @ P.T) @ area_weights(size[1], n).T
    D = 10 * np.log10(M + 1e-12)
    A = (D - D.max() + top_db) / top_db
    return np.floor(np.clip(np.flipud(A), 0, 1) * 255 + 0.5).astype(np.uint8)


def jet(m=256):
    """与 MATLAB jet(m) 相同的颜色表（m x 3）。"""
    n = int(np.ceil(m / 4))
    u = np.concatenate([np.arange(1, n + 1) / n, np.ones(n - 1), np.arange(n, 0, -1) / n])
    g = int(np.ceil(n / 2)) - (m % 4 == 1) + np.arange(1, u.size + 1)
    r, b = g + n, g - n
    g, r, b = g[g <= m], r[r <= m], b[b >= 1]
    J = np.zeros((m, 3))
    J[r - 1, 0] = u[:r.size]
    J[g - 1, 1] = u[:g.size]
    J[b - 1, 2] = u[u.size - b.size:]
    return J


JET = jet(256).astype(np.float32)


def to_rgb(idx):
    """颜色索引 (..., H, W) uint8 -> RGB float32 (..., H, W, 3)。"""
    return JET[np.asarray(idx)]


def make_image(x, fs=None, kind='set'):
    """按信号长度自动选择算法：kind='set' 时 ≤ 8192 点用精确算法，更长用快速算法；kind='mel' 为对数梅尔谱。"""
    if kind == 'mel':
        return logmel_image(x, fs or 16000)
    return set_image(x) if np.size(x) <= 8192 else set_image_fast(x)
