"""自检（约 1 分钟，不需要 GPU 和数据文件）。用法：python -m kdsj selftest"""
import os
import tempfile
import numpy as np


def run(log=print):
    from . import audio, tfr, vmd, optim, lssvm
    fails = 0

    def check(ok, msg):
        nonlocal fails
        log(f'[{" OK " if ok else "FAIL"}] {msg}')
        fails += not ok

    rng = np.random.default_rng(1)
    t = np.arange(1000) / 1000
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 200 * t) + 0.05 * rng.standard_normal(1000)
    u, om, _ = vmd.vmd(x, 2000, 2)
    check(abs(om[0] * 1000 - 50) < 3 and abs(om[1] * 1000 - 200) < 3,
          f'VMD 中心频率 {om[0] * 1000:.1f} / {om[1] * 1000:.1f} Hz（应为 50 / 200）')
    check(np.linalg.norm(u.sum(0) - x) / np.linalg.norm(x) < 0.1, 'VMD 各模态之和可重构原信号')
    check(all(vmd.decode(vmd.encode(1234, K)) == (1234, K) for K in range(3, 11)), 'VMD 参数编解码互逆')
    check(vmd.permutation_entropy(rng.standard_normal(5000)) > 0.98 and vmd.permutation_entropy(np.arange(1000.)) < 1e-9,
          '排列熵：白噪声≈1，单调序列=0')
    y, keep, r = vmd.denoise(x + 0.5 * rng.standard_normal(1000), 2000, 4)
    check(len(keep) >= 1 and y.shape == x.shape, f'VMD 降噪保留模态 {list(keep)}')

    for name in ('ttao', 'bka'):
        ev = optim.Evaluator(lambda z: (float(np.sum((z - 0.3) ** 2)), None), max_evals=600)
        res = optim.optimize(name, ev, 2, 12, 1000, seed=0)
        check(res['best_fit'] < 1e-3, f'{name.upper()} 优化二次函数，最优值 {res["best_fit"]:.2e}')

    X = np.r_[rng.standard_normal((100, 2)), rng.standard_normal((100, 2)) + 4, rng.standard_normal((100, 2)) + [4, -4]]
    yy = np.repeat(np.arange(3), 100)
    m = lssvm.BKALSSVM(3, pop=6, iters=5).fit(X[::2], yy[::2])
    acc = np.mean(m.predict(X[1::2]) == yy[1::2])
    check(acc > 0.95, f'BKA-LSSVM 三分类准确率 {100 * acc:.1f}%')

    x = np.cos(2 * np.pi * 0.1 * np.arange(1024))
    A = tfr.set_matrix(x)
    noct = int(np.floor(np.log2(1024))) - 1
    sc = 2.0 ** (np.arange(1, noct * 32 + 1) / 32)
    fc = np.flipud(6 / (2 * np.pi * sc))
    check(abs(fc[A.sum(1).argmax()] - 0.1) < 0.01, '同步提取变换峰值频率正确')
    B = tfr.resize_area(np.ones((320, 2048)), (64, 64))
    check(np.abs(B - 1).max() < 1e-12, '面积平均缩放保持常数')
    xs = rng.standard_normal(160000)
    a, b = tfr.set_image_fast(xs).astype(int), tfr.make_image(xs).astype(int)
    check(np.array_equal(a, b) and a.shape == (64, 64), '长信号自动使用快速时频算法')
    ref = [[0, 0, 0.515625], [0.015625, 1, 0.984375], [0.515625, 1, 0.484375], [1, 0.359375, 0], [0.5, 0, 0]]
    check(np.abs(tfr.jet(256)[[0, 96, 128, 200, 255]] - ref).max() < 1e-12, 'jet 颜色表与 MATLAB 一致')

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 't.wav')
        sig = 0.5 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
        audio.write_wav(p, sig, 8000)
        r, fs = audio.read_wav(p)
        check(fs == 8000 and np.abs(r[:, 0] - sig).max() < 1e-4, 'wav 写入与读取一致')

    try:
        import torch
        from .recognizer import Recognizer
        Xi = np.concatenate([np.full((40, 64, 64), 40, np.uint8), np.full((40, 64, 64), 200, np.uint8)])
        Xi = (Xi + rng.integers(0, 20, Xi.shape)).astype(np.uint8)
        yi = np.repeat([0, 1], 40)
        rec = Recognizer(['正常', '异常'], 'resnet', 'cpu', epochs=3, patience=3,
                         lssvm_kw={'pop': 6, 'iters': 3})
        rec.fit(Xi[::2], yi[::2], Xi[1::2], yi[1::2], log=lambda s: None)
        with tempfile.TemporaryDirectory() as d:
            rec.save(os.path.join(d, 'm.pt'))
            rec2 = Recognizer.load(os.path.join(d, 'm.pt'), 'cpu')
        p1, _ = rec.predict(Xi[1::2]); p2, _ = rec2.predict(Xi[1::2])
        check(np.array_equal(p1, p2) and np.mean(p1 == yi[1::2]) > 0.9, '识别器训练、保存、加载后预测一致')
    except ImportError:
        log('[SKIP] 未安装 PyTorch，跳过识别器检查')

    log(f'\n自检完成：{fails} 项失败')
    return fails
