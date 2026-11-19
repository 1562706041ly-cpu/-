"""生成基准测试用的时频图库（CPU 密集，已在服务器上算好，一般不需要自己运行）。

数据：东南大学齿轮箱公开数据集（gearbox/gearset/*.csv，第 3 通道），
      https://github.com/cathysiyu/Mechanical-datasets
样本：与 v2 的 run_step1_samples.m 相同（去掉前 999 点、窗长 2048、步长 1000、每类 200 个）。
      20-0 工况生成的样本与你们的 data_total.mat 逐点一致（已核对）。
噪声：先在整段连续信号上加噪声（按该类整段信号功率计算 SNR），再切窗，
      这样重叠窗口里的噪声也是同一段，和真实录音一致。噪声种子固定，结果可复现。
      --noise white  高斯白噪声（默认）
      --noise pink   粉红噪声（功率谱 ∝ 1/f，低频能量多，更接近机械背景噪声）
      --noise_wav    用一段真实噪声录音（如矿区背景声），每类随机截取一段叠加；
                     录音采样率需与信号相同（同一个麦克风录的就满足）
阶次归一化（--order_ref 20）：把转速为 S Hz 的工况按 S/20 的比例重采样，
      使每个窗口覆盖的转数与 20 Hz 工况相同，齿轮啮合频率对齐到同一位置。
      只需要知道（或估计出）转速。噪声先加在原始信号上，再重采样。

输出：banks/<工况>_<噪声><tag>.npz，含 X (1000x64x64 uint8 颜色索引)、y (1..5)、class_names，
      并同时导出 MATLAB 用的 .mat。

用法：
  python make_banks.py --csv_dir 东南大学数据目录 --out banks --workers 4
  python make_banks.py ... --levels snr10,snr5,snr0 --noise_seed 2027 --tag _n2     # 换一份噪声做确认实验
  python make_banks.py ... --levels snr5,snr0 --noise pink --tag _pink               # 粉红噪声
  python make_banks.py ... --conds 30_2 --levels clean,snr5 --order_ref 20 --tag _ord  # 阶次归一化
  python make_banks.py ... --noise_wav 矿区背景.wav --levels snr5,snr0 --tag _mine   # 真实噪声录音
  python make_banks.py --check_mat ../data/data_total.mat ...   # 额外核对 20-0 样本与 data_total.mat 一致
"""
import argparse, os, re, time, wave
# 每个子进程只用 1 个 BLAS 线程，否则多进程 x 多线程会严重抢占 CPU
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from multiprocessing import Pool
from tfr import tfr_index

CLASSES = ['Chipped', 'Health', 'Miss', 'Root', 'Surface']
CONDS = ['20_0', '30_2']
SPEED = {'20_0': 20, '30_2': 30}   # 转速（Hz），工况名的第一部分
LEVELS = ['clean', 'snr20', 'snr10', 'snr5', 'snr0', 'snr-5']
CHANNEL = 3          # 第 3 通道（1 起算），与 cfg.smp.channel 相同
SKIP = 999           # cfg.smp.skip_head = 1000（1 起算）
WIN, STEP, NPC = 2048, 1000, 200
NEED = (NPC - 1) * STEP + WIN


def read_channel(path, ch=CHANNEL):
    cache = path + f'.ch{ch}.npy'
    if os.path.exists(cache):
        return np.load(cache)
    with open(path, 'r', encoding='latin-1') as fh:
        lines = fh.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('Data')) + 1
    vals = []
    for l in lines[start:]:
        p = [t for t in re.split(r'[\t,]', l) if t.strip()]   # 部分文件用逗号分隔
        if len(p) >= ch:
            vals.append(float(p[ch - 1]))
    x = np.asarray(vals, dtype=np.float64)
    np.save(cache, x)
    return x


def read_wav(path):
    """读取 PCM wav（8/16/32 位），多声道取平均。"""
    with wave.open(path, 'rb') as w:
        n, ch, sw = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if sw == 1:
        x -= 128
    return x.reshape(-1, ch).mean(1)


def snr_db(level):
    return None if level == 'clean' else float(level[3:])


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(X.size); f[0] = 1
    x = np.fft.irfft(X / np.sqrt(f), n)
    return (x - x.mean()) / x.std()


def make_noise(n, rng, kind, wav):
    if wav is not None:
        reps = int(np.ceil((n + 1) / wav.size)) + 1
        long = np.tile(wav, reps)
        s = rng.integers(0, long.size - n)
        z = long[s: s + n]
        return (z - z.mean()) / (z.std() + 1e-12)
    return pink(n, rng) if kind == 'pink' else rng.standard_normal(n)


def class_windows(x, cond, level, cls_i, a, wav=None):
    ratio = SPEED[cond] / a.order_ref if a.order_ref else 1.0
    n_orig = int(np.ceil(NEED / ratio)) + 64 if ratio != 1.0 else NEED
    seg = x[SKIP: SKIP + n_orig].copy()
    s = snr_db(level)
    if s is not None:
        # 默认参数（白噪声、种子 2026）下与 v2.2 第一版图库完全相同
        rng = np.random.default_rng([a.noise_seed, CONDS.index(cond), int(round(s)) + 100, cls_i])
        seg = seg + make_noise(seg.size, rng, a.noise, wav) * np.sqrt(np.mean(seg ** 2) / 10 ** (s / 10))
    if ratio != 1.0:
        from fractions import Fraction
        from scipy.signal import resample_poly
        fr = Fraction(ratio).limit_denominator(100)
        seg = resample_poly(seg, fr.numerator, fr.denominator)[:NEED]
    return np.stack([seg[k * STEP: k * STEP + WIN] for k in range(NPC)])


def _img(w):
    return tfr_index(w)


def write_mat(npz_path):
    """同时导出 MATLAB 版（src/load_tfr_images.m 可直接读取）。需要 scipy，没有时跳过。"""
    m = npz_path[:-4] + '.mat'
    if os.path.exists(m):
        return
    try:
        import scipy.io as sio
    except ImportError:
        return
    d = np.load(npz_path)
    sio.savemat(m, {'X': d['X'], 'y': d['y'].astype(np.float64).reshape(-1, 1),
                    'class_names': np.array(list(d['class_names']), dtype=object)}, do_compression=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_dir', required=True)
    ap.add_argument('--out', default='banks')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--conds', default=','.join(CONDS))
    ap.add_argument('--levels', default=','.join(LEVELS))
    ap.add_argument('--noise', default='white', choices=['white', 'pink'])
    ap.add_argument('--noise_wav', default='')
    ap.add_argument('--noise_seed', type=int, default=2026)
    ap.add_argument('--order_ref', type=float, default=0, help='阶次归一化的参考转速（Hz），0 表示不做')
    ap.add_argument('--tag', default='')
    ap.add_argument('--check_mat', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conds = a.conds.split(',')
    wav = read_wav(a.noise_wav) if a.noise_wav else None
    raw = {(c, k): read_channel(os.path.join(a.csv_dir, f'{k}_{c}.csv')) for c in conds for k in CLASSES}

    if a.check_mat and '20_0' in conds:
        import scipy.io as sio
        R = sio.loadmat(a.check_mat)['Rawdata']
        W = np.concatenate([class_windows(raw[('20_0', k)], '20_0', 'clean', i, a) for i, k in enumerate(CLASSES)])
        print('20-0 样本与 data_total.mat 最大差异:', np.abs(W - R[:, :WIN]).max())

    with Pool(a.workers) as pool:
        for c in conds:
            for lv in a.levels.split(','):
                t0 = time.time()
                f = os.path.join(a.out, f'{c}_{lv}{a.tag}.npz')
                if os.path.exists(f):
                    write_mat(f); print(f'{f} 已存在，跳过'); continue
                W = np.concatenate([class_windows(raw[(c, k)], c, lv, i, a, wav) for i, k in enumerate(CLASSES)])
                X = np.stack(pool.map(_img, list(W), chunksize=20))
                y = np.repeat(np.arange(1, 6), NPC).astype(np.int64)
                np.savez_compressed(f, X=X, y=y, class_names=np.array(CLASSES))
                write_mat(f)
                print(f'{f}: {X.shape} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
