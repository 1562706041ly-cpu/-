"""生成基准测试用的时频图库（CPU 密集，已在服务器上算好，一般不需要自己运行）。

数据：东南大学齿轮箱公开数据集（gearbox/gearset/*.csv，第 3 通道），
      https://github.com/cathysiyu/Mechanical-datasets
样本：与 v2 的 run_step1_samples.m 相同（去掉前 999 点、窗长 2048、步长 1000、每类 200 个）。
      20-0 工况生成的样本与你们的 data_total.mat 逐点一致（已核对）。
噪声：先在整段连续信号上加高斯白噪声（按该类整段信号功率计算 SNR），再切窗，
      这样重叠窗口里的噪声也是同一段，和真实录音一致。噪声种子固定，结果可复现。

输出：banks/<工况>_<噪声>.npz，含 X (1000x64x64 uint8 颜色索引)、y (1..5)、class_names。

用法：
  python make_banks.py --csv_dir 东南大学数据目录 --out banks --workers 4
  python make_banks.py --check_mat ../data/data_total.mat ...   # 额外核对 20-0 样本与 data_total.mat 一致
"""
import argparse, os, re, time
# 每个子进程只用 1 个 BLAS 线程，否则多进程 x 多线程会严重抢占 CPU
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from multiprocessing import Pool
from tfr import tfr_index

CLASSES = ['Chipped', 'Health', 'Miss', 'Root', 'Surface']
CONDS = ['20_0', '30_2']
LEVELS = ['clean', 'snr20', 'snr10', 'snr5', 'snr0', 'snr-5']
CHANNEL = 3          # 第 3 通道（1 起算），与 cfg.smp.channel 相同
SKIP = 999           # cfg.smp.skip_head = 1000（1 起算）
WIN, STEP, NPC = 2048, 1000, 200


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


def snr_db(level):
    return None if level == 'clean' else float(level[3:])


def windows(seg):
    return np.stack([seg[k * STEP: k * STEP + WIN] for k in range(NPC)])


def class_windows(x, cond_i, level, cls_i):
    seg = x[SKIP: SKIP + (NPC - 1) * STEP + WIN].copy()
    s = snr_db(level)
    if s is not None:
        rng = np.random.default_rng([2026, cond_i, int(round(s)) + 100, cls_i])
        seg = seg + rng.standard_normal(seg.size) * np.sqrt(np.mean(seg ** 2) / 10 ** (s / 10))
    return windows(seg)


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
    ap.add_argument('--check_mat', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = {(c, k): read_channel(os.path.join(a.csv_dir, f'{k}_{c}.csv')) for c in CONDS for k in CLASSES}

    if a.check_mat:
        import scipy.io as sio
        R = sio.loadmat(a.check_mat)['Rawdata']
        W = np.concatenate([class_windows(raw[('20_0', k)], 0, 'clean', i) for i, k in enumerate(CLASSES)])
        print('20-0 样本与 data_total.mat 最大差异:', np.abs(W - R[:, :WIN]).max())

    with Pool(a.workers) as pool:
        for ci, c in enumerate(CONDS):
            for lv in LEVELS:
                t0 = time.time()
                f = os.path.join(a.out, f'{c}_{lv}.npz')
                if os.path.exists(f):
                    write_mat(f); print(f'{f} 已存在，跳过'); continue
                W = np.concatenate([class_windows(raw[(c, k)], ci, lv, i) for i, k in enumerate(CLASSES)])
                X = np.stack(pool.map(_img, list(W), chunksize=20))
                y = np.repeat(np.arange(1, 6), NPC).astype(np.int64)
                np.savez_compressed(f, X=X, y=y, class_names=np.array(CLASSES))
                write_mat(f)
                print(f'{f}: {X.shape} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
