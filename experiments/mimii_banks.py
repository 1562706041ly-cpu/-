"""把 MIMII 第 1 通道音频转成时频图库（CPU 密集，已在服务器上算好）。

每个 10 s 文件生成一张图（整段音频，不切窗，避免同一文件的窗口跨训练/测试集）：
  X_set : 同步提取变换图（kdsj.tfr.set_image_fast，与振动数据相同的方法，快速近似计算）
  X_mel : 对数梅尔谱图（kdsj.tfr.logmel_image，声音识别常用的对照）
都是 64x64 uint8 颜色索引。

分片计算（每片 300 个文件），可以中断后续跑；全部完成后合并为 <out>/<snr>_<machine>.npz。

用法：python mimii_banks.py --ch1_dir mimii_ch1 --out mimii_banks --workers 4 [--deadline 秒]
"""
import argparse, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from multiprocessing import Pool
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from kdsj.tfr import set_image_fast, logmel_image

SHARD = 300


def _one(x):
    x = x.astype(np.float64) / 32768
    return set_image_fast(x), logmel_image(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ch1_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--snr', default='0_dB')
    ap.add_argument('--machines', default='fan,pump,slider,valve')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--deadline', type=float, default=0, help='运行多少秒后停止（0 表示不限）')
    a = ap.parse_args()
    t_end = time.time() + a.deadline if a.deadline else float('inf')
    os.makedirs(os.path.join(a.out, 'shards'), exist_ok=True)
    with Pool(a.workers) as pool:
        for m in a.machines.split(','):
            final = os.path.join(a.out, f'{a.snr}_{m}.npz')
            if os.path.exists(final):
                continue
            src = np.load(os.path.join(a.ch1_dir, f'{a.snr}_{m}.npz'), mmap_mode='r')
            n = len(src['state'])
            audio = src['audio']
            shards = []
            for s0 in range(0, n, SHARD):
                f = os.path.join(a.out, 'shards', f'{a.snr}_{m}_{s0:05d}.npz')
                shards.append(f)
                if os.path.exists(f):
                    continue
                if time.time() > t_end - 120:
                    print('到时间，停止'); return
                t0 = time.time()
                res = pool.map(_one, [np.array(audio[i]) for i in range(s0, min(s0 + SHARD, n))], chunksize=4)
                np.savez(f, X_set=np.stack([r[0] for r in res]), X_mel=np.stack([r[1] for r in res]))
                print(f'{m} {s0}-{min(s0 + SHARD, n)} / {n} ({time.time() - t0:.0f} s)', flush=True)
            parts = [np.load(f) for f in shards]
            np.savez_compressed(final, X_set=np.concatenate([p['X_set'] for p in parts]),
                                X_mel=np.concatenate([p['X_mel'] for p in parts]),
                                id=src['id'], state=src['state'], name=src['name'])
            print('完成', final, flush=True)


if __name__ == '__main__':
    main()
