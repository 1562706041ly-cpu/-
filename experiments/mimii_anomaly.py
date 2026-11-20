"""MIMII 异常检测实验：每台设备只用正常声音训练，检测这台设备的异常。

划分（每台设备，即每个 设备类型 × 编号）：
  正常录音按文件编号排序，前 60% 训练、中间 20% 定阈值（验证）、最后 20% 测试；异常录音全部用于测试。
  训练和定阈值只用正常录音，异常录音只在测试时出现。
方法（--methods）：
  gauss  : 频带统计 + 马氏距离（不需要 GPU，几秒钟）
  ae     : 自编码器（MIMII 官方基线的思路）
  idclf  : 同类型 4 台设备的正常声音训练 ResNet 区分设备编号，分数 = -log p(真实编号)
  idmaha : 同一个网络的 fc1 特征 + 每台设备的马氏距离（和 idclf 一起算，不额外训练）
特征（--features）：mel | set
指标（每台设备）：AUC、pAUC（误报率 ≤ 10%）、阈值取验证集正常分数 95% 分位数时的检出率和误报率。

结果逐行写入 CSV，中断后重跑会跳过已完成的组。
用法：python mimii_anomaly.py --banks ..\\mimii_banks --device cuda
"""
import argparse, csv, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--banks', default='mimii_banks')
ap.add_argument('--out', default='results/mimii_anomaly.csv')
ap.add_argument('--device', default='cuda')
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--features', default='mel,set')
ap.add_argument('--methods', default='gauss,ae,idclf')
ap.add_argument('--machines', default='fan,pump,slider,valve')
ap.add_argument('--seeds', default='1-5')
ap.add_argument('--q', type=float, default=0.95, help='阈值取验证集正常分数的分位数')
args = ap.parse_args()
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = str(args.threads)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
from kdsj.anomaly import GaussianDetector, AEDetector, IDClassifierDetector, auc, pauc

torch.set_num_threads(args.threads)
DEV = args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
FIELDS = ['machine', 'id', 'feature', 'method', 'seed', 'auc', 'pauc', 'detect', 'false_alarm',
          'n_train', 'n_val', 'n_test_normal', 'n_test_abnormal', 'time_s']


def parse_seeds(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def split_normal(idx):
    n = idx.size
    return idx[:int(0.6 * n)], idx[int(0.6 * n):int(0.8 * n)], idx[int(0.8 * n):]


def evaluate(s_val, s_te_n, s_te_a):
    thr = np.quantile(s_val, args.q)
    return dict(auc=auc(s_te_n, s_te_a), pauc=pauc(s_te_n, s_te_a), detect=float(np.mean(s_te_a > thr)),
                false_alarm=float(np.mean(s_te_n > thr)))


def main():
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline='', encoding='utf-8') as fh:
            done = {(r['machine'], int(r['id']), r['feature'], r['method'], int(r['seed'])) for r in csv.DictReader(fh)}

    def write(row):
        new = not os.path.exists(args.out)
        with open(args.out, 'a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in row.items()})
        print(f"{row['machine']:6s} id{row['id']:02d} {row['feature']} {row['method']:6s} seed {row['seed']}: "
              f"AUC {100 * row['auc']:.1f}  pAUC {100 * row['pauc']:.1f}  检出 {100 * row['detect']:.1f}%  "
              f"误报 {100 * row['false_alarm']:.1f}%  ({row['time_s']:.0f} s)", flush=True)

    methods = args.methods.split(',')
    print(f'设备 {DEV}，已完成 {len(done)} 组', flush=True)
    for m in args.machines.split(','):
        d = np.load(os.path.join(args.banks, f'0_dB_{m}.npz'))
        ids, state = d['id'].astype(int), d['state'].astype(int)
        parts = {}
        for i in np.unique(ids):
            tr, va, te = split_normal(np.where((ids == i) & (state == 0))[0])
            parts[i] = (tr, va, te, np.where((ids == i) & (state == 1))[0])
        for feat in args.features.split(','):
            X = d['X_mel'] if feat == 'mel' else d['X_set']
            for seed in parse_seeds(args.seeds):
                for meth in ('gauss', 'ae'):
                    if meth not in methods:
                        continue
                    for i, (tr, va, te, ab) in parts.items():
                        if (m, i, feat, meth, seed) in done:
                            continue
                        if meth == 'gauss' and seed != 1:          # 高斯模型没有随机性，只算一次
                            continue
                        t0 = time.time()
                        det = GaussianDetector() if meth == 'gauss' else AEDetector(device=DEV)
                        det.fit(X[tr], seed=seed)
                        r = evaluate(det.score(X[va]), det.score(X[te]), det.score(X[ab]))
                        write(dict(machine=m, id=i, feature=feat, method=meth, seed=seed, **r, n_train=len(tr),
                                   n_val=len(va), n_test_normal=len(te), n_test_abnormal=len(ab),
                                   time_s=time.time() - t0))
                if 'idclf' in methods and any((m, i, feat, 'idclf', seed) not in done for i in parts):
                    t0 = time.time()
                    tr = np.concatenate([p[0] for p in parts.values()])
                    va = np.concatenate([p[1] for p in parts.values()])
                    det = IDClassifierDetector(device=DEV).fit(X[tr], ids[tr], X[va], ids[va], seed=seed)
                    dt = time.time() - t0
                    for meth in ('idclf', 'idmaha'):
                        for i, (tr_i, va_i, te_i, ab_i) in parts.items():
                            sc = lambda idx: det.score(X[idx], ids[idx], meth)
                            r = evaluate(sc(va_i), sc(te_i), sc(ab_i))
                            write(dict(machine=m, id=i, feature=feat, method=meth, seed=seed, **r, n_train=len(tr_i),
                                       n_val=len(va_i), n_test_normal=len(te_i), n_test_abnormal=len(ab_i),
                                       time_s=dt / len(parts)))


if __name__ == '__main__':
    main()
