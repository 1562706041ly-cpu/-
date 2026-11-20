"""MIMII 工业设备声音识别实验（使用 kdsj 软件包本身的识别器）。

任务：8 类 = 4 种设备（风机、水泵、滑轨、阀门）× 2 种状态（正常、异常）。每个 10 s 录音一张时频图。
数据：MIMII 0 dB 子集第 1 通道（Hitachi, Ltd., CC BY-SA 4.0, https://zenodo.org/records/3384388），
      录音中已混入真实工厂背景噪声。

划分方式（--protocol）：
  file : 每台设备、每种状态的录音按文件编号排序，前 60% 训练、中间 20% 验证、最后 20% 测试
  id   : 设备编号 00/02/04 训练（每组前 80% 训练、后 20% 验证），编号 06 全部用于测试
         （测试的是没见过的同型号另一台设备，类似"换一台设备还能不能认出来"）
特征（--features）：set（同步提取变换）| mel（对数梅尔谱）

除了 8 类准确率，还报告"告警"指标（只看正常/异常，不管设备类型）：
  检出率 = 异常录音被判为任意"异常"类的比例；误报率 = 正常录音被判为任意"异常"类的比例。

结果逐行追加到 CSV，中断后重跑会跳过已完成的组。

本机（RTX 2060）用法：
  python mimii_bench.py --banks mimii_banks --device cuda
  python mimii_bench.py --banks mimii_banks --device cuda --protocol id --features set --models resnet --seeds 1-3
"""
import argparse, csv, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--banks', default='mimii_banks')
ap.add_argument('--out', default='results/mimii_bench.csv')
ap.add_argument('--device', default='cuda')
ap.add_argument('--threads', type=int, default=4, help='CPU 线程上限（防止笔记本卡顿）')
ap.add_argument('--protocol', default='file,id')
ap.add_argument('--features', default='set,mel')
ap.add_argument('--models', default='cnn,resnet')
ap.add_argument('--seeds', default='1-5')
ap.add_argument('--no_lssvm', action='store_true')
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--patience', type=int, default=8)
ap.add_argument('--save_dir', default='models', help='每种 (划分, 特征, 网络) 保存第一个完成的模型，给界面用；设为空字符串则不保存')
args = ap.parse_args()
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = str(args.threads)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
from kdsj.recognizer import Recognizer, metrics

torch.set_num_threads(args.threads)
MACHINES = [('fan', '风机'), ('pump', '水泵'), ('slider', '滑轨'), ('valve', '阀门')]
CLASSES = [f'{cn}-{s}' for _, cn in MACHINES for s in ('正常', '异常')]
FIELDS = ['protocol', 'feature', 'model', 'seed', 'epochs', 'train_s', 'n_train', 'n_val', 'n_test',
          'net_acc', 'net_bal_acc', 'net_f1', 'net_detect', 'net_false_alarm',
          'lssvm_acc', 'lssvm_bal_acc', 'lssvm_f1', 'lssvm_detect', 'lssvm_false_alarm',
          'gam', 'sig2', 'cv_error']


def parse_seeds(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def load(banks):
    X = {'set': [], 'mel': []}
    y, ids = [], []
    for mi, (m, _) in enumerate(MACHINES):
        d = np.load(os.path.join(banks, f'0_dB_{m}.npz'))
        X['set'].append(d['X_set']); X['mel'].append(d['X_mel'])
        y.append(2 * mi + d['state'].astype(int)); ids.append(d['id'].astype(int))
    return {k: np.concatenate(v) for k, v in X.items()}, np.concatenate(y), np.concatenate(ids)


def split(y, ids, protocol):
    tr, va, te = [], [], []
    for c in np.unique(y):
        for i in np.unique(ids[y == c]):
            idx = np.where((y == c) & (ids == i))[0]          # 已按文件编号排序
            n = idx.size
            if protocol == 'file':
                a, b = int(0.6 * n), int(0.8 * n)
                tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
            elif i == 6:
                te += list(idx)
            else:
                a = int(0.8 * n)
                tr += list(idx[:a]); va += list(idx[a:])
    return map(np.array, (tr, va, te))


def alarm_rates(y, p):
    abn_true = y % 2 == 1
    abn_pred = (p >= 0) & (p % 2 == 1)
    return float(abn_pred[abn_true].mean()), float(abn_pred[~abn_true].mean())


def main():
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline='', encoding='utf-8') as fh:
            done = {(r['protocol'], r['feature'], r['model'], int(r['seed'])) for r in csv.DictReader(fh)}
    X, y, ids = load(args.banks)
    dev = args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    print(f'共 {len(y)} 个录音，设备 {dev}，已完成 {len(done)} 组', flush=True)
    for proto in args.protocol.split(','):
        itr, iva, ite = split(y, ids, proto)
        for seed in parse_seeds(args.seeds):
            for feat in args.features.split(','):
                for model in args.models.split(','):
                    if (proto, feat, model, seed) in done:
                        continue
                    t0 = time.time()
                    rec = Recognizer(CLASSES, model, dev, epochs=args.epochs, patience=args.patience,
                                     use_lssvm=not args.no_lssvm, class_weight='balanced',
                                     feature={'kind': feat, 'win_s': 10.0, 'fs': 16000},
                                     alarm_classes=[c for c in CLASSES if c.endswith('异常')])
                    rec.fit(X[feat][itr], y[itr], X[feat][iva], y[iva], seed=seed, log=lambda s: None)
                    F, P = rec._forward(X[feat][ite])
                    pn = P.argmax(1)
                    mn = metrics(y[ite], pn, len(CLASSES))
                    dn, fn = alarm_rates(y[ite], pn)
                    row = dict(protocol=proto, feature=feat, model=model, seed=seed, epochs=rec.history['epochs'],
                               train_s=round(rec.history['train_s'], 1), n_train=len(itr), n_val=len(iva),
                               n_test=len(ite), net_acc=mn['acc'], net_bal_acc=mn['bal_acc'], net_f1=mn['macro_f1'],
                               net_detect=dn, net_false_alarm=fn)
                    if rec.lssvm is not None:
                        pl = rec.lssvm.predict(F)
                        ml = metrics(y[ite], pl, len(CLASSES))
                        dl, fl = alarm_rates(y[ite], pl)
                        row.update(lssvm_acc=ml['acc'], lssvm_bal_acc=ml['bal_acc'], lssvm_f1=ml['macro_f1'],
                                   lssvm_detect=dl, lssvm_false_alarm=fl, gam=rec.lssvm.gam,
                                   sig2=rec.lssvm.sig2, cv_error=rec.lssvm.cv_error)
                    new = not os.path.exists(args.out)
                    with open(args.out, 'a', newline='', encoding='utf-8') as fh:
                        w = csv.DictWriter(fh, FIELDS, extrasaction='ignore')
                        if new:
                            w.writeheader()
                        w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in row.items()})
                    if args.save_dir:
                        os.makedirs(args.save_dir, exist_ok=True)
                        mp = os.path.join(args.save_dir, f'{proto}_{feat}_{model}.pt')
                        if not os.path.exists(mp):
                            rec.save(mp)
                    msg = (f'[{proto:4s}|{feat}] {model:6s} seed {seed}: 网络 {100 * mn["acc"]:.1f}% '
                           f'(平衡 {100 * mn["bal_acc"]:.1f}%, 检出 {100 * dn:.1f}%, 误报 {100 * fn:.1f}%)')
                    if rec.lssvm is not None:
                        msg += f' | +LSSVM {100 * ml["acc"]:.1f}% (平衡 {100 * ml["bal_acc"]:.1f}%)'
                    print(msg + f'  {rec.history["epochs"]} 轮, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
