"""汇总 mimii_anomaly.py 的结果。

每台设备（16 台）先对种子取平均，再在设备之间做配对比较（双侧精确 Wilcoxon 符号秩检验，n = 16），
这样显著性不依赖种子数。
用法：python mimii_anomaly_summary.py results/mimii_anomaly.csv [--out results/mimii_anomaly_summary.md]
"""
import argparse, csv, itertools
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('csv', nargs='+')
ap.add_argument('--out', default='')
a = ap.parse_args()
rows = []
for f in a.csv:
    rows += list(csv.DictReader(open(f, newline='', encoding='utf-8')))

acc = defaultdict(list)
for r in rows:
    acc[(r['feature'], r['method'], r['machine'], int(r['id']))].append(
        [float(r[k]) for k in ('auc', 'pauc', 'detect', 'false_alarm')])
mean = {k: np.mean(v, 0) for k, v in acc.items()}
nseed = {k: len(v) for k, v in acc.items()}
units = sorted({(k[2], k[3]) for k in mean})
combos = sorted({(k[0], k[1]) for k in mean}, key=lambda c: (c[0], ['gauss', 'ae', 'idclf', 'idmaha'].index(c[1])))
NAMES = {'gauss': '频带统计+马氏距离', 'ae': '自编码器', 'idclf': '设备编号分类', 'idmaha': '编号分类特征+马氏距离'}
MCN = {'fan': '风机', 'pump': '水泵', 'slider': '滑轨', 'valve': '阀门'}


def wilcoxon_p(d):
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    n = d.size
    if n == 0:
        return 1.0
    ab = np.abs(d)
    s = np.sort(ab)
    ranks = np.array([np.mean(np.where(np.isclose(s, v))[0]) + 1 for v in ab])
    obs, mu = ranks[d > 0].sum(), ranks.sum() / 2
    if n > 20:
        from math import erf, sqrt
        z = abs(obs - mu) / np.sqrt((ranks ** 2).sum() / 4)
        return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    W = np.array(list(itertools.product([0, 1], repeat=n))) @ ranks
    return float(np.mean(np.abs(W - mu) >= abs(obs - mu) - 1e-9))


def vec(c, j):
    return np.array([mean[(c[0], c[1], *u)][j] for u in units if (c[0], c[1], *u) in mean])


out = ['# MIMII 异常检测汇总（只用正常声音训练，每台设备单独定阈值）', '',
       '数值为百分比。先对种子取平均，再对 16 台设备取平均（括号内为设备间标准差）。'
       '检出率和误报率是阈值取验证集正常分数 95% 分位数时的结果。', '',
       '| 特征 | 方法 | AUC | pAUC | 检出率 | 误报率 | 种子数 |', '|---|---|---|---|---|---|---|']
for c in combos:
    cells = [f'{100 * vec(c, j).mean():.1f} ({100 * vec(c, j).std(ddof=1):.1f})' for j in range(4)]
    ns = sorted({nseed[(c[0], c[1], *u)] for u in units if (c[0], c[1], *u) in nseed})
    out.append(f'| {c[0]} | {NAMES[c[1]]} | ' + ' | '.join(cells) + f' | {"/".join(map(str, ns))} |')

out += ['', '## 各设备 AUC（%）', '', '| 设备 | ' + ' | '.join(f'{c[0]}/{c[1]}' for c in combos) + ' |',
        '|---' * (len(combos) + 1) + '|']
for u in units:
    out.append(f'| {MCN[u[0]]} id{u[1]:02d} | ' + ' | '.join(
        f'{100 * mean[(c[0], c[1], *u)][0]:.1f}' if (c[0], c[1], *u) in mean else '-' for c in combos) + ' |')

out += ['', '## 配对检验（AUC，按设备配对，n = 16）', '', '| 比较 | 平均差（百分点） | 胜/平/负 | p |', '|---|---|---|---|']
pairs = []
for f in ('mel', 'set'):
    pairs += [((f, 'idclf'), (f, 'ae')), ((f, 'idclf'), (f, 'gauss')), ((f, 'gauss'), (f, 'ae')),
              ((f, 'idclf'), (f, 'idmaha'))]
for m in ('gauss', 'ae', 'idclf', 'idmaha'):
    pairs.append((('mel', m), ('set', m)))
for A, B in pairs:
    if A not in combos or B not in combos:
        continue
    d = vec(A, 0) - vec(B, 0)
    out.append(f'| {A[0]}/{NAMES[A[1]]} vs {B[0]}/{NAMES[B[1]]} | {100 * d.mean():+.2f} | '
               f'{(d > 1e-12).sum()}/{(abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()} | {wilcoxon_p(d):.4f} |')
text = '\n'.join(out) + '\n'
print(text)
if a.out:
    open(a.out, 'w', encoding='utf-8').write(text)
