"""汇总 mimii_bench.py 的结果：均值 ± 标准差，以及按种子配对的双侧精确 Wilcoxon 符号秩检验。

用法：python mimii_summary.py results/mimii_bench.csv [--out results/mimii_summary.md]
"""
import argparse, csv, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('csv', nargs='+')
ap.add_argument('--out', default='')
a = ap.parse_args()
rows = []
for f in a.csv:
    rows += list(csv.DictReader(open(f, newline='', encoding='utf-8')))


def wilcoxon_p(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.abs(d) > 1e-12]
    n = d.size
    if n == 0:
        return 1.0
    ab = np.abs(d)
    ranks = np.array([np.mean(np.where(np.isclose(np.sort(ab), v))[0]) + 1 for v in ab])
    obs, mean = ranks[d > 0].sum(), ranks.sum() / 2
    W = np.array(list(itertools.product([0, 1], repeat=n))) @ ranks
    return float(np.mean(np.abs(W - mean) >= abs(obs - mean) - 1e-9))


def vals(p, f, m, col):
    return {int(r['seed']): float(r[col]) for r in rows
            if r['protocol'] == p and r['feature'] == f and r['model'] == m and r.get(col, '') != ''}


def ms(v):
    v = 100 * np.array(list(v.values()))
    return f'{v.mean():.1f} ± {v.std(ddof=1):.1f}' if v.size > 1 else (f'{v[0]:.1f}' if v.size else '-')


cols = [('net_acc', '网络 准确率'), ('lssvm_acc', '+LSSVM 准确率'), ('lssvm_bal_acc', '+LSSVM 平衡准确率'),
        ('lssvm_detect', '+LSSVM 检出率'), ('lssvm_false_alarm', '+LSSVM 误报率')]
out = ['# MIMII 实验汇总', '', '数值为百分比，均值 ± 标准差（按种子）。', '']
combos = sorted({(r['protocol'], r['feature'], r['model']) for r in rows})
out += ['| 划分 | 特征 | 网络 | 次数 | ' + ' | '.join(c[1] for c in cols) + ' |', '|---' * (4 + len(cols)) + '|']
for p, f, m in combos:
    out.append(f'| {p} | {f} | {m} | {len(vals(p, f, m, "net_acc"))} | ' +
               ' | '.join(ms(vals(p, f, m, c)) for c, _ in cols) + ' |')
out += ['', '## 配对检验（+LSSVM 准确率，按种子配对）', '', '| 划分 | 比较 | 平均差（百分点） | 胜/平/负 | p |', '|---|---|---|---|---|']


def comp(p, A, B, col='lssvm_acc', lab=None):
    x, y = vals(p, *A, col), vals(p, *B, col)
    k = sorted(set(x) & set(y))
    if len(k) < 2:
        return
    d = np.array([x[i] - y[i] for i in k])
    out.append(f'| {p} | {lab or f"{A[0]}/{A[1]} vs {B[0]}/{B[1]}"} | {100 * d.mean():+.2f} | '
               f'{(d > 1e-12).sum()}/{(abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()} | '
               f'{wilcoxon_p([x[i] for i in k], [y[i] for i in k]):.4f} |')


for p in sorted({c[0] for c in combos}):
    for f in ('set', 'mel'):
        comp(p, (f, 'resnet'), (f, 'cnn'))
    for m in ('cnn', 'resnet'):
        comp(p, ('set', m), ('mel', m))
    for f in ('set', 'mel'):
        for m in ('cnn', 'resnet'):
            x, y = vals(p, f, m, 'lssvm_acc'), vals(p, f, m, 'net_acc')
            k = sorted(set(x) & set(y))
            if len(k) >= 2:
                d = np.array([x[i] - y[i] for i in k])
                out.append(f'| {p} | {f}/{m}：+LSSVM vs 网络 | {100 * d.mean():+.2f} | '
                           f'{(d > 1e-12).sum()}/{(abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()} | '
                           f'{wilcoxon_p([x[i] for i in k], [y[i] for i in k]):.4f} |')
text = '\n'.join(out) + '\n'
print(text)
if a.out:
    open(a.out, 'w', encoding='utf-8').write(text)
