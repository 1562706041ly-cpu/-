"""汇总 run_bench.py 的结果：每个场景各模型的均值 ± 标准差，以及按种子配对的 Wilcoxon 符号秩检验。

用法：python summarize.py results/bench.csv [更多 csv ...] --out results/bench_summary.md
只依赖 numpy。检验为精确检验（枚举所有符号组合，处理了并列秩），种子数 <= 20 时适用。
"""
import argparse, csv, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('csv', nargs='+')
ap.add_argument('--out', default='')
args = ap.parse_args()

rows = []
for f in args.csv:
    with open(f, newline='') as fh:
        rows += list(csv.DictReader(fh))
MODELS = [m for m in ['cnn', 'cnn_wide', 'resnet', 'eca_resnet'] if any(r['model'] == m for r in rows)]
NAMES = {'cnn': 'CNN（v2 基线）', 'cnn_wide': '加宽 CNN', 'resnet': 'ResNet', 'eca_resnet': 'ECA-ResNet'}
SCN_ORDER = ['clean', 'snr20', 'snr10', 'snr5', 'snr0', 'snr-5']
scns = sorted({r['scenario'] for r in rows},
              key=lambda s: (('_few' in s), SCN_ORDER.index(s.split('+')[0].partition('_few')[0]) if s.split('+')[0].partition('_few')[0] in SCN_ORDER else 99, s))


def signed_rank_p(x, y):
    """双侧精确 Wilcoxon 符号秩检验 p 值（去掉差为 0 的对，并列取平均秩）。"""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.abs(d) > 1e-12]
    n = d.size
    if n == 0:
        return 1.0
    a = np.abs(d); order = a.argsort(); ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(a[order[j + 1]] - a[order[i]]) < 1e-12:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    obs = ranks[d > 0].sum(); mean = ranks.sum() / 2
    if n > 20:
        sdv = np.sqrt((ranks ** 2).sum() / 4)
        from math import erf, sqrt
        z = abs(obs - mean) / sdv
        return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    signs = np.array(list(itertools.product([0, 1], repeat=n)))
    W = signs @ ranks
    return float(np.mean(np.abs(W - mean) >= abs(obs - mean) - 1e-9))


def get(scn, model, col):
    v = sorted(((int(r['seed']), float(r[col])) for r in rows
                if r['scenario'] == scn and r['model'] == model and r.get(col, '') != ''))
    return dict(v)


def paired(a, b):
    s = sorted(set(a) & set(b))
    return [a[k] for k in s], [b[k] for k in s]


def fmt(v):
    v = 100 * np.asarray(list(v.values()))
    return f'{v.mean():.2f} ± {v.std(ddof=1) if v.size > 1 else 0:.2f}' if v.size else '-'


out = ['# 基准测试汇总', '',
       '每格为准确率（%），10 个种子的均值 ± 标准差。只写模型名的列是网络自己的 softmax 输出；"+LSSVM" 列是 fc1 特征 + BKA-LSSVM。"-" 表示该场景没跑这个模型。',
       'test = 20-0 工况时间上最后的 200 个样本；cross = 30-2 工况全部 1000 个样本；xtail = 30-2 工况时间上最后的 200 个样本（+mix/+adabn 场景比较跨工况时看这一列）。', '']
for split in ['test', 'cross', 'xtail']:
    if not any(r.get(f'cnn_{split}_acc', '') for r in rows):
        continue
    out += [f'## {split}', '', '| 场景 | ' + ' | '.join(f'{NAMES[m]} | {NAMES[m]}+LSSVM' for m in MODELS) + ' | 次数 |',
            '|---' * (2 * len(MODELS) + 2) + '|']
    for scn in scns:
        cells, n = [], 0
        for m in MODELS:
            c = get(scn, m, f'cnn_{split}_acc'); l = get(scn, m, f'lssvm_{split}_acc'); n = max(n, len(c))
            cells += [fmt(c), fmt(l)]
        out.append(f'| {scn} | ' + ' | '.join(cells) + f' | {n} |')
    out.append('')

out += ['## 配对检验（test，按种子配对，双侧精确 Wilcoxon 符号秩检验）', '',
        '| 场景 | 比较 | 输出 | 平均差（百分点） | 胜/平/负 | p |', '|---|---|---|---|---|---|']
pairs = [('eca_resnet', 'cnn'), ('resnet', 'cnn'), ('eca_resnet', 'resnet'), ('cnn_wide', 'cnn'), ('resnet', 'cnn_wide')]
for scn in scns:
    for a, b in pairs:
        if a not in MODELS or b not in MODELS:
            continue
        for head in ['cnn', 'lssvm']:
            x, y = paired(get(scn, a, f'{head}_test_acc'), get(scn, b, f'{head}_test_acc'))
            if not x:
                continue
            d = np.array(x) - np.array(y)
            out.append(f'| {scn} | {NAMES[a]} vs {NAMES[b]} | {"softmax" if head == "cnn" else "+LSSVM"} | '
                       f'{100 * d.mean():+.2f} | {(d > 1e-12).sum()}/{(abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()} | '
                       f'{signed_rank_p(x, y):.4f} |')
    for m in MODELS:
        x, y = paired(get(scn, m, 'lssvm_test_acc'), get(scn, m, 'cnn_test_acc'))
        if x:
            d = np.array(x) - np.array(y)
            out.append(f'| {scn} | {NAMES[m]}：+LSSVM vs softmax | - | {100 * d.mean():+.2f} | '
                       f'{(d > 1e-12).sum()}/{(abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()} | {signed_rank_p(x, y):.4f} |')

out += ['', '## 模型规模与训练', '', '| 模型 | 参数量 | 平均训练轮数 | 平均训练时间（s） |', '|---|---|---|---|']
for m in MODELS:
    rs = [r for r in rows if r['model'] == m]
    out.append(f"| {NAMES[m]} | {rs[0]['n_params']} | {np.mean([float(r['epochs']) for r in rs]):.1f} | "
               f"{np.mean([float(r['train_s']) for r in rs]):.1f} |")
text = '\n'.join(out) + '\n'
print(text)
if args.out:
    open(args.out, 'w', encoding='utf-8').write(text)
