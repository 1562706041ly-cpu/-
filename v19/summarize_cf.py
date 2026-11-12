"""Summarise the centrifugal-pump GCD runs (tags CF_<combo>_f<fold>_<method>_seed<s>). ASCII-only output.

Usage:  python summarize_cf.py [--root runs_paderborn] [--out summary_cf.txt] [--csv summary_cf.csv]
Reports, per method: mean +- std over (combo, fold, seed); per-combo means; per unknown-class recall;
where unknown windows go (healthy / known fault / unknown); paired differences vs E0 and E0b
(same combo, fold and seed). Std over runs is NOT a confidence interval.
"""
import argparse, csv, glob, json, math, os, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='runs_paderborn')
ap.add_argument('--out', default='summary_cf.txt')
ap.add_argument('--csv', default='summary_cf.csv')
ap.add_argument('--baselines', default='E0,E0b,B', help='comma-separated methods used as paired baselines')
ap.add_argument('--prefix', default='CF', help='tag prefix, e.g. CF or V16')
a = ap.parse_args()

TAG = re.compile(a.prefix + r'_(C\d)_f(\d)_([A-Za-z0-9]+)_seed(\d+)$')
METRICS = ['known_acc', 'unk_recall', 'known_false_unk', 'H', 'unk_cluster_acc', 'nmi',
           'false_unk_class0', 'false_unk_class1', 'unk_recall_class2', 'unk_recall_class3', 'unk_recall_class4',
           'unk_to_healthy', 'unk_to_known_fault']
rows = []
for f in sorted(glob.glob(os.path.join(a.root, a.prefix + '_*_seed*', 'results_exp2.json'))):
    m = TAG.search(os.path.basename(os.path.dirname(f)))
    if not m: continue
    combo, fold, method, seed = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    r = list(json.load(open(f, encoding='utf-8')).values())[0]
    o = dict(r['openset']['V'])
    o['unk_cluster_acc'] = r['M2_GCD+半监督KMeans']['V']['new']          # Hungarian among unknown clusters
    o['nmi'] = r['M2_GCD+半监督KMeans']['V']['nmi']
    p = np.load(os.path.join(os.path.dirname(f), f'pred_PADERBORN_seed{seed}.npz'), allow_pickle=True)
    y, c = p['yv'], np.asarray(p['raw_V'])
    u = y >= 2
    o['unk_to_healthy'] = float((c[u] == 0).mean()); o['unk_to_known_fault'] = float((c[u] == 1).mean())
    rows.append(dict(combo=combo, fold=fold, method=method, seed=seed, **{k: o.get(k, float('nan')) for k in METRICS}))

lines = []
P = lambda s='': lines.append(s)
ms = lambda v: (float(np.nanmean(v)), float(np.nanstd(v))) if len(v) else (float('nan'), float('nan'))
ORDER = ['E0', 'E0b', 'KNN', 'DE', 'B', 'A3g', 'A3d', 'A3', 'A3c', 'A3nm', 'A3ns', 'R4', 'F']
methods = sorted({r['method'] for r in rows}, key=lambda x: ORDER.index(x) if x in ORDER else 99)
P(f'runs found: {len(rows)}  methods: {methods}')
P('combo labels: 0 healthy, 1 known fault, 2..4 unknown faults (see centrifugal/splits/manifest.json)')
P('  C1: known BPFO       | unknown 2 impeller, 3 pump bearing, 4 soft foot')
P('  C2: known BPFI       | unknown 2 impeller, 3 pump bearing, 4 motor BPFO')
P('  C3: known soft foot  | unknown 2 impeller, 3 pump bearing, 4 motor BPFI')
P()
head = ['known_acc', 'unk_recall', 'known_false_unk', 'H', 'unk_cluster_acc']
P('=== overall (mean +- std over combo x fold x seed)')
P(f"{'method':8s}{'n':>4s}  " + '  '.join(f'{h:>15s}' for h in head))
for me in methods:
    R = [r for r in rows if r['method'] == me]
    P(f'{me:8s}{len(R):4d}  ' + '  '.join('%15s' % ('%.3f+-%.3f' % ms([r[h] for r in R])) for h in head))
P()
P('=== per combo (mean over folds x seeds)')
for combo in sorted({r['combo'] for r in rows}):
    P(f'-- {combo}')
    for me in methods:
        R = [r for r in rows if r['method'] == me and r['combo'] == combo]
        if R: P(f'   {me:8s}' + '  '.join(f'{h}={ms([r[h] for r in R])[0]:.3f}' for h in head))
P()
P('=== per class (mean over everything); unknown classes 2..4 = recall, known 0/1 = false-unknown rate')
for me in methods:
    R = [r for r in rows if r['method'] == me]
    P(f'   {me:8s}' + '  '.join(f'{h.replace("unk_recall_class", "rec").replace("false_unk_class", "fu")}={ms([r[h] for r in R])[0]:.3f}'
                               for h in ['false_unk_class0', 'false_unk_class1', 'unk_recall_class2', 'unk_recall_class3', 'unk_recall_class4']))
P()
P('=== where do unknown windows go? (fraction of unknown windows predicted healthy / known fault), by fold')
P('    fold k evaluates healthy k and severity k; healthy 1 was measured right before impeller 1')
for me in methods:
    for k in (1, 2, 3):
        R = [r for r in rows if r['method'] == me and r['fold'] == k]
        if R: P(f'   {me:8s} fold{k}: to_healthy={ms([r["unk_to_healthy"] for r in R])[0]:.3f}  to_known_fault={ms([r["unk_to_known_fault"] for r in R])[0]:.3f}')
P()
for base in a.baselines.split(','):
    if base not in methods: continue
    P(f'=== paired differences vs {base} (same combo, fold, seed; method - baseline)')
    B = {(r['combo'], r['fold'], r['seed']): r for r in rows if r['method'] == base}
    for me in methods:
        if me == base: continue
        pairs = [(r, B[(r['combo'], r['fold'], r['seed'])]) for r in rows if r['method'] == me and (r['combo'], r['fold'], r['seed']) in B]
        if not pairs: continue
        P(f'-- {me}: {len(pairs)} pairs')
        for h in head:
            d = np.array([x[h] - y[h] for x, y in pairs]); d = d[~np.isnan(d)]
            better = (d < 0).sum() if h == 'known_false_unk' else (d > 0).sum()
            P(f'   {h:>16s}: mean {d.mean():+.3f} (std {d.std():.3f}), better in {better}/{len(d)}')
    P()
txt = '\n'.join(lines); print(txt)
open(a.out, 'w', encoding='utf-8').write(txt + '\n')
with open(a.csv, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=['combo', 'fold', 'method', 'seed'] + METRICS); w.writeheader(); w.writerows(rows)
