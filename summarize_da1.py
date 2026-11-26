"""Collect E0 / E0' results into a CSV and a text table. Usage: python summarize_da1.py [--root runs_paderborn] [--out summary_da1.csv]
Primary metric (A1-D7): known_acc / unk_recall / H of KR.openset_metrics on T_eval. Secondary: known_recall_mean,
known_false_unk, unknown clustering accuracy ('new') and NMI of M2 (fixed known centres)."""
import argparse, csv, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='runs_paderborn'); ap.add_argument('--out', default='summary_da1.csv')
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.root, 'DA1_E0*', 'results_exp2.json'))):
    tag = os.path.basename(os.path.dirname(f))
    m = re.match(r'DA1_(E0p?)_(C\d)_f(\d)_seed(\d+)$', tag)
    if not m: continue
    r = json.load(open(f, encoding='utf-8'))
    for key, v in r.items():
        o, mm = v['openset']['V'], v['M2_GCD+半监督KMeans']['V']
        rows.append(dict(tag=tag, method=m.group(1), combo=m.group(2), fold=int(m.group(3)), seed=int(key.split('seed')[1]),
                         n_eval=v['n_val'], n_unlabeled=v['n_unlabeled'], KNO=o['known_acc'], UNK=o['unk_recall'], H=o['H'],
                         KNO_class_mean=o.get('known_recall_mean'), known_false_unk=o['known_false_unk'],
                         unk_cluster_acc=mm['new'], NMI=mm['nmi']))
cols = ['tag', 'method', 'combo', 'fold', 'seed', 'n_eval', 'n_unlabeled', 'KNO', 'UNK', 'H', 'KNO_class_mean', 'known_false_unk', 'unk_cluster_acc', 'NMI']
with open(a.out, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} runs -> {a.out}')
for meth in ('E0', 'E0p'):
    for c in ('C1', 'C2', 'C3'):
        for fo in (1, 3):
            hs = [r['H'] for r in rows if r['method'] == meth and r['combo'] == c and r['fold'] == fo]
            if hs: print(f'{meth:4s} {c} fold{fo}: n={len(hs)} H mean={sum(hs) / len(hs):.4f} ' + ' '.join(f'{h:.4f}' for h in hs))
