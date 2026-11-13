"""Summarise runs produced by m2_paderborn_kr.py (ASCII output, safe on any Windows console).

Usage (per-seed tags such as E0_seed0 ... E0_seed5):
    python summarize_runs.py --exps E0 E3 --baseline E0
    python summarize_runs.py --exps E0 E3 E3k --baseline E0 --root runs_paderborn --csv summary.csv

Reports per-seed open-set metrics, mean +- population std, and PAIRED differences against the
baseline (same seed => same labelled windows). Std over seeds is NOT a confidence interval.
"""
import argparse, csv, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--exps', nargs='+', required=True)
ap.add_argument('--baseline', default=None)
ap.add_argument('--root', default='runs_paderborn')
ap.add_argument('--split', default='PADERBORN')
ap.add_argument('--csv', default=None)
a = ap.parse_args()

METRICS = ['known_acc', 'unk_recall', 'known_false_unk', 'unk_precision', 'H']
DIAG = ['auroc_K_unknown_vs_known', 'auroc_R_correct_routing', 'zone_reliable_unknown_unknown_frac',
        'zone_reliable_known_unknown_frac', 'label_neg_precision', 'label_pos_precision',
        'final_auroc_K', 'init_zone_unknown_frac', 'expand_known_frac', 'expand_class_correct',
        'false_unk_class0', 'false_unk_class1', 'auto_unk_chosen', 'guard_reassigned_val']


def load(exp):
    out = {}
    for f in sorted(Path(a.root).glob(f'{exp}_seed*/results_exp2.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        for key, r in d.items():
            seed = int(key.split('seed')[-1])
            row = dict(r['openset']['V'])
            if r.get('kr_log'):
                row.update({k: v for k, v in r['kr_log'][-1].items() if k in DIAG})
            fk = r.get('final_kr_diag_train_EVAL_ONLY')
            if fk:
                row['final_auroc_K'] = fk.get('auroc_K_unknown_vs_known')
            ke = r.get('known_expand')
            if ke:
                row['expand_known_frac'] = ke.get('selected_known_frac_EVAL_ONLY')
                row['expand_class_correct'] = ke.get('selected_class_correct_EVAL_ONLY')
            au = r.get('auto_unknown')
            if au:
                row['auto_unk_chosen'] = float(au.get('chosen'))
            kg = r.get('known_guard')
            if kg:
                row['guard_reassigned_val'] = float(kg.get('reassigned_val'))
            ui = r.get('unk_init')
            if ui:
                row['init_zone_unknown_frac'] = ui.get('zone_unknown_frac_EVAL_ONLY')
            row['_labeled'] = tuple(r.get('labeled_idx', []))
            out[seed] = row
    return out


def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v: return float('nan'), float('nan')
    m = sum(v) / len(v)
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


data = {e: load(e) for e in a.exps}
rows = []
for e, runs in data.items():
    print(f'\n=== {e}  (n_seeds={len(runs)}) ===')
    print('seed  ' + '  '.join(f'{m:>15s}' for m in METRICS))
    for s in sorted(runs):
        print(f'{s:>4d}  ' + '  '.join(f'{runs[s].get(m, float("nan")):>15.3f}' for m in METRICS))
        rows.append(dict(exp=e, seed=s, **{m: runs[s].get(m) for m in METRICS + DIAG}))
    print('mean  ' + '  '.join('%15s' % ('%.3f+-%.3f' % ms([runs[s].get(m) for s in runs])) for m in METRICS))
    diag = [d for d in DIAG if any(d in runs[s] for s in runs)]
    for d in diag:
        print(f'  {d}: %.3f+-%.3f' % ms([runs[s].get(d) for s in runs]))
    collapsed = [s for s in sorted(runs) if runs[s].get('unk_recall', 1) == 0]
    if collapsed: print(f'  seeds with unknown recall == 0 (cluster collapse): {collapsed}')

if a.baseline and a.baseline in data:
    base = data[a.baseline]
    print(f'\n=== paired differences vs {a.baseline} (method - baseline) ===')
    for e in a.exps:
        if e == a.baseline: continue
        common = sorted(set(base) & set(data[e]))
        same_lab = all(base[s]['_labeled'] == data[e][s]['_labeled'] for s in common)
        print(f'{e}: {len(common)} paired seeds, identical labelled windows: {same_lab}')
        for m in METRICS:
            diffs = [data[e][s][m] - base[s][m] for s in common
                     if not (math.isnan(data[e][s][m]) or math.isnan(base[s][m]))]
            if not diffs: continue
            mu, sd = ms(diffs)
            better = sum(d > 0 for d in diffs) if m != 'known_false_unk' else sum(d < 0 for d in diffs)
            print(f'   {m:>16s}: mean diff {mu:+.3f} (std {sd:.3f}), better in {better}/{len(diffs)} seeds,'
                  f' per-seed: ' + ' '.join(f'{d:+.2f}' for d in diffs))

if a.csv:
    keys = ['exp', 'seed'] + METRICS + DIAG
    with open(a.csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f'\nwrote {a.csv}')
