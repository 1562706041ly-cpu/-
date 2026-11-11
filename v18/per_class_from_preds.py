"""Per-class breakdown from ALREADY SAVED predictions (no retraining).

Answers: when `known_false_unk` is ~0.5, is it one whole known class (e.g. all healthy windows) that is
pushed into the unknown cluster, or are errors spread over both classes?
Optionally also breaks the windows down by any metadata field stored in the data NPZ (bearing / record).

Usage:
    python per_class_from_preds.py --exps S_E0 S_N0k S_N0kc2
    python per_class_from_preds.py --exps S_N0k --data data\\paderborn_singlecondition_n15_m07_f10_v1.npz
Evaluation only.
"""
import argparse, glob, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--exps', nargs='+', required=True)
ap.add_argument('--root', default='runs_paderborn')
ap.add_argument('--split', default='PADERBORN')
ap.add_argument('--known', default='0,1')
ap.add_argument('--data', default=None, help='optional: NPZ with per-window metadata for the val set')
a = ap.parse_args()
known = [int(v) for v in a.known.split(',')]
n_known = len(known)

meta = {}
if a.data:
    d = np.load(a.data, allow_pickle=False)
    nv = len(d['y_val'])
    meta = {k: np.asarray(d[k]).astype(str) for k in d.files
            if k not in ('X_train', 'y_train', 'X_val', 'y_val') and d[k].ndim == 1 and len(d[k]) == nv}
    print('val metadata fields:', list(meta) or 'none found')

for exp in a.exps:
    print(f'\n=== {exp}: fraction of validation windows predicted UNKNOWN ===')
    files = sorted(glob.glob(os.path.join(a.root, f'{exp}_seed*', f'pred_{a.split}_seed*.npz')))
    classes = None
    for f in files:
        p = np.load(f, allow_pickle=True)
        y, raw = p['yv'], np.asarray(p['raw_V'])
        pu = raw >= n_known
        if classes is None:
            classes = sorted(np.unique(y).tolist())
            print('seed  ' + '  '.join(f'class{c}(n={int((y == c).sum())})' for c in classes) + '   [known classes: '
                  + ','.join(map(str, known)) + ']')
        seed = f.rsplit('seed', 1)[-1].split('.')[0]
        print(f'{seed:>4}  ' + '  '.join(f'{pu[y == c].mean():>{len(f"class{c}(n={int((y == c).sum())})")}.2f}' for c in classes))
        for key, vals in meta.items():
            print(f'      by {key}: ' + ', '.join(f'{g}={pu[vals == g].mean():.2f}' for g in np.unique(vals)))
print('\nknown classes: value = false-unknown rate; unknown classes: value = unknown recall.')
