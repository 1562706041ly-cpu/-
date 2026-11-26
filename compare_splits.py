"""Fallback for a failed regression check (A1 section 7 item 3): are X_train / X_val (and labels) of the old split
files (splits_ch1245) and the v2 split files (splits_da_v2) element-wise identical?
Usage: python compare_splits.py <dir_splits_ch1245> <dir_splits_da_v2> [name_stem ...]   (default: the 9 files)
Needs numpy only. Prints one line per file and per array; exit 1 if anything differs."""
import hashlib, os, sys
import numpy as np

old, new = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or [f'motor2_100_ch1245_{c}_fold{f}' for c in ('C1', 'C2', 'C3') for f in (1, 2, 3)]
bad = False
for n in names:
    po, pn = os.path.join(old, n + '.npz'), os.path.join(new, n + '.npz')
    if not (os.path.exists(po) and os.path.exists(pn)):
        print(n, 'MISSING', po if not os.path.exists(po) else pn); bad = True; continue
    do, dn = np.load(po, allow_pickle=False), np.load(pn, allow_pickle=False)
    for k in ('X_train', 'y_train', 'X_val', 'y_val'):
        if k not in do.files or k not in dn.files:
            print(n, k, 'KEY MISSING'); bad = True; continue
        same = do[k].shape == dn[k].shape and np.array_equal(do[k], dn[k])
        info = ''
        if not same and do[k].shape == dn[k].shape and k.startswith('X'):
            # same windows in another order?
            ho = sorted(hashlib.sha1(np.ascontiguousarray(w).tobytes()).hexdigest() for w in do[k])
            hn = sorted(hashlib.sha1(np.ascontiguousarray(w).tobytes()).hexdigest() for w in dn[k])
            info = ' (same windows in a different order)' if ho == hn else ' (different windows)'
        print(f'{n} {k} {"IDENTICAL" if same else "DIFFERENT"}{info}')
        bad |= not same
print('RESULT:', 'ALL IDENTICAL' if not bad else 'DIFFERENCES FOUND')
sys.exit(1 if bad else 0)
