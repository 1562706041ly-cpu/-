"""Materialise the 9 GCD splits (C1-C3 x fold 1-3) from the shared multi-channel pool. Cheap: only copies arrays.

Usage:  python make_splits_from_pool.py [--pool pool/pool_motor2_100_ch1245.npz] [--out splits_ch1245]
Installations / classes / folds / combinations are exactly those of build_splits.py (fixed before any result).
Output NPZ (readable by m2_paderborn_kr.py, 3-D input): X_train (N, 4, 2048), y_train, X_val, y_val + metadata
  {train,val}_installation, _fault_type, _severity, _measurement, _column, _start
"""
import argparse, json, os, hashlib
import numpy as np

TYPES = {'healthy': ['healthy 1', 'healthy 2', 'healthy 3'],
         'impeller': ['impeller 1', 'impeller 2', 'impeller 3'],
         'pump bearing': ['bearing pump 1', 'bearing pump 2', 'bearing pump 3'],
         'motor BPFO': ['bearing bpfo 1', 'bearing bpfo 2', 'bearing bpfo 3'],
         'motor BPFI': ['bearing bpfi 1', 'bearing bpfi 2', 'bearing bpfi 3'],
         'soft foot': ['soft foot 1', 'soft foot 2']}
COMBOS = {'C1': dict(known_fault='motor BPFO', unknown=['impeller', 'pump bearing', 'soft foot']),
          'C2': dict(known_fault='motor BPFI', unknown=['impeller', 'pump bearing', 'motor BPFO']),
          'C3': dict(known_fault='soft foot', unknown=['impeller', 'pump bearing', 'motor BPFI'])}
META = ('installation', 'fault_type', 'severity', 'measurement', 'column', 'start')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=os.path.join('pool', 'pool_motor2_100_ch1245.npz'))
    ap.add_argument('--out', default='splits_ch1245')
    a = ap.parse_args()
    man = json.load(open(a.pool[:-4] + '.json'))
    sha = hashlib.sha256(open(a.pool, 'rb').read()).hexdigest()
    if sha != man['sha256']:
        raise SystemExit(f'pool sha256 mismatch ({sha}); re-assemble the pool file')
    P = np.load(a.pool)
    X, inst = P['X'], P['installation']
    os.makedirs(a.out, exist_ok=True)
    for cname, cfg in COMBOS.items():
        used = ['healthy', cfg['known_fault']] + cfg['unknown']
        for k in (1, 2, 3):
            ev = [TYPES[t][(k - 1) % len(TYPES[t])] for t in used]
            tr = [n for t in used for n in TYPES[t] if n not in ev]
            assert not set(ev) & set(tr)
            def part(names):
                idx = np.concatenate([np.where(inst == n)[0] for n in names])
                y = np.array([used.index(str(P['fault_type'][i])) for i in idx], np.int64)
                return idx, y
            itr, ytr = part(tr); iev, yev = part(ev)
            out = os.path.join(a.out, f'motor2_100_ch1245_{cname}_fold{k}.npz')
            np.savez(out, X_train=X[itr], y_train=ytr, X_val=X[iev], y_val=yev, class_names=np.array(used),
                     known_labels=np.array([0, 1]), unknown_labels=np.array([2, 3, 4]), channels=P['channels'],
                     sample_rate_hz=P['sample_rate_hz'], window_size=P['window_size'],
                     train_pool_index=itr.astype(np.int64), val_pool_index=iev.astype(np.int64),
                     **{f'train_{m}': P[m][itr] for m in META}, **{f'val_{m}': P[m][iev] for m in META})
            print(f'{out}: train {X[itr].shape} ({len(tr)} inst), eval {X[iev].shape} | eval = {ev}')


if __name__ == '__main__':
    main()
