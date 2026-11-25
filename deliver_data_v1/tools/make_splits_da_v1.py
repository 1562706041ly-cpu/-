"""Materialise the 9 splits (C1-C3 x fold 1-3) with the target T_adapt / T_eval assignment (rule R1).

Usage:
  python make_splits_da_v1.py --pool <dir>/pool_motor2_100_ch1245.npz --out <dir>/splits_da_v1
         [--target-split <deliver_data_v1>/target_split/target_split_v1.json]
         [--logspec <dir>/pool_motor2_100_ch1245_logspec.npy]

Same installations / classes / folds / combinations and the same keys and file names
(motor2_100_ch1245_{C}_fold{k}.npz) as centrifugal/make_splits_from_pool.py, so existing code reads the files
unchanged. Extra keys:
  val_recording   (N_val,) str   recording key 'installation|measurement' of every val (target) window
  val_part        (N_val,) int8  0 = T_adapt (unlabelled adaptation data), 1 = T_eval (evaluation only)
  train_recording (N_train,) str recording key of every train (source) window
  split_rule      str            'R1'
  split_feasible  bool           False if some target class has windows on one side only (see check report)
  X_train_logspec, X_val_logspec (only with --logspec) (N, 4, 1025) float32 log spectrum of the windows
"""
import argparse
import hashlib
import json
import os

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
HERE = os.path.dirname(os.path.abspath(__file__))
PART = {'adapt': 0, 'eval': 1}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=os.path.join('pool', 'pool_motor2_100_ch1245.npz'))
    ap.add_argument('--out', default='splits_da_v1')
    ap.add_argument('--target-split', default=os.path.join(os.path.dirname(HERE), 'target_split',
                                                           'target_split_v1.json'))
    ap.add_argument('--logspec', default=None)
    a = ap.parse_args()
    man = json.load(open(a.pool[:-4] + '.json', encoding='utf-8'))
    sha = sha256_file(a.pool)
    if sha != man['sha256']:
        raise SystemExit(f'pool sha256 mismatch ({sha}); re-assemble the pool file')
    T = json.load(open(a.target_split, encoding='utf-8'))
    if T['pool_sha256'] != sha:
        raise SystemExit('target_split json was made from a different pool file')
    if T['rule_version'] != 'R1':
        raise SystemExit(f'unexpected rule version {T["rule_version"]}')
    tsplit = {s['split']: s for s in T['splits']}
    LS = None
    if a.logspec:
        lman = json.load(open(a.logspec[:-4] + '.json', encoding='utf-8'))
        lsha = sha256_file(a.logspec)
        if lsha != lman['sha256'] or lman['pool_sha256'] != sha:
            raise SystemExit(f'logspec sha256 mismatch ({lsha}) or logspec made from another pool')
        LS = np.load(a.logspec, mmap_mode='r')
    P = np.load(a.pool)
    X, inst = P['X'], P['installation']
    if LS is not None and LS.shape[0] != X.shape[0]:
        raise SystemExit('logspec and pool have different numbers of windows')
    rec = np.array([f'{i}|{int(m)}' for i, m in zip(inst, P['measurement'])])
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
            s = tsplit[f'{cname}_fold{k}']
            if [int(i) for i in iev] != s['target_pool_index'] or s['class_names'] != used:
                raise SystemExit(f'{cname}_fold{k}: target_split json does not match the split definition')
            val_rec = np.array(s['target_recording_key'])
            if not (val_rec == rec[iev]).all():
                raise SystemExit(f'{cname}_fold{k}: recording keys in json differ from pool metadata')
            val_part = np.array([PART[x] for x in s['target_side']], np.int8)
            extra = dict(val_recording=val_rec, val_part=val_part, train_recording=rec[itr],
                         split_rule=np.array('R1'), split_feasible=np.array(s['verdict'] == 'FEASIBLE'))
            if LS is not None:
                extra.update(X_train_logspec=np.asarray(LS[itr]), X_val_logspec=np.asarray(LS[iev]))
            out = os.path.join(a.out, f'motor2_100_ch1245_{cname}_fold{k}.npz')
            np.savez(out, X_train=X[itr], y_train=ytr, X_val=X[iev], y_val=yev, class_names=np.array(used),
                     known_labels=np.array([0, 1]), unknown_labels=np.array([2, 3, 4]), channels=P['channels'],
                     sample_rate_hz=P['sample_rate_hz'], window_size=P['window_size'],
                     train_pool_index=itr.astype(np.int64), val_pool_index=iev.astype(np.int64),
                     **{f'train_{m}': P[m][itr] for m in META}, **{f'val_{m}': P[m][iev] for m in META}, **extra)
            na, ne = int((val_part == 0).sum()), int((val_part == 1).sum())
            print(f'{out}: train {X[itr].shape}, eval {X[iev].shape} T_adapt/T_eval {na}/{ne} '
                  f'feasible={s["verdict"] == "FEASIBLE"} | eval = {ev}')


if __name__ == '__main__':
    main()
