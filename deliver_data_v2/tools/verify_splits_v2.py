"""Verify the split files written by make_splits_da_v2.py against the pre-registered expectations (prereg-da-v1).

Usage:
  python verify_splits_v2.py --splits <dir>/splits_da_v2 --pool <dir>/pool_motor2_100_ch1245.npz
         [--phase-jump-dir ...] [--expected-sha <deliver_data_v2>/target_split/split_assignment_r1m_v1.sha256]
Exit code 1 if any check fails.  Only numpy and the standard library are needed.
"""
import argparse
import collections
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# prereg-da-v1, section 2 (T_adapt windows, T_eval windows, T_adapt records, T_eval records)
EXPECT = {'C1_fold1': (491, 309, 15, 11), 'C1_fold3': (485, 315, 14, 11),
          'C2_fold1': (486, 314, 15, 11), 'C2_fold3': (479, 321, 13, 11),
          'C3_fold1': (495, 305, 15, 11), 'C3_fold3': (514, 286, 13, 10),
          'C1_fold2': (511, 289, 13, 10), 'C2_fold2': (532, 268, 13, 10), 'C3_fold2': (547, 253, 13, 9)}
PRIMARY = ['C1_fold1', 'C1_fold3', 'C2_fold1', 'C2_fold3', 'C3_fold1', 'C3_fold3']
SIDE = {0: 'adapt', 1: 'eval'}
fails = []


def check(ok, msg):
    print(('OK   ' if ok else 'FAIL ') + msg)
    if not ok:
        fails.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--phase-jump-dir', default=os.path.join(ROOT, 'tools', 'phase_jump_meta', 'Motor-2_100'))
    ap.add_argument('--expected-sha', default=os.path.join(ROOT, 'target_split', 'split_assignment_r1m_v1.sha256'))
    a = ap.parse_args()
    P = np.load(a.pool)
    n_pool = P['X'].shape[0]
    tsv = ['split\tpool_index\trecording\tside']
    for name in sorted(EXPECT):
        p = os.path.join(a.splits, f'motor2_100_ch1245_{name}.npz')
        if not os.path.exists(p):
            check(False, f'{name}: file missing')
            continue
        S = np.load(p)
        check(str(S['split_rule']) == 'R1m', f'{name}: split_rule is R1m')
        part = S['val_part']
        rec = S['val_recording']
        iev, itr = S['val_pool_index'], S['train_pool_index']
        y = S['y_val']
        na, ne = int((part == 0).sum()), int((part == 1).sum())
        ra = len(set(rec[part == 0].tolist())); re_ = len(set(rec[part == 1].tolist()))
        check((na, ne, ra, re_) == EXPECT[name],
              f'{name}: T_adapt/T_eval windows {na}/{ne}, records {ra}/{re_} (expected {EXPECT[name]})')
        check(len(set(iev.tolist())) == len(iev) and len(set(itr.tolist())) == len(itr), f'{name}: no duplicate pool index')
        check(not (set(iev.tolist()) & set(itr.tolist())), f'{name}: source and target windows are disjoint')
        both = set(rec[part == 0].tolist()) & set(rec[part == 1].tolist())
        check(not both, f'{name}: no recording key on both sides')
        # nominal measurement and phase-jump measurement must not be shared across sides either
        inst = S['val_installation']
        nom = [(i, int(m)) for i, m in zip(inst, S['val_measurement'])]
        pjm = [(i, int(m)) for i, m in zip(inst, S['val_pj_measurement'])]
        for label, keys in (('nominal measurement', nom), ('phase-jump measurement', pjm)):
            sides = collections.defaultdict(set)
            for k, s in zip(keys, part.tolist()):
                sides[k].add(s)
            check(all(len(v) == 1 for v in sides.values()), f'{name}: no {label} on both sides')
        feas = bool(S['split_feasible'])
        check(feas == (name in PRIMARY), f'{name}: split_feasible = {feas}')
        if name in PRIMARY:
            ok = True
            for lab in range(5):
                m = y == lab
                ok &= bool(((part[m] == 0).any()) and ((part[m] == 1).any()))
            check(ok, f'{name}: every one of the 5 target classes has windows on both sides')
        # metadata consistency with the pool
        check((P['installation'][iev] == S['val_installation']).all() and (P['column'][iev] == S['val_column']).all(),
              f'{name}: val metadata equals pool rows')
        check(np.array_equal(S['X_val'], P['X'][iev]) and np.array_equal(S['X_train'], P['X'][itr]),
              f'{name}: X_val / X_train equal pool rows')
        if 'X_val_logspec' in S.files:
            check(S['X_val_logspec'].shape == (len(iev), 4, 1025) and S['X_train_logspec'].shape == (len(itr), 4, 1025),
                  f'{name}: logspec arrays have the expected shape')
        for i, r, s in zip(iev.tolist(), rec.tolist(), part.tolist()):
            tsv.append(f'{name}\t{i}\t{r}\t{SIDE[s]}')
    body = '\n'.join(tsv) + '\n'
    sha = hashlib.sha256(body.encode('utf-8')).hexdigest()
    print('assignment sha256 (rebuilt from the npz files):', sha)
    if os.path.exists(a.expected_sha):
        exp = open(a.expected_sha, encoding='utf-8').read().split()[0].strip().lower()
        check(sha == exp, f'assignment sha256 equals the pre-registered value {exp}')
    else:
        print('expected-sha file not found: comparison skipped')
    print('ALL CHECKS PASSED' if not fails else f'{len(fails)} CHECK(S) FAILED')
    raise SystemExit(1 if fails else 0)


if __name__ == '__main__':
    main()
