"""Materialise the 9 splits (C1-C3 x fold 1-3) with the target T_adapt / T_eval assignment of rule R1-merged (R1m).

Replaces make_splits_da_v1.py (which only accepts rule R1).  Same installations / classes / folds / combinations,
same keys and file names (motor2_100_ch1245_{C}_fold{k}.npz) as v1, so existing code reads the files unchanged.

Usage:
  python make_splits_da_v2.py --pool <dir>/pool_motor2_100_ch1245.npz --out <dir>/splits_da_v2
         [--logspec <dir>/pool_motor2_100_ch1245_logspec.npy]
         [--phase-jump-dir <deliver_data_v2>/tools/phase_jump_meta/Motor-2_100]
         [--target-split <deliver_data_v2>/target_split/target_split_v1.json]

Rule R1-merged (pre-registered, prereg-da-v1, decision D2):
  Per installation, the pool columns that occur in the pool are nodes.  Two columns are linked if they share the
  nominal measurement (column // 5) OR the phase-jump measurement (measurement_of_column).  Connected components are
  the recordings; recording key = installation + '|m' + component index, components numbered by their smallest
  column.  Recordings are sorted by index; even rank -> T_adapt, odd rank -> T_eval.  No random numbers.
  A split is feasible iff every target installation has >= 2 recordings.

Extra keys compared with the original make_splits_from_pool.py:
  val_recording        (N_val,) str   recording key of every val (target) window
  val_part             (N_val,) int8  0 = T_adapt (unlabelled adaptation data), 1 = T_eval (evaluation only)
  val_pj_measurement   (N_val,) int   phase-jump measurement number of the column of every val window
  train_recording      (N_train,) str recording key of every train (source) window
  split_rule           str            'R1m'
  split_feasible       bool           False if some target installation has a single recording
  X_train_logspec, X_val_logspec (only with --logspec) (N, 4, 1025) float32
Also writes <out>/split_assignment_r1m_v1.tsv  (split, pool_index, recording key, side) for all val windows.
"""
import argparse
import collections
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
ROOT = os.path.dirname(HERE)
SIDE = {0: 'adapt', 1: 'eval'}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def r1m_components(cols, pj):
    """component index of every used column (cols: iterable of ints; pj: measurement_of_column list)"""
    cols = sorted(set(int(c) for c in cols))
    parent = {c: c for c in cols}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in cols:
        for b in cols:
            if a < b and (a // 5 == b // 5 or pj[a] == pj[b]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = collections.defaultdict(list)
    for c in cols:
        groups[find(c)].append(c)
    order = sorted(groups, key=lambda r: min(groups[r]))
    return {c: i for i, r in enumerate(order) for c in groups[r]}


def r1m_assignment(inst, col, pjdir):
    """record key (str), side (0 adapt / 1 eval), phase-jump measurement and number of records per installation"""
    n = len(inst)
    key = np.empty(n, dtype=object)
    side = np.zeros(n, np.int8)
    pjm = np.zeros(n, np.int64)
    nrec = {}
    for name in sorted(set(inst.tolist())):
        idx = np.where(inst == name)[0]
        pj = json.load(open(os.path.join(pjdir, f'{name}.json'), encoding='utf-8'))['measurement_of_column']
        cmap = r1m_components(col[idx].tolist(), pj)
        nrec[name] = len(set(cmap.values()))
        for i in idx:
            r = cmap[int(col[i])]
            key[i] = f'{name}|m{r}'
            side[i] = r % 2
            pjm[i] = pj[int(col[i])]
    return key.astype(str), side, pjm, nrec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=os.path.join('pool', 'pool_motor2_100_ch1245.npz'))
    ap.add_argument('--out', default='splits_da_v2')
    ap.add_argument('--phase-jump-dir', default=os.path.join(ROOT, 'tools', 'phase_jump_meta', 'Motor-2_100'))
    ap.add_argument('--target-split', default=os.path.join(ROOT, 'target_split', 'target_split_v1.json'),
                    help='cross-check against option_R1_merged of this json (skipped if the file does not exist)')
    ap.add_argument('--logspec', default=None)
    a = ap.parse_args()
    man = json.load(open(a.pool[:-4] + '.json', encoding='utf-8'))
    sha = sha256_file(a.pool)
    if sha != man['sha256']:
        raise SystemExit(f'pool sha256 mismatch ({sha}); re-assemble the pool file')
    LS = None
    if a.logspec:
        lman = json.load(open(a.logspec[:-4] + '.json', encoding='utf-8'))
        lsha = sha256_file(a.logspec)
        if lsha != lman['sha256'] or lman['pool_sha256'] != sha:
            raise SystemExit(f'logspec sha256 mismatch ({lsha}) or logspec made from another pool')
        LS = np.load(a.logspec, mmap_mode='r')
    P = np.load(a.pool)
    X, inst, col = P['X'], np.array([str(x) for x in P['installation']]), P['column'].astype(np.int64)
    if not (P['measurement'].astype(np.int64) == col // 5).all():
        raise SystemExit('pool field measurement is not column // 5')
    if LS is not None and LS.shape[0] != X.shape[0]:
        raise SystemExit('logspec and pool have different numbers of windows')

    rec, side, pjm, nrec = r1m_assignment(inst, col, a.phase_jump_dir)
    if os.path.exists(a.target_split):
        T = json.load(open(a.target_split, encoding='utf-8'))
        if T['pool_sha256'] != sha:
            raise SystemExit('target_split json was made from a different pool file')
        bad = 0
        for name, pi in T['option_R1_merged']['per_installation'].items():
            for i, k, s in zip(pi['pool_index'], pi['window_recording_key'], pi['window_side']):
                if rec[i] != k or SIDE[int(side[i])] != s:
                    bad += 1
        if bad:
            raise SystemExit(f'{bad} windows differ between this script and option_R1_merged of {a.target_split}')
        print('cross-check with option_R1_merged of the json: 2720 windows, 0 differences')
    else:
        print('target_split json not found: cross-check skipped')

    os.makedirs(a.out, exist_ok=True)
    tsv = []
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
            itr, ytr = part(tr)
            iev, yev = part(ev)
            feasible = all(nrec[n] >= 2 for n in ev)
            val_part = side[iev]
            extra = dict(val_recording=rec[iev], val_part=val_part, val_pj_measurement=pjm[iev],
                         train_recording=rec[itr], split_rule=np.array('R1m'), split_feasible=np.array(feasible))
            if LS is not None:
                extra.update(X_train_logspec=np.asarray(LS[itr]), X_val_logspec=np.asarray(LS[iev]))
            out = os.path.join(a.out, f'motor2_100_ch1245_{cname}_fold{k}.npz')
            np.savez(out, X_train=X[itr], y_train=ytr, X_val=X[iev], y_val=yev, class_names=np.array(used),
                     known_labels=np.array([0, 1]), unknown_labels=np.array([2, 3, 4]), channels=P['channels'],
                     sample_rate_hz=P['sample_rate_hz'], window_size=P['window_size'],
                     train_pool_index=itr.astype(np.int64), val_pool_index=iev.astype(np.int64),
                     **{f'train_{m}': P[m][itr] for m in META}, **{f'val_{m}': P[m][iev] for m in META}, **extra)
            for i in iev:
                tsv.append(f'{cname}_fold{k}\t{int(i)}\t{rec[i]}\t{SIDE[int(side[i])]}')
            na, ne = int((val_part == 0).sum()), int((val_part == 1).sum())
            print(f'{out}: train {X[itr].shape}, eval {X[iev].shape} T_adapt/T_eval {na}/{ne} '
                  f'feasible={feasible} | eval = {ev}')
    tsv_path = os.path.join(a.out, 'split_assignment_r1m_v1.tsv')
    with open(tsv_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('split\tpool_index\trecording\tside\n' + '\n'.join(tsv) + '\n')
    print(f'wrote {tsv_path} sha256 {sha256_file(tsv_path)}')


if __name__ == '__main__':
    main()
