"""Sandbox-side check of the deliver_data_v1 folder. Exit code 0 = all checks passed.

Usage:  python verify_delivery.py [--scratch <dir for merged files and rebuilt splits>]

Checks:
  1. every file in deliver_data_v1 is < 100 MB (100 000 000 bytes)
  2. every part matches size and sha256 in centrifugal/parts_sha256.json and no unlisted part exists
  3. concatenated parts reproduce the pool sha256 and the VMD bank_sha256 of the json manifests
  4. log spectrum sha256 matches its manifest; recomputation from the merged pool gives the same values
  5. target_split_check.py rerun gives byte-identical target_split json / csv / report
  6. make_splits_da_v1.py (with --logspec) writes 9 splits; for each: all original keys present and equal to
     the arrays of the original split builder (centrifugal/make_splits_from_pool.py logic), extra keys present,
     X_train / X_val / logspec slices equal pool rows, val_recording = installation|measurement,
     T_adapt / T_eval recordings disjoint, feasible splits have every class on both sides, split_feasible and
     per-class counts equal target_split_v1.json
  7. merge_data_v1.ps1 is pure ASCII; user documents contain no em dash
  8. nothing in the delivery is or references 75 % speed data
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True           # keep tools/__pycache__ out of the delivery
sys.path.insert(0, HERE)
import build_logspec_pool as BL          # noqa: E402
import make_splits_da_v1 as MS           # noqa: E402

LIMIT = 100_000_000
ORIG_KEYS = {'X_train', 'y_train', 'X_val', 'y_val', 'class_names', 'known_labels', 'unknown_labels', 'channels',
             'sample_rate_hz', 'window_size', 'train_pool_index', 'val_pool_index'} | \
            {f'{p}_{m}' for p in ('train', 'val') for m in MS.META}
EXTRA_KEYS = {'val_recording', 'val_part', 'train_recording', 'split_rule', 'split_feasible'}
LS_KEYS = {'X_train_logspec', 'X_val_logspec'}
SEALED = re.compile(r'Motor-2[_/ ]75|Motor-2/75|_75_time|/75/|SEALED_test', re.I)

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)
    return cond


def concat(parts, out):
    h = hashlib.sha256()
    with open(out, 'wb') as o:
        for p in parts:
            with open(os.path.join(ROOT, p['file']), 'rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    h.update(block)
                    o.write(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', default='/projects/sandbox/.scratch_data/verify_delivery')
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    files = sorted(os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, '/')
                   for d, _, fs in os.walk(ROOT) for f in fs if '__pycache__' not in d)

    # 1 sizes
    big = [(f, os.path.getsize(os.path.join(ROOT, f))) for f in files if os.path.getsize(os.path.join(ROOT, f)) >= LIMIT]
    check(not big, f'all {len(files)} files < 100 MB' + (f' (too big: {big})' if big else ''))

    # 2 + 3 parts
    PM = json.load(open(os.path.join(ROOT, 'centrifugal', 'parts_sha256.json'), encoding='utf-8'))
    listed = {p['file'] for e in ('pool', 'vmd') for p in PM[e]['parts']}
    on_disk = {f for f in files if re.search(r'\.part\d+$', f)}
    check(listed == on_disk, f'parts on disk == parts listed ({len(listed)})')
    merged = {}
    for e in ('pool', 'vmd'):
        ent = PM[e]
        ok = True
        for p in ent['parts']:
            fp = os.path.join(ROOT, p['file'])
            ok &= os.path.getsize(fp) == p['size_bytes'] and BL.sha256_file(fp) == p['sha256']
        check(ok, f'{e}: size and sha256 of {len(ent["parts"])} parts match parts_sha256.json')
        man = json.load(open(os.path.join(ROOT, ent['manifest']), encoding='utf-8'))
        out = os.path.join(a.scratch, os.path.basename(ent['target']))
        got = concat(ent['parts'], out)
        check(got == man[ent['manifest_sha256_key']] == ent['sha256'],
              f'{e}: concatenated parts sha256 {got} == manifest {man[ent["manifest_sha256_key"]]}')
        check(os.path.getsize(out) == ent['size_bytes'], f'{e}: merged size {ent["size_bytes"]}')
        merged[e] = out
    pool = merged['pool']
    import shutil
    shutil.copy(os.path.join(ROOT, 'centrifugal', 'pool', 'pool_motor2_100_ch1245.json'), pool[:-4] + '.json')
    vman = json.load(open(os.path.join(ROOT, 'centrifugal', 'pool', 'pool_motor2_100_ch1245_vmd.json')))
    pman = json.load(open(pool[:-4] + '.json'))
    check(vman['pool_sha256'] == pman['sha256'], 'VMD manifest refers to the delivered pool')
    V = np.load(merged['vmd'], mmap_mode='r')
    check(list(V.shape) == vman['vmd_shape'] and str(V.dtype) == vman['dtype'], f'VMD bank shape {V.shape} {V.dtype}')

    # 4 logspec
    lsp = os.path.join(ROOT, 'centrifugal', 'pool', 'pool_motor2_100_ch1245_logspec.npy')
    LM = json.load(open(lsp[:-4] + '.json', encoding='utf-8'))
    check(BL.sha256_file(lsp) == LM['sha256'] and LM['pool_sha256'] == pman['sha256'],
          'logspec sha256 matches manifest and manifest refers to the delivered pool')
    S = np.load(lsp)
    check(S.shape == (2720, 4, 1025) and S.dtype == np.float32 and list(S.shape) == LM['shape'],
          f'logspec shape {S.shape} {S.dtype}')
    tc = LM['torch_check']
    check(tc.get('done') and tc['max_abs_diff_vs_torch_float64'] <= 1e-4,
          f'manifest torch check: max abs diff vs torch float64 {tc.get("max_abs_diff_vs_torch_float64")}, '
          f'vs torch float32 {tc.get("max_abs_diff_vs_torch_float32")}')
    P = np.load(pool)
    X = P['X']
    R = np.empty_like(S)
    z = BL.zn(X)
    for i in range(0, len(z), 256):
        R[i:i + 256] = BL.log_spectrum_np(z[i:i + 256])
    d = float(np.abs(R - S).max())
    check(d <= 1e-6, f'logspec recomputed from merged pool: max abs diff {d:.3g}')
    del z, R

    # 5 target split rerun
    ts_out = os.path.join(a.scratch, 'target_split')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'target_split_check.py'), '--pool', pool,
                        '--out', ts_out], capture_output=True, text=True)
    check(r.returncode == 0, 'target_split_check.py rerun exit 0' + ('' if r.returncode == 0 else r.stderr[-2000:]))
    for f in ('target_split_v1.json', 'target_split_v1.csv', 'check_report_v1.md'):
        check(BL.sha256_file(os.path.join(ts_out, f)) == BL.sha256_file(os.path.join(ROOT, 'target_split', f)),
              f'rerun {f} byte-identical to delivered file')
    T = json.load(open(os.path.join(ROOT, 'target_split', 'target_split_v1.json'), encoding='utf-8'))

    # 6 splits
    sp_out = os.path.join(a.scratch, 'splits_da_v1')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'make_splits_da_v1.py'), '--pool', pool, '--out', sp_out,
                        '--target-split', os.path.join(ROOT, 'target_split', 'target_split_v1.json'),
                        '--logspec', lsp], capture_output=True, text=True)
    check(r.returncode == 0, 'make_splits_da_v1.py exit 0' + ('' if r.returncode == 0 else r.stderr[-2000:]))
    inst = np.array([str(v) for v in P['installation']])
    rec = np.array([f'{i}|{int(m)}' for i, m in zip(inst, P['measurement'])])
    n_feas = 0
    for s in T['splits']:
        name = s['split']
        f = os.path.join(sp_out, s['split_file'])
        if not check(os.path.exists(f), f'{name}: file {s["split_file"]} written'):
            continue
        Z = np.load(f)
        keys = set(Z.files)
        check(ORIG_KEYS | EXTRA_KEYS | LS_KEYS == keys, f'{name}: keys = original + extra + logspec')
        # original builder logic, recomputed here independently
        c, k = name.split('_fold')
        cfg = MS.COMBOS[c]
        used = ['healthy', cfg['known_fault']] + cfg['unknown']
        ev = [MS.TYPES[t][(int(k) - 1) % len(MS.TYPES[t])] for t in used]
        tr = [n for t in used for n in MS.TYPES[t] if n not in ev]
        itr = np.concatenate([np.where(inst == n)[0] for n in tr])
        iev = np.concatenate([np.where(inst == n)[0] for n in ev])
        ok = (np.array_equal(Z['train_pool_index'], itr) and np.array_equal(Z['val_pool_index'], iev)
              and np.array_equal(Z['X_train'], X[itr]) and np.array_equal(Z['X_val'], X[iev])
              and np.array_equal(Z['X_train_logspec'], S[itr]) and np.array_equal(Z['X_val_logspec'], S[iev])
              and Z['class_names'].tolist() == used
              and np.array_equal(Z['y_val'], [used.index(str(P['fault_type'][i])) for i in iev])
              and np.array_equal(Z['y_train'], [used.index(str(P['fault_type'][i])) for i in itr])
              and all(np.array_equal(Z[f'{p}_{m}'], P[m][ix]) for p, ix in (('train', itr), ('val', iev))
                      for m in MS.META))
        check(ok, f'{name}: indices, X, logspec, labels and metadata equal the pool rows')
        vr, vp = Z['val_recording'], Z['val_part']
        ok = (np.array_equal(vr, rec[iev]) and np.array_equal(Z['train_recording'], rec[itr])
              and vp.dtype == np.int8 and set(vp.tolist()) <= {0, 1} and str(Z['split_rule']) == 'R1'
              and Z['split_feasible'].dtype == np.bool_)
        check(ok, f'{name}: recording keys, val_part int8 in {{0,1}}, split_rule R1')
        ra, re_ = set(vr[vp == 0].tolist()), set(vr[vp == 1].tolist())
        check(not ra & re_, f'{name}: T_adapt and T_eval recordings disjoint ({len(ra)} / {len(re_)})')
        both = all((Z['y_val'][vp == 0] == lab).any() and (Z['y_val'][vp == 1] == lab).any() for lab in range(5))
        feas = bool(Z['split_feasible'])
        check(feas == both == (s['verdict'] == 'FEASIBLE'),
              f'{name}: split_feasible={feas}, every class on both sides={both}, report verdict {s["verdict"]}')
        n_feas += feas
        ok = True
        for cl in s['per_class']:
            m = Z['y_val'] == cl['label']
            ok &= int((m & (vp == 0)).sum()) == cl['n_win_adapt'] and int((m & (vp == 1)).sum()) == cl['n_win_eval']
            ok &= len(set(vr[m & (vp == 0)].tolist())) == cl['n_rec_adapt']
            ok &= len(set(vr[m & (vp == 1)].tolist())) == cl['n_rec_eval']
        ok &= int((vp == 0).sum()) == s['totals']['n_win_adapt'] and int((vp == 1).sum()) == s['totals']['n_win_eval']
        check(ok, f'{name}: per-class and total counts equal target_split_v1.json '
                  f'(T_adapt {int((vp == 0).sum())}, T_eval {int((vp == 1).sum())})')
    check(len(T['splits']) == 9, f'9 splits checked, {n_feas} feasible')

    # 7 ASCII / em dash
    ps = open(os.path.join(ROOT, 'merge_data_v1.ps1'), 'rb').read()
    check(all(b < 128 for b in ps), 'merge_data_v1.ps1 is pure ASCII')
    for f in ('README_data_v1.md', 'target_split/check_report_v1.md'):
        t = open(os.path.join(ROOT, f), encoding='utf-8').read()
        check('\u2014' not in t and '\u2013' not in t, f'{f}: no em / en dash')

    # 8 sealed 75 % data
    hits = [f for f in files if SEALED.search(f)]
    for f in files:
        if f == 'tools/verify_delivery.py':      # contains the search pattern itself
            continue
        if f.endswith(('.md', '.py', '.json', '.csv', '.ps1', '.txt')):
            if SEALED.search(open(os.path.join(ROOT, f), encoding='utf-8').read()):
                hits.append(f)
    check(not hits, 'no file name or text references 75 % speed data paths' + (f' ({hits})' if hits else ''))
    pj_dir = os.path.join(HERE, 'phase_jump_meta')
    setups = {json.load(open(os.path.join(d, f), encoding='utf-8'))['setup']
              for d, _, fs in os.walk(pj_dir) for f in fs if f.endswith('.json')}
    check(setups == {'Motor-2_100'} and os.listdir(pj_dir) == ['Motor-2_100'],
          f'phase-jump metadata only from setup {sorted(setups)}')
    check(pman['sha256'] == '4be16646c108805062fcc19c523b346cafd7fad812262a3cdfcea8418157afa4',
          'pool is the 100 % speed development pool of v19-B')

    print(f'\n{"ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED"}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
