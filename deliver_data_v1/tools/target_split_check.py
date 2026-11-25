"""Target-split feasibility check and T_adapt / T_eval assignment (rule R1), metadata only.

Usage:
  python target_split_check.py --pool <path>/pool_motor2_100_ch1245.npz
         [--pj-dir <dir with Motor-2_100/<installation>.json>] [--out <output dir>]

Reads only the pool metadata arrays (installation, fault_type, severity, measurement, column, start).
The waveform array X is never loaded. No random numbers, no model output. Two runs give identical files.

Rule R1 (fixed before looking at any counts):
  The assignment is made per target installation and depends only on that installation's metadata.
  recordings = distinct pool 'measurement' values among the installation's pool windows, sorted ascending
  as integers; recording at rank r (0-based) -> T_adapt if r is even, T_eval if r is odd.
  Every window inherits the side of its recording. Recording key = installation + '|' + measurement.

Outputs (in --out): target_split_v1.json, target_split_v1.csv, check_report_v1.md
"""
import argparse
import hashlib
import json
import os

import numpy as np

# ---- copied verbatim from centrifugal/make_splits_from_pool.py ----
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
# --------------------------------------------------------------------

RULE_VERSION = 'R1'
RULE_TEXT = ("R1: assignment per target installation, metadata only. recordings = distinct pool 'measurement' "
             "values among the installation's pool windows, sorted ascending as integers; the recording at rank r "
             "(0-based) goes to T_adapt if r is even and to T_eval if r is odd. Every window inherits the side of "
             "its recording. Recording key = installation + '|' + measurement. A split is FEASIBLE iff every target "
             "installation has >= 2 recordings (then each class has >= 1 recording on each side).")
WARN_MIN_WINDOWS = 16          # 10 % of 160
WARN_SHARE = (0.25, 0.75)      # allowed T_adapt share of a class's target windows
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_meta(pool):
    z = np.load(pool, allow_pickle=False)
    meta = {k: z[k] for k in META}          # lazy npz: 'X' is never read
    meta['installation'] = np.array([str(v) for v in meta['installation']])
    meta['fault_type'] = np.array([str(v) for v in meta['fault_type']])
    return meta


def r1_sides(rec_ids):
    """rec_ids: sequence of int recording ids. Returns sorted unique ids and {id: side}."""
    uniq = sorted({int(v) for v in rec_ids})
    return uniq, {r: ('adapt' if rank % 2 == 0 else 'eval') for rank, r in enumerate(uniq)}


def warnings_for(n_adapt, n_eval):
    w = []
    n = n_adapt + n_eval
    if min(n_adapt, n_eval) < WARN_MIN_WINDOWS:
        w.append(f'side with < {WARN_MIN_WINDOWS} windows (adapt {n_adapt}, eval {n_eval})')
    share = n_adapt / n if n else 0.0
    if not (WARN_SHARE[0] <= share <= WARN_SHARE[1]):
        w.append(f'T_adapt share {share:.4f} outside [{WARN_SHARE[0]}, {WARN_SHARE[1]}]')
    return w


def assign_installation(name, idx, rec_ids, key_prefix):
    """Apply the even/odd rule to one installation. rec_ids aligned with idx."""
    uniq, side_of = r1_sides(rec_ids)
    rec_ids = np.asarray(rec_ids, np.int64)
    recs = []
    for r in uniq:
        recs.append({'key': f'{name}|{key_prefix}{r}', 'id': r, 'n_windows': int((rec_ids == r).sum()),
                     'side': side_of[r]})
    sides = [side_of[int(r)] for r in rec_ids]
    keys = [f'{name}|{key_prefix}{int(r)}' for r in rec_ids]
    n_wa = sum(s == 'adapt' for s in sides)
    n_we = len(sides) - n_wa
    return {'n_windows': len(idx), 'n_recordings': len(uniq),
            'n_rec_adapt': sum(x['side'] == 'adapt' for x in recs),
            'n_rec_eval': sum(x['side'] == 'eval' for x in recs),
            'n_win_adapt': n_wa, 'n_win_eval': n_we,
            'adapt_share': round(n_wa / len(idx), 4),
            'both_sides': len(uniq) >= 2, 'warnings': warnings_for(n_wa, n_we),
            'recordings': recs, '_sides': sides, '_keys': keys}


def public(d):
    return {k: v for k, v in d.items() if not k.startswith('_')}


def option_A(name, idx, col):
    """Column-level (12-s block) even/odd rule over sorted column ids."""
    return assign_installation(name, idx, col[idx], 'c')


def option_B(name, idx, col, start, meas):
    """Time halves inside each recording: windows ordered by (column, start, pool_index); first ceil(n/2) -> adapt."""
    sides = {}
    boundaries = []
    for r in sorted({int(v) for v in meas[idx]}):
        sub = [i for i in idx if int(meas[i]) == r]
        sub.sort(key=lambda i: (int(col[i]), int(start[i]), int(i)))
        h = (len(sub) + 1) // 2
        for j, i in enumerate(sub):
            sides[int(i)] = 'adapt' if j < h else 'eval'
        if 0 < h < len(sub):
            a, e = sub[h - 1], sub[h]
            same_col = int(col[a]) == int(col[e])
            gap = int(start[e]) - (int(start[a]) + 2048) if same_col else None
            boundaries.append({'recording': f'{name}|{r}', 'last_adapt': [int(col[a]), int(start[a])],
                               'first_eval': [int(col[e]), int(start[e])], 'same_column': same_col,
                               'gap_samples_if_same_column': gap})
    n_wa = sum(sides[int(i)] == 'adapt' for i in idx)
    n_we = len(idx) - n_wa
    cols_a = sorted({int(col[i]) for i in idx if sides[int(i)] == 'adapt'})
    cols_e = sorted({int(col[i]) for i in idx if sides[int(i)] == 'eval'})
    return {'n_windows': len(idx), 'n_win_adapt': n_wa, 'n_win_eval': n_we,
            'adapt_share': round(n_wa / len(idx), 4), 'columns_adapt': cols_a, 'columns_eval': cols_e,
            'columns_on_both_sides': sorted(set(cols_a) & set(cols_e)), 'boundaries': boundaries,
            'warnings': warnings_for(n_wa, n_we)}


def pj_check(name, idx, col, meas, pj_dir):
    path = os.path.join(pj_dir, f'{name}.json')
    d = json.load(open(path, encoding='utf-8'))
    moc = d['measurement_of_column']
    cols = col[idx]
    if int(cols.max()) >= len(moc):
        raise SystemExit(f'{path}: measurement_of_column too short for column {int(cols.max())}')
    pj = np.array([int(moc[int(c)]) for c in cols], np.int64)
    nom = meas[idx].astype(np.int64)
    pairs = sorted({(int(a), int(b)) for a, b in zip(nom, pj)})
    nom_of_pj, pj_of_nom = {}, {}
    for a, b in pairs:
        nom_of_pj.setdefault(b, set()).add(a)
        pj_of_nom.setdefault(a, set()).add(b)
    same_partition = all(len(v) == 1 for v in nom_of_pj.values()) and all(len(v) == 1 for v in pj_of_nom.values())
    pj_refines_nominal = all(len(v) == 1 for v in nom_of_pj.values())
    return pj, {'json_file': f'{name}.json', 'json_sha256': sha256_file(path),
                'n_columns_json': int(d['n_columns']), 'n_measurements_json': int(d['n_measurements']),
                'n_columns_used': int(len(set(cols.tolist()))),
                'n_nominal_measurements': int(len(set(nom.tolist()))),
                'n_phase_jump_measurements_used': int(len(set(pj.tolist()))),
                'nominal_equals_column_div_5': bool((nom == cols // 5).all()),
                'same_partition': bool(same_partition),
                'phase_jump_refines_nominal': bool(pj_refines_nominal)}


def merged_components(cols, nom, pj):
    """R1-merged recording ids. cols, nom, pj aligned per window of one installation.
    Two pool columns are linked if they share the nominal measurement (column // 5) OR the phase-jump
    measurement_of_column. Returns (component id per window, list of column lists per component);
    components are numbered in order of their smallest column."""
    ucols = sorted({int(c) for c in cols})
    parent = {c: c for c in ucols}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nom_of_col = {int(c): int(m) for c, m in zip(cols, nom)}
    pj_of_col = {int(c): int(p) for c, p in zip(cols, pj)}
    for attr in (nom_of_col, pj_of_col):
        first = {}
        for c in ucols:
            v = attr[c]
            if v in first:
                union(first[v], c)
            else:
                first[v] = c
    groups = {}
    for c in ucols:
        groups.setdefault(find(c), []).append(c)
    comps = sorted(groups.values(), key=lambda g: g[0])
    comp_of_col = {c: k for k, g in enumerate(comps) for c in g}
    return np.array([comp_of_col[int(c)] for c in cols], np.int64), comps


def build_splits(inst):
    out = []
    for cname, cfg in COMBOS.items():
        used = ['healthy', cfg['known_fault']] + cfg['unknown']
        for k in (1, 2, 3):
            ev = [TYPES[t][(k - 1) % len(TYPES[t])] for t in used]
            tr = [n for t in used for n in TYPES[t] if n not in ev]
            assert not set(ev) & set(tr)
            iev = np.concatenate([np.where(inst == n)[0] for n in ev])
            itr = np.concatenate([np.where(inst == n)[0] for n in tr])
            out.append(dict(name=f'{cname}_fold{k}', combo=cname, fold=k, used=used, ev=ev, tr=tr,
                            iev=iev, itr=itr))
    return out


def fmt_list(v):
    return ', '.join(str(x) for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--pj-dir', default=os.path.join(HERE, 'phase_jump_meta', 'Motor-2_100'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(HERE), 'target_split'))
    a = ap.parse_args()

    pool_sha = sha256_file(a.pool)
    man_path = a.pool[:-4] + '.json'
    if os.path.exists(man_path):
        man = json.load(open(man_path, encoding='utf-8'))
        if man.get('sha256') != pool_sha:
            raise SystemExit(f'pool sha256 mismatch ({pool_sha} vs manifest {man.get("sha256")})')
    meta = load_meta(a.pool)
    inst, ftype = meta['installation'], meta['fault_type']
    meas, col, start = meta['measurement'], meta['column'], meta['start']
    inst_names = list(dict.fromkeys(inst.tolist()))       # pool order

    # ---- per-installation assignment (primary R1 and phase-jump variant) ----
    per_inst, pj_ids, merged_inst = {}, {}, {}
    for n in inst_names:
        idx = np.where(inst == n)[0]
        prim = assign_installation(n, idx, meas[idx], '')
        pj, chk = pj_check(n, idx, col, meas, a.pj_dir)
        pj_ids[n] = pj
        alt = assign_installation(n, idx, pj, 'pj')
        # cross-definition leakage: recordings of one definition that end up on both sides of the other
        chk['phase_jump_recordings_on_both_sides_under_primary'] = sorted(
            {int(p) for p in pj if {sd for q, sd in zip(pj, prim['_sides']) if q == p} == {'adapt', 'eval'}})
        nom = meas[idx]
        chk['nominal_recordings_on_both_sides_under_phase_jump'] = sorted(
            {int(m) for m in nom if {sd for q, sd in zip(nom, alt['_sides']) if q == m} == {'adapt', 'eval'}})
        per_inst[n] = {'fault_type': str(ftype[idx[0]]), 'pool_index_first': int(idx[0]),
                       'pool_index_last': int(idx[-1]), 'primary_R1': prim, 'phase_jump_R1': alt,
                       'measurement_id_check': chk}
        comp, comps = merged_components(col[idx], meas[idx], pj)
        mg = assign_installation(n, idx, comp, 'm')
        for r in mg['recordings']:
            r['columns'] = comps[r['id']]
        merged_inst[n] = mg

    splits = build_splits(inst)
    target_insts = sorted({n for s in splits for n in s['ev']}, key=inst_names.index)
    infeasible_insts = [n for n in target_insts if not per_inst[n]['primary_R1']['both_sides']]
    infeasible_insts_pj = [n for n in target_insts if not per_inst[n]['phase_jump_R1']['both_sides']]

    # ---- options for installations that R1 cannot split ----
    options = {}
    for n in infeasible_insts:
        idx = np.where(inst == n)[0]
        A = option_A(n, idx, col)
        B = option_B(n, idx, col, start, meas)
        options[n] = {
            'A_column_level': dict(public(A), key='installation|c<column>',
                                   loses='T_adapt and T_eval windows come from adjacent 12-s columns of the same '
                                         'continuous recording (same mounting, same run, same day); only the '
                                         'column boundary separates them'),
            'B_time_halves': dict(B, key='installation|measurement, split by window time order',
                                  loses='both halves come from the same recording; if the boundary is inside one '
                                        'column the last adapt and first eval windows are seconds apart'),
            'C_i_eval_only': {'n_win_adapt': 0, 'n_win_eval': len(idx),
                              'loses': 'class absent from T_adapt: the adaptation stage never sees this class '
                                       'in the target domain, so the per-class condition (class on both sides) '
                                       'fails for it; evaluation stays fully recording-independent'},
            'C_ii_exclude_from_eval': {'n_win_adapt': len(idx), 'n_win_eval': 0,
                                       'loses': 'class has no T_eval data: no metric for this unknown class in '
                                                'that split; T_eval has one class fewer'},
        }

    # ---- per split ----
    split_out, csv_rows, summary = [], {}, []
    for s in splits:
        used, ev, tr = s['used'], s['ev'], s['tr']
        iev, itr = s['iev'], s['itr']
        per_class, keys, sides = [], [], []
        alt_classes = []
        for lab, (t, n) in enumerate(zip(used, ev)):
            pr = per_inst[n]['primary_R1']
            al = per_inst[n]['phase_jump_R1']
            keys += pr['_keys']
            sides += pr['_sides']
            per_class.append(dict(label=lab, class_name=t, role='known' if lab < 2 else 'unknown',
                                  installation=n, **{k: v for k, v in public(pr).items()}))
            alt_classes.append(dict(label=lab, class_name=t, installation=n,
                                    **{k: v for k, v in public(al).items() if k != 'recordings'}))
        assert len(keys) == len(iev)
        ka = {k for k, sd in zip(keys, sides) if sd == 'adapt'}
        ke = {k for k, sd in zip(keys, sides) if sd == 'eval'}
        assert not ka & ke
        failing = [c['class_name'] for c in per_class if not c['both_sides']]
        verdict = 'FEASIBLE' if not failing else 'INFEASIBLE'
        alt_failing = [c['class_name'] for c in alt_classes if not c['both_sides']]
        alt_verdict = 'FEASIBLE' if not alt_failing else 'INFEASIBLE'
        totals = {'n_windows': len(iev), 'n_win_adapt': sides.count('adapt'), 'n_win_eval': sides.count('eval'),
                  'n_recordings': len(ka) + len(ke), 'n_rec_adapt': len(ka), 'n_rec_eval': len(ke)}
        alt_totals = {'n_win_adapt': sum(c['n_win_adapt'] for c in alt_classes),
                      'n_win_eval': sum(c['n_win_eval'] for c in alt_classes),
                      'n_rec_adapt': sum(c['n_rec_adapt'] for c in alt_classes),
                      'n_rec_eval': sum(c['n_rec_eval'] for c in alt_classes)}
        # source side
        source = []
        for lab, t in enumerate(used):
            names = [n for n in TYPES[t] if n in tr]
            recs = sum(per_inst[n]['primary_R1']['n_recordings'] for n in names)
            recs_pj = sum(per_inst[n]['phase_jump_R1']['n_recordings'] for n in names)
            wins = int(sum((inst == n).sum() for n in names))
            source.append({'label': lab, 'class_name': t, 'role': 'known' if lab < 2 else 'unknown',
                           'n_installations': len(names), 'installations': names, 'n_recordings': recs,
                           'n_recordings_phase_jump': recs_pj, 'n_windows': wins})
        # options effect on totals
        opt_totals = {}
        for n in [n for n in ev if n in options]:
            other_a = totals['n_win_adapt'] - per_inst[n]['primary_R1']['n_win_adapt']
            other_e = totals['n_win_eval'] - per_inst[n]['primary_R1']['n_win_eval']
            o = options[n]
            opt_totals[n] = {lbl: {'n_win_adapt': other_a + o[lbl]['n_win_adapt'],
                                   'n_win_eval': other_e + o[lbl]['n_win_eval']}
                             for lbl in ('A_column_level', 'B_time_halves', 'C_i_eval_only',
                                         'C_ii_exclude_from_eval')}
        split_out.append({
            'split': s['name'], 'combo': s['combo'], 'fold': s['fold'], 'split_file': f'motor2_100_ch1245_{s["name"]}.npz',
            'class_names': used, 'known_labels': [0, 1], 'unknown_labels': [2, 3, 4],
            'target_installations': ev, 'source_installations': tr,
            'verdict': verdict, 'failing_classes': failing,
            'totals': totals, 'per_class': per_class, 'source_per_class': source,
            'phase_jump_variant': {'verdict': alt_verdict, 'failing_classes': alt_failing, 'totals': alt_totals,
                                   'per_class': alt_classes},
            'option_totals_for_failing_installations': opt_totals,
            'target_pool_index': [int(i) for i in iev], 'target_recording_key': keys, 'target_side': sides})
        summary.append((s['name'], verdict, failing, totals, alt_verdict, alt_failing))
        for i, k, sd in zip(iev, keys, sides):
            i = int(i)
            row = (i, inst[i], ftype[i], int(meas[i]), int(col[i]), int(start[i]), k, sd)
            assert csv_rows.setdefault(i, row) == row
    # ---- reported option R1-merged (not adopted; primary R1 above is unchanged) ----
    merged_splits = []
    for s in splits:
        pcs = []
        for lab, (t, n) in enumerate(zip(s['used'], s['ev'])):
            pcs.append(dict(label=lab, class_name=t, role='known' if lab < 2 else 'unknown', installation=n,
                            **{k: v for k, v in public(merged_inst[n]).items() if k != 'recordings'}))
        mfail = [c['class_name'] for c in pcs if not c['both_sides']]
        merged_splits.append({
            'split': s['name'], 'verdict': 'FEASIBLE' if not mfail else 'INFEASIBLE', 'failing_classes': mfail,
            'totals': {k: sum(c[k] for c in pcs) for k in ('n_win_adapt', 'n_win_eval', 'n_rec_adapt',
                                                            'n_rec_eval')},
            'per_class': pcs})
    merged_doc_inst = {}
    for n in inst_names:
        mg = merged_inst[n]
        idx = np.where(inst == n)[0]
        pj = pj_ids[n]
        # by construction no nominal and no phase-jump recording can lie on both sides
        for ids in (meas[idx], pj):
            for v in set(ids.tolist()):
                assert len({sd for q, sd in zip(ids, mg['_sides']) if q == v}) == 1, n
        merged_doc_inst[n] = dict(public(mg), pool_index=[int(i) for i in idx], window_recording_key=mg['_keys'],
                                  window_side=mg['_sides'])
    option_merged = {
        'status': 'OPTION ONLY, NOT ADOPTED. The primary rule is R1 (keys rule_version, splits, per_installation).',
        'name': 'R1-merged',
        'rule_text': ("R1-merged: per installation, connected components of the pool columns, two columns being "
                      "linked if they share the nominal measurement (column // 5) OR the phase-jump "
                      "measurement_of_column; recording key = installation + '|m' + component index, components "
                      "numbered by their smallest column; the same even/odd rank rule as R1 (rank r even -> "
                      "T_adapt, odd -> T_eval). No window of a nominal or of a phase-jump recording can be on "
                      "both sides."),
        'summary': [{'split': m['split'], 'verdict': m['verdict'], 'failing_classes': m['failing_classes']}
                    for m in merged_splits],
        'splits': merged_splits, 'per_installation': merged_doc_inst}
    for n in per_inst:
        for v in ('primary_R1', 'phase_jump_R1'):
            per_inst[n][v] = public(per_inst[n][v])

    doc = {'rule_version': RULE_VERSION, 'rule_text': RULE_TEXT,
           'pool_file': os.path.basename(a.pool), 'pool_sha256': pool_sha,
           'metadata_fields_used': list(META), 'waveform_X_loaded': False,
           'warning_thresholds': {'min_windows_per_side': WARN_MIN_WINDOWS, 'adapt_share_range': list(WARN_SHARE),
                                  'note': 'warnings only, not part of the feasibility definition'},
           'definitions': {'recording_primary': "installation + '|' + pool measurement (= column // 5)",
                           'recording_phase_jump': "installation + '|pj' + measurement_of_column[column] from "
                                                   "centrifugal/npy/Motor-2_100/<installation>.json",
                           'target': "split 'val' side (5 installations, one per class)",
                           'feasible': 'every target installation has >= 2 recordings'},
           'n_splits': len(split_out),
           'summary': [{'split': n, 'verdict': v, 'failing_classes': f, 'phase_jump_verdict': av}
                       for n, v, f, t, av, af in summary],
           'infeasible_installations_primary': infeasible_insts,
           'infeasible_installations_phase_jump': infeasible_insts_pj,
           'options_for_infeasible_installations': options,
           'per_installation': per_inst, 'splits': split_out,
           'option_R1_merged': option_merged}

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'target_split_v1.json'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(doc, ensure_ascii=False, indent=1) + '\n')
    with open(os.path.join(a.out, 'target_split_v1.csv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('pool_index,installation,fault_type,measurement,column,start,recording_key,side\n')
        for i in sorted(csv_rows):
            f.write(','.join(str(x) for x in csv_rows[i]) + '\n')
    with open(os.path.join(a.out, 'check_report_v1.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(report(doc, per_inst, split_out, options, a))
    for s in summary:
        print(f'{s[0]}: {s[1]} failing={s[2]} adapt/eval windows {s[3]["n_win_adapt"]}/{s[3]["n_win_eval"]} '
              f'recordings {s[3]["n_rec_adapt"]}/{s[3]["n_rec_eval"]} | phase-jump variant {s[4]} {s[5]}')
    for m in merged_splits:
        t = m['totals']
        print(f'  option R1-merged {m["split"]}: {m["verdict"]} failing={m["failing_classes"]} '
              f'adapt/eval windows {t["n_win_adapt"]}/{t["n_win_eval"]} recordings {t["n_rec_adapt"]}/{t["n_rec_eval"]}')
    print('written to', os.path.abspath(a.out))


ROLE_ZH = {'known': '已知', 'unknown': '未知'}
SIDE_ZH = {'adapt': 'T_adapt', 'eval': 'T_eval'}


def warn_zh(w):
    out = []
    for x in w:
        if x.startswith('side with'):
            out.append(f'某一侧窗口数 < {WARN_MIN_WINDOWS}')
        else:
            out.append('T_adapt 占比超出 [0.25, 0.75]')
    return '；'.join(out) if out else '无'


def report(doc, per_inst, splits, options, a):
    L = []
    w = L.append
    w('# 第一步数据准备：目标域 T_adapt / T_eval 拆分可行性检查（规则 R1，v1）\n')
    w('本报告由 `deliver_data_v1/tools/target_split_check.py` 自动生成，只读取池文件的元数据'
      '（installation、fault_type、severity、measurement、column、start），不读取波形 X，不使用任何模型输出，'
      '不使用随机数。\n')
    w('## 1. 数据来源\n')
    w(f'- 数据：Bruinsma 离心泵振动数据，Motor-2，100 % 转速，通道 [1, 2, 4, 5]，采样率 20 kHz，窗长 2048。')
    w(f'- 池文件：`{doc["pool_file"]}`，sha256 `{doc["pool_sha256"]}`，来自分支 `v19-B`'
      '（分卷 part00、part01 合并后校验一致）。')
    w('- 分支 `v18-centrifugal-data` 顶层是 Paderborn 轴承数据，并不是离心泵数据'
      '（它的 `v19/` 子目录里有同一份泵数据的副本，git blob 与 `v19-B` 完全相同）。本报告使用 `v19-B`。')
    w('- 75 % 转速数据是封存测试集，本步骤没有打开。')
    w('- 划分定义与 `centrifugal/make_splits_from_pool.py` 相同（TYPES、COMBOS 原样复制，'
      '第 k 折每类的目标安装为 `TYPES[t][(k-1) % len(TYPES[t])]`）。\n')
    w('## 2. 定义\n')
    w('- 安装（installation）：一次故障或健康状态的安装，池中每个安装 160 个窗口。')
    w('- 测量记录（recording）：键为 `安装名|measurement`，measurement 是池文件里的字段，等于 column // 5，'
      '即 5 个 12 s 列组成的名义 60 s 测量。不同安装的 measurement 编号会重复，所以键里必须带安装名。')
    w('- 目标域：每个划分的 val 一侧，共 5 个安装，每类正好 1 个。T_adapt 是目标域里给适应阶段用的无标签部分，'
      'T_eval 是只用于评估的部分。')
    w('- 可行（FEASIBLE）：每个目标安装至少有 2 条记录。因为每类在每折只有 1 个目标安装，这等价于'
      '"每个已知类和未知类在 T_adapt 和 T_eval 两边都至少有 1 条记录"。')
    w(f'- 警告（不影响可行性判定）：某类某一侧窗口数少于 {WARN_MIN_WINDOWS}（160 的 10 %），'
      f'或该类 T_adapt 窗口占比不在 [{WARN_SHARE[0]}, {WARN_SHARE[1]}] 之内。\n')
    w('## 3. 规则 R1（在看到任何计数之前固定）\n')
    w('1. 拆分按目标安装逐个进行，只依赖该安装自己的元数据。所以同一个安装在它作为目标安装的所有划分里拆法相同。')
    w('2. 记录 = 该安装 160 个池窗口中出现的不同 measurement 值，按整数升序排列。')
    w('3. 排名 r（从 0 开始）为偶数的记录进 T_adapt，为奇数的进 T_eval。')
    w('4. 每个窗口跟随它所在记录的一侧。同一记录的窗口不会出现在两边。\n')
    w(f'英文原文（写入 json 的 rule_text）：{doc["rule_text"]}\n')

    # 4. measurement id check
    w('## 4. 测量编号核对（池字段 column // 5 与相位跳变检测结果）\n')
    w('相位跳变检测结果来自 `centrifugal/npy/Motor-2_100/<安装>.json` 的 `measurement_of_column`'
      '（已复制到 `deliver_data_v1/tools/phase_jump_meta/Motor-2_100/`），只统计池实际用到的列。\n')
    w('| 安装 | 池用到的列数 | 名义记录数 (column//5) | 相位跳变记录数（池用到的列） | json 中 n_measurements | 两种分组一致 | 相位跳变是名义分组的细分 |')
    w('|---|---|---|---|---|---|---|')
    n_dis = 0
    for n, d in per_inst.items():
        c = d['measurement_id_check']
        n_dis += not c['same_partition']
        w(f'| {n} | {c["n_columns_used"]} | {c["n_nominal_measurements"]} | {c["n_phase_jump_measurements_used"]} | '
          f'{c["n_measurements_json"]} | {"是" if c["same_partition"] else "否"} | '
          f'{"是" if c["phase_jump_refines_nominal"] else "否"} |')
    w('')
    all_div5 = all(d['measurement_id_check']['nominal_equals_column_div_5'] for d in per_inst.values())
    w(f'- 池字段 measurement 与 column // 5 在全部 2720 个窗口上{"一致" if all_div5 else "不一致"}。')
    w(f'- 17 个安装中有 {n_dis} 个安装的两种分组不一致（需要你注意）。主定义仍然是池文件的 measurement 字段'
      '（划分文件里带的就是这个字段），没有改。相位跳变定义下的 R1 结果在第 9 节单独列出，供你决定。')
    not_ref = [n for n, d in per_inst.items() if not d['measurement_id_check']['phase_jump_refines_nominal']]
    w(f'- 相位跳变分组不是名义分组的细分的安装（某条相位跳变记录横跨两条名义记录）：{fmt_list(not_ref) or "无"}。')
    w('- 交叉检查：按主定义 R1 拆分后，有没有某条相位跳变记录的窗口同时出现在 T_adapt 和 T_eval；'
      '反过来，按相位跳变定义 R1 拆分后，有没有某条名义记录横跨两边：\n')
    w('| 安装 | 主定义 R1 下横跨两边的相位跳变记录 | 相位跳变定义 R1 下横跨两边的名义记录 |')
    w('|---|---|---|')
    n_leak = 0
    for n, d in per_inst.items():
        c = d['measurement_id_check']
        a1, a2 = c['phase_jump_recordings_on_both_sides_under_primary'], c['nominal_recordings_on_both_sides_under_phase_jump']
        if a1 or a2:
            n_leak += 1
            w(f'| {n} | {fmt_list("pj" + str(x) for x in a1) or "无"} | {fmt_list(a2) or "无"} |')
    if not n_leak:
        w('| （全部安装） | 无 | 无 |')
    w('')
    w('- 解读：如果相位跳变检测到的才是真实的记录边界，那么上表第二列不为空的安装在主定义下，'
      '同一条真实记录的窗口会分到 T_adapt 和 T_eval 两边；第三列不为空表示改用相位跳变定义时，'
      '同一条名义 60 s 记录会被分到两边。两种定义哪个更接近真实记录，需要你根据数据采集方式判断。\n')

    # 5. source side
    w('## 5. 源域（训练安装）统计\n')
    for s in splits:
        w(f'### {s["split"]}\n')
        w('| 标签 | 类别 | 已知/未知 | 源安装数 | 源安装 | 记录数（名义） | 记录数（相位跳变） | 窗口数 |')
        w('|---|---|---|---|---|---|---|---|')
        for c in s['source_per_class']:
            w(f'| {c["label"]} | {c["class_name"]} | {ROLE_ZH[c["role"]]} | {c["n_installations"]} | '
              f'{fmt_list(c["installations"])} | {c["n_recordings"]} | {c["n_recordings_phase_jump"]} | {c["n_windows"]} |')
        w('')
    one_src = sorted({(s['split'], c['class_name'], ROLE_ZH[c['role']]) for s in splits for c in s['source_per_class']
                      if c['n_installations'] == 1})
    w('源域只有 1 个安装的类（第二步"源域留一安装验证"时，这些类在留出该安装后源域里就没有该类了）：\n')
    for sp, cn, rl in one_src:
        w(f'- {sp}：{cn}（{rl}类）')
    w('')
    w('- soft foot 只有 2 个安装，所以第 3 折重新使用 soft foot 1 作为目标安装（与第 1 折相同）。'
      '在 C1 和 C3 中，第 1 折和第 3 折的 soft foot 目标安装都是 soft foot 1，拆分也相同'
      '（源域里的 soft foot 都是 soft foot 2）。C2 不使用 soft foot。\n')

    # 6. target side
    w('## 6. 目标域逐划分统计（主定义：池 measurement 字段）\n')
    for s in splits:
        t = s['totals']
        w(f'### {s["split"]}：{s["verdict"]}\n')
        w('| 标签 | 类别 | 已知/未知 | 目标安装 | 记录 id（窗口数，侧） | T_adapt 记录/窗口 | T_eval 记录/窗口 | T_adapt 占比 | 警告 |')
        w('|---|---|---|---|---|---|---|---|---|')
        for c in s['per_class']:
            recs = '; '.join(f'{r["id"]}({r["n_windows"]},{"A" if r["side"] == "adapt" else "E"})'
                             for r in c['recordings'])
            w(f'| {c["label"]} | {c["class_name"]} | {ROLE_ZH[c["role"]]} | {c["installation"]} | {recs} | '
              f'{c["n_rec_adapt"]}/{c["n_win_adapt"]} | {c["n_rec_eval"]}/{c["n_win_eval"]} | '
              f'{c["adapt_share"]:.4f} | {warn_zh(c["warnings"])} |')
        w(f'\n合计：目标窗口 {t["n_windows"]}，T_adapt {t["n_win_adapt"]} 个窗口 / {t["n_rec_adapt"]} 条记录，'
          f'T_eval {t["n_win_eval"]} 个窗口 / {t["n_rec_eval"]} 条记录。'
          '记录 id 后面括号里是窗口数和侧（A = T_adapt，E = T_eval）。\n')

    # 7. verdict summary
    w('## 7. 九个划分的判定汇总\n')
    w('| 划分 | 判定（主定义） | 不满足的类 | T_adapt 窗口 | T_eval 窗口 | T_adapt 记录 | T_eval 记录 | 有警告的类 | 判定（相位跳变定义） |')
    w('|---|---|---|---|---|---|---|---|---|')
    for s in splits:
        t = s['totals']
        warned = [c['class_name'] for c in s['per_class'] if c['warnings']]
        w(f'| {s["split"]} | {s["verdict"]} | {fmt_list(s["failing_classes"]) or "无"} | {t["n_win_adapt"]} | '
          f'{t["n_win_eval"]} | {t["n_rec_adapt"]} | {t["n_rec_eval"]} | {fmt_list(warned) or "无"} | '
          f'{s["phase_jump_variant"]["verdict"]} |')
    fails = [s for s in splits if s['verdict'] != 'FEASIBLE']
    w('')
    if fails:
        w(f'结论：9 个划分中 {len(fails)} 个不可行：{fmt_list(s["split"] for s in fails)}。'
          f'其余 {len(splits) - len(fails)} 个可行。')
    else:
        w('结论：9 个划分全部可行。')
    w('')

    # 8. options
    w('## 8. 不可行划分与备选方案（需要你在写预注册之前选定）\n')
    if not fails:
        w('没有不可行的划分，不需要备选方案。\n')
    for n, o in options.items():
        aff = [s['split'] for s in splits if n in s['target_installations']]
        cls = per_inst[n]['fault_type']
        w(f'### 目标安装 {n}（类别 {cls}），受影响的划分：{fmt_list(aff)}\n')
        p = per_inst[n]['primary_R1']
        pj = per_inst[n]['phase_jump_R1']
        w(f'- 原因：{n} 在池中只有 {p["n_recordings"]} 条名义记录（{p["n_windows"]} 个窗口），'
          f'相位跳变定义下也只有 {pj["n_recordings"]} 条，所以按记录拆分时该类只能落在一边。')
        w(f'- 在 {fmt_list(aff)} 中，{cls} 都是未知类。\n')
        A, B = o['A_column_level'], o['B_time_halves']
        w('| 方案 | 做法 | 该类 T_adapt 窗口 | 该类 T_eval 窗口 | 失去的独立性 |')
        w('|---|---|---|---|---|')
        colsA = '; '.join(f'c{r["id"]}({r["n_windows"]},{"A" if r["side"] == "adapt" else "E"})' for r in A['recordings'])
        w(f'| A 列级拆分 | 键为"安装名 + 竖线 + 列号"（json 中写作 installation\\|c&lt;column&gt;），对排序后的列号用同样的偶数/奇数规则：{colsA} | {A["n_win_adapt"]} | '
          f'{A["n_win_eval"]} | 两边来自同一条连续记录中相邻的 12 s 列（同一次安装、同一次运行），只隔一个列边界 |')
        bd = B['boundaries'][0] if B['boundaries'] else None
        if bd:
            if bd['same_column']:
                g = bd['gap_samples_if_same_column']
                bdesc = (f'分界在第 {bd["first_eval"][0]} 列内部（T_adapt 最后一个窗口起点 {bd["last_adapt"][1]}，'
                         f'T_eval 第一个窗口起点 {bd["first_eval"][1]}，两窗口之间相隔 {g} 个采样点，'
                         f'约 {g / 20000:.2f} s）')
            else:
                bdesc = f'分界在第 {bd["last_adapt"][0]} 列与第 {bd["first_eval"][0]} 列之间'
        else:
            bdesc = ''
        w(f'| B 记录内按时间对半 | 按 (列, 起点) 排序，前一半进 T_adapt，后一半进 T_eval；{bdesc}；'
          f'两边都有的列：{fmt_list(B["columns_on_both_sides"]) or "无"} | {B["n_win_adapt"]} | {B["n_win_eval"]} | '
          '两边来自同一条记录的同一个 60 s 测量；分界两侧的窗口在时间上相隔不到 1 s（见做法一栏） |')
        w(f'| C(i) 只放 T_eval | 该类全部窗口进 T_eval，T_adapt 中没有该类 | 0 | {o["C_i_eval_only"]["n_win_eval"]} | '
          '评估仍与适应数据按记录独立，但适应阶段在目标域看不到该类，"每类两边都有"的条件对该类不成立 |')
        w(f'| C(ii) 不参与评估 | 该类全部窗口只进 T_adapt，从 T_eval 中排除 | {o["C_ii_exclude_from_eval"]["n_win_adapt"]} | 0 | '
          '该划分中没有这个未知类的评估指标，T_eval 只剩 4 类，与其他划分不可直接比较 |')
        w('')
        w('各方案下受影响划分的目标窗口合计（T_adapt / T_eval）：\n')
        w('| 划分 | A | B | C(i) | C(ii) |')
        w('|---|---|---|---|---|')
        for s in splits:
            ot = s['option_totals_for_failing_installations'].get(n)
            if ot:
                cells = [f'{ot[k]["n_win_adapt"]} / {ot[k]["n_win_eval"]}' for k in
                         ('A_column_level', 'B_time_halves', 'C_i_eval_only', 'C_ii_exclude_from_eval')]
                w(f'| {s["split"]} | ' + ' | '.join(cells) + ' |')
        w('')
    if fails:
        w('以上方案只是供选择的选项，脚本没有采用其中任何一个，R1 的主规则也没有改。'
          '另一种做法是不使用这几个划分的第 2 折（只报告可行的划分）。选定之后再写入第二步的预注册。\n')

    # 9. phase-jump variant
    w('## 9. 相位跳变定义下的 R1 结果（对照，不是主定义）\n')
    w('把记录定义换成 `安装|pj<measurement_of_column[列]>`，其余规则不变：\n')
    w('| 划分 | 判定 | 不满足的类 | T_adapt 窗口 | T_eval 窗口 | T_adapt 记录 | T_eval 记录 |')
    w('|---|---|---|---|---|---|---|')
    for s in splits:
        v = s['phase_jump_variant']
        t = v['totals']
        w(f'| {s["split"]} | {v["verdict"]} | {fmt_list(v["failing_classes"]) or "无"} | {t["n_win_adapt"]} | '
          f'{t["n_win_eval"]} | {t["n_rec_adapt"]} | {t["n_rec_eval"]} |')
    w('')
    w('各目标安装在两种定义下的 R1 计数：\n')
    w('| 安装 | 名义记录数 | 名义 T_adapt/T_eval 窗口 | 相位跳变记录数 | 相位跳变 T_adapt/T_eval 窗口 |')
    w('|---|---|---|---|---|')
    for n, d in per_inst.items():
        p, q = d['primary_R1'], d['phase_jump_R1']
        w(f'| {n} | {p["n_recordings"]} | {p["n_win_adapt"]}/{p["n_win_eval"]} | {q["n_recordings"]} | '
          f'{q["n_win_adapt"]}/{q["n_win_eval"]} |')
    w('')

    # 10. R1-merged option
    om = doc['option_R1_merged']
    w('## 10. 备选记录定义 R1-merged（只是选项，没有采用）\n')
    n_dis = sum(not d['measurement_id_check']['same_partition'] for d in per_inst.values())
    fewer = [f'{n}（{per_inst[n]["primary_R1"]["n_recordings"]} 条变为 {d["n_recordings"]} 条）'
             for n, d in om['per_installation'].items()
             if d['n_recordings'] < per_inst[n]['primary_R1']['n_recordings']]
    w(f'第 4 节显示两种记录定义（名义 column // 5 与相位跳变）在 {n_dis} 个安装上不一致，主定义 R1 下有相位跳变记录横跨两边。'
      'R1-merged 把两种定义合并：在每个安装内，把池用到的列看成节点，两列只要名义 measurement 相同'
      '或相位跳变 measurement_of_column 相同就连一条边，取连通分量作为记录，键为 `安装名|m<分量编号>`，'
      '分量按其最小列号排序编号，然后用与 R1 相同的偶数/奇数规则。这样无论按哪种定义，同一条记录的窗口都不会分到两边。'
      f'代价是记录更粗，记录数比名义定义少的安装：{fmt_list(fewer) or "无"}。'
      'impeller 2 在两种定义下都只有 1 条记录，所以 R1-merged 不能解决第 2 折的问题。\n')
    w('这一节只是给你的选项。主规则仍是 R1，json 中主规则的条目没有变化，R1-merged 的结果单独放在 json 的 '
      '`option_R1_merged` 键下。\n')
    w('| 安装 | R1-merged 记录数 | 各记录（列号，窗口数，侧） | T_adapt/T_eval 窗口 |')
    w('|---|---|---|---|')
    for n, d in om['per_installation'].items():
        recs = '; '.join(f'm{r["id"]}[c{r["columns"][0]}..c{r["columns"][-1]}, {len(r["columns"])} 列]'
                         f'({r["n_windows"]},{"A" if r["side"] == "adapt" else "E"})' for r in d['recordings'])
        w(f'| {n} | {d["n_recordings"]} | {recs} | {d["n_win_adapt"]}/{d["n_win_eval"]} |')
    w('\n（`c起..c止` 表示该分量的最小和最大列号，分量内的列不一定连续，完整列表见 json。）\n')
    w('| 划分 | R1-merged 判定 | 不满足的类 | T_adapt 窗口 | T_eval 窗口 | T_adapt 记录 | T_eval 记录 | 有警告的类 |')
    w('|---|---|---|---|---|---|---|---|')
    for m in om['splits']:
        t = m['totals']
        warned = [c['class_name'] for c in m['per_class'] if c['warnings']]
        w(f'| {m["split"]} | {m["verdict"]} | {fmt_list(m["failing_classes"]) or "无"} | {t["n_win_adapt"]} | '
          f'{t["n_win_eval"]} | {t["n_rec_adapt"]} | {t["n_rec_eval"]} | {fmt_list(warned) or "无"} |')
    w('')
    w('各划分每类计数（T_adapt 记录/窗口，T_eval 记录/窗口）：\n')
    w('| 划分 | ' + ' | '.join(f'标签 {k}' for k in range(5)) + ' |')
    w('|---|---|---|---|---|---|')
    for m in om['splits']:
        cells = [f'{c["class_name"]}（{c["installation"]}）：{c["n_rec_adapt"]}/{c["n_win_adapt"]}，'
                 f'{c["n_rec_eval"]}/{c["n_win_eval"]}' for c in m['per_class']]
        w(f'| {m["split"]} | ' + ' | '.join(cells) + ' |')
    w('')
    nf = [m['split'] for m in om['splits'] if m['verdict'] != 'FEASIBLE']
    w(f'R1-merged 下 9 个划分中 {len(nf)} 个不可行：{fmt_list(nf) or "无"}。\n')

    # 11. command
    w('## 11. 输出文件与复现命令\n')
    w('- `target_split_v1.json`：规则文本、规则版本 R1、池 sha256、每个划分的目标池索引、每个窗口的记录键和侧、'
      '每类计数、判定、逐安装拆分表、备选方案计数。')
    w('- `target_split_v1.csv`：所有曾作为目标窗口的池窗口，每行一个（pool_index, installation, fault_type, '
      'measurement, column, start, recording_key, side）。同一安装在所有划分中拆法相同，所以每个窗口只有一行。')
    w('- `check_report_v1.md`：本报告。\n')
    w('沙箱中的复现命令：\n')
    w('```')
    w('/projects/sandbox/.venv/bin/python /projects/sandbox/deliver_data_v1/tools/target_split_check.py '
      '--pool /projects/sandbox/.scratch_data/pool/pool_motor2_100_ch1245.npz')
    w('```\n')
    w('本地（Windows）在 deliver_data_v1 目录下，合并好池文件后：\n')
    w('```')
    w('python tools\\target_split_check.py --pool <池文件路径>\\pool_motor2_100_ch1245.npz')
    w('```\n')
    w('脚本不使用随机数，连续运行两次得到的 json 和 csv 字节完全相同。')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
