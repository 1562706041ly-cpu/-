"""v28 analysis (criteria: v28_endtoend_PREREGISTERED.txt). ASCII-only output. Exploratory screen: 9 splits x seed 0.
Usage:
  python analyze_v28.py --check-only [--root runs_paderborn]      GPU consistency check of the reused Bg4; exit code 0 = PASS (reuse), 3 = FAIL or not evaluable (re-run Bg4)
  python analyze_v28.py [--root runs_paderborn] [--splits centrifugal\\splits_ch1245] [--out summary_v28.txt] [--csv summary_v28.csv] [--force]
"""
import argparse, csv, glob, itertools, json, math, os, re, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='runs_paderborn'); ap.add_argument('--splits', default=os.path.join('centrifugal', 'splits_ch1245'))
ap.add_argument('--out', default='summary_v28.txt'); ap.add_argument('--csv', default='summary_v28.csv'); ap.add_argument('--pairing-out', default='pairing_report_v28.txt')
ap.add_argument('--check-out', default='bg4_check_v28.txt'); ap.add_argument('--invalid', default='invalid_v28.txt'); ap.add_argument('--logs', default='.')
ap.add_argument('--check-only', action='store_true'); ap.add_argument('--force', action='store_true'); ap.add_argument('--tol', type=float, default=0.005)
a = ap.parse_args()

def read_log(path):
    """PowerShell '*>' writes UTF-16LE logs; decode by BOM / NUL bytes."""
    b = open(path, 'rb').read()
    if b[:2] in (b'\xff\xfe', b'\xfe\xff') or b[1:2] == b'\x00': return b.decode('utf-16', errors='ignore')
    return b.decode('utf-8', errors='ignore')

def load_run(tag):
    f = os.path.join(a.root, tag, 'results_exp2.json')
    if not os.path.exists(f): return None
    r = list(json.load(open(f, encoding='utf-8')).values())[0]; o = r['openset']['V']
    return dict(tag=tag, args=r['args'], lab=list(r['labeled_idx']), n=(r['n_labeled'], r['n_unlabeled'], r['n_val']),
                vals=dict(H=o['H'], known_acc=o['known_acc'], unk_recall=o['unk_recall'], known_false_unk=o['known_false_unk'],
                          unk_cluster_acc=r['M2_GCD+半监督KMeans']['V']['new'], nmi=r['M2_GCD+半监督KMeans']['V']['nmi']))

# ------------------------------------------------------------------ GPU consistency check
if a.check_only:
    new, old = load_run('V28CHK_C1_f1_Bg4_seed0'), load_run('V20_C1_f1_Bg4_seed0')
    lines = []
    if new is None or old is None:
        lines.append(f"cannot evaluate: V28CHK found={new is not None}, V20_C1_f1_Bg4_seed0 found={old is not None} -> treated as FAIL (the 9 Bg4 runs are re-run with the new code)"); verdict = 3
    else:
        dh = new['vals']['H'] - old['vals']['H']
        lines.append(f"H new code (GPU, default band flags) {new['vals']['H']:.6f}   H old Bg4 {old['vals']['H']:.6f}   difference {dh:+.6f}   tolerance {a.tol}")
        lines.append(f"labelled windows identical: {new['lab'] == old['lab']}")
        verdict = 0 if (abs(dh) <= a.tol and new['lab'] == old['lab']) else 3
        lines.append('PASS: the old Bg4 runs are reused' if verdict == 0 else 'FAIL: the old Bg4 runs are NOT reused; all 9 Bg4 are re-run with the new code')
    open(a.check_out, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines)); sys.exit(verdict)

# ------------------------------------------------------------------ collect runs
splits = [(c, f) for c in ('C1', 'C2', 'C3') for f in (1, 2, 3)]
ARMS = ['Bg4', 'BI65', 'BC2', 'BI2']
new_bg4 = all(load_run(f'V28_{c}_f{f}_Bg4_seed0') is not None for c, f in splits)
bg4_source = 'V28 (re-run with the new code after a failed GPU check)' if new_bg4 else 'V20 (existing runs, reused; GPU check passed)'
runs = {}
for c, f in splits:
    for arm in ARMS:
        tag = (f'V28_{c}_f{f}_Bg4_seed0' if new_bg4 else f'V20_{c}_f{f}_Bg4_seed0') if arm == 'Bg4' else f'V28_{c}_f{f}_{arm}_seed0'
        r = load_run(tag)
        if r is not None: runs[(arm, c, f)] = r
EXPECT = {'BI65': dict(aug_band='indep', band_gain=0.65), 'BC2': dict(aug_band='coord', band_gain=2.0), 'BI2': dict(aug_band='indep', band_gain=2.0)}
IGN = {'tag', 'seeds', 'data', 'aug_band', 'band_gain'}
invalid_txt = set()
if os.path.exists(a.invalid):
    for ln in open(a.invalid, encoding='utf-8-sig'):
        if ln.strip(): invalid_txt.add(ln.split()[0])
rep = []; P = rep.append; fails = []; valid = {}
for (arm, c, f), v in runs.items():
    bad = v['tag'] in invalid_txt or any((x != x) or x is None for x in v['vals'].values())
    lg = os.path.join(a.logs, f"log_{v['tag']}.txt")
    if os.path.exists(lg) and re.search(r'loss=(nan|inf|-inf)', read_log(lg)): bad = True
    valid[(arm, c, f)] = not bad
P(f'Bg4 source: {bg4_source}')
P('runs found per arm (valid / total of 9): ' + '  '.join(f"{arm} {sum(valid.get((arm, c, f), False) for c, f in splits)}/{sum((arm, c, f) in runs for c, f in splits)}" for arm in ARMS))
bad_list = sorted(v['tag'] for (arm, c, f), v in runs.items() if not valid[(arm, c, f)])
if bad_list: P('INVALID runs (excluded): ' + ', '.join(bad_list))
failed_arm = {arm: sum(valid.get((arm, c, f), False) for c, f in splits) < 8 for arm in ARMS}
for arm in ARMS:
    if failed_arm[arm]: fails.append(f'arm {arm}: fewer than 8 valid runs -> FAILED, no comparison with it is interpreted')
for c, f in splits:
    ref = runs.get(('Bg4', c, f))
    if ref is None: continue
    for arm in ('BI65', 'BC2', 'BI2'):
        v = runs.get((arm, c, f))
        if v is None: continue
        if v['lab'] != ref['lab']: fails.append(f'{arm} {c} f{f}: labelled window indices differ from Bg4')
        if v['n'] != ref['n']: fails.append(f'{arm} {c} f{f}: n_labeled/n_unlabeled/n_val {v["n"]} vs Bg4 {ref["n"]}')
        for k, want in EXPECT[arm].items():
            if v['args'].get(k) != want: fails.append(f'{arm} {c} f{f}: {k}={v["args"].get(k)!r}, expected {want!r}')
        diff = {k: (ref['args'].get(k), v['args'].get(k)) for k in set(ref['args']) | set(v['args']) if k not in IGN and ref['args'].get(k) != v['args'].get(k)}
        if diff: fails.append(f'{arm} {c} f{f}: args differ from Bg4: {diff}')
hard = [x for x in fails if not x.startswith('arm ')]
P('pairing checks: ' + ('PASSED' if not hard else f'FAILED ({len(hard)} problems)'))
for x in fails[:60]: P('  - ' + x)
open(a.pairing_out, 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))
if hard and not a.force:
    print('STOP: pairing not verified. Send pairing_report_v28.txt to Kiro. Do not interpret.'); sys.exit(2)

out = []; Q = out.append
Q('v28 exploratory screen: 9 splits x seed 0 (criteria: v28_endtoend_PREREGISTERED.txt). Screening, not confirmation; five comparisons, no multiplicity correction.')
Q('KNOWN LIMITATION: BI65 is not feature-matched to BC2 in the three C3 splits (independent about 2x weaker there); BC2 at kappa = 1 is shrunk by the fidelity rule, so its effective perturbation is weaker than nominal.')
Q(f'Bg4 source: {bg4_source}')
METS = ['H', 'known_acc', 'unk_recall', 'known_false_unk', 'unk_cluster_acc', 'nmi']
Q(''); Q('arm means over valid splits: ' + ', '.join(METS))
for arm in ARMS:
    sp = [(c, f) for c, f in splits if valid.get((arm, c, f), False)]
    if sp: Q(f'  {arm:5s} n={len(sp)}  ' + '  '.join(f"{m} {np.mean([runs[(arm, c, f)]['vals'][m] for c, f in sp]):.3f}" for m in METS) + ('   [FAILED arm]' if failed_arm[arm] else ''))
TCRIT = {8: 2.306, 7: 2.365}                                     # two-sided 95 %, df = n - 1 (n = 9 or 8)
rows = []; verdicts = {}
def compare(X, Y, label, kind):
    if failed_arm[X] or failed_arm[Y]: Q(f'{label}: {X} - {Y}: not interpreted (failed arm)'); verdicts[label] = 'not interpreted'; return
    sp = [(c, f) for c, f in splits if valid.get((X, c, f), False) and valid.get((Y, c, f), False)]
    n = len(sp)
    if n not in (8, 9): Q(f'{label}: {X} - {Y}: only {n} valid pairs -> not interpreted'); verdicts[label] = 'not interpreted'; return
    d = np.array([runs[(X, c, f)]['vals']['H'] - runs[(Y, c, f)]['vals']['H'] for c, f in sp]); mean = float(d.mean()); sd = float(d.std(ddof=1)); se = sd / math.sqrt(n)
    lo, hi = mean - TCRIT[n - 1] * se, mean + TCRIT[n - 1] * se; pos = int((d > 0).sum()); neg = int((d < 0).sum()); need = 7
    cand = mean >= 0.02 and pos >= need and lo > 0; harm = mean <= -0.02 and neg >= need
    verdict = 'CANDIDATE' if cand else ('HARM (harmful in this setting)' if harm else 'no evidence of a difference at this resolution')
    sec = {m: float(np.mean([runs[(X, c, f)]['vals'][m] - runs[(Y, c, f)]['vals'][m] for c, f in sp])) for m in METS[1:]}
    Q(f'{label} [{kind}]: {X} - {Y} (n={n}): mean dH {mean:+.4f}  sd {sd:.4f}  se {se:.4f}  95% t-interval [{lo:+.4f}, {hi:+.4f}]  positive in {pos}/{n}, negative in {neg}/{n}')
    Q('   per-split dH (' + ' '.join(f'{c}f{f}' for c, f in sp) + '): ' + ' '.join(f'{x:+.3f}' for x in d)); Q('   secondary mean differences: ' + '  '.join(f'{k} {v:+.3f}' for k, v in sec.items()))
    Q(f'   VERDICT (pre-registered rule): {verdict}')
    verdicts[label] = verdict
    rows.append(dict(label=label, kind=kind, X=X, Y=Y, n=n, mean_dH=mean, sd=sd, se=se, ci_lo=lo, ci_hi=hi, positive=pos, negative=neg, verdict=verdict))
    if label == 'P1':
        fam = {'C1/C2': [i for i, (c, f) in enumerate(sp) if c != 'C3'], 'C3': [i for i, (c, f) in enumerate(sp) if c == 'C3']}
        Q('   DESCRIPTIVE ONLY, enters no rule - by split family: ' + '; '.join(f"{k}: n={len(v)} mean dH {np.mean(d[v]):+.4f}" if v else f'{k}: none' for k, v in fam.items()))
        Q('   (in C3 the coordinated arm is both more strongly perturbed and has the larger feature change than BI65; an advantage coming mainly from C3 is not attributed to coordination)')
Q('')
for X, Y, lab, kind in (('BC2', 'BI65', 'P1', 'primary, feature-level matched (limited in C3)'), ('BC2', 'BI2', 'P2', 'primary, signal-level matched'),
                        ('BI65', 'Bg4', 'S1', 'secondary'), ('BC2', 'Bg4', 'S2', 'secondary (decides the stop rule)'), ('BI2', 'Bg4', 'S3', 'secondary')):
    compare(X, Y, lab, kind); Q('')
Q('WHAT THE VERDICTS MEAN (fixed in the pre-registration)')
v = verdicts
if v.get('S2') == 'CANDIDATE': Q('  S2 BC2 - Bg4 is a CANDIDATE: "coordinated band gain improves performance" becomes a candidate for a separately pre-registered multi-seed stage. Not a confirmation.')
else: Q('  STOP RULE: S2 BC2 - Bg4 is not a CANDIDATE -> the line "coordinated band gain improves performance" is stopped; recorded as "no benefit observed in this setting". No threshold change, no arm added, no further run on the same splits and seed.')
for lab in ('P1', 'P2'):
    if v.get(lab) == 'CANDIDATE' and v.get('S2') != 'CANDIDATE': Q(f'  {lab} is a CANDIDATE but S2 is not: coordination is less harmful than that independent arm; this is NOT a performance candidate.')
for lab in ('S1', 'S3'):
    if v.get(lab) == 'CANDIDATE': Q(f'  {lab} is a CANDIDATE: independent band gain becomes a candidate, unrelated to coordination.')

# ------------------------------------------------------------------ unknown-class separability P1-AUROC from the feature files (labels for evaluation only)
from sklearn.metrics import roc_auc_score
NPOS = NNEG = 100000
def sample_pairs(cls, rec, rng):
    cells = {}
    for i, (c_, r_) in enumerate(zip(cls, rec)): cells.setdefault((c_, r_), []).append(i)
    keys = list(cells); arr = [np.array(cells[k]) for k in keys]; kc = np.array([k[0] for k in keys])
    pos_c = [np.array([q for q in range(len(keys)) if kc[q] == kc[j] and q != j]) for j in range(len(keys))]
    neg_c = [np.array([q for q in range(len(keys)) if kc[q] != kc[j]]) for j in range(len(keys))]
    def draw(n, cand):
        ok = [j for j in range(len(keys)) if len(cand[j]) > 0]
        if not ok: return None
        j = np.array(ok)[rng.integers(0, len(ok), n)]; aa = np.empty(n, int); bb = np.empty(n, int)
        for k in np.unique(j):
            sel = np.where(j == k)[0]; aa[sel] = arr[k][rng.integers(0, len(arr[k]), len(sel))]; q = cand[k][rng.integers(0, len(cand[k]), len(sel))]
            for qq in np.unique(q):
                s2 = sel[q == qq]; bb[s2] = arr[qq][rng.integers(0, len(arr[qq]), len(s2))]
        return aa, bb
    p = draw(NPOS, pos_c); n_ = draw(NNEG, neg_c)
    if p is None or n_ is None: return None
    return np.concatenate([p[0], n_[0]]), np.concatenate([p[1], n_[1]]), np.concatenate([np.ones(NPOS), np.zeros(NNEG)])
def p1_for_run(run, c, f, ci):
    ff = os.path.join(a.root, run['tag'], 'feat_gcd_PADERBORN_seed0.npz'); sf = os.path.join(a.splits, f'motor2_100_ch1245_{c}_fold{f}.npz')
    if not os.path.exists(ff): return None, 'feature file missing'
    if not os.path.exists(sf): return None, 'split file missing'
    d = np.load(sf, allow_pickle=True); z = np.load(ff, allow_pickle=True); known = [int(v_) for v_ in str(run['args']['known']).split(',')]
    ytr, yva = d['y_train'], d['y_val']; unl = np.setdiff1d(np.arange(len(ytr)), np.array(run['lab']))
    if len(z['yu']) != len(unl) or not np.array_equal(z['yu'], ytr[unl]) or not np.array_equal(z['yv'], yva): return None, 'labels in the feature file do not match the split (integrity check failed)'
    rec_u = np.char.add(np.char.add(np.asarray(d['train_installation']).astype(str)[unl], '|'), np.asarray(d['train_measurement']).astype(str)[unl])
    rec_v = np.char.add(np.char.add(np.asarray(d['val_installation']).astype(str), '|'), np.asarray(d['val_measurement']).astype(str))
    res = {}
    for nm, Fz, yy, rec in (('P1_U', z['FU'], z['yu'], rec_u), ('P1_V', z['FV'], z['yv'], rec_v)):
        um = ~np.isin(yy, known); rng = np.random.default_rng(900 + 10 * ci + (1 if nm == 'P1_V' else 0))
        pr = sample_pairs(yy[um], rec[um], rng)
        if pr is None: res[nm] = float('nan'); continue
        ia, ib, lab = pr; Fn = Fz[um] / np.maximum(np.linalg.norm(Fz[um], axis=1, keepdims=True), 1e-12)
        res[nm] = float(roc_auc_score(lab, (Fn[ia] * Fn[ib]).sum(1)))
    return res, 'ok'
Q(''); Q('Unknown-class separability P1-AUROC (pair AUROC between different recordings; unknown windows of the unlabelled training set P1_U and of the held-out set P1_V; labels for evaluation only; the same pairs for every arm of a split)')
p1 = {}; notes = {}
for ci, (c, f) in enumerate(splits):
    for arm in ARMS:
        r = runs.get((arm, c, f))
        if r is None: continue
        res, note = p1_for_run(r, c, f, ci)
        if res is None: notes.setdefault(arm, set()).add(note)
        else: p1[(arm, c, f)] = res
for arm in ARMS:
    have = [(c, f) for c, f in splits if (arm, c, f) in p1]
    if have: Q(f'  {arm:5s} n={len(have)}  P1_U {np.nanmean([p1[(arm, c, f)]["P1_U"] for c, f in have]):.4f}  P1_V {np.nanmean([p1[(arm, c, f)]["P1_V"] for c, f in have]):.4f}')
    else: Q(f'  {arm:5s} not available: ' + '; '.join(sorted(notes.get(arm, {'no runs'}))) + ('  (Bg4 is NOT re-run for this; P1 is compared among the new arms only)' if arm == 'Bg4' else ''))
for X, Y in (('BC2', 'BI65'), ('BC2', 'BI2'), ('BI65', 'Bg4'), ('BC2', 'Bg4'), ('BI2', 'Bg4')):
    sp = [(c, f) for c, f in splits if (X, c, f) in p1 and (Y, c, f) in p1]
    if sp: Q(f'  dP1 {X} - {Y} (n={len(sp)}): P1_U {np.nanmean([p1[(X, c, f)]["P1_U"] - p1[(Y, c, f)]["P1_U"] for c, f in sp]):+.4f}  P1_V {np.nanmean([p1[(X, c, f)]["P1_V"] - p1[(Y, c, f)]["P1_V"] for c, f in sp]):+.4f}   (descriptive)')

# ------------------------------------------------------------------ in-training implementation check (decoded logs)
Q(''); Q('In-training implementation check (last print of each run; mean over valid runs; NOT outcomes): band_sig = median relative signal change, band_shrunk, band_fallback, band_feat_e3 = 1000 x median feature change')
pat = {k: re.compile(k + r':([0-9.eE+-]+|nan)') for k in ('band_sig', 'band_shrunk', 'band_fallback', 'band_feat_e3')}
for arm in ('BI65', 'BC2', 'BI2'):
    acc = {k: [] for k in pat}; fam = {'C1/C2': [], 'C3': []}
    for c, f in splits:
        r = runs.get((arm, c, f))
        if r is None: continue
        lg = os.path.join(a.logs, f"log_{r['tag']}.txt")
        if not os.path.exists(lg): continue
        last = [ln for ln in read_log(lg).splitlines() if ' aug=' in ln and 'band_sig' in ln]
        if not last: continue
        for k, p_ in pat.items():
            mm = p_.search(last[-1])
            if mm: acc[k].append(float(mm.group(1)))
        mm = pat['band_feat_e3'].search(last[-1])
        if mm: fam['C3' if c == 'C3' else 'C1/C2'].append(float(mm.group(1)))
    Q(f'  {arm:5s} ' + ('  '.join(f'{k} {np.mean(v_):.3f}' if v_ else f'{k} n/a' for k, v_ in acc.items()) + f'  (logs: {len(acc["band_sig"])} runs)') + '   feature change e3 by family: ' + ', '.join(f'{k} {np.mean(v_):.2f}' if v_ else f'{k} n/a' for k, v_ in fam.items()))
open(a.out, 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
if rows:
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
