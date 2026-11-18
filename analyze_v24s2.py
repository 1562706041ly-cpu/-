"""v24s2 analysis (criteria: v24s2_PREREGISTERED.txt). ASCII-only output. Exploratory screen: 9 splits x seed 0.
Usage: python analyze_v24s2.py [--root runs_paderborn] [--out summary_v24s2.txt] [--csv summary_v24s2.csv] [--force]
"""
import argparse, csv, glob, json, math, os, re, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='runs_paderborn'); ap.add_argument('--out', default='summary_v24s2.txt'); ap.add_argument('--csv', default='summary_v24s2.csv')
ap.add_argument('--pairing-out', default='pairing_report_v24s2.txt'); ap.add_argument('--invalid', default='invalid_v24s2.txt'); ap.add_argument('--logs', default='.'); ap.add_argument('--force', action='store_true')
a = ap.parse_args()
ARMS = ['Bg4', 'Rk', 'Rk2', 'Rk4', 'Rks2', 'Rks4']
TAG = re.compile(r'V20_(C\d)_f(\d)_(Bg4|Rk|Rk2|Rk4|Rks2|Rks4)_seed0$')
EXPECT = {'Rk2': dict(mode='adaptive_kr2', kr2_gain=2.0, kr2_shuf_scope='both'), 'Rk4': dict(mode='adaptive_kr2', kr2_gain=4.0, kr2_shuf_scope='both'),
          'Rks2': dict(mode='shuffled_kr2', kr2_gain=2.0, kr2_shuf_scope='sample'), 'Rks4': dict(mode='shuffled_kr2', kr2_gain=4.0, kr2_shuf_scope='sample'),
          'Rk': dict(mode='adaptive_kr2', kr2_gain=1.0, kr2_shuf_scope='both')}
IGN = {'tag', 'seeds', 'mode', 'kr2_gain', 'kr2_shuf_scope', 'data'}
METS = ['H', 'known_acc', 'unk_recall', 'known_false_unk', 'unk_cluster_acc', 'nmi']
splits = [(c, f) for c in ('C1', 'C2', 'C3') for f in (1, 2, 3)]
invalid_txt = set()
if os.path.exists(a.invalid):
    for ln in open(a.invalid, encoding='utf-8-sig'):
        if ln.strip(): invalid_txt.add(ln.split()[0])
runs = {}
for fpath in sorted(glob.glob(os.path.join(a.root, 'V20_*_seed0', 'results_exp2.json'))):
    tag = os.path.basename(os.path.dirname(fpath)); m = TAG.search(tag)
    if not m: continue
    r = list(json.load(open(fpath, encoding='utf-8')).values())[0]; o = r['openset']['V']
    vals = dict(H=o['H'], known_acc=o['known_acc'], unk_recall=o['unk_recall'], known_false_unk=o['known_false_unk'], unk_cluster_acc=r['M2_GCD+半监督KMeans']['V']['new'], nmi=r['M2_GCD+半监督KMeans']['V']['nmi'])
    runs[(m.group(3), m.group(1), int(m.group(2)))] = dict(tag=tag, vals=vals, args=r['args'], lab=list(r['labeled_idx']), n=(r['n_labeled'], r['n_unlabeled'], r['n_val']))
rep = []; P = rep.append; fails = []
valid = {}
for (arm, c, f), v in runs.items():
    bad = v['tag'] in invalid_txt or any((x != x) or x is None for x in v['vals'].values())
    lg = os.path.join(a.logs, f"log_{v['tag']}.txt")
    if os.path.exists(lg) and re.search(r'loss=(nan|inf|-inf)', open(lg, encoding='utf-8', errors='ignore').read()): bad = True
    valid[(arm, c, f)] = not bad
P('runs found per arm (valid / total of 9): ' + '  '.join(f"{arm} {sum(valid.get((arm, c, f), False) for c, f in splits)}/{sum((arm, c, f) in runs for c, f in splits)}" for arm in ARMS))
invalid_list = sorted(v['tag'] for (arm, c, f), v in runs.items() if not valid[(arm, c, f)]) + sorted(t for t in invalid_txt if t not in {v['tag'] for v in runs.values()})
if invalid_list: P('INVALID runs (excluded): ' + ', '.join(invalid_list))
failed_arm = {arm: sum(valid.get((arm, c, f), False) for c, f in splits) < 8 for arm in ARMS}
for arm in ARMS:
    if failed_arm[arm]: fails.append(f'arm {arm}: fewer than 8 valid runs -> FAILED, no comparison with it is interpreted')
for c, f in splits:
    ref = runs.get(('Rk', c, f))
    if ref is None: continue
    for arm in ARMS:
        v = runs.get((arm, c, f))
        if v is None or arm == 'Rk': continue
        if v['lab'] != ref['lab']: fails.append(f'{arm} {c} f{f}: labelled window indices differ from Rk')
        if v['n'] != ref['n']: fails.append(f'{arm} {c} f{f}: n_labeled/n_unlabeled/n_val {v["n"]} vs Rk {ref["n"]}')
        if arm == 'Bg4': continue
        for k, want in EXPECT[arm].items():
            got = v['args'].get(k, 'both' if k == 'kr2_shuf_scope' else None)
            if got != want: fails.append(f'{arm} {c} f{f}: {k}={got!r}, expected {want!r}')
        diff = {k: (ref['args'].get(k), v['args'].get(k)) for k in set(ref['args']) | set(v['args']) if k not in IGN and ref['args'].get(k) != v['args'].get(k)}
        if diff: fails.append(f'{arm} {c} f{f}: args differ from Rk: {diff}')
for (arm, c, f), v in runs.items():
    if arm == 'Rk':
        for k, want in EXPECT['Rk'].items():
            got = v['args'].get(k, 'both' if k == 'kr2_shuf_scope' else None)
            if got != want: fails.append(f'Rk {c} f{f}: {k}={got!r}, expected {want!r}')
P('pairing checks: ' + ('PASSED' if not [x for x in fails if not x.startswith('arm ')] else 'FAILED'))
for x in fails[:60]: P('  - ' + x)
open(a.pairing_out, 'w').write('\n'.join(rep) + '\n'); print('\n'.join(rep))
if [x for x in fails if not x.startswith('arm ')] and not a.force:
    print('STOP: pairing not verified. Send pairing_report_v24s2.txt to Kiro. Do not interpret.'); sys.exit(2)

out = []; Q = out.append
Q('v24s2 exploratory screen: 9 splits x seed 0 (criteria: v24s2_PREREGISTERED.txt). Candidates only; no multiplicity correction.')
Q('')
Q('arm means over valid splits: ' + ', '.join(METS))
for arm in ARMS:
    sp = [(c, f) for c, f in splits if valid.get((arm, c, f), False)]
    if not sp: continue
    Q(f'  {arm:5s} n={len(sp)}  ' + '  '.join(f"{m} {np.mean([runs[(arm, c, f)]['vals'][m] for c, f in sp]):.3f}" for m in METS) + ('   [FAILED arm]' if failed_arm[arm] else ''))
rng = np.random.default_rng(0); rows = []
def cmp(X, Y, label):
    if failed_arm[X] or failed_arm[Y]: Q(f'{label}: {X} - {Y}: not interpreted (failed arm)'); return
    sp = [(c, f) for c, f in splits if valid.get((X, c, f), False) and valid.get((Y, c, f), False)]
    d = np.array([runs[(X, c, f)]['vals']['H'] - runs[(Y, c, f)]['vals']['H'] for c, f in sp]); n = len(d)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)]); lo, hi = np.percentile(b, [2.5, 97.5]); mean = float(d.mean()); pos = int((d > 0).sum())
    need_pos = 7 if n == 9 else math.ceil(7 * n / 9)
    verdict = 'WORTH CONFIRMING' if (mean >= 0.02 and pos >= need_pos and lo > 0) else ('HARM' if (mean <= -0.02 and hi < 0) else 'no evidence of a difference at this resolution')
    sec = {m: float(np.mean([runs[(X, c, f)]['vals'][m] - runs[(Y, c, f)]['vals'][m] for c, f in sp])) for m in METS[1:]}
    Q(f'{label}: {X} - {Y} (n={n}): mean dH {mean:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  positive in {pos}/{n} splits')
    Q('   per-split dH: ' + ' '.join(f'{x:+.3f}' for x in d)); Q('   secondary mean differences: ' + '  '.join(f'{k} {v:+.3f}' for k, v in sec.items())); Q(f'   VERDICT (pre-registered rule): {verdict}')
    rows.append(dict(label=label, X=X, Y=Y, n=n, mean_dH=mean, ci_lo=lo, ci_hi=hi, positive_splits=pos, verdict=verdict))
Q('')
for X, Y, lab in (('Rk2', 'Rk', 'C1 dose'), ('Rk4', 'Rk', 'C1 dose'), ('Rk2', 'Bg4', 'C2 baseline'), ('Rk4', 'Bg4', 'C2 baseline'), ('Rk2', 'Rks2', 'C3 allocation'), ('Rk4', 'Rks4', 'C3 allocation')):
    cmp(X, Y, lab); Q('')
Q('mechanism logs (last print of each run, mean over runs): kap_mean kap_p10 kap_p90 relchg_med fallback mean_shrink')
pat = {k: re.compile(k + r':([0-9.eE+-]+|nan)') for k in ('kap_mean', 'kap_p10', 'kap_p90', 'relchg_med', 'fallback', 'mean_shrink')}
for arm in ('Rk', 'Rk2', 'Rk4', 'Rks2', 'Rks4'):
    acc = {k: [] for k in pat}
    for c, f in splits:
        v = runs.get((arm, c, f))
        if v is None: continue
        lg = os.path.join(a.logs, f"log_{v['tag']}.txt")
        if not os.path.exists(lg): continue
        last = [ln for ln in open(lg, encoding='utf-8', errors='ignore') if ' aug=' in ln]
        if not last: continue
        for k, p in pat.items():
            mm = p.search(last[-1])
            if mm: acc[k].append(float(mm.group(1)))
    Q(f'  {arm:5s} ' + ('  '.join(f'{k} {np.mean(v):.3f}' if v else f'{k} n/a' for k, v in acc.items()) + f'  (logs: {len(acc["fallback"])} runs)'))
open(a.out, 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
if rows:
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
