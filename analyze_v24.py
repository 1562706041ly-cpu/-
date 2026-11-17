"""v24 analysis: Rg (existing) vs Rk (pure NN) vs Rd (distance energy) on the Bruinsma pump. ASCII-only output.
Usage: python analyze_v24.py [--root runs_paderborn] [--out summary_v24.txt] [--csv summary_v24.csv] [--force]
Step 1 verifies pairing from the stored args / labelled indices (aborts on failure unless --force; --force results are NOT valid evidence).
Step 2 applies the pre-registered rules of v24_endtoend_PREREGISTERED.txt.
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='runs_paderborn'); ap.add_argument('--out', default='summary_v24.txt'); ap.add_argument('--csv', default='summary_v24.csv')
ap.add_argument('--pairing-out', default='pairing_report_v24.txt'); ap.add_argument('--force', action='store_true')
a = ap.parse_args()
TAG = re.compile(r'V20_(C\d)_f(\d)_(Rg|Rk|Rd|Rgrep)_seed(\d+)$')
EXPECT_KST = {'Rg': 'gated', 'Rk': 'knn', 'Rd': None, 'Rgrep': 'gated'}
IGN = {'tag', 'seeds', 'k_score_train'}
runs = {}
for f in sorted(glob.glob(os.path.join(a.root, 'V20_*_seed*', 'results_exp2.json'))):
    m = TAG.search(os.path.basename(os.path.dirname(f)))
    if not m: continue
    key = (m.group(3), m.group(1), int(m.group(2)), int(m.group(4)))
    r = list(json.load(open(f, encoding='utf-8')).values())[0]
    o = r['openset']['V']
    runs[key] = dict(H=o['H'], known_false_unk=o['known_false_unk'], unk_recall=o['unk_recall'], known_acc=o['known_acc'],
                     nmi=r['M2_GCD+半监督KMeans']['V']['nmi'], args=r['args'], lab=list(r['labeled_idx']),
                     n=(r['n_labeled'], r['n_unlabeled'], r['n_val']))
rep = []; P = rep.append
fails = []
splits = [(c, f) for c in ('C1', 'C2', 'C3') for f in (1, 2, 3)]
keys27 = [(c, f, s) for c, f in splits for s in (0, 1, 2)]
have = {me: sum((me, c, f, s) in runs for c, f, s in keys27) for me in ('Rg', 'Rk', 'Rd')}
P(f'runs found (of 27): {have};  Rgrep: {sum(k[0] == "Rgrep" for k in runs)} (of 3)')
for me in ('Rg', 'Rk', 'Rd'):
    if have[me] != 27: fails.append(f'{me}: {have[me]} of 27 runs present')
ref = runs.get(('Rg', 'C1', 1, 0))
for (me, c, f, s), v in sorted(runs.items()):
    kst = v['args'].get('k_score_train')
    if kst != EXPECT_KST[me]: fails.append(f'{me} {c} f{f} s{s}: k_score_train={kst!r}, expected {EXPECT_KST[me]!r}')
    if ref is not None:
        diff = {k: (ref['args'].get(k), v['args'].get(k)) for k in set(ref['args']) | set(v['args']) if k not in IGN and ref['args'].get(k) != v['args'].get(k)}
        # split-specific data path differs by design
        diff.pop('data', None)
        if diff: fails.append(f'{me} {c} f{f} s{s}: args differ from Rg C1 f1 s0: {diff}')
    base = runs.get(('Rg', c, f, s))
    if base is not None and me != 'Rg':
        if v['lab'] != base['lab']: fails.append(f'{me} {c} f{f} s{s}: labelled indices differ from Rg')
        if v['n'] != base['n']: fails.append(f'{me} {c} f{f} s{s}: n_labeled/n_unlabeled/n_val {v["n"]} vs Rg {base["n"]}')
P('pairing checks: ' + ('PASSED' if not fails else f'FAILED ({len(fails)} problems)'))
for x in fails[:60]: P('  - ' + x)
open(a.pairing_out, 'w').write('\n'.join(rep) + '\n')
print('\n'.join(rep))
if fails and not a.force:
    print('STOP: pairing not verified. Do not interpret. Send pairing_report_v24.txt to Kiro. (Re-run all arms on the GPU if Rg differs.)'); sys.exit(2)

out = []; Q = out.append
Q('v24 pump end-to-end comparison (criteria: v24_endtoend_PREREGISTERED.txt)')
if fails: Q('WARNING: --force used, pairing NOT verified, results are not valid evidence')
Q('')
mets = ['H', 'known_acc', 'unk_recall', 'known_false_unk', 'nmi']
Q('overall mean +- std over 27 runs')
for me in ('Rg', 'Rk', 'Rd'):
    Q(f'  {me:3s} ' + '  '.join(f'{k} {np.mean([runs[(me, c, f, s)][k] for c, f, s in keys27 if (me, c, f, s) in runs]):.3f}' for k in mets))
rng = np.random.default_rng(0); rows = []
def compare(X, Y):
    d = np.array([runs[(X, c, f, s)]['H'] - runs[(Y, c, f, s)]['H'] for c, f, s in keys27])
    sp = d.reshape(9, 3).mean(1)
    boots = np.array([sp[rng.integers(0, 9, 9)].mean() for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5]); mean = float(d.mean()); wins = int((d > 0).sum()); posp = int((sp > 0).sum())
    if mean >= 0.02 and wins >= 18 and posp >= 6 and lo > 0: verdict = 'better'
    elif mean <= -0.02 and hi < 0: verdict = 'worse'
    elif mean >= -0.01 and lo >= -0.02: verdict = 'no worse (not shown better)'
    else: verdict = 'inconclusive'
    sec = {k: float(np.mean([runs[(X, c, f, s)][k] - runs[(Y, c, f, s)][k] for c, f, s in keys27])) for k in mets[1:]}
    return mean, lo, hi, wins, posp, sp, verdict, sec
Q('')
for X, Y, name in (('Rk', 'Rg', 'P1'), ('Rd', 'Rg', 'P2'), ('Rk', 'Rd', 'P3')):
    mean, lo, hi, wins, posp, sp, verdict, sec = compare(X, Y)
    Q(f'{name} {X} - {Y}: mean dH {mean:+.4f}  95% interval over splits [{lo:+.4f}, {hi:+.4f}]  higher in {wins}/27 runs, positive in {posp}/9 splits')
    Q('   per-split mean dH: ' + ' '.join(f'{v:+.3f}' for v in sp))
    Q('   secondary mean differences: ' + '  '.join(f'{k} {v:+.3f}' for k, v in sec.items()))
    Q(f'   VERDICT (pre-registered rule): {verdict}')
    rows.append(dict(comparison=name, X=X, Y=Y, mean_dH=mean, ci_lo=lo, ci_hi=hi, wins27=wins, pos_splits9=posp, verdict=verdict))
reps = [(k, runs[k]['H'] - runs[('Rg',) + k[1:]]['H']) for k in sorted(runs) if k[0] == 'Rgrep' and ('Rg',) + k[1:] in runs]
Q('')
if reps:
    ad = [abs(v) for _, v in reps]
    Q('Rgrep noise floor |H(Rgrep) - H(Rg)| on 3 repeats: ' + ' '.join(f'{x:.3f}' for x in ad) + f'  mean {np.mean(ad):.3f}')
    if np.mean(ad) > 0.03: Q('  mean above 0.03: any |mean dH| below the noise floor is NOT interpretable (pre-registered caveat)')
else: Q('Rgrep repeats not found')
open(a.out, 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
with open(a.csv, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
