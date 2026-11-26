"""A1 section 7 item 3: baseline port regression check. Reads the E0 run on C1_fold1 (seed 0) that evaluated ALL
800 target windows (--eval-part all) and compares H with 0.694737 (tolerance 0.005).
Usage: python check_regression.py [--root runs_paderborn] [--tag DA1_REG_E0_C1_f1_seed0_all] [--data <npz>] [--out regression_result.txt]
Exit code 0 = PASS, 1 = FAIL / not evaluable."""
import argparse, hashlib, json, os, sys, datetime

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='runs_paderborn')
ap.add_argument('--tag', default='DA1_REG_E0_C1_f1_seed0_all')
ap.add_argument('--data', default=None, help='split file used (its sha256 is recorded)')
ap.add_argument('--out', default='regression_result.txt')
ap.add_argument('--target', type=float, default=0.694737)
ap.add_argument('--tol', type=float, default=0.005)
a = ap.parse_args()

lines, ok = [], False
try:
    r = json.load(open(os.path.join(a.root, a.tag, 'results_exp2.json'), encoding='utf-8'))['PADERBORN_seed0']
    h = r['openset']['V']['H']
    n_val = r['n_val']
    lines.append(f"tag={a.tag} n_val={n_val} n_labeled={r['n_labeled']} n_unlabeled={r['n_unlabeled']}")
    lines.append(f"known_acc={r['openset']['V']['known_acc']:.6f} unk_recall={r['openset']['V']['unk_recall']:.6f} H={h:.6f}")
    lines.append(f"target={a.target} tol={a.tol} |dH|={abs(h - a.target):.6f}")
    ok = (n_val == 800) and abs(h - a.target) <= a.tol and not r['args'].get('use_adapt')
    if n_val != 800: lines.append('n_val is not 800: the run did not use --eval-part all')
except Exception as e:
    lines.append(f'cannot evaluate: {e!r}')
if a.data and os.path.exists(a.data):
    lines.append('data sha256 ' + hashlib.sha256(open(a.data, 'rb').read()).hexdigest())
lines.append('RESULT: ' + ('PASS' if ok else 'FAIL'))
txt = '\n'.join(lines)
print(txt)
with open(a.out, 'w', encoding='utf-8') as f:
    f.write(datetime.datetime.now().isoformat(timespec='seconds') + '\n' + txt + '\n')
sys.exit(0 if ok else 1)
