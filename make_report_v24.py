"""Tables + figures for report-v24. Reads only the user's GPU summaries (csv/txt). Run: MPLCONFIGDIR=/tmp/mpl python make_report_v24.py
Inputs: /projects/sandbox/results_v23/summary_v23_cf.csv (all pump methods, 27 runs each), summary_v24.txt / summary_v24.csv (v24 GPU summary)."""
import re, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/projects/sandbox/pump/fonts/NotoSansSC.otf')
plt.rcParams['font.family'] = fm.FontProperties(fname='/projects/sandbox/pump/fonts/NotoSansSC.otf').get_name(); plt.rcParams['axes.unicode_minus'] = False
d = pd.read_csv('/projects/sandbox/results_v23/summary_v23_cf.csv')
K = ['combo', 'fold', 'seed']
def get(m, col): return d[d.method == m].set_index(K)[col].sort_index()
def boot(x, n=10000):
    sp = x.groupby(level=['combo', 'fold']).mean().values; rng = np.random.default_rng(0)
    b = [sp[rng.integers(0, len(sp), len(sp))].mean() for _ in range(n)]; return np.percentile(b, [2.5, 97.5]), sp
def paired(X, Y, col):
    x = get(X, col) - get(Y, col); ci, sp = boot(x)
    return dict(X=X, Y=Y, metric=col, mean=x.mean(), lo=ci[0], hi=ci[1], higher=int((x > 0).sum()), pos_splits=int((sp > 0).sum()), per_split=sp)
SPEC = [('Bg4', 'B', 'v20 pre-registered (guard)'), ('EDdg', 'EDrg', 'v20 pre-registered (energy direction vs random)'), ('EDdg', 'Bg4', 'v20 pre-registered'),
        ('Gd', 'Bg4', 'v22 pre-registered (control)'), ('Gfs', 'Gd', 'v22 pre-registered rule 1'), ('Gfa', 'Gd', 'v22 pre-registered rule 1'), ('Gfs', 'Bg4', 'v22 pre-registered'), ('Gfa', 'Bg4', 'v22 pre-registered'),
        ('Mx', 'Bg4', 'v23 pre-registered'), ('Mx', 'Rg', 'v23 pre-registered'), ('Rg', 'Bg4', 'POST-HOC'), ('Rg', 'B', 'POST-HOC')]
rows = []
for X, Y, st in SPEC:
    for col in ('H', 'known_false_unk', 'unk_recall', 'unk_cluster_acc', 'nmi'):
        r = paired(X, Y, col); r['status'] = st; rows.append(r)
T = pd.DataFrame(rows).drop(columns='per_split'); T.to_csv('table_paired_v23data.csv', index=False)
M = d[d.method.isin(['B', 'Bg4', 'Rg', 'Mx', 'EDdg', 'EDrg'])].groupby('method')[['H', 'known_acc', 'unk_recall', 'known_false_unk', 'unk_cluster_acc', 'nmi']].mean().round(3)
M.to_csv('table_means_v23data.csv'); print(M); print(T[T.metric == 'H'].round(4).to_string())
txt = open('summary_v24.txt').read(); P = {}
for name in ('P1', 'P2', 'P3'):
    m = re.search(name + r' .*?\n\s+per-split mean dH: (.*?)\n', txt); P[name] = np.array([float(v) for v in m.group(1).split()])
v24 = pd.read_csv('summary_v24.csv')
labels = {'P1': 'Rk - Rg', 'P2': 'Rd - Rg', 'P3': 'Rk - Rd'}
fig, ax = plt.subplots(figsize=(7.5, 4))
for i, (k, v) in enumerate(P.items()):
    r = v24[v24.comparison == k].iloc[0]
    ax.plot(np.full(9, i) + np.linspace(-0.12, 0.12, 9), v, 'o', color='#8fa8c8', ms=5)
    ax.errorbar(i, r.mean_dH, yerr=[[r.mean_dH - r.ci_lo], [r.ci_hi - r.mean_dH]], fmt='s', color='#c0392b', capsize=5, ms=7)
ax.axhline(0, color='k', lw=0.8); ax.axhline(-0.02, color='gray', ls='--', lw=0.8); ax.text(2.45, -0.0185, '不劣界值 -0.02', fontsize=8, ha='right', color='gray')
ax.set_xticks(range(3)); ax.set_xticklabels([f'{k}\n{labels[k]}' for k in P]); ax.set_ylabel('ΔH(留出安装)')
ax.set_title('评分设计对照(v24,预注册):9 个划分各一点,红为均值和 95% 区间'); fig.tight_layout(); fig.savefig('figA_v24_scoring.png', dpi=160); plt.close(fig)
x = paired('Rg', 'Bg4', 'H'); y = paired('Mx', 'Bg4', 'H'); z = paired('Mx', 'Rg', 'H')
fig, ax = plt.subplots(figsize=(8, 4)); w = 0.27; idx = np.arange(9)
for j, (r, lab, c) in enumerate(((x, 'Rg - Bg4(事后)', '#c0392b'), (y, 'Mx - Bg4', '#3b6ea5'), (z, 'Mx - Rg', '#7f8c8d'))):
    ax.bar(idx + (j - 1) * w, r['per_split'], w, color=c, label=f"{lab}: 均值 {r['mean']:+.3f}")
ax.axhline(0, color='k', lw=0.8); ax.set_xticks(idx); ax.set_xticklabels([f'{c}f{f}' for c in ('C1', 'C2', 'C3') for f in (1, 2, 3)])
ax.set_ylabel('ΔH(3 个种子平均)'); ax.legend(fontsize=8); ax.set_title('第三章相关对照,按划分(Bg4 = 无第三章)'); fig.tight_layout(); fig.savefig('figB_chapter3_per_split.png', dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4.6))
for m in ('B', 'Bg4', 'Rg', 'Mx'):
    ax.scatter(M.loc[m, 'known_false_unk'], M.loc[m, 'unk_cluster_acc'], s=70); ax.annotate(m, (M.loc[m, 'known_false_unk'], M.loc[m, 'unk_cluster_acc']), xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('已知误判为未知(越低越好)'); ax.set_ylabel('未知聚类准确率(越高越好)'); ax.set_title('保护机制的权衡:B → Bg4'); fig.tight_layout(); fig.savefig('figC_guard_tradeoff.png', dpi=160); plt.close(fig)
