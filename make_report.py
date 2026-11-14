"""Figures + tables for the centrifugal-pump report. Reads the user's GPU summaries (csv) only.
Run:  MPLCONFIGDIR=/tmp/mpl python make_report.py   (from /projects/sandbox/report_v22)"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle

fm.fontManager.addfont('/projects/sandbox/pump/fonts/NotoSansSC.otf')
plt.rcParams['font.family'] = fm.FontProperties(fname='/projects/sandbox/pump/fonts/NotoSansSC.otf').get_name()
plt.rcParams['axes.unicode_minus'] = False
R = '/projects/sandbox/'
old = pd.read_csv(R + 'results_v18/summary_v18_cf.csv'); old = old[old.method == 'B'].assign(method='B_old')
v19 = pd.read_csv(R + 'results_v19/summary_v19_cf_all.csv'); v19 = v19[v19.method != 'B'].assign(method=lambda x: 'v19_' + x.method)
v22 = pd.read_csv(R + 'results_v22/summary_v22_cf.csv')
d = pd.concat([old, v19, v22], ignore_index=True)
K = ['combo', 'fold', 'seed']
rng = np.random.default_rng(0)


def split_boot(x, n=5000):
    """x: Series indexed by (combo, fold, seed). Bootstrap over the 9 (combo, fold) splits -> 95 % CI of the mean."""
    cell = x.groupby(level=['combo', 'fold']).mean().values
    bs = [rng.choice(cell, len(cell)).mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


def get(m, col): return d[d.method == m].set_index(K)[col]


def paired(a, b, col):
    x = (get(a, col) - get(b, col)).dropna()
    return x.mean(), split_boot(x), int((x > 0).sum()), len(x)


# ---------------------------------------------------------------- table 1: all methods
cols = ['H', 'known_acc', 'unk_recall', 'known_false_unk', 'unk_cluster_acc']
names = {'B_old': '旧基线：通道4 + 波形', 'B': '新基线B：4通道 + 频谱', 'Bg3': 'B + 留一安装保护 q=0.3', 'Bg4': 'B + 留一安装保护 q=0.4（主方法）',
         'Bg5': 'B + 留一安装保护 q=0.5', 'v19_A3g': '第三章增强 A3g（能量—距离门控K）', 'v19_A3d': '第三章增强 A3d（无能量门控）',
         'v19_A3c': '第三章增强 A3c（预算打乱对照）', 'EDdg': '第三章：能量决定方向 EDdg', 'EDrg': '第三章：随机方向 EDrg（对照）',
         'Gd': '保护换深度kNN距离 Gd', 'Gfs': '保护融合人工特征（频带）Gfs', 'Gfa': '保护融合人工特征（全部）Gfa'}
t1 = d.groupby('method')[cols].mean().loc[list(names)].round(3)
t1.insert(0, '方法', [names[m] for m in t1.index]); t1.insert(1, 'n', [int((d.method == m).sum()) for m in t1.index])
t1.to_csv('table1_all_methods.csv', encoding='utf-8-sig')

# ---------------------------------------------------------------- table 2: paired comparisons
pairs = [('B', 'B_old', '新基线 vs 旧基线（4通道+频谱）'), ('Bg4', 'B', '留一安装保护 vs 新基线'), ('Bg4', 'B_old', '主方法 vs 旧基线'),
         ('v19_A3g', 'B', '第三章 A3g vs B'), ('v19_A3g', 'v19_A3c', 'A3g vs 预算打乱 A3c'), ('v19_A3g', 'v19_A3d', 'A3g vs 无能量门控 A3d'),
         ('EDdg', 'EDrg', '能量方向 vs 随机方向'), ('EDdg', 'Bg4', '能量方向 vs Bg4'), ('Gd', 'Bg4', '深度kNN保护 vs Bg4'),
         ('Gfs', 'Gd', '融合人工特征（频带）vs Gd'), ('Gfa', 'Gd', '融合人工特征（全部）vs Gd')]
rows = []
for a, b, lab in pairs:
    r = {'比较': lab}
    for col, nm in (('H', 'H'), ('known_false_unk', '已知误判未知'), ('unk_recall', '未知召回')):
        m, ci, w, N = paired(a, b, col)
        r[nm + ' 平均差'] = round(m, 3); r[nm + ' 95%区间'] = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'
        if col == 'H': r['H 更好的对数'] = f'{w}/{N}'
    rows.append(r)
pd.DataFrame(rows).to_csv('table2_paired.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- fig1: split design (C1)
TYPES = [('健康', ['healthy 1', 'healthy 2', 'healthy 3'], '已知'), ('电机轴承外圈 BPFO（已知故障）', ['bpfo 1', 'bpfo 2', 'bpfo 3'], '已知'),
         ('叶轮', ['impeller 1', 'impeller 2', 'impeller 3'], '未知'), ('泵轴承', ['pump bearing 1', 'pump bearing 2', 'pump bearing 3'], '未知'),
         ('软脚', ['soft foot 1', 'soft foot 2'], '未知'), ('电机轴承内圈 BPFI', ['bpfi 1', 'bpfi 2', 'bpfi 3'], '未使用')]
fig, ax = plt.subplots(figsize=(8.2, 6.4))
y = 0; yt = []; yl = []
for ti, (tn, inst, role) in enumerate(TYPES):
    for j, nm in enumerate(inst):
        for f in (1, 2, 3):
            if role == '未使用': col = '#dddddd'
            else:
                ev = (j + 1 == f) if len(inst) == 3 else (j == (f - 1) % 2)
                col = '#e4572e' if ev else '#4c8bc9'
            ax.add_patch(Rectangle((f - 0.45, -y - 0.42), 0.9, 0.84, color=col, ec='white'))
        yt.append(-y); yl.append(nm); y += 1
    ax.text(3.62, -(y - 1 - (len(inst) - 1) / 2), f'{tn}\n[{role}类]', va='center', fontsize=9, linespacing=1.5)
    y += 0.5
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8.5); ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['第1折', '第2折', '第3折'])
ax.set_xlim(0.4, 5.6); ax.set_ylim(-y + 0.3, 0.8); ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False); ax.tick_params(length=0)
ax.add_patch(Rectangle((0, 0), 0, 0, color='#4c8bc9', label='训练（每个窗口只在已知类的10%上有标签）'))
ax.add_patch(Rectangle((0, 0), 0, 0, color='#e4572e', label='评估（该安装在这一折里从未出现在训练中）'))
ax.add_patch(Rectangle((0, 0), 0, 0, color='#dddddd', label='本组合未使用'))
ax.legend(loc='lower center', bbox_to_anchor=(0.45, -0.17), ncol=1, frameon=False, fontsize=9)
ax.set_title('数据划分：按"安装"划分的3折（以组合C1为例）', fontsize=12)
plt.tight_layout(); plt.savefig('fig1_split.png', dpi=170); plt.close()

# ---------------------------------------------------------------- fig2: pipeline
fig, ax = plt.subplots(figsize=(10.6, 3.9)); ax.axis('off'); ax.set_xlim(0, 10.5); ax.set_ylim(0, 4)
boxes = [(0.1, '振动窗口\n4通道×2048点\n(0.1 s, 约2.5转)', '#f4e3b2'), (1.85, '汉宁窗 + FFT\n对数幅度谱\n(每通道1025点)', '#cfe5cf'),
         (3.6, '频率轴一维CNN\n对比学习\n(仅10%已知标签)', '#cfe0f4'), (5.35, '已知性 K\n距离能量/kNN\n可靠度 R', '#e8d3ee'),
         (7.1, '半监督k-means\n已知:2子中心/类\n未知:2个中心', '#f4d3d3'), (8.85, '已知类保护\n阈值τ按\n"留一安装"校准', '#f9c9a8')]
for x, t, c in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.5), 1.45, 1.6, boxstyle='round,pad=0.04', fc=c, ec='#555')); ax.text(x + 0.725, 2.3, t, ha='center', va='center', fontsize=8.6)
for x in (1.57, 3.32, 5.07, 6.82, 8.57): ax.annotate('', xy=(x + 0.27, 2.3), xytext=(x, 2.3), arrowprops=dict(arrowstyle='->', lw=1.4))
ax.text(5.25, 3.65, '本文流程（离心泵，输入为多通道频谱）', ha='center', fontsize=12)
ax.text(5.25, 0.95, '输出：已知类（健康 / 已知故障）或"未知"；未知窗口再分到2个未知簇', ha='center', fontsize=9.5)
ax.text(5.25, 0.45, '4通道+频谱输入：H 0.441→0.646；留一安装保护校准：H持平(0.646→0.650)，已知误判未知 0.304→0.177，未知召回 0.744→0.626', ha='center', fontsize=9, color='#a02020')
plt.savefig('fig2_pipeline.png', dpi=170, bbox_inches='tight'); plt.close()

# ---------------------------------------------------------------- fig3: progress (H, known_false_unk, unk_recall)
ms = ['B_old', 'B', 'Bg4']; lab = ['旧基线\n通道4+波形', '新基线B\n4通道+频谱', '主方法Bg4\n+留一安装保护']
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.7))
for ax, (col, ttl, better) in zip(axs, [('H', 'H（越高越好）', 'up'), ('known_false_unk', '已知误判为未知（越低越好）', 'down'), ('unk_recall', '未知召回', 'up')]):
    mean = [get(m, col).mean() for m in ms]; ci = np.array([split_boot(get(m, col)) for m in ms])
    ax.bar(range(3), mean, yerr=[np.array(mean) - ci[:, 0], ci[:, 1] - np.array(mean)], color=['#9a9a9a', '#4c8bc9', '#e4572e'], capsize=4)
    for i, v in enumerate(mean): ax.text(i + 0.12, ci[i, 1] + 0.012, f'{v:.3f}', ha='left', fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=8); ax.set_title(ttl, fontsize=10); ax.set_ylim(0, max(ci[:, 1]) * 1.18)
fig.suptitle('离心泵 27组配对实验：平均值与95%区间（对9个划分自助重采样）', fontsize=11)
plt.tight_layout(); plt.savefig('fig3_progress.png', dpi=170); plt.close()

# ---------------------------------------------------------------- fig4: guard operating curve
fig, ax = plt.subplots(figsize=(5.6, 4.3))
pts = [('B', '无校准 B'), ('Bg3', 'q=0.3'), ('Bg4', 'q=0.4'), ('Bg5', 'q=0.5')]
xs = [get(m, 'known_false_unk').mean() for m, _ in pts]; ys = [get(m, 'unk_recall').mean() for m, _ in pts]
ax.plot(xs[1:], ys[1:], '-o', color='#e4572e', label='留一安装校准（q越大，保护越弱）'); ax.plot(xs[0], ys[0], 's', color='#4c8bc9', ms=9, label='无校准（旧阈值）')
for (m, t), x_, y_ in zip(pts, xs, ys): ax.annotate(f'{t}\nH={get(m, "H").mean():.3f}', (x_, y_), textcoords='offset points', xytext=(8, -4), fontsize=8)
ax.set_xlim(0.13, 0.36); ax.set_xlabel('已知误判为未知（越低越好）'); ax.set_ylabel('未知召回（越高越好）'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='lower right')
ax.set_title('保护阈值是一条工作曲线：用未知召回换已知准确率\n（H基本不变）', fontsize=10)
plt.tight_layout(); plt.savefig('fig4_guard_curve.png', dpi=170); plt.close()

# ---------------------------------------------------------------- fig5: negative results (paired delta H)
neg = [('v19_A3g', 'B', '第三章 A3g − B'), ('v19_A3g', 'v19_A3c', 'A3g − 预算打乱 A3c'), ('v19_A3g', 'v19_A3d', 'A3g − 无能量门控 A3d (能量的贡献)'),
       ('EDdg', 'EDrg', '能量方向 − 随机方向'), ('EDdg', 'Bg4', '能量方向 − Bg4'), ('Gd', 'Bg4', '深度kNN保护 − Bg4'),
       ('Gfs', 'Gd', '融合人工特征(频带) − Gd'), ('Gfa', 'Gd', '融合人工特征(全部) − Gd')]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for i, (a, b, t) in enumerate(neg):
    m, ci, w, N = paired(a, b, 'H')
    ax.errorbar(m, -i, xerr=[[m - ci[0]], [ci[1] - m]], fmt='o', color='#e4572e' if ci[1] < 0 else '#444', capsize=4)
    ax.text(0.405, -i, f'{w}/{N}', va='center', fontsize=8, clip_on=False)
ax.axvline(0, color='k', lw=1); ax.set_yticks([-i for i in range(len(neg))]); ax.set_yticklabels([t for *_, t in neg], fontsize=8.5)
ax.set_xlim(-0.45, 0.4); ax.set_xlabel('H 的配对平均差（95%区间，对9个划分重采样）'); ax.text(0.405, 0.8, 'H更好的对数', fontsize=8, clip_on=False)
ax.set_title('第三章及其变体：均未显示稳定的改进（离心泵，27对）', fontsize=11); ax.grid(axis='x', alpha=.3)
plt.tight_layout(); plt.savefig('fig5_negative.png', dpi=170); plt.close()

# ---------------------------------------------------------------- fig6: per fold
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7))
for ax, (col, ttl) in zip(axs, [('H', 'H'), ('known_false_unk', '已知误判为未知')]):
    for k, (m, l, c) in enumerate([('B_old', '旧基线', '#9a9a9a'), ('B', '新基线B', '#4c8bc9'), ('Bg4', '主方法Bg4', '#e4572e')]):
        v = [get(m, col).xs(f, level='fold').mean() for f in (1, 2, 3)]
        ax.bar(np.arange(3) + (k - 1) * 0.27, v, 0.26, color=c, label=l)
    ax.set_xticks(range(3)); ax.set_xticklabels(['第1折\n(健康时段1)', '第2折', '第3折']); ax.set_title(ttl, fontsize=10)
axs[0].set_ylim(0, 0.9); axs[0].legend(fontsize=8, ncol=3, loc='upper center'); fig.suptitle('分折结果（第1折的健康类是健康时段1，它在试验中紧接叶轮之前测量）', fontsize=11)
plt.tight_layout(); plt.savefig('fig6_per_fold.png', dpi=170); plt.close()
print(t1.to_string()); print(pd.DataFrame(rows).to_string())
d.to_csv('all_runs_used.csv', index=False)
