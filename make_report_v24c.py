"""Figures D and E for report-v24c. Reads summary_v24s2.txt and v24d_dose.json (user's GPU summary and the sandbox dose scan). Run: MPLCONFIGDIR=/tmp/mpl python make_report_v24c.py"""
import re, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/projects/sandbox/pump/fonts/NotoSansSC.otf')
plt.rcParams['font.family'] = fm.FontProperties(fname='/projects/sandbox/pump/fonts/NotoSansSC.otf').get_name(); plt.rcParams['axes.unicode_minus'] = False
t = open('summary_v24s2.txt', encoding='utf-8').read()
def block(label):
    m = re.search(re.escape(label) + r' \(n=9\): mean dH ([+-][0-9.]+)\s+95% \[([+-][0-9.]+), ([+-][0-9.]+)\].*?\n\s+per-split dH: (.*?)\n', t)
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), np.array([float(v) for v in m.group(4).split()])
items = [('C1 dose: Rk2 - Rk', 'Rk2-Rk\n(剂量)'), ('C1 dose: Rk4 - Rk', 'Rk4-Rk\n(剂量)'), ('C2 baseline: Rk2 - Bg4', 'Rk2-Bg4\n(基线)'), ('C2 baseline: Rk4 - Bg4', 'Rk4-Bg4\n(基线)'),
         ('C3 allocation: Rk2 - Rks2', 'Rk2-Rks2\n(分配)'), ('C3 allocation: Rk4 - Rks4', 'Rk4-Rks4\n(分配)')]
fig, ax = plt.subplots(figsize=(9, 4.4))
for i, (lab, nm) in enumerate(items):
    mean, lo, hi, sp = block(lab)
    ax.plot(np.full(9, i) + np.linspace(-0.14, 0.14, 9), sp, 'o', color='#8fa8c8', ms=5)
    ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt='s', color='#c0392b', capsize=5, ms=7)
ax.axhline(0, color='k', lw=0.8); ax.axhline(0.02, color='gray', ls=':', lw=0.8); ax.axhline(-0.02, color='gray', ls=':', lw=0.8)
ax.set_xticks(range(len(items))); ax.set_xticklabels([n for _, n in items], fontsize=9); ax.set_ylabel('ΔH(留出安装,种子 0)')
ax.set_title('阶段二(探索性,单种子):9 个划分各一点,红为均值和 95% 区间;虚线为 ±0.02')
fig.tight_layout(); fig.savefig('figD_stage2_per_split.png', dpi=160); plt.close(fig)
rows = json.load(open('v24d_dose.json')); G = [0.0, 1.0, 2.0, 4.0, 8.0]
sig = [np.mean([r['gains'][str(g)]['relchg']['med'] for r in rows]) for g in G]; rr = [np.mean([r['gains'][str(g)]['r'] for r in rows]) for g in G]
shr = [np.mean([r['gains'][str(g)]['frac_shrunk'] for r in rows]) for g in G]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
ax[0].plot(G, sig, 'o-', color='#3b6ea5', label='实际信号变化(中位数)'); ax[0].plot(G, shr, 's--', color='#c0392b', label='被保真检查收缩的窗口比例')
ax[0].set_xlabel('--kr2-gain'); ax[0].set_title('剂量与实际扰动(预热模型,27 个平均)'); ax[0].legend(fontsize=8)
ax[1].plot(G, rr, 'o-', color='#2d7f5e'); ax[1].axhline(1, color='gray', ls=':'); ax[1].set_xlabel('--kr2-gain'); ax[1].set_ylabel('r = VMD 扰动的特征变化 / 两个标准视图的差异'); ax[1].set_title('特征响应(r=1:与两个标准视图的差异相当)')
fig.tight_layout(); fig.savefig('figE_dose_response.png', dpi=160); plt.close(fig)
