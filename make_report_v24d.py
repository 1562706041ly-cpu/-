"""Figure F for report-v24d: unknown-class separability change under the VMD view (v26). Reads v26_separability.json. Run: MPLCONFIGDIR=/tmp/mpl python make_report_v24d.py"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/projects/sandbox/pump/fonts/NotoSansSC.otf')
plt.rcParams['font.family'] = fm.FontProperties(fname='/projects/sandbox/pump/fonts/NotoSansSC.otf').get_name(); plt.rcParams['axes.unicode_minus'] = False
rows = json.load(open('v26_separability.json')); G = ['1.0', '2.0', '4.0']
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for r in rows: ax[0].plot([1, 2, 4], [r['gains'][g]['dP1'] for g in G], color='#8fa8c8', lw=0.8, alpha=0.7)
ax[0].plot([1, 2, 4], [np.mean([r['gains'][g]['dP1'] for r in rows]) for g in G], 'o-', color='#c0392b', lw=2, label='27 个模型的均值')
ax[0].axhline(0, color='k', lw=0.8); ax[0].set_xlabel('--kr2-gain'); ax[0].set_ylabel('ΔP1 = P1(A(x_vmd)) − P1(A(x))'); ax[0].set_xticks([1, 2, 4]); ax[0].legend(fontsize=8)
ax[0].set_title('未知类可分性随剂量的变化(预热模型)')
d2 = np.array([r['gains']['2.0']['dP1'] for r in rows]); nm = np.array([r['noise_mag'] for r in rows]); order = np.argsort(d2)
ax[1].bar(range(27), d2[order], color=['#c0392b' if d2[i] <= -(nm[i] + 0.01) else '#8fa8c8' for i in order])
ax[1].plot(range(27), -(nm[order] + 0.01), 'k_', ms=8, label='阈值 −(噪声 + 0.01)'); ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_xlabel('模型(按 ΔP1 排序)'); ax[1].set_title('gain 2:红色 17/27 低于阈值(规则要求 ≥ 23/27)'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig('figF_v26_separability.png', dpi=160)
