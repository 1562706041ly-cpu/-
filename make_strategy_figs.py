"""Strategy overview figures: energy attempts (two tables), week timeline, formula sheets (two pages).
Numbers come from the summaries already used in the report (see 汇报材料 / 能量公式清单).
Run: MPLCONFIGDIR=/tmp/mpl python make_strategy_figs.py"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch

fm.fontManager.addfont('/projects/sandbox/pump/fonts/NotoSansSC.otf')
plt.rcParams['font.family'] = fm.FontProperties(fname='/projects/sandbox/pump/fonts/NotoSansSC.otf').get_name()
plt.rcParams['axes.unicode_minus'] = False
COL = {'沿用': '#4c8bc9', '变差': '#e4572e', '无差异': '#9a9a9a', '淘汰': '#c9a227'}


import re
_TOK = re.compile(r"[A-Za-z0-9_.%+\-−→/≈≤²½·′()=<>:,~^*\[\]]+|\s+|.", re.S)
_NOSTART = set('）)。，、；：！？”')


def wrap(s, lim):
    """wrap by display width (CJK = 1, ascii = 0.55) without breaking ascii tokens; closing punctuation never starts a line"""
    out, cur, w = [], '', 0.0
    for tok in _TOK.findall(s):
        if tok == '\n': out.append(cur); cur, w = '', 0.0; continue
        tw = sum(1.0 if ord(c) > 255 else 0.55 for c in tok)
        if w + tw > lim and cur and not (tok[0] in _NOSTART and w + tw <= lim + 1.2):
            out.append(cur); cur, w = '', 0.0
            if tok.isspace(): continue
        cur += tok; w += tw
    out.append(cur)
    return '\n'.join(out)


def table(rows, title, fname, widths, header, fs=9.2):
    """widths in INCHES (data units == inches). Characters per line follow from the font size."""
    cw = fs / 72 * 1.04                                  # width of one CJK character in inches
    lim = [(w - 0.2) / cw for w in widths[:-1]]
    lh = fs / 72 * 1.5                                   # line height in inches
    wrapped = [[wrap(str(c), l) for c, l in zip(r[:-1], lim)] for r in rows]
    hs = [max(c.count('\n') + 1 for c in cells) * lh + 0.22 for cells in wrapped]
    W = sum(widths); head_h = 0.42; title_h = 0.6; H = title_h + head_h + sum(hs) + 0.1
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(-H, 0); ax.axis('off')
    ax.text(W / 2, -title_h / 2, title, ha='center', va='center', fontsize=14)
    y = -title_h; x = 0
    for h, w in zip(header, widths):
        ax.add_patch(Rectangle((x, y - head_h), w, head_h, color='#333')); ax.text(x + w / 2, y - head_h / 2, h, ha='center', va='center', color='w', fontsize=fs + 0.8); x += w
    y -= head_h
    for cells, r, h in zip(wrapped, rows, hs):
        x = 0
        for c, w in zip(cells, widths[:-1]):
            ax.add_patch(Rectangle((x, y - h), w, h, fc='#fafafa', ec='#cfcfcf', lw=0.6)); ax.text(x + 0.1, y - h / 2, c, ha='left', va='center', fontsize=fs, linespacing=1.4); x += w
        ax.add_patch(Rectangle((x, y - h), widths[-1], h, fc=COL[r[-1]], ec='white', lw=0.6))
        ax.text(x + widths[-1] / 2, y - h / 2, r[-1], ha='center', va='center', color='w', fontsize=fs + 1); y -= h
    plt.savefig(fname, dpi=150); plt.close()


HEAD = ['编号', '能量 / 距离的形式（怎么算）', '用在哪里（角色）', '数据与规模', '结果', '判定']
WID = [0.5, 4.4, 2.6, 2.6, 3.8, 0.8]
rows_a = [
    ('1', '分类头自由能\nE_h = −log Σ_c exp(w_c·f(x)+b_c)', '已知性 K 的一部分（最早的版本）', 'Paderborn 单工况', '能量的 AUROC 只有 0.44（低于随机），原型分数 0.64：分类头能量方向反了', '淘汰'),
    ('2', '距离能量（软 kNN）\nE_d = −T·log Σ_j exp(cos(f(x), l_j)/T)，T=0.05；T→0 即 kNN', '已知性 K（后处理主线），沿用至今', 'Paderborn、离心泵', '训练时纯距离 K 的 AUROC 0.80，高于门控 K 的 0.66。形式接近 Hopfield 能量和 kNN，不是创新点', '沿用'),
    ('3', '门控融合\nK = w·q(−E_h) + (1−w)·q(kNN)，w = q(kNN)²', '控制每个样本的扰动预算（A3g）', 'Paderborn 12 个种子；离心泵 27 对', 'A3g 对 A3d（K=距离能量）：Paderborn −0.001（4/12），离心泵 +0.001（11/27）', '无差异'),
    ('4', '余弦分类器能量（S1）\nt·log Σ_c exp(cos(f, w_c)/t)，t=0.1', '已知性 K（离线筛选）', '预热后的模型，2 个数据集', '离线筛选后淘汰（本次整理时没有保留具体数值）', '淘汰'),
    ('5', '远处无标注样本校准的头能量（S2b）', '已知性 K（离线筛选）', '同上', 'Paderborn 与离心泵的结果不一致', '淘汰'),
    ('6', '伪未知校准的头能量（S2a）\n头在冻结特征上重训：CE + 能量间隔损失', '已知性 K（离线筛选）', '同上', '能量本身明显变好（Paderborn 0.46→0.62；离心泵 C2 0.33→0.85），但与距离融合后超不过纯距离', '无差异'),
    ('7', '能量—距离一致性\nR3 = R0·(1 − |q(−E_S2a) − q(kNN)|)', '并入可靠度 R（离线筛选）', '同上', '路由 AUROC 变差', '淘汰'),
]
table(rows_a, '能量 / 距离作为"已知性 K"：试过的形式', 'fig7a_energy_forms.png', WID, HEAD)

rows_b = [
    ('8', 'E1 能量间隔损失\nλ·[ReLU(E_in−m_in)² + ReLU(m_out−E_out)²]\nE = −log Σ exp(head(f))', '损失项：VMD 伪未知窗口能量要高，已标注窗口要低', 'Paderborn 6 个种子（CPU 初筛）', 'H 0.732（A3g 0.831），已知误判 0.13→0.40，0/6 更好；去掉能量的对照 E1d 几乎一样（0.729）', '变差'),
    ('9', 'E2 能量语义保真\n保留条件：预测类别不变，且 E(x′) ≤ E(x) + tol', '约束：对判为已知的样本', '同上', 'H 0.838（A3g 0.831）；6 个种子里 4 个与 A3g 完全相同，约束几乎不起作用', '无差异'),
    ('10', 'E3 模态能量敏感度\nS_m = ½(|E(x+0.5c_m)−E(x)| + |E(x−0.5c_m)−E(x)|)', '给每个样本、每个 VMD 模态分配扰动预算', '同上', 'H 0.771（A3g 0.831），1/5 更好', '变差'),
    ('11', 'EDd 距离能量梯度方向\nu_m = λ·2·sign(s_K·∂E_d/∂a_m) + (1−λ)ε_m', '决定扰动方向（判为已知：能量升；判为未知：能量降），幅度上限不变', '离心泵 27 对（GPU）', '对随机方向 −0.005（12/27）；对 Bg4 −0.013（7/27）。沙箱初筛曾有 7/9 优于随机，没有重现', '无差异'),
    ('12', 'EDh 分类头能量梯度方向\n同上，E 取 E_h', '同上', '离心泵 9 个划分，种子 0（沙箱）', 'H 0.617，比随机方向好 5/9；没有上 GPU', '无差异'),
    ('13', 'EDd2：EDd + λ=1（纯能量方向）+ 扰动幅度加倍', '同上（事后探索）', '离心泵 9 个划分，种子 0（沙箱）', 'H 0.547（B 0.605），已知误判 0.56；扰动越强越差', '变差'),
    ('14', '保护里换成距离 / 融合人工特征\nGd：深度 kNN；Gfs / Gfa：再加人工特征 kNN', '已知类保护的评分', '离心泵 27 对（GPU）', 'Gd 对 Bg4 −0.002（12/27）；Gfs −0.25、Gfa −0.35，C3 上未知召回接近 0', '变差'),
]
table(rows_b, '能量 / 距离进入增强与保护：试过的用法', 'fig7b_energy_roles.png', WID, HEAD)

# ---------------------------------------------------------------- timeline
stages = [
    ('① 周初：第三章 v16–v17', '记录差异校准的 VMD 增强；\n门控能量 K 控制扰动预算', 'Paderborn：比 B +0.024（4/6）；\nA3g≈A3d，头能量门控无贡献', '无差异'),
    ('② 能量的离线筛选', '余弦能量 S1、远样本校准 S2b、\n能量—距离一致性 S3、伪未知校准 S2a', '只有 S2a 让能量本身变好；\n其余淘汰', '淘汰'),
    ('③ 让能量进入增强：E1 / E2 / E3', '能量间隔损失、语义保真、\n模态敏感度预算', '三个都没通过：E1 把已知误判\n推到 0.40，E3 变差，E2 不起作用', '变差'),
    ('④ 换数据集，定位瓶颈', '离心泵：波形输入在新安装上\nAUROC 0.26–0.34（训练安装 0.99）', '改 4 通道频谱：H 0.441→0.646\n（27 对中 24 对更好）', '沿用'),
    ('⑤ 保护校准 + 能量决定方向', '保护阈值按"留一安装"校准；\n能量梯度决定 VMD 扰动方向', '已知误判 0.304→0.177（H 持平）；\n能量方向对随机方向 −0.005（12/27）', '无差异'),
    ('⑥ 保护里换距离 / 融合人工特征', '深度 kNN；\n深度 + 人工特征融合', 'kNN 与距离能量一样（−0.002）；\n融合人工特征明显变差', '变差'),
]
fig = plt.figure(figsize=(12.6, 8.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 12.6); ax.set_ylim(0, 8.0)
ax.text(6.3, 7.65, '本周能量引导策略的演进', ha='center', fontsize=15)
for k, (n, c) in enumerate([('有效 / 沿用', COL['沿用']), ('无差异', COL['无差异']), ('淘汰', COL['淘汰']), ('变差', COL['变差'])]):
    ax.add_patch(Rectangle((3.9 + k * 1.5, 7.02), 0.2, 0.2, color=c)); ax.text(4.18 + k * 1.5, 7.12, n, fontsize=9, va='center')
for i, (t, did, res, v) in enumerate(stages):
    cx, cy = (i % 3) * 4.2 + 0.1, 4.0 - (i // 3) * 2.95
    ax.add_patch(FancyBboxPatch((cx, cy), 3.9, 2.65, boxstyle='round,pad=0.02', fc='#fafafa', ec='#999'))
    ax.add_patch(Rectangle((cx, cy + 2.15), 3.9, 0.5, color=COL[v])); ax.text(cx + 0.12, cy + 2.4, t, va='center', color='w', fontsize=10.4)
    ax.text(cx + 0.12, cy + 1.98, '做了什么', fontsize=8.6, color='#666', va='top'); ax.text(cx + 0.12, cy + 1.72, did, fontsize=9.6, va='top', linespacing=1.4)
    ax.text(cx + 0.12, cy + 1.05, '结果', fontsize=8.6, color='#666', va='top'); ax.text(cx + 0.12, cy + 0.79, res, fontsize=9.6, va='top', linespacing=1.4)
    if i % 3 < 2: ax.annotate('', xy=(cx + 4.17, cy + 1.3), xytext=(cx + 3.92, cy + 1.3), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(6.3, 0.42, '结论：能量作为"已知性""扰动多少""扰动方向"，在两个数据集上都没有稳定收益；\n瓶颈是跨安装差异，针对它的改动（4 通道频谱、留一安装校准）才有效。',
        ha='center', va='center', fontsize=10.2, color='#a02020', linespacing=1.5)
plt.savefig('fig8_timeline.png', dpi=150); plt.close()


# ---------------------------------------------------------------- formula sheets
def formula_sheet(items, title, fname, pitch=1.5):
    H = 0.9 + pitch * len(items)
    fig = plt.figure(figsize=(12.8, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 12.8); ax.set_ylim(0, H)
    ax.text(6.4, H - 0.4, title, ha='center', va='center', fontsize=14)
    y = H - 0.95
    for name, tex, note in items:
        ax.text(0.2, y, name, fontsize=10.4, color='#111', va='center')
        ax.text(0.55, y - 0.45, tex, fontsize=13, va='center')
        ax.text(0.55, y - 0.85, wrap(note, 112), fontsize=8.8, color='#666', va='top', linespacing=1.3)
        ax.plot([0.15, 12.65], [y - pitch + 0.18, y - pitch + 0.18], color='#ccc', lw=0.8)
        y -= pitch
    plt.savefig(fname, dpi=150); plt.close()


FA = [
    ('校准（所有已知性分数共用）', r'$q(v)=\sigma\!\left(\frac{v-v_{0.05}}{s}\right),\quad s=\frac{\mathrm{med}(v)-v_{0.05}}{\ln 19}$', 'v_0.05、med(v)：已标注窗口分数的 5% 分位数和中位数。校准后 q(5% 分位数)=0.5，q(中位数)=0.95'),
    ('分类头自由能 E_h（越高越不像已知）', r'$E_h(x)=-\log\sum_c \exp\!\left(w_c^{\top}f(x)+b_c\right)$', 'f(x)：编码器 64 维特征。代码里的已知性分数用 +logsumexp，即 −E_h，越大越像已知'),
    ('距离能量 E_d（越高越不像已知）', r'$E_d(x)=-T\log\sum_j \exp\!\left(\cos(f(x),l_j)/T\right),\quad T=0.05$', 'l_j：已标注窗口的特征。T→0 时变成 −max_j cos(f(x), l_j)，即 kNN。已标注窗口自己评分时排除自己（留一）'),
    ('kNN 分数（用于门控和保护）', r'$d(x)=\max_j \cos(f(x),l_j)$', '越大越像已知。保护阈值用同样的分数，只是校准时排除同一安装的窗口'),
    ('门控融合 K（A3g）', r'$K=w\,q(-E_h)+(1-w)\,q(d),\quad w=q(d)^{\gamma},\ \gamma=2$', '距离先判断样本是否落在已知支撑内，只有落在里面时头能量才有较大权重。A3d 则直接用 q(−E_d) 作为 K'),
    ('可靠度 R', r'$R=\left(1-|K_1-K_2|\right)\cdot|K_1+K_2-1|$', 'K_1、K_2：同一窗口的两个弱增强视图的 K。前一项是稳定性，后一项是决断性（K 离 0.5 越远越大）'),
    ('样本级扰动预算 κ', r'$\kappa=\kappa_{\min}+(1-\kappa_{\min})\,R\,[K+\rho(1-K)],\quad \kappa_{\min}=0.25,\ \rho=0.5$', '已标注窗口 κ=1。判为已知且可靠的样本扰动最大，不确定的样本最小'),
]
formula_sheet(FA, '能量 / 距离公式（一）：已知性、可靠度与预算', 'fig9a_formulas_K.png')

FB = [
    ('VMD 模态幅度扰动', r'$x^{\prime}=x+\sum_m\left(e^{a_m}-1\right)c_m,\qquad a_m=\frac{1}{2}\,u_m\,\kappa\,s_m$', 'c_m：第 m 个 VMD 模态；s_m = σ_rec,m·g_m：同类不同记录之间模态能量的自然差异；u_m ~ N(0,1) 截断到 ±2'),
    ('E1 能量间隔损失', r'$L_{E1}=\lambda\left\{\mathrm{ReLU}(E_{in}-m_{in})^2+\mathrm{ReLU}(m_{out}-E_{out})^2\right\}$', 'E = −log Σ exp(head(f))；m_in = 已标注窗口能量中位数，m_out = m_in + max(2, 2·std)；λ=0.1；E_out 用 VMD 生成的伪未知窗口'),
    ('E2 能量语义保真（接受条件）', r'$\mathrm{argmax}_c\,h_c(x^{\prime})=\mathrm{argmax}_c\,h_c(x)\ \wedge\ E(x^{\prime})\leq E(x)+\mathrm{tol}$', '只对判为已知的样本；h_c 是分类头输出；tol = 已标注窗口能量的标准差；不满足就把指数减半，最多 3 次，仍不满足则退回原窗口'),
    ('E3 模态能量敏感度', r'$S_m=\frac{1}{2}\left(|E(x+0.5c_m)-E(x)|+|E(x-0.5c_m)-E(x)|\right)$', '预算 s_{x,m} = σ_rec,m · g_{x,m}，g_x = 1/(1 + S_m/mean(S))，再按平均值重新缩放，使平均预算和类级预算一致'),
    ('能量对模态幅度的斜率', r'$g_m=\frac{\partial E}{\partial a_m}=\langle\nabla_x E,\;c_m\rangle$', '在 a=0 处求值：x 对 a_m 的导数正好是模态 c_m，所以一次反向传播就得到所有模态的斜率。E 取 E_d（EDd）或 E_h（EDh）'),
    ('能量决定方向（EDd / EDh）', r'$u_m=\lambda\cdot2\cdot\mathrm{sign}\left(s_K\,g_m\right)+(1-\lambda)\,\varepsilon_m$', 's_K = +1（判为已知，K≥0.5：让能量升）或 −1（判为未知：让能量降）；λ=0.5；ε_m ~ N(0,1) 截断到 ±2；幅度上限仍由 κ·s_m 给出。随机对照 EDr 用随机符号'),
    ('余弦分类器能量 S1（离线筛选）', r'$E_{cos}(x)=-t\log\sum_c\exp\!\left(\cos(f(x),w_c)/t\right),\quad t=0.1$', '代码里用 +t·logsumexp 作为已知性分数（越大越像已知）；w_c：分类头权重'),
    ('能量—距离一致性 S3（离线筛选）', r'$R_3=R_0\cdot\left(1-|q(-E_{S2a})-q(d)|\right)$', 'E_S2a：头在冻结特征上用伪未知窗口重训后的能量；R_0 就是上面的可靠度 R'),
]
formula_sheet(FB, '能量 / 距离公式（二）：进入增强的用法', 'fig9b_formulas_aug.png')
print('ok')
