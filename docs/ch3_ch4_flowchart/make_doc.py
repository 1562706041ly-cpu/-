# -*- coding: utf-8 -*-
"""生成《第三、四章流程图公式解释》Word 文档。公式渲染为高清图片后嵌入。"""
import os, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "第三四章流程图_公式解释_v3.docx")
CN = "微软雅黑"

doc = Document()
st = doc.styles["Normal"]
st.font.name = CN; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), CN)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2)

def run_font(r, size=None, bold=None, color=None):
    r.font.name = CN
    r._element.rPr.rFonts.set(qn("w:eastAsia"), CN)
    if size: r.font.size = Pt(size)
    if bold is not None: r.bold = bold
    if color: r.font.color.rgb = RGBColor(*color)

def S(kind, text):
    col={"开题已明确":(0x2E,0x7D,0x32),"待确定":(0xD3,0x54,0x00),"推测":(0xC0,0x39,0x2B)}[kind]
    p = doc.add_paragraph()
    r = p.add_run("【"+kind+"】"); run_font(r, 9.5, True, col)
    r = p.add_run(text); run_font(r, 9.5, color=(0x55,0x55,0x55))
    p.paragraph_format.space_after = Pt(4)

def H(text, lvl=1):
    p = doc.add_paragraph()
    r = p.add_run(text)
    run_font(r, {0: 22, 1: 16, 2: 13, 3: 11.5}[lvl], True,
             (0x1F, 0x3A, 0x5F) if lvl < 3 else (0x2E, 0x7D, 0x32))
    p.paragraph_format.space_before = Pt(12 if lvl else 0)
    p.paragraph_format.space_after = Pt(4)
    if lvl == 0: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    return p

def T(text, bold_lead=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead); run_font(r, bold=True)
    r = p.add_run(text); run_font(r); r.italic = italic
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.3
    return p

def F(latex, width=None):
    """渲染 latex 公式为图片并居中插入"""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=19)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=False,
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    w, h = Image.open(buf).size
    buf.seek(0)
    cm_w = min(w / 300 * 2.54 * 0.62, 15.5)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(buf, width=Cm(cm_w))
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_together = True

def table(rows, widths, header=True):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            cell = t.cell(i, j); cell.width = Cm(widths[j])
            cell.text = ""
            r = cell.paragraphs[0].add_run(c); run_font(r, 9.5, bold=(i == 0 and header))
    doc.add_paragraph()

# ============ 封面/说明 ============
H("轴向柱塞泵开放环境故障诊断", 0)
H("第三章、第四章技术路线图：公式与原理解释（对照开题报告修订版 v3）", 0)
T("本文按流程图中每一个方框逐项给出：目的、公式、含义。公式编号 (3-x) 对应第三章增强方法，(4-x) 对应第四章比较方法。", italic=True)
T("v3 已对照你的开题报告（第1.3.2节研究内容与图1-1）修订：模块划分、输入输出、保真指标、难负样本策略、输出设定等均已与开题一致。开题只给出文字性的方法描述，没有给出公式，因此下面的公式仍是我根据文字描述写的“一种可行的数学表达”，不是你已经确定的公式。每小节开头的标注含义：绿色“开题已明确”表示开题文字明确写了这一点；橙色“待确定”表示开题没给出具体形式，需要你定；红色“推测”表示函数形式、阈值是我的建议。", bold_lead="说明：")
T("与 v2 相比的主要修正：(1) 增强预算方向：已知性低且可靠的样本适度加大扰动，已知性高且可靠的样本弱扰动，证据冲突/不确定样本保守扰动（v2 方向写反了）；(2) 难负样本改为“在可信负关系中混合合成”，难度预算由关系可信度决定，并有相似度上限与回退（v2 写成了 Top-H 挖掘）；(3) 未知类聚类按“未知类别数给定”的设定做半监督聚类，不再用轮廓系数选类别数；(4) 联合优化改为“已知类监督 + 同源视图 + 可靠跨样本关系”；(5) 保真约束与开题一致：相对扰动量、频谱形状相似性、RMS 相对变化；失败处理为预算缩减与保守视图回退。", italic=False)

# 流程图页(横向)
sec = doc.add_section()
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = sec.page_height, sec.page_width
sec.left_margin = sec.right_margin = Cm(1.5)
H("图 1  第三、四章技术路线图", 1)
doc.add_picture(os.path.join(HERE, "tech_route_ch3_ch4.png"), width=Cm(26))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

sec = doc.add_section()
sec.orientation = WD_ORIENT.PORTRAIT
sec.page_width, sec.page_height = sec.page_height, sec.page_width
sec.left_margin = sec.right_margin = Cm(2.2)

# ============ 符号表 ============
H("0  符号说明", 1)
table([
    ["符号", "含义"],
    ["x, s", "一条振动样本 / 其一维时域信号"],
    ["f_θ, z", "特征编码器及其输出特征 z = f_θ(x)"],
    ["C, μ_c", "已知故障类别数；第 c 类的原型（类中心）"],
    ["l_c(x), T", "分类 logit；能量温度"],
    ["K(x)", "已知性：样本像已知类的程度，越大越像已知"],
    ["R(x)", "可靠度：对 K(x) 这一判断有多可信，越大越可信"],
    ["b(x)", "增强预算：控制扰动强度与扰动模态数量"],
    ["u_k(t)", "VMD 分解得到的第 k 个本征模态分量"],
    ["z₁, z₂", "同一样本的两个视图（弱视图与自适应增强视图）的特征"],
    ["q_ij", "样本对 (i, j) 的关系可信度"],
], [3.5, 12.5])

# ============ 第三章 ============
H("第三章  已知性与可靠度联合控制的自适应增强", 1)
T("整体思路：先给每个样本打两个分——“像不像已知类”(K) 和“这个判断靠不靠谱”(R)；再用这两个分决定增强多强、扰动几个模态；"
  "最后用保真检查保证增强后的信号仍是真实的故障信号，不把类别改掉。")

H("3.1  输入", 2)
S('待确定', '第三章原稿尚未提供，输入的具体设定以原稿为准。')
T("少量带标签的已知类样本 𝒟_l = {(x_i, y_i)} 与大量无标签的混合样本 𝒟_u = {x_j}（其中既有已知类也含未知类）。"
  "编码器 f_θ 与分类头给出特征 z 与 logit l_c(x)；已知类原型取有标签样本特征的类内均值 μ_c。")

H("3.2  已知性 K(x)：能量 + 原型距离", 2)
S('开题已明确', '前期实验用过“与已知类原型的最大相似度”作为已知性分数（对应下面的 s(x)）；能量项以及两项加权的形式、权重均为推测，不代表论文最终公式。')
T("用两路证据加权：一路是“能量”（已知类样本的能量低），一路是到最近已知原型的距离（越近越像已知）。")
F(r"E(x)=-T\log\sum_{c=1}^{C}\exp\left(l_c(x)/T\right)\qquad(3\text{-}1)")
F(r"e(x)=\sigma\left(\frac{-E(x)-\tau_E}{\gamma}\right)\qquad(3\text{-}2)")
F(r"s(x)=\exp\left(-\frac{d_{\min}^{2}(x)}{2\sigma_d^{2}}\right),\quad d_{\min}(x)=\min_{c}\|z-\mu_c\|_2\qquad(3\text{-}3)")
F(r"K(x)=\lambda\,e(x)+(1-\lambda)\,s(x),\quad K(x)\in[0,1]\qquad(3\text{-}4)")
T("含义：E(x) 为能量分数（式 3-1），σ 为 sigmoid，τ_E、γ 是能量阈值与尺度；e(x) 把“能量低”映射成 0~1 的已知性证据；"
  "s(x) 是高斯核形式的“离已知原型有多近”；λ 为两路证据的权重。K 越接近 1 越像已知类，越接近 0 越像未知。")

H("3.3  可靠度 R(x)：三项一致性", 2)
S('开题已明确', '开题明确 R 由三类证据决定：能量与原型两类证据的一致性、跨弱增强视图稳定性、分流确定性。三项的具体公式和几何平均合并方式为推测；“分流”若指已知/未知分流，p(x) 的来源（此处令 p=K）仍待你确定。')
T("K 只是一个分数，它可能因为噪声而不准。可靠度衡量“多路证据是否互相印证”，包含三项：")
F(r"R_c(x)=1-\left|e(x)-s(x)\right|\qquad(3\text{-}5)")
F(r"R_v(x)=\frac{1}{2}\left(1+\frac{z^{(1)\top}z^{(2)}}{\|z^{(1)}\|\,\|z^{(2)}\|}\right)\qquad(3\text{-}6)")
F(r"R_s(x)=\left|2p(x)-1\right|,\quad p(x)=K(x)\qquad(3\text{-}7)")
F(r"R(x)=\left(R_c(x)\,R_v(x)\,R_s(x)\right)^{1/3}\qquad(3\text{-}8)")
T("含义：R_c 是“能量与距离一致性”——两路证据给的结论越接近，值越大（式 3-5）；"
  "R_v 是“两个弱视图稳定性”——同一样本的两个轻度增强视图 z⁽¹⁾、z⁽²⁾ 的特征余弦相似度，映射到 [0,1]（式 3-6）；"
  "R_s 是“分流确定性”——分流概率 p(x) 越靠近 0.5 越摇摆，|2p−1| 就越小（式 3-7）。三项取几何平均得到 R（式 3-8），任何一项很差，R 都会明显下降。")

H("3.4  样本证据分区（软权重，不做硬划分）", 2)
S('开题已明确', '开题写明按已知性与可靠度把样本分为：已知性高且可靠、已知性低且可靠、证据冲突或不确定三种情形。软权重写法是我的表达。')
F(r"w_{k}(x)=K(x)R(x),\quad w_{u}(x)=\left(1-K(x)\right)R(x),\quad w_{n}(x)=1-R(x)\qquad(3\text{-}9)")
T("含义：w_k、w_u、w_n 依次为“可靠已知 / 可靠未知 / 不确定”的隶属程度，三者之和为 1。")

H("3.5  样本级增强预算 b(x)", 2)
S('开题已明确', '开题的规则：已知性高且可靠 → 较弱扰动；已知性低且可靠 → 适度增加；证据冲突或不确定 → 保守扰动。下面用 b_k（弱）、b_u（适度加强）、b_c（保守）三个档位的软插值表达，档位数值为推测。')
F(r"b(x)=w_k\,b_k+w_u\,b_u+w_n\,b_c,\qquad b_k\leq b_c<b_u\qquad(3\text{-}11)")
F(r"\varepsilon(x)=\varepsilon_{\max}\,b(x),\qquad m(x)=\max\left(1,\lceil M\,b(x)\rceil\right)\qquad(3\text{-}12)")
T("含义：b(x) 即 w_k、w_u、w_n 对三档预算的加权平均。R 低时 w_n 大，b 趋向保守档 b_c；R 高且 K 高时趋向弱扰动 b_k；R 高且 K 低时趋向 b_u。ε(x) 为扰动幅度上限，m(x) 为被扰动的模态个数（M 为 VMD 模态数）。开题强调“增强控制机制与具体变换算子解耦”，因此 b(x) 只是控制量，换成别的增强算子时公式 3-11 不变。")

H("3.6  VMD 分解：得到各本征模态分量", 2)
S('开题已明确', 'VMD 本身是标准方法。本文档用 M 表示模态数，以免与已知性 K(x) 混淆（前期实验里的 K=8 指模态数）。前期 VMD 为信号级可行性实验，不等于已完成 VMD 增强模型训练。')
T("变分模态分解把信号拆成 M 个中心频率各异的窄带分量，每个分量对应一种相对独立的振动成分，便于“分模态”地做扰动。")
F(r"\min_{\{u_k\},\{\omega_k\}}\ \sum_{k=1}^{M}\left\|\partial_t\left[\left(\delta(t)+\frac{j}{\pi t}\right)*u_k(t)\right]e^{-j\omega_k t}\right\|_2^{2}\qquad(3\text{-}13)")
F(r"\mathrm{s.t.}\quad \sum_{k=1}^{M}u_k(t)=s(t)\qquad(3\text{-}14)")
T("含义：u_k、ω_k 是第 k 个模态及其中心频率；约束式 3-14 保证各模态之和等于原信号。模态数 M 与惩罚因子按常规方法或中心频率观察法确定。")

H("3.7  按预算扰动模态幅值，并重构增强视图", 2)
S('推测', '按能量选模态、均匀随机缩放的具体做法是示意，待核对。')
T("按能量大小挑出 m(x) 个模态，在其幅值上乘以随机缩放系数，其余模态保持不变，再把所有模态相加得到增强视图。")
F(r"\mathcal{S}(x)=\mathrm{TopK}_{k}\left(\|u_k\|_2^{2},\ m(x)\right)\qquad(3\text{-}15)")
F(r"\tilde u_k=(1+\epsilon_k)u_k\ (k\in\mathcal{S}),\quad \tilde u_k=u_k\ (k\notin\mathcal{S}),\quad \epsilon_k\sim U\left(-\varepsilon(x),\varepsilon(x)\right)\qquad(3\text{-}16)")
F(r"\tilde s(t)=\sum_{k=1}^{M}\tilde u_k(t)\qquad(3\text{-}17)")
T("含义：只动关键模态、并限定幅度上限 ε(x)，使增强后的信号 s̃ 仍保留原有的故障频谱结构，同时带来足够的多样性。")

H("3.8  保真检查：相对扰动量 / 频谱结构 / 整体振动能量", 2)
S('开题已明确', '开题明确设置相对扰动量、频谱形状相似性和 RMS 相对变化三项约束，并有预算缩减与保守视图回退。JS 散度作为“频谱形状相似性”的度量是我的选择，阈值待定。')
T("流程图菱形框中的三个检查对应三个指标：")
F(r"\delta_{\mathrm{rel}}=\frac{\|\tilde s-s\|_2}{\|s\|_2}\qquad(3\text{-}18)")
F(r"D_{\mathrm{spec}}=\mathrm{JS}\left(P_s\,\|\,P_{\tilde s}\right),\quad P_s(f)=\frac{|\mathcal{F}[s](f)|^{2}}{\sum_f|\mathcal{F}[s](f)|^{2}}\qquad(3\text{-}19)")
F(r"\rho_E=\left|\frac{\|\tilde s\|_2^{2}}{\|s\|_2^{2}}-1\right|\qquad(3\text{-}20)")
F(r"\mathrm{pass}\Longleftrightarrow \delta_{\mathrm{rel}}\leq\delta_{\max}\ \wedge\ D_{\mathrm{spec}}\leq\tau_s\ \wedge\ \rho_E\leq\tau_e\qquad(3\text{-}21)")
T("含义：δ_rel 是相对扰动量（式 3-18），衡量改了多少；D_spec 用 Jensen–Shannon 散度比较增强前后的归一化功率谱，衡量频谱结构有没有变形（式 3-19）；"
  "ρ_E 是整体振动能量的相对变化（式 3-20）。三项都不超限才算通过（式 3-21），通过后把增强视图送入第四章。")
F(r"b(x)\leftarrow\eta\,b(x)\ (0<\eta<1),\ \mathrm{retry}\leq T_{\max};\quad \mathrm{else}\ \tilde s\leftarrow\mathrm{ConservativeView}(s)\qquad(3\text{-}22)")
T("含义：未通过时按系数 η 缩减预算并重新扰动（预算缩减），多次仍不合格则回退到保守视图（例如仅轻微扰动或直接使用弱视图）。通过后得到基础弱视图 z₁ 与受控自适应视图 z₂，送入第四章。")

# ============ 第四章 ============
H("第四章  分流引导的可靠正负关系构建与对比学习", 1)
T("整体思路：对比学习需要“哪些样本对该拉近、哪些该推远”。无标签数据里这类关系往往不可靠，错拉/错推会损伤特征空间。"
  "所以本章的原则是“先判断关系是否可信，可信才拉近或推远”，不可信的关系降权或不用。")

H("4.1  输入", 2)
S('待确定', '第四章输入与符号以原稿为准。')
T("第三章得到的两个视图特征 z₁、z₂（弱视图与自适应增强视图）、每个样本的分流权重 (K, R)，以及已知类标签 y。"
  "记 i、j 为批内两个样本，余弦相似度为：")
F(r"s_{ij}=\frac{z_i^{\top}z_j}{\|z_i\|\,\|z_j\|}\qquad(4\text{-}1)")

H("4.2  三类关系证据", 2)
S('开题已明确', '结合标签锚点、局部邻域、跨视图信息判断关系，这个动机有依据；三项证据的具体公式（Jaccard 等）为推测。')
T("标签锚点：有标签样本之间关系最确定。", bold_lead="① ")
F(r"a_{ij}=+1\ (y_i=y_j);\quad -1\ (y_i\neq y_j);\quad 0\ (\mathrm{unlabeled})\qquad(4\text{-}2)")
T("局部邻域：互为 k 近邻且共享邻居多 → 正对候选；彼此都不是对方近邻、且几乎没有共享邻居 → 负对候选。", bold_lead="② ")
F(r"J_{ij}=\frac{|\mathcal{N}_k(i)\cap\mathcal{N}_k(j)|}{|\mathcal{N}_k(i)\cup\mathcal{N}_k(j)|}\qquad(4\text{-}3)")
F(r"n_{ij}=+J_{ij}\ \ (j\in\mathcal{N}_k(i)\ \mathrm{and}\ i\in\mathcal{N}_k(j))\qquad(4\text{-}4)")
F(r"n_{ij}=-(1-J_{ij})\ \ (j\notin\mathcal{N}_k(i),\ i\notin\mathcal{N}_k(j),\ J_{ij}<\tau_J);\quad n_{ij}=0\ \ (\mathrm{otherwise})")
T("跨视图一致性：弱视图与自适应视图下，这对样本的关系判断应该一致。", bold_lead="③ ")
F(r"A_{ij}=\mathbf{1}\left[(s^{(1)}_{ij}-\tau_s)(s^{(2)}_{ij}-\tau_s)>0\right],\quad c_{ij}=A_{ij}\left(1-\left|s^{(1)}_{ij}-s^{(2)}_{ij}\right|\right)\qquad(4\text{-}5)")
T("含义：J_ij 是 Jaccard 邻域重叠度，n_ij 的正负分别表示“倾向正对 / 倾向负对”；s⁽¹⁾、s⁽²⁾ 是两个视图下的相似度，"
  "A_ij 判断两个视图是否站在阈值 τ_s 的同一侧（关系方向一致），c_ij 在方向一致的基础上进一步要求相似度数值接近。")

H("4.3  关系可信度评估与可靠正/负/不确定划分", 2)
S('开题已明确', '开题：结合标签锚点、分流结果、暂定类别、邻域联系、跨视图关系一致性，划分可靠正关系、可靠负关系与不确定关系。融合公式与阈值为推测。使用真实标签的 O1/O2 只是上限分析，不能当作方法。')
T("“暂定类别”记为 ŷ_i（由分类头或原型最近邻给出），再加入一项“暂定类别是否一致” g_ij：")
F(r"g_{ij}=+\sqrt{\hat p_i^{\top}\hat p_j}\ \ (\hat y_i=\hat y_j);\quad -\left(1-\hat p_i^{\top}\hat p_j\right)\ \ (\hat y_i\neq\hat y_j)\qquad(4\text{-}6)")
F(r"t_{ij}=\alpha_1 a_{ij}+\alpha_2 g_{ij}+\alpha_3 n_{ij}\qquad(4\text{-}7)")
F(r"q_{ij}=c_{ij}\cdot\sqrt{R_iR_j}\cdot\min\left(1,|t_{ij}|\right)\qquad(4\text{-}8)")
F(r"\mathcal{P}=\{t_{ij}>0,\ q_{ij}\geq\tau_q\},\quad \mathcal{N}=\{t_{ij}<0,\ q_{ij}\geq\tau_q\},\quad \mathcal{U}=\mathrm{rest}\qquad(4\text{-}9)")
T("含义：p̂_i 为样本 i 的类别概率向量；t_ij 的符号表示正/负倾向，q_ij 要求跨视图一致、两个样本自身分流可靠且证据足够强。按 q_ij 是否超过 τ_q 划分可靠正关系 𝒫、可靠负关系 𝒩、不确定关系 𝒰。")
F(r"\omega_{ij}=q_{ij}\ \ ((i,j)\in\mathcal{P}\cup\mathcal{N});\quad \omega_{ij}=\lambda_u q_{ij}\ \ ((i,j)\in\mathcal{U}),\ \lambda_u\in[0,1)\qquad(4\text{-}10)")
T("含义：对三类关系实施差异化约束。可靠关系按可信度 q_ij 加权；不确定关系掩码或强降权（λ_u 取 0 即完全掩码，对应流程图灰色虚线框“降权或暂不使用”）。")

H("4.4  可信负关系中的受控难负样本构造", 2)
S('开题已明确', '开题：在可信负关系中筛选混合来源，把关系可信度映射为合成负样本的难度预算，用难度增益与相似度上限筛选受控难负样本，避免对低可信关系激进难化，无合格候选则回退。混合系数的映射形式为推测。')
T("动机：与锚点最相似的“难负样本”在混合数据里最可能是同类样本，直接推远会破坏未知类内部聚合，所以只对可信负关系合成难负样本，且难度由可信度控制。")
F(r"\lambda_{ij}=\lambda_{\max}\,\phi(q_{ij}),\quad \phi(q)=\left(\frac{q-\tau_q}{1-\tau_q}\right)_{+}\qquad(4\text{-}11)")
F(r"h_{ij}=\frac{(1-\lambda_{ij})\,z_j+\lambda_{ij}\,z_i}{\left\|(1-\lambda_{ij})\,z_j+\lambda_{ij}\,z_i\right\|}\qquad(j\in\mathcal{N}_i)\qquad(4\text{-}12)")
F(r"\Delta_{ij}=s(z_i,h_{ij})-s_{ij}\geq\delta_{\min},\qquad s(z_i,h_{ij})\leq s_{\max}\qquad(4\text{-}13)")
F(r"\mathcal{H}_i=\{h_{ij}:\ \mathrm{满足式\ 4\text{-}13}\}\ ;\quad \mathcal{H}_i=\varnothing\ \Rightarrow\ \mathrm{回退为原始负样本}\qquad(4\text{-}14)".replace(r"\mathrm{满足式\ 4\text{-}13}", r"\mathrm{qualified}").replace(r"\mathrm{回退为原始负样本}", r"\mathrm{fallback\ to\ original\ negatives}"))
T("含义：q_ij 越高，混合系数 λ_ij 越大，合成的负样本越靠近锚点（越“难”）；q 在阈值附近时 λ≈0，几乎不难化。式 4-12 在特征空间把负样本 z_j 与锚点 z_i 混合并归一化；式 4-13 要求难度增益 Δ 不小于 δ_min 且相似度不超过上限 s_max（防止把可能的同类样本当作负样本）；没有合格候选时回退到原始负样本（式 4-14）。")

H("4.5  分阶段联合训练", 2)
S('开题已明确', '开题：结合已知类监督、同源视图及可靠跨样本关系，分阶段启用，并周期更新特征、原型和关系。损失的具体形式和权重为推测。')
F(r"\mathcal{L}_{sup}=\mathcal{L}_{SupCon}(B_l)+\mathcal{L}_{CE}(B_l)\qquad(4\text{-}15)")
F(r"\mathcal{L}_{self}=-\frac{1}{|B|}\sum_{i\in B}\log\frac{\exp(s(z_i^{(1)},z_i^{(2)})/\tau)}{\sum_{k\neq i}\exp(s(z_i^{(1)},z_k^{(2)})/\tau)}\qquad(4\text{-}16)")
F(r"\mathcal{L}_{rel}=-\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\omega_{ij}\log\frac{\exp(s_{ij}/\tau)}{\exp(s_{ij}/\tau)+\sum_{h\in\mathcal{H}_i}\exp(s(z_i,h)/\tau)+\sum_{k\in\mathcal{N}_i}\omega_{ik}\exp(s_{ik}/\tau)}\qquad(4\text{-}17)")
F(r"\mathcal{L}=\mathcal{L}_{sup}+\lambda_s(e)\,\mathcal{L}_{self}+\lambda_r(e)\,\mathcal{L}_{rel}\qquad(4\text{-}18)")
T("含义：L_sup 为已知类监督（有标签样本的监督对比与交叉熵）；L_self 为同源视图对比（同一样本的 z⁽¹⁾、z⁽²⁾ 互为正对）；L_rel 为可靠关系加权对比，分母中包含合成难负样本 𝓗_i 与可信负关系。λ_s(e)、λ_r(e) 随训练轮数 e 分阶段启用：前期只开 L_sup 和 L_self，特征稳定后再逐步打开 L_rel，避免早期的错误关系放大。每隔若干轮用当前编码器重新计算特征、已知类原型和关系（式 4-19）。")
F(r"\{z,\ \mu_c,\ K,\ R,\ \hat y,\ q_{ij}\}^{(t+1)}\leftarrow f_{\theta^{(t+1)}}\qquad(4\text{-}19)")

H("4.6  输出：已知类识别与未知类聚类", 2)
S('开题已明确', '开题主实验给定未知类别数量（正常、滑靴松动为已知，其余三种为未知）；未知类真实标签只用于最终评价，不参与关系构建、调参或模型选择。基线为 GCD 特征学习加半监督 K-means。聚类细节为推测。')
F(r"\hat y(x)=\mathrm{arg\,max}_{c\in\{1,\dots,C\}}\ p(c\mid x),\quad x\in\mathcal{X}_k\qquad(4\text{-}20)")
F(r"\min_{\{\mu'_m\}}\sum_{x\in\mathcal{X}_{all}}\min_{m\in\{1,\dots,C+U\}}\|z-\mu'_m\|^2,\quad \mu'_c\ \mathrm{init\ from\ labeled},\ U=3\qquad(4\text{-}21)")
T("含义：已知类用分类头识别（式 4-20）。未知类在全部无标签样本特征上做半监督 K-means：前 C 个中心由有标签的已知类样本初始化，另设 U 个（本文为 3）未知中心，类别总数 C+U 给定（式 4-21）。评价指标按开题分为已知类识别、未知类聚类和整体类别发现三类。")

H("附：整体流程（算法概要）", 1)
for i, st_ in enumerate([
    "用少量有标签样本训练编码器，得到已知类原型 μ_c。",
    "弱视图特征 → 分类能量与原型距离 → 已知性 K(x)；再由证据一致性、跨弱视图稳定性、分流确定性得到可靠度 R(x)。",
    "按 K、R 分为可靠已知 / 可靠未知 / 不确定，映射为样本级增强预算 b(x)。",
    "VMD 分解，按预算扰动模态幅值并重构；做相对扰动量、频谱形状、RMS 相对变化检查，不合格则预算缩减，仍不合格回退保守视图。",
    "用标签锚点、分流与暂定类别、邻域、跨视图一致性计算关系可信度，划分可靠正 / 可靠负 / 不确定关系。",
    "在可信负关系中按可信度预算合成受控难负样本（有相似度上限，无合格候选则回退）。",
    "分阶段联合优化：已知类监督 + 同源视图对比 + 可靠关系加权对比；周期更新特征、原型和关系，回到第 2 步。",
    "输出：已知类用分类头识别；全部样本做半监督 K-means（未知类别数给定）发现未知故障簇。"], 1):
    T(st_, bold_lead=f"{i}. ")

H("与第五章消融/对照的对应关系", 1)
table([
    ["开题第5章的检验项","对应本文模块"],
    ["能量—原型联合估计","式 3-1~3-4 (K)"],
    ["可信度门控","式 3-5~3-9 (R 与分区)"],
    ["信号保真约束","式 3-18~3-22"],
    ["可靠关系构建","式 4-2~4-10"],
    ["难度预算耦合","式 4-11~4-14"],
    ["相同平均增强强度 / 固定难度预算 / 打乱条件对应关系","把 b(x)、λ_ij 替换为常数或随机打乱，检验自适应控制是否必要"],
], [7, 9])

doc.save(OUT)
print("saved", OUT)
