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
OUT = os.path.join(os.path.dirname(HERE), "第三四章流程图_公式解释_v2.docx")
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
    col={"部分有依据":(0x2E,0x7D,0x32),"待核对":(0xD3,0x54,0x00),"推测":(0xC0,0x39,0x2B)}[kind]
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
H("第三章、第四章技术路线图：公式与原理解释（草稿，待核对原稿）", 0)
T("本文按流程图中每一个方框逐项给出：目的、公式、含义。公式编号 (3-x) 对应第三章增强方法，(4-x) 对应第四章比较方法。", italic=True)
T("本文档是在没有读到你论文原稿的情况下，根据流程图写的草稿。每一小节开头标注了状态：绿色“部分有依据”表示动机或某个指标有前期实验支持；橙色“待核对”表示要以原稿为准；红色“推测”表示公式和参数是我按常规做法设的，不能当作论文已定稿的方法。前期可行性实验和探索性对照，不等于第三、四章方法已经定稿或验证。符号 K、R、b、p 的正式写法也待与原稿统一。", bold_lead="重要说明：")

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
S('待核对', '第三章原稿尚未提供，输入的具体设定以原稿为准。')
T("少量带标签的已知类样本 𝒟_l = {(x_i, y_i)} 与大量无标签的混合样本 𝒟_u = {x_j}（其中既有已知类也含未知类）。"
  "编码器 f_θ 与分类头给出特征 z 与 logit l_c(x)；已知类原型取有标签样本特征的类内均值 μ_c。")

H("3.2  已知性 K(x)：能量 + 原型距离", 2)
S('部分有依据', '前期实验用过“与已知类原型的最大相似度”作为已知性分数（对应下面的 s(x)）；能量项以及两项加权的形式、权重均为推测，不代表论文最终公式。')
T("用两路证据加权：一路是“能量”（已知类样本的能量低），一路是到最近已知原型的距离（越近越像已知）。")
F(r"E(x)=-T\log\sum_{c=1}^{C}\exp\left(l_c(x)/T\right)\qquad(3\text{-}1)")
F(r"e(x)=\sigma\left(\frac{-E(x)-\tau_E}{\gamma}\right)\qquad(3\text{-}2)")
F(r"s(x)=\exp\left(-\frac{d_{\min}^{2}(x)}{2\sigma_d^{2}}\right),\quad d_{\min}(x)=\min_{c}\|z-\mu_c\|_2\qquad(3\text{-}3)")
F(r"K(x)=\lambda\,e(x)+(1-\lambda)\,s(x),\quad K(x)\in[0,1]\qquad(3\text{-}4)")
T("含义：E(x) 为能量分数（式 3-1），σ 为 sigmoid，τ_E、γ 是能量阈值与尺度；e(x) 把“能量低”映射成 0~1 的已知性证据；"
  "s(x) 是高斯核形式的“离已知原型有多近”；λ 为两路证据的权重。K 越接近 1 越像已知类，越接近 0 越像未知。")

H("3.3  可靠度 R(x)：三项一致性", 2)
S('推测', '前期仅观察到低可靠度的已知样本更容易分错；R 由哪几项构成、如何合并（此处用几何平均）均待核对。其中 R_s 里令 p(x)=K(x) 纯属假设，“分流”的含义和分流概率的来源尚未确定，请勿当作确定结构。')
T("K 只是一个分数，它可能因为噪声而不准。可靠度衡量“多路证据是否互相印证”，包含三项：")
F(r"R_c(x)=1-\left|e(x)-s(x)\right|\qquad(3\text{-}5)")
F(r"R_v(x)=\frac{1}{2}\left(1+\frac{z^{(1)\top}z^{(2)}}{\|z^{(1)}\|\,\|z^{(2)}\|}\right)\qquad(3\text{-}6)")
F(r"R_s(x)=\left|2p(x)-1\right|,\quad p(x)=K(x)\qquad(3\text{-}7)")
F(r"R(x)=\left(R_c(x)\,R_v(x)\,R_s(x)\right)^{1/3}\qquad(3\text{-}8)")
T("含义：R_c 是“能量与距离一致性”——两路证据给的结论越接近，值越大（式 3-5）；"
  "R_v 是“两个弱视图稳定性”——同一样本的两个轻度增强视图 z⁽¹⁾、z⁽²⁾ 的特征余弦相似度，映射到 [0,1]（式 3-6）；"
  "R_s 是“分流确定性”——分流概率 p(x) 越靠近 0.5 越摇摆，|2p−1| 就越小（式 3-7）。三项取几何平均得到 R（式 3-8），任何一项很差，R 都会明显下降。")

H("3.4  样本证据分区（软权重，不做硬划分）", 2)
S('推测', '“可靠已知/可靠未知/不确定”的软权重写法是按流程图推的，是否这样定义待核对。')
T("流程图中的四类分区对应：高 K·高 R → 可靠已知；低 K·高 R → 可靠未知；低 R → 不确定样本。为避免硬阈值带来的不稳定，用连续的软权重表示：")
F(r"w_{k}(x)=K(x)R(x),\quad w_{u}(x)=\left(1-K(x)\right)R(x),\quad w_{n}(x)=1-R(x)\qquad(3\text{-}9)")
F(r"w_{k}(x)+w_{u}(x)+w_{n}(x)=1\qquad(3\text{-}10)")
T("含义：w_k、w_u、w_n 依次为“可靠已知 / 可靠未知 / 不确定”的隶属程度，三者之和为 1（式 3-10）。样本不会被硬性归入某一类，而是按权重参与后续步骤。")

H("3.5  增强预算分配 b(x) = f(K, R)", 2)
S('推测', '研究方向（用 K、R 控制增强程度）与设想一致，但 b(x) 的函数形式、扰动强度和模态数的映射均为推测。')
T("预算越大，增强越强。原则是：可靠度越高，越敢做强增强；不确定样本（R 低）只做弱增强，避免把噪声放大。")
F(r"b(x)=R(x)^{\gamma_b}\left[\kappa_u+(\kappa_k-\kappa_u)K(x)\right]\qquad(3\text{-}11)")
F(r"\varepsilon(x)=\varepsilon_{\max}\,b(x),\qquad m(x)=\max\left(1,\lceil M\,b(x)\rceil\right)\qquad(3\text{-}12)")
T("含义：γ_b 控制 R 的影响陡峭程度，κ_k、κ_u 分别是已知、未知方向的基础预算系数（可令 κ_k>κ_u，对未知样本更保守，因为其语义更容易被破坏）。"
  "式 3-12 把预算翻译成两个具体量：扰动幅度上限 ε(x) 与被扰动的模态个数 m(x)（M 为 VMD 总模态数）。R 低 ⇒ b 小 ⇒ 弱增强，对应流程图“不确定样本 → 弱增强”。")

H("3.6  VMD 分解：得到各本征模态分量", 2)
S('部分有依据', 'VMD 本身是标准方法。本文档用 M 表示模态数，以免与已知性 K(x) 混淆（前期实验里的 K=8 指模态数）。前期 VMD 为信号级可行性实验，不等于已完成 VMD 增强模型训练。')
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
S('部分有依据', '前期 VMD 实验报告过相对扰动量、RMS 相对变化和频谱形状相似度，并发现模态扰动会改变频谱形状，所以需要保真约束。JS 散度写法和全部阈值均未确认，请勿把实验观察值当阈值。')
T("流程图菱形框中的三个检查对应三个指标：")
F(r"\delta_{\mathrm{rel}}=\frac{\|\tilde s-s\|_2}{\|s\|_2}\qquad(3\text{-}18)")
F(r"D_{\mathrm{spec}}=\mathrm{JS}\left(P_s\,\|\,P_{\tilde s}\right),\quad P_s(f)=\frac{|\mathcal{F}[s](f)|^{2}}{\sum_f|\mathcal{F}[s](f)|^{2}}\qquad(3\text{-}19)")
F(r"\rho_E=\left|\frac{\|\tilde s\|_2^{2}}{\|s\|_2^{2}}-1\right|\qquad(3\text{-}20)")
F(r"\mathrm{pass}\Longleftrightarrow \delta_{\mathrm{rel}}\leq\delta_{\max}\ \wedge\ D_{\mathrm{spec}}\leq\tau_s\ \wedge\ \rho_E\leq\tau_e\qquad(3\text{-}21)")
T("含义：δ_rel 是相对扰动量（式 3-18），衡量改了多少；D_spec 用 Jensen–Shannon 散度比较增强前后的归一化功率谱，衡量频谱结构有没有变形（式 3-19）；"
  "ρ_E 是整体振动能量的相对变化（式 3-20）。三项都不超限才算通过（式 3-21），通过后把增强视图送入第四章。")
F(r"\varepsilon(x)\leftarrow\eta\,\varepsilon(x)\ (0<\eta<1),\quad \text{重试 }T_{\max}\text{ 次；仍失败则回退 }b(x)=b_{\min}\qquad(3\text{-}22)".replace(r"\text{重试 }T_{\max}\text{ 次；仍失败则回退 }", r"\ \mathrm{retry}\leq T_{\max},\ \mathrm{else}\ "))
T("含义：未通过时按系数 η 降低扰动强度并重新扰动（对应流程图“降低强度，重试”），多次失败则回退到最弱增强 b_min（“回退弱增强”）。")

# ============ 第四章 ============
H("第四章  分流引导的可靠正负关系构建与对比学习", 1)
T("整体思路：对比学习需要“哪些样本对该拉近、哪些该推远”。无标签数据里这类关系往往不可靠，错拉/错推会损伤特征空间。"
  "所以本章的原则是“先判断关系是否可信，可信才拉近或推远”，不可信的关系降权或不用。")

H("4.1  输入", 2)
S('待核对', '第四章输入与符号以原稿为准。')
T("第三章得到的两个视图特征 z₁、z₂（弱视图与自适应增强视图）、每个样本的分流权重 (K, R)，以及已知类标签 y。"
  "记 i、j 为批内两个样本，余弦相似度为：")
F(r"s_{ij}=\frac{z_i^{\top}z_j}{\|z_i\|\,\|z_j\|}\qquad(4\text{-}1)")

H("4.2  三类关系证据", 2)
S('部分有依据', '结合标签锚点、局部邻域、跨视图信息判断关系，这个动机有依据；三项证据的具体公式（Jaccard 等）为推测。')
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

H("4.3  关系可信度评估（正 / 负关系是否可信）", 2)
S('部分有依据', '动机有依据：前期诊断发现相似度很高的“负样本”可能实际同类，仅按相似度剔除负样本也不稳定。融合公式、阈值与正负对选择规则为推测。使用真实标签的 O1/O2 只是上限分析，不是可直接使用的方法。')
T("综合锚点、邻域、跨视图一致性与分流权重，给出关系的方向 t_ij 与可信度 q_ij：")
F(r"t_{ij}=\alpha_1 a_{ij}+\alpha_2 n_{ij}\qquad(4\text{-}6)")
F(r"q_{ij}=c_{ij}\cdot\sqrt{R_iR_j}\cdot\min\left(1,|t_{ij}|\right)\qquad(4\text{-}7)")
F(r"\mathcal{P}=\{(i,j):t_{ij}>0,\ q_{ij}\geq\tau_q\},\quad \mathcal{N}=\{(i,j):t_{ij}<0,\ q_{ij}\geq\tau_q\},\quad \mathcal{U}=\text{rest}\qquad(4\text{-}8)".replace(r"\text{rest}", r"\mathrm{rest}"))
F(r"w_{ij}=q_{ij}\ \ ((i,j)\in\mathcal{P}\cup\mathcal{N});\quad w_{ij}=\lambda_u q_{ij}\ \ ((i,j)\in\mathcal{U},\ \lambda_u\ \mathrm{small\ or}\ 0)\qquad(4\text{-}9)")
T("含义：t_ij 的符号决定是正对倾向还是负对倾向，α₁、α₂ 为证据权重；q_ij 要求三件事同时成立——跨视图一致 (c_ij)、两个样本自身分流可靠 (√(R_iR_j))、证据足够强 (|t_ij|)。"
  "按 q_ij 是否超过阈值 τ_q，把样本对分成可靠正对 𝒫（拉近）、可靠负对 𝒩（推远）、不确定关系 𝒰（降权或暂不使用，对应流程图中的灰色虚线框）。")

H("4.4  选择性难负对学习", 2)
S('推测', '难负对的挖掘和假负剔除规则为推测。')
T("难负对指“看起来很像、却不是同一类”的样本，对学习判别边界最有用；但它也最容易混入“其实是同类”的假负样本。因此只在可靠负对中挖掘，并剔除疑似假负：")
F(r"\mathcal{H}_i=\mathrm{TopH}_{k\in\mathcal{N}_i}\left(s_{ik}\right)\ \ \mathrm{s.t.}\ \ s_{ik}\leq\tau_{fn}\qquad(4\text{-}10)")
T("含义：𝒩_i 是与 i 构成可靠负对的样本集合；取其中相似度最高的 H 个作为难负样本；τ_fn 以上的视为“太像了，可能是假负”而被剔除。")

H("4.5  联合优化", 2)
S('待核对', '损失函数是否恰为这三项、权重如何，以原稿为准。')
T("联合优化由三部分组成：有监督对比（已知锚点）、加权无监督对比（可靠关系）、已知类分类损失。")
F(r"\mathcal{L}_{ucl}=-\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}w_{ij}\log\frac{\exp(s_{ij}/\tau)}{\exp(s_{ij}/\tau)+\sum_{k\in\mathcal{H}_i}w_{ik}\exp(s_{ik}/\tau)}\qquad(4\text{-}11)")
F(r"\mathcal{L}_{sup}=-\frac{1}{|B_l|}\sum_{i\in B_l}\frac{1}{|P_l(i)|}\sum_{p\in P_l(i)}\log\frac{\exp(s_{ip}/\tau)}{\sum_{a\ne i}\exp(s_{ia}/\tau)}\qquad(4\text{-}12)")
F(r"\mathcal{L}_{cls}=-\frac{1}{|B_l|}\sum_{i\in B_l}\log p(y_i\mid x_i)\qquad(4\text{-}13)")
F(r"\mathcal{L}=\mathcal{L}_{sup}+\lambda_1\,\mathcal{L}_{ucl}+\lambda_2\,\mathcal{L}_{cls}\qquad(4\text{-}14)")
T("含义：式 4-11 是加权的 InfoNCE：分子是可靠正对的拉近项，分母是难负对的推远项，w_ij 用关系可信度加权，不可信的关系几乎不起作用；"
  "式 4-12 是 SupCon 形式的有监督对比，B_l 为批内有标签样本，P_l(i) 为与 i 同类的有标签样本；式 4-13 为已知类交叉熵；"
  "式 4-14 为总损失，λ₁、λ₂ 平衡各项。")

H("4.6  输出：已知类识别与未知类聚类发现", 2)
S('部分有依据', '前期基线用过 GCD 特征学习加半监督 K-means；也探索过冻结编码器后设两个未知聚类中心（均映射为“未知”），但那只是探索性对照，不代表最终类别数。用轮廓系数选聚类数为推测。')
T("已知类：用分类头直接识别；未知类：把“可靠未知”的样本取出来做聚类，发现多个未知故障类别，而不是笼统地归为一个“未知类”。")
F(r"\hat y(x)=\mathrm{arg\,max}_{c}\ p(c\mid x),\quad x\in\mathcal{X}_{k}=\{x:K(x)\geq\tau_K\}\qquad(4\text{-}15)")
F(r"\mathcal{X}_{u}=\{x:K(x)<\tau_K,\ R(x)\geq\tau_R\}\qquad(4\text{-}16)")
F(r"\hat k=\mathrm{arg\,max}_{k}\ \mathrm{Sil}(k),\quad \mathrm{Sil}(k)=\frac{1}{n}\sum_{i=1}^{n}\frac{b_i-a_i}{\max(a_i,b_i)}\qquad(4\text{-}17)")
T("含义：𝒳_k 为判为已知的样本，用分类头做准确识别（式 4-15）；𝒳_u 为“低 K 且高 R”的可靠未知样本（式 4-16），"
  "在其特征上做聚类（如 K-means 或高斯混合），并用轮廓系数 Sil 选出聚类个数 k̂（式 4-17），a_i、b_i 分别是簇内平均距离和最近邻簇平均距离。")

H("4.7  迭代：更新编码器 → 重新评估 K、R", 2)
S('待核对', '迭代方式以原稿为准。')
T("流程图底部的虚线表示闭环。编码器经过联合优化后特征空间更好，需要重新计算原型 μ_c、K(x)、R(x)，再进入第三章生成新的增强，如此循环：")
F(r"\theta^{(t+1)}=\theta^{(t)}-\eta_{lr}\nabla_\theta\mathcal{L},\quad \{\mu_c,K,R\}^{(t+1)}\leftarrow f_{\theta^{(t+1)}}\qquad(4\text{-}18)")

H("附：整体流程（算法概要）", 1)
for i, s in enumerate([
    "初始化编码器 f_θ，用有标签样本得到原型 μ_c。",
    "计算每个样本的 K(x)、R(x) 及软权重 (w_k, w_u, w_n)。",
    "按预算 b(x) 决定扰动幅度 ε(x) 与模态数 m(x)；VMD 分解 → 扰动 → 重构得到增强视图。",
    "保真检查：不通过则降低强度重试，仍不通过回退到弱增强。",
    "用两个视图构建三类关系证据，计算关系可信度 q_ij，划分可靠正对 / 可靠负对 / 不确定关系。",
    "在可靠负对中做选择性难负对挖掘，联合优化 L = L_sup + λ₁L_ucl + λ₂L_cls。",
    "更新编码器，重新估计 K、R，回到第 2 步迭代；收敛后输出已知类识别结果，并对可靠未知样本聚类发现多个未知类别。"], 1):
    T(s, bold_lead=f"{i}. ")

doc.save(OUT)
print("saved", OUT)
