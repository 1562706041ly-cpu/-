"""生成可在 draw.io 中编辑的框架图：数据准备 → 阶段①预训练 → 阶段②自适应 → 推理。"""
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

cells = []
_id = [10]


def nid():
    _id[0] += 1
    return f"c{_id[0]}"


BLUE = "#1f77d0"   # 源域数据流
ORANGE = "#ef7d1a" # 目标域数据流
GREEN = "#2e9e4f"
GREY = "#666666"


def box(x, y, w, h, text, fill="#ffffff", stroke="#666666", font=12, bold=False,
        rounded=True, dashed=False, align="center", fontcolor="#000000", extra=""):
    i = nid()
    style = (f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor={fill};"
             f"strokeColor={stroke};fontSize={font};fontColor={fontcolor};align={align};"
             f"verticalAlign=middle;arcSize=12;{'dashed=1;' if dashed else ''}"
             f"{'fontStyle=1;' if bold else ''}{extra}")
    cells.append(("v", i, text, style, (x, y, w, h)))
    return i


def panel(x, y, w, h, title, fill, stroke):
    box(x, y, w, h, "", fill=fill, stroke=stroke, rounded=True, dashed=True,
        extra="container=0;opacity=60;")
    box(x + 8, y + 6, 360, 26, title, fill="none", stroke="none", font=15, bold=True,
        align="left", fontcolor=stroke)


def trap(x, y, w, h, text, fill, stroke):
    i = nid()
    style = (f"shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;"
             f"fixedSize=1;size=18;direction=south;fillColor={fill};strokeColor={stroke};"
             f"fontSize=12;fontStyle=1;")
    cells.append(("v", i, text, style, (x, y, w, h)))
    return i


def edge(s, t, color=GREY, dashed=False, label="", width=2, end="block",
         exit_=None, entry=None, curved=False):
    i = nid()
    style = (f"endArrow={end};html=1;strokeColor={color};strokeWidth={width};"
             f"fontSize=11;fontColor={color};"
             f"{'dashed=1;' if dashed else ''}"
             f"{'curved=1;' if curved else 'edgeStyle=orthogonalEdgeStyle;rounded=1;'}")
    if exit_:
        style += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
    if entry:
        style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    cells.append(("e", i, label, style, (s, t)))
    return i


def loss(x, y, text, color):
    return box(x, y, 92, 34, text, fill="#ffffff", stroke=color, font=13, bold=True,
               fontcolor=color, rounded=False)


# ---------------------------------------------------------------- 标题
box(20, 8, 1720, 34,
    "跨域广义类别发现框架：固定 ETF 预训练 + 可靠度门控自适应（Bruinsma 离心泵 → 柱塞泵）",
    fill="none", stroke="none", font=18, bold=True)

# ---------------------------------------------------------------- 面板 A：数据准备
panel(20, 50, 400, 900, "A  数据准备", "#f7f9fc", "#4a6fa5")
src = box(40, 95, 360, 92,
          "<b>源域 S：训练安装</b><br>已知类窗口 10% 有标签 (x<sub>s</sub>, y<sub>s</sub>)<br>其余无标签（可能含未知故障）",
          fill="#e3efff", stroke=BLUE)
tgt = box(40, 205, 360, 112,
          "<b>目标域 T：新设备（留出安装）</b><br>全部无标签 x<sub>t</sub>，含已知 + 未知故障<br>按测量记录拆分：<br>T<sub>adapt</sub>（训练用） | T<sub>eval</sub>（只评估）",
          fill="#fff0e0", stroke=ORANGE)
prep = box(40, 340, 360, 78,
           "<b>预处理</b><br>Motor-2，100% 转速，4 个振动通道<br>分窗 → 4 通道对数谱（主线）<br>对照：4 通道 STFT/CWT 时频图",
           fill="#ffffff", stroke=GREY)
weak = box(40, 445, 170, 70, "<b>弱视图 x<sup>w</sup></b><br>归一化", fill="#ffffff", stroke=GREY)
strong = box(225, 445, 175, 70, "<b>强视图 x<sup>s</sup></b><br>来自第三章模块", fill="#ffffff", stroke="#c0392b")
ch3 = box(40, 540, 360, 150,
          "<b>第三章模块：可靠度感知的强视图生成器</b><br>VMD 模态扰动（沙箱预计算）+ 保真检查<br>"
          "扰动强度按门控分组分配：<br>可靠已知 → 较强<br>可靠未知 → 中等<br>不确定 → 较弱 / 通用增强<br>"
          "保真不通过 → 退回弱视图",
          fill="#fdecea", stroke="#c0392b", align="center")
note_a = box(40, 710, 360, 86,
             "未知类标签只用于评估与事后诊断<br>阈值、超参数只用源域留一安装验证选取<br>75% 转速数据封存",
             fill="#ffffff", stroke="#999999", dashed=True, font=11)
edge(src, prep, BLUE, exit_=(0.25, 1), entry=(0.25, 0))
edge(tgt, prep, ORANGE, exit_=(0.75, 1), entry=(0.75, 0))
edge(prep, weak, GREY, exit_=(0.25, 1), entry=(0.5, 0))
edge(prep, strong, GREY, exit_=(0.75, 1), entry=(0.5, 0))
edge(ch3, strong, "#c0392b", dashed=True, exit_=(0.75, 0), entry=(0.5, 1), label="生成")

# ---------------------------------------------------------------- 面板 B：阶段①预训练
panel(440, 50, 1300, 330, "B  阶段 ① 预训练（第 0–300 步）", "#f3f8ff", "#1f5fa8")
v1 = box(465, 100, 120, 44, "视图 v", fill="#ffffff", stroke=GREY)
v2 = box(465, 270, 120, 44, "视图 v′", fill="#ffffff", stroke=GREY)
enc_on = trap(615, 85, 150, 74, "编码器 f<sub>θ</sub><br>一维 CNN<br>(对照: 一维 ResNet)", "#d6e8ff", BLUE)
enc_tg = trap(615, 255, 150, 74, "目标编码器 f<sub>ξ</sub><br>(EMA)", "#d6e8ff", BLUE)
proj_on = box(795, 100, 110, 44, "投影 MLP", fill="#eeeeee", stroke=GREY)
pred_on = box(925, 100, 110, 44, "预测 MLP q<sub>θ</sub>", fill="#eeeeee", stroke=GREY)
proj_tg = box(795, 270, 110, 44, "投影 MLP", fill="#eeeeee", stroke=GREY)
sg = box(925, 270, 110, 44, "sg(z′)", fill="#fff7cc", stroke=GREY)
l_byol = loss(1060, 185, "L<sub>BYOL</sub>", BLUE)
etf_head = box(1150, 100, 140, 44, "ETF 投影头 g<br>→ L2 归一化 z", fill="#e8f5e9", stroke=GREEN)
etf = box(1315, 85, 200, 80,
          "<b>固定 ETF 原型</b>（不训练）<br>C 个已知原型 + 3 个未知槽<br>cos = −1/(K−1)",
          fill="#ffffff", stroke=GREEN)
l_src1 = loss(1560, 108, "L<sub>src</sub>", GREEN)
note_b = box(1150, 190, 560, 120,
             "<b>目的</b>：① 不依赖伪标签先学出可用特征（两个域全部窗口参与 BYOL）<br>"
             "② 给阶段 ② 的门控与伪标签打底，避免确认偏差<br>"
             "③ 源域有标签样本拉向自己的已知原型，同时被推离 3 个未知槽 → 未知位置预留<br>"
             "L = L<sub>BYOL</sub> + L<sub>src</sub>（类均衡采样）",
             fill="#ffffff", stroke="#999999", dashed=True, align="left", font=11)
edge(v1, enc_on, BLUE)
edge(v2, enc_tg, BLUE)
edge(enc_on, proj_on, BLUE)
edge(proj_on, pred_on, BLUE)
edge(enc_tg, proj_tg, BLUE)
edge(proj_tg, sg, BLUE)
edge(enc_on, enc_tg, GREY, dashed=True, label="EMA", exit_=(0.3, 1), entry=(0.3, 0), end="open")
edge(pred_on, l_byol, BLUE, dashed=True, exit_=(1, 0.5), entry=(0.5, 0))
edge(sg, l_byol, BLUE, dashed=True, exit_=(1, 0.5), entry=(0.5, 1))
edge(enc_on, etf_head, GREEN, exit_=(0.85, 0), entry=(0.5, 0), label="源域有标签")
edge(etf_head, etf, GREEN)
edge(etf, l_src1, GREEN, dashed=True)

# ---------------------------------------------------------------- 面板 C：阶段②自适应
panel(440, 395, 1300, 555, "C  阶段 ② 自适应（第 300–1500 步，伪标签每 100 步刷新）", "#fff8f0", "#c0661a")
enc2 = trap(470, 470, 160, 190, "编码器 f<br>（继承阶段 ①，<br>可训练）", "#d6e8ff", BLUE)
zs_src = box(660, 450, 150, 44, "源域有标签 z<sub>s</sub>", fill="#e3efff", stroke=BLUE)
zw = box(660, 540, 150, 44, "目标弱视图 z<sup>w</sup>", fill="#fff0e0", stroke=ORANGE)
zst = box(660, 640, 150, 44, "目标强视图 z<sup>s</sup>", fill="#fdecea", stroke="#c0392b")
etf2 = box(840, 450, 170, 234,
           "<b>固定 ETF 原型</b><br>（与阶段 ① 同一组）<br><br>s<sub>k</sub> = cos(z, m<sub>k</sub>)/τ<br>τ = 0.1",
           fill="#ffffff", stroke=GREEN)
gate = box(1040, 520, 210, 84,
           "<b>门控</b>：k(x)=max<sub>c≤C</sub> cos(z<sup>w</sup>, m<sub>c</sub>)<br>阈值 t<sub>hi</sub>/t<sub>lo</sub>：源域留一安装校准",
           fill="#fff7cc", stroke="#b8860b")
g_known = box(1280, 470, 170, 50, "可靠已知<br>→ 已知类伪标签", fill="#e3efff", stroke=BLUE)
g_unk = box(1280, 540, 170, 66, "可靠未知<br>→ 放宽均衡分配<br>到未知槽 u1/u2/u3", fill="#fde2e2", stroke="#c0392b")
g_unc = box(1280, 622, 170, 44, "不确定 → 只参与 BYOL", fill="#f0f0f0", stroke=GREY)
l_src2 = loss(1630, 452, "L<sub>src</sub>", GREEN)
l_ta = loss(1630, 500, "L<sub>ta</sub>", BLUE)
l_unk = loss(1630, 552, "L<sub>unk</sub>", "#c0392b")
l_new = box(1630, 604, 92, 34, "L<sub>new</sub><br><font style='font-size:9px'>(损失创新，待定)</font>",
            fill="#ffffff", stroke="#7b3fa0", font=12, bold=True, fontcolor="#7b3fa0", dashed=True, rounded=False)
l_byol2 = loss(1630, 656, "0.1·L<sub>BYOL</sub>", GREY)
note_c = box(470, 760, 1250, 100,
             "<b>总损失</b>：L = L<sub>src</sub> + β(t)·(L<sub>ta</sub> + L<sub>unk</sub> [+ L<sub>new</sub>]) + 0.1·L<sub>BYOL</sub>，β 在第 300–500 步由 0 升到 1<br>"
             "L<sub>ta</sub>：可靠已知样本的强视图拟合弱视图伪标签（拉向已知原型 = 按类跨域对齐，不加 DANN）<br>"
             "L<sub>unk</sub>：可靠未知样本的强视图拟合所分配的未知槽，按槽位类均衡；未知槽固定，无需匹配器",
             fill="#ffffff", stroke="#999999", dashed=True, align="left", font=11)
edge(enc2, zs_src, BLUE, exit_=(1, 0.1), entry=(0, 0.5))
edge(enc2, zw, ORANGE, exit_=(1, 0.45), entry=(0, 0.5))
edge(enc2, zst, "#c0392b", exit_=(1, 0.9), entry=(0, 0.5))
edge(zs_src, etf2, BLUE, entry=(0, 0.1))
edge(zw, etf2, ORANGE, entry=(0, 0.4))
edge(zst, etf2, "#c0392b", entry=(0, 0.85))
edge(etf2, gate, ORANGE, exit_=(1, 0.4), entry=(0, 0.5))
edge(gate, g_known, BLUE, exit_=(1, 0.2), entry=(0, 0.5))
edge(gate, g_unk, "#c0392b", exit_=(1, 0.5), entry=(0, 0.5))
edge(gate, g_unc, GREY, exit_=(1, 0.85), entry=(0, 0.5))
edge(etf2, l_src2, GREEN, dashed=True, exit_=(1, 0.03), entry=(0, 0.5))
edge(g_known, l_ta, BLUE, dashed=True, label="监督强视图")
edge(g_unk, l_unk, "#c0392b", dashed=True, label="监督强视图")
edge(g_unk, l_new, "#7b3fa0", dashed=True, exit_=(1, 0.9), entry=(0, 0.5))
edge(g_unc, l_byol2, GREY, dashed=True)
edge(g_unc, ch3, "#c0392b", dashed=True, label="分组决定扰动强度", exit_=(0.3, 1), entry=(1, 0.75))

# 跨面板连线
edge(weak, enc2, ORANGE, exit_=(0.5, 1), entry=(0, 0.45), label="弱视图")
edge(strong, enc2, "#c0392b", exit_=(1, 0.5), entry=(0, 0.85), label="强视图")
edge(src, v1, BLUE, exit_=(1, 0.3), entry=(0, 0.5), label="两个域全部窗口")
edge(enc_on, enc2, BLUE, dashed=True, exit_=(0.5, 1), entry=(0.5, 0), label="继承参数")

# ---------------------------------------------------------------- 面板 D：推理
panel(20, 965, 1720, 140, "D  推理（目标域 T_eval）", "#f5f5f5", "#555555")
d1 = box(40, 1010, 210, 70, "测试窗口 → 弱处理<br>→ 编码器 + ETF 头 → z", fill="#ffffff", stroke=GREY)
d2 = box(290, 1010, 260, 70, "k(x) = max 已知余弦<br>≥ t<sub>guard</sub>？（新特征上重新校准）", fill="#fff7cc", stroke="#b8860b")
d3 = box(600, 990, 230, 44, "是 → 已知类 argmax", fill="#e3efff", stroke=BLUE)
d4 = box(600, 1046, 230, 44, "否 → 最近未知槽：未知 1 / 2 / 3", fill="#fde2e2", stroke="#c0392b")
d5 = box(880, 1010, 380, 70, "评估：H、未知聚类准确率（匈牙利匹配）、<br>已知误判为未知比例、NMI；对照：拒绝样本 k-means",
         fill="#ffffff", stroke=GREY, font=11)
d6 = box(1290, 1000, 430, 90,
         "<b>基线</b>：Bg4（新设定重跑）、GCD 式（对比学习 + 半监督 k-means）、TRAILER、NCDLR、OVANet + k-means<br>"
         "<b>核心消融</b>：固定 vs 可学习原型；3 槽 vs 1 槽；有 / 无门控；第三章强视图 vs 通用增强",
         fill="#ffffff", stroke="#999999", dashed=True, align="left", font=11)
edge(d1, d2, GREY)
edge(d2, d3, BLUE, exit_=(1, 0.3), entry=(0, 0.5))
edge(d2, d4, "#c0392b", exit_=(1, 0.7), entry=(0, 0.5))
edge(d3, d5, GREY, exit_=(1, 0.5), entry=(0, 0.3))
edge(d4, d5, GREY, exit_=(1, 0.5), entry=(0, 0.7))

# ---------------------------------------------------------------- 图例
box(20, 1115, 1720, 40, "", fill="#ffffff", stroke="#cccccc", rounded=False)
lx = 40
for color, txt, dash in [(BLUE, "源域数据流", False), (ORANGE, "目标域弱视图", False),
                         ("#c0392b", "目标域强视图 / 未知", False), (GREEN, "固定 ETF 相关", False),
                         (GREY, "损失 / 监督（虚线）", True), ("#7b3fa0", "待定的创新损失", True)]:
    a = box(lx, 1128, 1, 1, "", fill="none", stroke="none")
    b = box(lx + 50, 1128, 1, 1, "", fill="none", stroke="none")
    edge(a, b, color, dashed=dash)
    box(lx + 56, 1120, 200, 18, txt, fill="none", stroke="none", font=11, align="left")
    lx += 285

# ---------------------------------------------------------------- 输出 XML
mxfile = ET.Element("mxfile", host="app.diagrams.net")
diagram = ET.SubElement(mxfile, "diagram", name="框架总图", id="framework")
model = ET.SubElement(diagram, "mxGraphModel", dx="1600", dy="1000", grid="1", gridSize="10",
                      guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1",
                      pageScale="1", pageWidth="1760", pageHeight="1180", math="0", shadow="0")
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", id="0")
ET.SubElement(root, "mxCell", id="1", parent="0")
for kind, i, text, style, geo in cells:
    if kind == "v":
        c = ET.SubElement(root, "mxCell", id=i, value=text, style=style, vertex="1", parent="1")
        x, y, w, h = geo
        ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
    else:
        s, t = geo
        c = ET.SubElement(root, "mxCell", id=i, value=text, style=style, edge="1",
                          parent="1", source=s, target=t)
        ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})

ET.ElementTree(mxfile).write("framework_etf_gcd.drawio", encoding="utf-8", xml_declaration=True)
print("cells:", len(cells))
