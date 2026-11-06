# -*- coding: utf-8 -*-
"""生成第三、四章技术路线图（同时输出 SVG 矢量图 和 高清 PNG）。
用法: python make_flowchart.py
依赖: pip install pillow  (字体用同目录下 NotoSansSC-*.otf)
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 2440, 1340
S = 2  # PNG 放大倍数 (最终 4880x2680)

FONT_R = os.path.join(HERE, "NotoSansSC-Regular.otf")
FONT_B = os.path.join(HERE, "NotoSansSC-Bold.otf")
_fc = {}
def font(size, bold):
    k = (size, bold)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FONT_B if bold else FONT_R, int(size * S))
    return _fc[k]

svg = []
img = Image.new("RGB", (W * S, H * S), "white")
d = ImageDraw.Draw(img)

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def P(*v): return [x * S for x in v]

def rrect(x1, y1, x2, y2, fill, stroke, r=14, sw=2.5, dash=False):
    dash_attr = ' stroke-dasharray="10,7"' if dash else ""
    svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="{r}" '
               f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>')
    if not dash:
        d.rounded_rectangle(P(x1, y1, x2, y2), radius=r * S, fill=fill, outline=stroke, width=int(sw * S))
    else:
        d.rounded_rectangle(P(x1, y1, x2, y2), radius=r * S, fill=fill)
        dashed_rect(x1, y1, x2, y2, stroke, sw)

def dashed_line_pil(pts, color, sw, on=10, off=7):
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        L = ((xb - xa) ** 2 + (yb - ya) ** 2) ** .5
        n = int(L // (on + off)) + 1
        for i in range(n):
            s = i * (on + off) / L
            e = min((i * (on + off) + on) / L, 1)
            if s >= 1: break
            d.line(P(xa + (xb - xa) * s, ya + (yb - ya) * s, xa + (xb - xa) * e, ya + (yb - ya) * e),
                   fill=color, width=int(sw * S))

def dashed_rect(x1, y1, x2, y2, color, sw):
    dashed_line_pil([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], color, sw)

def diamond(cx, cy, w, h, fill, stroke, sw=2.5):
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    svg.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="%s"/>'
               % (" ".join(f"{x},{y}" for x, y in pts), fill, stroke, sw))
    d.polygon([(x * S, y * S) for x in [0] for x, y in pts], fill=fill)
    d.polygon([(x * S, y * S) for x, y in pts], fill=fill, outline=stroke, width=int(sw * S))

def text_lines(cx, cy, lines, color="#222"):
    """lines: [(text,size,bold)]  居中排版"""
    lh = [s * 1.38 for _, s, _ in lines]
    total = sum(lh)
    y = cy - total / 2
    for (t, s, b), h in zip(lines, lh):
        yc = y + h / 2
        svg.append(f'<text x="{cx}" y="{yc}" font-size="{s}" font-weight="{"700" if b else "400"}" '
                   f'text-anchor="middle" dominant-baseline="central" fill="{color}">{esc(t)}</text>')
        f = font(s, b)
        w = d.textlength(t, font=f) / S
        d.text((cx * S, yc * S), t, font=f, fill=color, anchor="mm")
        y += h
    return

def box(x1, y1, x2, y2, lines, fill, stroke, dash=False, color="#222"):
    rrect(x1, y1, x2, y2, fill, stroke, dash=dash)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ls = []
    for i, t in enumerate(lines):
        ls.append((t, 27, True) if i == 0 else (t, 22, False))
    # 宽度检查
    for t, s, b in ls:
        if d.textlength(t, font=font(s, b)) / S > (x2 - x1 - 16):
            print("WARN 文本过宽:", t)
    text_lines(cx, cy, ls, color)

def arrow(pts, color="#555", sw=3, dash=False, head=True):
    svg.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s"%s/>'
               % (" ".join(f"{x},{y}" for x, y in pts), color, sw,
                  ' stroke-dasharray="10,7"' if dash else ""))
    body = pts[:]
    if head:
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        L = ((xb - xa) ** 2 + (yb - ya) ** 2) ** .5
        ux, uy = (xb - xa) / L, (yb - ya) / L
        hl, hw = 16, 8
        bx, by = xb - ux * hl, yb - uy * hl
        tri = [(xb, yb), (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)]
        svg.append('<polygon points="%s" fill="%s"/>' % (" ".join(f"{x},{y}" for x, y in tri), color))
        d.polygon([(x * S, y * S) for x, y in tri], fill=color)
        body[-1] = (bx + ux * 2, by + uy * 2)
    if dash:
        dashed_line_pil(body, color, sw)
    else:
        d.line([(x * S, y * S) for x, y in body], fill=color, width=int(sw * S), joint="curve")

def label(cx, cy, t, size=21, color="#555", bg="white", bold=False):
    f = font(size, bold)
    w = d.textlength(t, font=f) / S + 14
    h = size * 1.4
    svg.append(f'<rect x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" fill="{bg}"/>')
    d.rectangle(P(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), fill=bg)
    svg.append(f'<text x="{cx}" y="{cy}" font-size="{size}" text-anchor="middle" dominant-baseline="central" '
               f'fill="{color}" font-weight="{"700" if bold else "400"}">{esc(t)}</text>')
    d.text((cx * S, cy * S), t, font=f, fill=color, anchor="mm")

# ---------- 颜色 ----------
OR_BG, OR_BD, OR_BOX = "#FFF3E0", "#EE9B3A", "#FDE3C0"
GR_BG, GR_BD, GR_BOX = "#EAF5E5", "#69AE5A", "#D8EDD1"
PK, PK_BD = "#F9D0D0", "#D75C5C"
BL, BL_BD = "#D5E6F8", "#4A82C2"
GY, GY_BD = "#F1F1F1", "#8A8A8A"
WH = "#FFFFFF"
DK = "#444"

# ---------- 标题 ----------
text_lines(W / 2, 52, [("轴向柱塞泵开放环境故障诊断 · 技术路线（拟研究方案）", 42, True)])

# ================= 第三章 =================
rrect(40, 100, 2400, 620, OR_BG, OR_BD, r=22, sw=3)
rrect(40, 100, 140, 620, "#FBD9A6", OR_BD, r=16)
text_lines(90, 360, [("第三章", 26, True), ("增强方法", 23, True)])
rrect(170, 118, 2370, 172, PK, PK_BD, r=10)
text_lines(1270, 145, [("已知性与可靠度联合控制的自适应增强", 30, True)])

box(170, 280, 420, 450, ["输入", "少量标注样本", "+ 混合无标签", "样本 x"], WH, "#777")
box(450, 200, 780, 360, ["已知性 K(x)", "能量与已知原型距离", "加权评价对已知的相似度"], OR_BOX, OR_BD)
box(450, 390, 780, 570, ["可靠度 R(x)", "能量与距离一致性", "两个弱视图稳定性", "分流确定性（是否在0.5附近）"], OR_BOX, OR_BD)
box(840, 225, 1180, 535, ["样本证据分区", "高K·高R → 可靠已知", "低K·高R → 可靠未知", "低R → 不确定样本", "（软权重，不做硬划分）"], OR_BOX, OR_BD)
box(1240, 225, 1580, 535, ["增强预算分配", "b(x) = f(K, R)", "控制扰动强度", "与扰动模态数量", "不确定样本 → 弱增强"], OR_BOX, OR_BD)
box(1640, 200, 1960, 340, ["VMD 分解", "得到各本征模态分量"], WH, "#777")
box(1640, 380, 1960, 520, ["按预算扰动模态幅值", "重构得到增强视图"], WH, "#777")
# 菱形
diamond(2170, 310, 320, 240, WH, "#555")
text_lines(2170, 310, [("保真检查", 28, True), ("扰动量 · 频谱结构", 18, False), ("· 整体振动能量", 18, False)])
box(2010, 500, 2330, 595, ["未通过：降低强度", "重试或回退弱增强"], "#F4F4F4", GY_BD)

# 箭头(第三章)
arrow([(420, 365), (435, 365)], head=False)
arrow([(435, 280), (435, 450)], head=False)
arrow([(435, 280), (450, 280)]); arrow([(435, 450), (450, 450)])
arrow([(780, 280), (810, 280), (810, 450), (780, 450)], head=False)
arrow([(780, 280), (810, 280)], head=False); arrow([(780, 480), (810, 480)], head=False)
arrow([(810, 280), (810, 480)], head=False)
arrow([(810, 380), (840, 380)])
arrow([(1180, 380), (1240, 380)])
arrow([(1580, 380), (1610, 380)], head=False)
arrow([(1610, 270), (1610, 450)], head=False)
arrow([(1610, 270), (1640, 270)]); arrow([(1610, 450), (1640, 450)])
arrow([(1800, 340), (1800, 380)])
arrow([(1960, 450), (1985, 450), (1985, 310), (2010, 310)])
arrow([(2170, 430), (2170, 500)]); label(2194, 465, "否", 22, "#C0392B", "#FFF3E0", True)
arrow([(2010, 548), (1800, 548), (1800, 520)], dash=True); label(1900, 548, "重试", 20, "#555", "#FFF3E0")
# ================= 第四章 =================
rrect(40, 660, 2400, 1305, GR_BG, GR_BD, r=22, sw=3)
rrect(40, 660, 140, 1305, "#BFE0B5", GR_BD, r=16)
text_lines(90, 982, [("第四章", 26, True), ("比较方法", 23, True)])
rrect(450, 678, 2370, 732, PK, PK_BD, r=10)
text_lines(1410, 705, [("分流引导的可靠正负关系构建与对比学习（先判断关系是否可信，再拉近/推远）", 29, True)])

box(170, 880, 420, 1080, ["输入", "增强视图特征", "z1, z2", "分流权重 (K, R)", "已知类标签"], WH, "#777")
box(450, 755, 780, 885, ["标签锚点", "同一已知标签 → 正对候选", "少量的已知类样本"], GR_BOX, GR_BD)
box(450, 905, 780, 1075, ["局部邻域", "互为k近邻且共享邻居多", "→ 正对候选", "近邻稀疏且无共享邻居", "→ 负对候选"], GR_BOX, GR_BD)
box(450, 1095, 780, 1225, ["跨视图一致性", "弱视图与自适应视图", "关系是否一致"], GR_BOX, GR_BD)
box(840, 870, 1180, 1115, ["关系可信度评估", "综合锚点、邻域、", "跨视图一致性与分流权重", "判断正/负关系是否可信"], GR_BOX, GR_BD)
box(1260, 755, 1560, 865, ["可靠负对 → 推远"], WH, "#777")
box(1260, 895, 1560, 1005, ["可靠正对 → 拉近"], WH, "#777")
box(1260, 1040, 1560, 1160, ["不确定关系", "降权或暂不使用"], GY, GY_BD, dash=True)
box(1640, 755, 1960, 905, ["选择性难负对学习", "仅在可靠负对中挖掘", "剔除疑似假负样本"], GR_BOX, GR_BD)
box(1640, 965, 1960, 1165, ["联合优化", "有监督对比（已知锚点）", "+ 加权无监督对比", "+ 已知类分类损失"], GR_BOX, GR_BD)
box(2040, 800, 2330, 1125, ["输出", "已知类：准确识别", "未知类：聚类发现", "多个未知故障类别", "（而非单一“未知类”）"], BL, BL_BD)

# 箭头(第四章)
arrow([(420, 980), (435, 980)], head=False)
arrow([(435, 820), (435, 1160)], head=False)
for y in (820, 990, 1160): arrow([(435, y), (450, y)])
for y in (820, 990, 1160): arrow([(780, y), (810, y)], head=False)
arrow([(810, 820), (810, 1160)], head=False)
arrow([(810, 990), (840, 990)])
arrow([(1180, 990), (1210, 990)], head=False)
arrow([(1210, 810), (1210, 1100)], head=False)
arrow([(1210, 810), (1260, 810)]); arrow([(1210, 950), (1260, 950)]); arrow([(1210, 1100), (1260, 1100)], dash=True)
arrow([(1560, 810), (1640, 810)])
arrow([(1800, 905), (1800, 965)])
arrow([(1560, 950), (1600, 950), (1600, 1030), (1640, 1030)])
arrow([(1560, 1100), (1640, 1100)], dash=True)
arrow([(1960, 1065), (2040, 1065)])
# 迭代反馈
arrow([(2185, 1125), (2185, 1272), (155, 1272), (155, 520), (450, 520)], color="#D35400", sw=3.5, dash=True)
label(1170, 1272, "更新编码器 → 重新评估 K, R（迭代）", 22, "#D35400", GR_BG, True)

# 通过 -> 第四章
arrow([(2330, 310), (2372, 310), (2372, 640), (285, 640), (285, 880)], color="#2E7D32", sw=3.5)
label(2372, 440, "通过", 21, "#2E7D32", "#FFF3E0", True)
label(1300, 640, "通过 → 增强视图（z1, z2）送入第四章", 21, "#2E7D32", "white", True)


# ---------- 输出 ----------
img.save(os.path.join(HERE, "tech_route_ch3_ch4.png"), dpi=(300, 300))
hdr = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Microsoft YaHei, 微软雅黑, Noto Sans SC, PingFang SC, sans-serif">'
       f'<rect width="{W}" height="{H}" fill="white"/>')
with open(os.path.join(HERE, "tech_route_ch3_ch4.svg"), "w", encoding="utf-8") as f:
    f.write(hdr + "\n".join(svg) + "</svg>")
print("done")
