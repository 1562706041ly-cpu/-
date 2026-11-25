"""生成软著“文档鉴别材料”（用户手册）：docx + PDF，内容来自 manual_content.py，截图来自 make_screenshots.py。

用法：python tools/make_manual.py 截图目录 输出目录
"""
import csv
import html
import os
import re
import sys
from ruanzhu_common import TITLE, docx_page_setup, docx_header, set_run_font, add_field, qt_app, cjk_family
from manual_content import blocks
from kdsj import SOFTWARE_NAME, VERSION_LABEL, SHORT_NAME

CN = '一二三四五六七八九十'
MARGIN = dict(top=2.54, bottom=2.54, left=3.17, right=3.17)
TEXT_W_CM = 21.0 - MARGIN['left'] - MARGIN['right']


def number(B):
    """给标题、图、表编号，返回 [(类型, 显示文字, 原始块)]。"""
    out, ch, sec, fig, tab = [], 0, 0, 0, 0
    for b in B:
        if b[0] == 'h1':
            ch += 1; sec = fig = tab = 0
            out.append(('h1', f'第{CN[ch - 1]}章　{b[1]}', b))
        elif b[0] == 'h2':
            sec += 1
            out.append(('h2', f'{ch}.{sec}　{b[1]}', b))
        elif b[0] == 'img':
            fig += 1
            out.append(('img', f'图 {ch}-{fig}　{b[2]}', b))
        elif b[0] == 'table':
            tab += 1
            out.append(('table', f'表 {ch}-{tab}　{b[3]}', b))
        else:
            out.append((b[0], b[1] if len(b) > 1 else '', b))
    return out


def report_rows(csv_path):
    """把报告 CSV 拆成三部分：统计信息、告警事件、逐窗口结果（只保留前 5 行）。"""
    parts = [[]]
    for r in csv.reader(open(csv_path, encoding='utf-8-sig')):
        if not any(r):
            parts.append([])
        else:
            parts[-1].append(r)
    stats, events, wins = (parts + [[], [], []])[:3]
    stats = [r[:2] for r in stats]
    wins = wins[:6] + ([['……'] + [''] * (len(wins[0]) - 1)] if len(wins) > 6 else [])
    return {'stats': stats, 'events': events, 'windows': wins}


# =============================== docx ===============================
def write_docx(items, toc, img_dir, path):
    import docx
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = docx.Document()
    docx_page_setup(doc, **MARGIN)
    docx_header(doc)
    normal = doc.styles['Normal']
    normal.font.size = Pt(12)
    normal.font.name = 'Times New Roman'
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    for name, size in (('Heading 1', 16), ('Heading 2', 14)):
        st = doc.styles[name]
        st.font.size = Pt(size); st.font.bold = True; st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.name = '黑体'
        st.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        st.paragraph_format.space_before = Pt(12 if name == 'Heading 2' else 0)
        st.paragraph_format.space_after = Pt(6 if name == 'Heading 2' else 12)

    def para(text='', size=12, bold=False, align=None, indent=True, font=('Times New Roman', '宋体'), spacing=1.5):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = spacing
        pf.space_after = Pt(0)
        if indent:
            pf.first_line_indent = Pt(size * 2)
        if align is not None:
            p.alignment = align
        if text:
            set_run_font(p.add_run(text), font[0], font[1], size, bold)
        return p

    def page_break():
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # 封面
    body = doc.paragraphs[0] if doc.paragraphs else doc.add_paragraph()
    for _ in range(7):
        para(indent=False)
    para(SOFTWARE_NAME, 22, True, WD_ALIGN_PARAGRAPH.CENTER, False, ('黑体', '黑体'))
    para(f'[简称：{SHORT_NAME}]　{VERSION_LABEL}', 16, True, WD_ALIGN_PARAGRAPH.CENTER, False, ('黑体', '黑体'))
    for _ in range(4):
        para(indent=False)
    para('用户手册', 28, True, WD_ALIGN_PARAGRAPH.CENTER, False, ('黑体', '黑体'))
    page_break()
    # 目录（Word 打开时会提示更新域，更新后页码以 Word 排版为准）
    para('目　　录', 16, True, WD_ALIGN_PARAGRAPH.CENTER, False, ('黑体', '黑体'))
    width = Cm(TEXT_W_CM)
    first = True
    for kind, text, page in toc:
        p = doc.add_paragraph()
        p.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(0)
        if kind == 'h2':
            p.paragraph_format.left_indent = Pt(24)
        if first:
            add_field_begin(p, 'TOC \\o "1-2" \\h \\z \\u')
            first = False
        set_run_font(p.add_run(f'{text}\t{page}'), 'Times New Roman', '黑体' if kind == 'h1' else '宋体', 12,
                     kind == 'h1')
    add_field_end(doc.add_paragraph())
    settings = doc.settings.element
    uf = OxmlElement('w:updateFields'); uf.set(qn('w:val'), 'true'); settings.append(uf)

    for kind, text, b in items:
        if kind == 'h1':
            h = doc.add_heading(text, 1)
            h.paragraph_format.page_break_before = toc_first_h1(items, b)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif kind == 'h2':
            doc.add_heading(text, 2)
        elif kind == 'p':
            para(text)
        elif kind == 'li':
            p = para('', indent=False)
            p.paragraph_format.left_indent = Pt(24)
            p.paragraph_format.first_line_indent = Pt(-12)
            set_run_font(p.add_run('• ' + text), 'Times New Roman', '宋体', 12)
        elif kind == 'code':
            for line in text.split('\n'):
                p = para('', indent=False, spacing=1.15)
                p.paragraph_format.left_indent = Pt(12)
                shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:fill'), 'EDEFF2')
                p._p.get_or_add_pPr().append(shd)
                set_run_font(p.add_run(line), 'Consolas', '宋体', 9.5)
        elif kind == 'img':
            p = para('', indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0)
            p.paragraph_format.space_before = Pt(6)
            p.add_run().add_picture(os.path.join(img_dir, b[1]), width=Cm(min(b[3], TEXT_W_CM - 0.2)))
            c = para(text, 10.5, False, WD_ALIGN_PARAGRAPH.CENTER, False, ('Times New Roman', '宋体'))
            c.paragraph_format.space_after = Pt(6)
        elif kind == 'table':
            head, rows = b[1], b[2]
            c = para(text, 10.5, True, WD_ALIGN_PARAGRAPH.CENTER, False, ('Times New Roman', '黑体'))
            c.paragraph_format.space_before = Pt(6)
            ncol = len(head)
            show_head = any(head[1:])
            t = doc.add_table(rows=(1 if show_head else 0) + len(rows), cols=ncol)
            t.style = 'Table Grid'
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            all_rows = ([head] if show_head else []) + rows
            for i, r in enumerate(all_rows):
                for j, v in enumerate(r):
                    cell = t.cell(i, j)
                    cell.text = ''
                    cp = cell.paragraphs[0]
                    cp.paragraph_format.line_spacing = 1.2
                    set_run_font(cp.add_run(str(v)), 'Times New Roman', '宋体', 10.5, show_head and i == 0)
                    if show_head and i == 0:
                        shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:fill'), 'D9E2F3')
                        cell._tc.get_or_add_tcPr().append(shd)
            para('', indent=False, spacing=1.0)
    doc.save(path)


def add_field_begin(p, instr):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    r = OxmlElement('w:r'); f = OxmlElement('w:fldChar'); f.set(qn('w:fldCharType'), 'begin'); r.append(f); p._p.append(r)
    r = OxmlElement('w:r'); it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = f' {instr} '
    r.append(it); p._p.append(r)
    r = OxmlElement('w:r'); f = OxmlElement('w:fldChar'); f.set(qn('w:fldCharType'), 'separate'); r.append(f); p._p.append(r)


def add_field_end(p):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    r = OxmlElement('w:r'); f = OxmlElement('w:fldChar'); f.set(qn('w:fldCharType'), 'end'); r.append(f); p._p.append(r)


# =============================== PDF（Qt） ===============================
PX_PER_CM = 96 / 2.54


def build_html(items, toc_pages, img_dir):
    e = html.escape
    fam = cjk_family() or 'sans-serif'
    H = [f'<html><body style="font-family:\'{fam}\'; font-size:12pt;">']
    br = '<p style="page-break-before:always; margin:0;"></p>'
    H.append('<p>&nbsp;</p>' * 9)
    H.append(f'<p style="text-align:center; font-size:22pt; font-weight:bold;">{e(SOFTWARE_NAME)}</p>')
    H.append(f'<p style="text-align:center; font-size:16pt; font-weight:bold;">[简称：{e(SHORT_NAME)}]　{VERSION_LABEL}</p>')
    H.append('<p>&nbsp;</p>' * 5)
    H.append('<p style="text-align:center; font-size:28pt; font-weight:bold;">用户手册</p>')
    H.append(br)
    H.append('<p style="text-align:center; font-size:16pt; font-weight:bold; margin-bottom:12px;">目　　录</p>')
    H.append('<table width="100%" cellspacing="0" cellpadding="3">')
    for kind, text, page in toc_pages:
        st = 'font-weight:bold;' if kind == 'h1' else ''
        pad = '' if kind == 'h1' else '&nbsp;' * 6
        H.append(f'<tr><td style="{st}">{pad}{e(text)}</td><td align="right" style="{st}">{page}</td></tr>')
    H.append('</table>')
    for kind, text, b in items:
        if kind == 'h1':
            pb = 'page-break-before:always; ' if toc_first_h1(items, b) else 'margin-top:18px; '
            H.append(f'<h1 style="{pb}text-align:center; font-size:16pt; margin-bottom:14px;">{e(text)}</h1>')
        elif kind == 'h2':
            H.append(f'<h2 style="font-size:14pt; margin-top:14px; margin-bottom:6px;">{e(text)}</h2>')
        elif kind == 'p':
            H.append(f'<p style="text-indent:24pt; line-height:150%; margin:0;">{e(text)}</p>')
        elif kind == 'li':
            H.append(f'<p style="margin:0 0 0 24pt; text-indent:-12pt; line-height:150%;">• {e(text)}</p>')
        elif kind == 'code':
            H.append('<table width="100%" cellpadding="6" style="background-color:#edeff2; margin:4px 0 4px 12px;"><tr><td>'
                     f'<pre style="font-family:\'Noto Sans Mono\',\'Consolas\',\'{fam}\'; font-size:9pt; margin:0; white-space:pre-wrap;">{e(text)}</pre>'
                     '</td></tr></table>')
        elif kind == 'img':
            w = int(min(b[3], TEXT_W_CM - 0.2) * PX_PER_CM)
            src = os.path.abspath(os.path.join(img_dir, b[1]))
            H.append(f'<p align="center" style="margin-top:8px; margin-bottom:0;"><img src="{src}" width="{w}"></p>')
            H.append(f'<p align="center" style="font-size:10.5pt; margin-top:4px; margin-bottom:8px;">{e(text)}</p>')
        elif kind == 'table':
            head, rows = b[1], b[2]
            show_head = any(head[1:])
            H.append(f'<p align="center" style="font-size:10.5pt; font-weight:bold; margin-top:8px; margin-bottom:4px;">{e(text)}</p>')
            H.append('<table width="100%" border="1" cellspacing="0" cellpadding="4" style="border-color:#000; font-size:10.5pt;">')
            if show_head:
                H.append('<tr>' + ''.join(f'<td style="background-color:#d9e2f3; font-weight:bold;">{e(h)}</td>' for h in head) + '</tr>')
            for r in rows:
                H.append('<tr>' + ''.join(f'<td>{e(str(v))}</td>' for v in r) + '</tr>')
            H.append('</table><p style="margin:0;"></p>')
    H.append('</body></html>')
    return ''.join(H)


def layout(items, toc_pages, img_dir):
    from PySide6.QtGui import QTextDocument
    from PySide6.QtCore import QSizeF
    doc = QTextDocument()
    doc.setDocumentMargin(0)
    doc.setHtml(build_html(items, toc_pages, img_dir))
    w = int(TEXT_W_CM * PX_PER_CM)
    h = int((29.7 - MARGIN['top'] - MARGIN['bottom']) * PX_PER_CM)
    doc.setPageSize(QSizeF(w, h))
    return doc, w, h


def toc_first_h1(items, b):
    """只有第一章前强制分页（封面、目录之后），其余章节接着排，避免出现大半页空白。"""
    return next(x for k, _, x in items if k == 'h1') is b


def norm(t):
    return re.sub(r'\s+', ' ', t.replace('\u3000', ' ')).strip()


def heading_pages(doc, h, items):
    heads = [(k, t) for k, t, _ in items if k in ('h1', 'h2')]
    res, i = [], 0
    lay = doc.documentLayout()
    blk = doc.begin()
    while blk.isValid() and i < len(heads):
        if norm(blk.text()) == norm(heads[i][1]):
            y = lay.blockBoundingRect(blk).top()
            page = int(y // h) + 1
            if page > 2:                                  # 跳过封面和目录
                res.append((heads[i][0], heads[i][1], page))
                i += 1
        blk = blk.next()
    return res


def write_pdf(items, img_dir, path):
    from PySide6.QtGui import QPdfWriter, QPainter, QPageSize, QPageLayout, QFont, QPen, QColor
    from PySide6.QtCore import QMarginsF, QRectF, Qt
    qt_app()
    heads = [(k, t, 0) for k, t, _ in items if k in ('h1', 'h2')]
    doc, w, h = layout(items, [(k, t, '00') for k, t, _ in heads], img_dir)
    toc = heading_pages(doc, h, items)
    doc, w, h = layout(items, toc, img_dir)
    toc2 = heading_pages(doc, h, items)
    assert toc == toc2, '目录页码不稳定'
    pw = QPdfWriter(path)
    pw.setPageSize(QPageSize(QPageSize.A4))
    pw.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter)
    pw.setResolution(96)
    pw.setTitle(f'{TITLE} 用户手册')
    p = QPainter(pw)
    left, top = MARGIN['left'] * PX_PER_CM, MARGIN['top'] * PX_PER_CM
    W = 21.0 * PX_PER_CM
    hf = QFont(cjk_family() or '')
    hf.setPointSizeF(9)
    n = doc.pageCount()
    for i in range(n):
        if i:
            pw.newPage()
        p.setFont(hf)
        p.setPen(QColor('black'))
        hr = QRectF(left, 1.2 * PX_PER_CM, w, 1.0 * PX_PER_CM)
        p.drawText(hr, Qt.AlignLeft | Qt.AlignBottom, TITLE)
        p.drawText(hr, Qt.AlignRight | Qt.AlignBottom, f'第 {i + 1} 页')
        p.setPen(QPen(QColor('black'), 0.8))
        y = 2.25 * PX_PER_CM
        p.drawLine(int(left), int(y), int(W - left), int(y))
        p.save()
        p.translate(left, top - i * h)
        p.setClipRect(QRectF(0, i * h, w, h))
        doc.drawContents(p, QRectF(0, i * h, w, h - 1))
        p.restore()
    p.end()
    return toc, n


def main(img_dir, out):
    os.makedirs(out, exist_ok=True)
    items = number(blocks(report_rows(os.path.join(img_dir, 'report_example.csv'))))
    base = os.path.join(out, f'{TITLE} 用户手册')
    toc, n = write_pdf(items, img_dir, base + '.pdf')
    write_docx(items, toc, img_dir, base + '.docx')
    print(f'用户手册 {n} 页 -> {base}.pdf / .docx')
    for k, t, pg in toc:
        print(('  ' if k == 'h2' else '') + f'{t} …… {pg}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
