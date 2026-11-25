"""软著材料生成的公共部分：软件名称、页面设置、Qt 字体、docx 页眉页码。"""
import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kdsj import SOFTWARE_NAME, VERSION_LABEL  # noqa: E402

TITLE = f'{SOFTWARE_NAME} {VERSION_LABEL}'


def disp_width(s):
    """显示宽度：中文等全角字符算 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def wrap_line(line, width):
    """按显示宽度折行。续行前面加 4 个空格，保持可读。"""
    out, cur, w = [], '', 0
    for ch in line.replace('\t', '    '):
        cw = 2 if unicodedata.east_asian_width(ch) in 'WF' else 1
        if w + cw > width:
            out.append(cur)
            cur, w = '    ', 4
        cur += ch
        w += cw
    out.append(cur)
    return out


# ---------------- docx ----------------
def docx_page_setup(doc, top=2.2, bottom=2.0, left=2.0, right=2.0):
    from docx.shared import Cm
    for s in doc.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.top_margin, s.bottom_margin, s.left_margin, s.right_margin = Cm(top), Cm(bottom), Cm(left), Cm(right)
        s.header_distance, s.footer_distance = Cm(1.0), Cm(1.0)


def set_run_font(run, latin='宋体', east='宋体', size=None, bold=None):
    from docx.oxml.ns import qn
    from docx.shared import Pt
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        from docx.oxml import OxmlElement
        fonts = OxmlElement('w:rFonts')
        rpr.append(fonts)
    for k in ('w:ascii', 'w:hAnsi', 'w:cs'):
        fonts.set(qn(k), latin)
    fonts.set(qn('w:eastAsia'), east)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def add_field(paragraph, instr, placeholder='1'):
    """插入 Word 域（例如 PAGE、NUMPAGES、TOC）。"""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    def fld(t):
        r = OxmlElement('w:r'); f = OxmlElement('w:fldChar'); f.set(qn('w:fldCharType'), t); r.append(f); return r
    r_instr = OxmlElement('w:r')
    it = OxmlElement('w:instrText')
    it.set(qn('xml:space'), 'preserve')
    it.text = f' {instr} '
    r_instr.append(it)
    r_txt = OxmlElement('w:r')
    t = OxmlElement('w:t'); t.text = placeholder; r_txt.append(t)
    for el in (fld('begin'), r_instr, fld('separate'), r_txt, fld('end')):
        paragraph._p.append(el)


def docx_header(doc, text=TITLE, size=9):
    """页眉：左边软件名称和版本号，右边“第 N 页”，下面一条细线。"""
    from docx.enum.text import WD_TAB_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for s in doc.sections:
        s.header.is_linked_to_previous = False
        p = s.header.paragraphs[0]
        p.text = ''
        width = s.page_width - s.left_margin - s.right_margin
        p.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
        r = p.add_run(text + '\t第 ')
        set_run_font(r, '宋体', '宋体', size)
        add_field(p, 'PAGE')
        r2 = p.add_run(' 页')
        set_run_font(r2, '宋体', '宋体', size)
        ppr = p._p.get_or_add_pPr()
        bdr = OxmlElement('w:pBdr')
        b = OxmlElement('w:bottom')
        for k, v in (('w:val', 'single'), ('w:sz', '6'), ('w:space', '1'), ('w:color', '000000')):
            b.set(qn(k), v)
        bdr.append(b)
        ppr.append(bdr)


# ---------------- Qt PDF ----------------
def qt_app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase
    app = QApplication.instance() or QApplication(sys.argv)
    for f in filter(None, os.environ.get('KDSJ_FONT', '').split(os.pathsep)):
        QFontDatabase.addApplicationFont(f)
    return app


def cjk_family():
    from PySide6.QtGui import QFontDatabase
    have = set(QFontDatabase.families())
    return next((f for f in ('Microsoft YaHei', 'SimSun', 'Noto Sans SC', 'Noto Sans CJK SC') if f in have), None)


def mono_families():
    from PySide6.QtGui import QFontDatabase
    have = set(QFontDatabase.families())
    mono = next((f for f in ('Consolas', 'Noto Sans Mono', 'DejaVu Sans Mono', 'Courier New') if f in have), 'Monospace')
    return [mono] + [f for f in [cjk_family()] if f]
