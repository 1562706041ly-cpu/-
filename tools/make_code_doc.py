"""生成软著“程序鉴别材料”（源代码文档）：docx + PDF。

规则（中国版权保护中心的一般要求）：
  - 每页 50 行，页眉写软件名称和版本号，带连续页码；
  - 源代码不足 60 页时提交全部，超过 60 页时提交前 30 页和后 30 页；
  - 去掉空行（空行不计入有效行数）；保留注释。
用法：python tools/make_code_doc.py 输出目录
"""
import os
import sys
from ruanzhu_common import ROOT, TITLE, wrap_line, docx_page_setup, docx_header, set_run_font, qt_app, mono_families

LINES_PER_PAGE = 50
WIDTH = 92                     # 每行最多 92 个显示列（中文算 2 列），超出折行
ORDER = ['__init__.py', '__main__.py', 'audio.py', 'tfr.py', 'vmd.py', 'optim.py', 'nets.py', 'lssvm.py',
         'recognizer.py', 'anomaly.py', 'monitor.py', 'cli.py', 'selftest.py',
         'gui/__init__.py', 'gui/style.py', 'gui/widgets.py', 'gui/auth.py', 'gui/store.py', 'gui/workers.py',
         'gui/login.py', 'gui/main_window.py']


def collect():
    pkg = os.path.join(ROOT, 'kdsj')
    found = sorted(os.path.relpath(os.path.join(d, f), pkg).replace(os.sep, '/')
                   for d, _, fs in os.walk(pkg) if '__pycache__' not in d for f in fs if f.endswith('.py'))
    missing = set(found) - set(ORDER)
    if missing:
        sys.exit(f'这些文件不在 ORDER 里，请补上：{sorted(missing)}')
    lines, stats = [], []
    for rel in ORDER:
        src = open(os.path.join(pkg, rel), encoding='utf-8').read().splitlines()
        code = [l.rstrip() for l in src if l.strip()]
        stats.append((rel, len(src), len(code)))
        lines.append(f'# ==================== kdsj/{rel} ====================')
        for l in code:
            lines.extend(wrap_line(l, WIDTH))
    return lines, stats


def paginate(lines):
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    if len(pages) > 60:
        return pages[:30] + pages[-30:], len(pages), True
    return pages, len(pages), False


def write_docx(pages, path):
    import docx
    from docx.shared import Pt
    from docx.enum.text import WD_BREAK
    doc = docx.Document()
    docx_page_setup(doc)
    docx_header(doc)
    st = doc.styles['Normal']
    st.paragraph_format.space_before = Pt(0)
    st.paragraph_format.space_after = Pt(0)
    body = doc.paragraphs[0] if doc.paragraphs else doc.add_paragraph()
    first = True
    for pi, page in enumerate(pages):
        for li, line in enumerate(page):
            p = body if first else doc.add_paragraph()
            first = False
            pf = p.paragraph_format
            pf.line_spacing = Pt(13)                  # 固定行距，50 行正好一页
            pf.space_before = pf.space_after = Pt(0)
            pf.widow_control = False
            r = p.add_run(line if line else ' ')
            set_run_font(r, 'Consolas', '宋体', 9)
            if li == len(page) - 1 and pi < len(pages) - 1:
                r.add_break(WD_BREAK.PAGE)
    doc.save(path)


def write_pdf(pages, path):
    from PySide6.QtGui import QPdfWriter, QPainter, QPageSize, QPageLayout, QFont, QPen, QColor
    from PySide6.QtCore import QMarginsF, QRectF, Qt
    qt_app()
    w = QPdfWriter(path)
    w.setPageSize(QPageSize(QPageSize.A4))
    w.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter)
    w.setResolution(300)
    w.setTitle(f'{TITLE} 源程序')
    mm = 300 / 25.4
    W, H = 210 * mm, 297 * mm
    left, right, top = 20 * mm, 20 * mm, 22 * mm
    code_font = QFont()
    code_font.setFamilies(mono_families())
    code_font.setPointSizeF(8.6)
    head_font = QFont()
    head_font.setFamilies(mono_families()[1:] or mono_families())
    head_font.setPointSizeF(9)
    p = QPainter(w)
    step = 13 / 72 * 300                              # 13 pt 行距
    for i, page in enumerate(pages):
        if i:
            w.newPage()
        p.setFont(head_font)
        p.setPen(QColor('black'))
        hr = QRectF(left, 10 * mm, W - left - right, 8 * mm)
        p.drawText(hr, Qt.AlignLeft | Qt.AlignBottom, TITLE)
        p.drawText(hr, Qt.AlignRight | Qt.AlignBottom, f'第 {i + 1} 页')
        p.setPen(QPen(QColor('black'), 0.25 * mm))
        p.drawLine(int(left), int(18.6 * mm), int(W - right), int(18.6 * mm))
        p.setFont(code_font)
        for k, line in enumerate(page):
            p.drawText(QRectF(left, top + k * step, W - left - right, step), Qt.AlignLeft | Qt.AlignVCenter, line)
    p.end()


def main(out):
    os.makedirs(out, exist_ok=True)
    lines, stats = collect()
    pages, n_all, cut = paginate(lines)
    base = os.path.join(out, f'{TITLE} 源程序')
    write_docx(pages, base + '.docx')
    write_pdf(pages, base + '.pdf')
    total_src = sum(s[1] for s in stats)
    total_code = sum(s[2] for s in stats)
    with open(os.path.join(out, '源程序统计.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{TITLE}\n\n文件\t总行数\t非空行数\n')
        for rel, a, b in stats:
            f.write(f'kdsj/{rel}\t{a}\t{b}\n')
        f.write(f'\n合计 {len(stats)} 个文件，源程序总行数 {total_src} 行（含空行），非空行 {total_code} 行。\n')
        f.write(f'折行后共 {len(lines)} 行，每页 {LINES_PER_PAGE} 行，共 {n_all} 页；'
                f'{"提交前 30 页和后 30 页" if cut else "不足 60 页，全部提交"}（{len(pages)} 页）。\n')
    print(f'源程序 {total_src} 行（非空 {total_code} 行），{n_all} 页，提交 {len(pages)} 页 -> {base}.docx / .pdf')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, '软著材料'))
