"""界面配色、样式表、背景与面板绘制。"""
import os
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (QPainter, QColor, QLinearGradient, QRadialGradient, QPen, QPixmap, QFont,
                           QFontDatabase, QPainterPath)
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QVBoxLayout

CYAN = '#3fd0ff'
CYAN_DIM = '#1f6f99'
TEXT = '#d8f3ff'
MUTED = '#86a9c0'
OK = '#38e08c'
WARN = '#ff5a5a'
GOLD = '#ffb547'

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')

QSS = f"""
QWidget {{ color: {TEXT}; font-size: 13px; }}
QLabel {{ background: transparent; }}
QLineEdit {{ background: rgba(4,18,40,220); border: 1px solid #2a7fb0; border-radius: 4px;
            padding: 7px 10px; color: #eafaff; selection-background-color: #2a7fb0; }}
QLineEdit:focus {{ border: 1px solid {CYAN}; }}
QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e74ae, stop:1 #0b3a62);
              border: 1px solid #3aa9e0; border-radius: 4px; padding: 6px 14px; color: #eefbff; }}
QPushButton:hover {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2b8fd0, stop:1 #10507f);
                    border: 1px solid {CYAN}; }}
QPushButton:pressed {{ background: #0b3a62; }}
QPushButton:disabled {{ color: #5f7d90; border-color: #23475f; background: #0a2036; }}
QPushButton#primary {{ font-size: 15px; font-weight: bold; padding: 9px; }}
QListWidget, QTableWidget {{ background: rgba(3,14,32,170); border: 1px solid #1d5a80; outline: none;
                             gridline-color: #16405c; alternate-background-color: rgba(10,40,70,120); }}
QListWidget::item {{ padding: 9px 6px; border-bottom: 1px solid #12344d; }}
QListWidget::item:selected, QTableWidget::item:selected {{ background: rgba(63,208,255,70); color: white; }}
QHeaderView::section {{ background: #0b2d4c; color: {CYAN}; border: none; border-right: 1px solid #16405c;
                        padding: 5px; font-weight: bold; }}
QTableCornerButton::section {{ background: #0b2d4c; border: none; }}
QProgressBar {{ border: 1px solid #1d5a80; border-radius: 3px; background: #06192f; text-align: center;
               color: #eafaff; min-height: 16px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a8fd0, stop:1 {CYAN}); }}
QDialog, QMessageBox {{ background: #071a33; }}
QSpinBox, QDoubleSpinBox, QComboBox {{ background: #06192f; border: 1px solid #2a7fb0; border-radius: 3px;
                                       padding: 4px 6px; color: #eafaff; }}
QComboBox QAbstractItemView {{ background: #06192f; selection-background-color: #1e74ae; }}
QScrollBar:vertical {{ background: transparent; width: 9px; }}
QScrollBar::handle:vertical {{ background: #1d5a80; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; }}
QScrollBar::handle:horizontal {{ background: #1d5a80; border-radius: 4px; min-width: 24px; }}
QToolTip {{ background: #0b2d4c; color: #eafaff; border: 1px solid {CYAN}; }}
"""


def scaled_font(font, k=1.0, bold=None):
    """按比例放大/缩小字体，兼容用像素（样式表 font-size: 13px）和用磅值设置的字体。"""
    f = QFont(font)
    if f.pixelSize() > 0:
        f.setPixelSize(max(8, round(f.pixelSize() * k)))
    else:
        f.setPointSizeF(max(6.0, f.pointSizeF() * k))
    if bold is not None:
        f.setBold(bold)
    return f


def setup_fonts(app):
    """Windows 用微软雅黑；可用环境变量 KDSJ_FONT 指定字体文件（例如在没有中文字体的电脑上）。"""
    fam = None
    for f in filter(None, os.environ.get('KDSJ_FONT', '').split(os.pathsep)):
        fid = QFontDatabase.addApplicationFont(f)
        if fid >= 0 and fam is None:
            fam = QFontDatabase.applicationFontFamilies(fid)[0]
    if fam is None:
        have = set(QFontDatabase.families())
        fam = next((f for f in ('Microsoft YaHei UI', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans SC',
                                'Noto Sans CJK SC', 'SimHei') if f in have), app.font().family())
    font = QFont(fam, 10)
    app.setFont(font)
    return fam


def load_background():
    for name in ('background.jpg', 'background.jpeg', 'background.png'):
        p = os.path.join(ASSET_DIR, name)
        if os.path.exists(p):
            pix = QPixmap(p)
            if not pix.isNull():
                return pix
    return None


class Backdrop(QWidget):
    """全窗口背景：矿山照片（kdsj/assets/background.jpg，铺满裁剪）+ 深蓝渐变遮罩 + 细网格。
    没有照片时用程序生成的深蓝背景。overlay 越大背景越暗。"""

    def __init__(self, parent=None, overlay=0.62, grid=True):
        super().__init__(parent)
        self.overlay, self.grid = overlay, grid
        self._src = load_background()
        self._cache = None

    def resizeEvent(self, e):
        self._cache = None
        super().resizeEvent(e)

    def _render(self):
        pm = QPixmap(self.size())
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        if self._src is not None:
            s = self._src.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            p.drawPixmap(int((r.width() - s.width()) / 2), int((r.height() - s.height()) / 2), s)
        else:
            g = QRadialGradient(r.center(), max(r.width(), r.height()) * 0.7)
            g.setColorAt(0, QColor('#0c3a66'))
            g.setColorAt(1, QColor('#020a1a'))
            p.fillRect(r, g)
        g = QLinearGradient(0, 0, 0, r.height())
        a = int(255 * self.overlay)
        g.setColorAt(0, QColor(2, 12, 32, min(255, a + 40)))
        g.setColorAt(0.45, QColor(3, 20, 48, a))
        g.setColorAt(1, QColor(2, 10, 26, min(255, a + 60)))
        p.fillRect(r, g)
        if self.grid:
            p.setPen(QPen(QColor(63, 208, 255, 18), 1))
            step = 40
            for x in range(0, int(r.width()), step):
                p.drawLine(x, 0, x, int(r.height()))
            for y in range(0, int(r.height()), step):
                p.drawLine(0, y, int(r.width()), y)
        p.end()
        return pm

    def paintEvent(self, e):
        if self._cache is None or self._cache.size() != self.size():
            self._cache = self._render()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        p.end()


class Panel(QFrame):
    """科技风面板：半透明深蓝底、青色细边框、四角加粗折线、带标题条。"""

    def __init__(self, title='', parent=None, accent=CYAN):
        super().__init__(parent)
        self.title, self.accent = title, QColor(accent)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(12, 40 if title else 12, 12, 12)
        self.body.setSpacing(8)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(63, 208, 255, 90), 1))
        p.setBrush(QColor(5, 22, 48, 195))
        p.drawRoundedRect(r, 4, 4)
        c = QColor(self.accent)
        p.setPen(QPen(c, 2))
        L = 14
        for (x, y, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                               (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + dx * L, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * L))
        if self.title:
            g = QLinearGradient(r.left(), 0, r.right(), 0)
            g.setColorAt(0, QColor(30, 120, 190, 150))
            g.setColorAt(1, QColor(30, 120, 190, 0))
            p.fillRect(QRectF(r.left() + 2, r.top() + 2, r.width() - 4, 28), g)
            p.fillRect(QRectF(r.left() + 10, r.top() + 10, 4, 12), c)
            p.setFont(scaled_font(self.font(), 1.12, True))
            p.setPen(QColor('#e6f8ff'))
            p.drawText(QRectF(r.left() + 22, r.top() + 2, r.width() - 30, 28), Qt.AlignVCenter | Qt.AlignLeft,
                       self.title)


def glow_label(text, size=26, color=CYAN, bold=True):
    lab = QLabel(text)
    # 全局样式表用像素设置了字号，这里也必须用样式表，否则会被覆盖
    lab.setStyleSheet(f'color: {color}; font-size: {round(size * 1.333)}px; '
                      f'font-weight: {"bold" if bold else "normal"}; letter-spacing: 2px;')
    from PySide6.QtWidgets import QGraphicsDropShadowEffect
    eff = QGraphicsDropShadowEffect(lab)
    eff.setColor(QColor(color))
    eff.setBlurRadius(18)
    eff.setOffset(0, 0)
    lab.setGraphicsEffect(eff)
    return lab
