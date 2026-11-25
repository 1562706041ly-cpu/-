"""自绘控件：波形图、时频图、异常指数曲线、数值卡片、时钟（不依赖 matplotlib）。"""
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, QDateTime, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QImage, QFont, QPolygonF, QLinearGradient
from PySide6.QtWidgets import QWidget, QLabel, QFrame, QVBoxLayout
from .style import CYAN, TEXT, MUTED, OK, WARN, GOLD, scaled_font
from ..tfr import JET

AX = QColor(63, 208, 255, 60)
LBL = QColor('#86a9c0')


def _small(widget, k=0.85):
    return scaled_font(widget.font(), k)


class _Plot(QWidget):
    """有坐标边距的绘图区基类。"""
    margins = (44, 10, 12, 24)                       # 左 上 右 下

    def plot_rect(self):
        l, t, r, b = self.margins
        return QRectF(l, t, max(10, self.width() - l - r), max(10, self.height() - t - b))

    def _empty(self, p, text):
        p.setPen(LBL)
        p.drawText(self.rect(), Qt.AlignCenter, text)


class WaveformView(_Plot):
    """整段录音的波形。告警窗口标红，当前窗口用黄框标出；点击选择窗口。"""
    windowClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.x, self.fs, self.windows, self.current = None, 1, [], -1
        self._env = None

    def set_signal(self, x, fs):
        self.x, self.fs, self._env = np.asarray(x, np.float32), fs, None
        self.update()

    def set_windows(self, windows):
        """windows：[(开始秒, 结束秒, 是否告警), ...]"""
        self.windows = list(windows)
        self.update()

    def set_current(self, i):
        self.current = i
        self.update()

    def _envelope(self, w):
        if self._env is not None and self._env[0] == w:
            return self._env[1]
        n = self.x.size
        edges = np.linspace(0, n, w + 1).astype(int)
        mn = np.array([self.x[a:max(b, a + 1)].min() for a, b in zip(edges[:-1], edges[1:])])
        mx = np.array([self.x[a:max(b, a + 1)].max() for a, b in zip(edges[:-1], edges[1:])])
        self._env = (w, (mn, mx))
        return mn, mx

    def mousePressEvent(self, e):
        if self.x is None or not self.windows:
            return
        r = self.plot_rect()
        t = (e.position().x() - r.left()) / r.width() * self.x.size / self.fs
        hits = [i for i, w in enumerate(self.windows) if w[0] <= t < w[1]]
        if hits:
            # 重叠窗口时选中心离点击位置最近的
            self.windowClicked.emit(min(hits, key=lambda i: abs((self.windows[i][0] + self.windows[i][1]) / 2 - t)))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.x is None or self.x.size == 0:
            self._empty(p, '尚未分析录音')
            return
        r = self.plot_rect()
        dur = self.x.size / self.fs
        sx = lambda t: r.left() + t / dur * r.width()
        for i, (a, b, alarm) in enumerate(self.windows):
            if alarm:
                p.fillRect(QRectF(sx(a), r.top(), sx(b) - sx(a), r.height()), QColor(255, 70, 70, 55))
        w = int(r.width())
        mn, mx = self._envelope(w)
        amp = max(1e-12, float(np.abs(self.x).max()))
        cy, h = r.center().y(), r.height() / 2 * 0.92
        top = [QPointF(r.left() + i, cy - mx[i] / amp * h) for i in range(w)]
        bot = [QPointF(r.left() + i, cy - mn[i] / amp * h) for i in range(w - 1, -1, -1)]
        p.setPen(QPen(QColor(63, 208, 255, 220), 1))
        p.setBrush(QColor(63, 208, 255, 90))
        p.drawPolygon(QPolygonF(top + bot))
        if 0 <= self.current < len(self.windows):
            a, b, _ = self.windows[self.current]
            p.setPen(QPen(QColor(GOLD), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(sx(a), r.top() + 1, sx(b) - sx(a), r.height() - 2))
        self._axes(p, r, dur)

    def _axes(self, p, r, dur):
        p.setPen(QPen(AX, 1))
        p.drawRect(r)
        p.setFont(_small(self))
        p.setPen(LBL)
        n = 6
        for k in range(n + 1):
            t = dur * k / n
            x = r.left() + r.width() * k / n
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 18), Qt.AlignHCenter | Qt.AlignTop, f'{t:.0f} s' if dur >= 10 else f'{t:.1f} s')
        p.drawText(QRectF(0, r.top(), self.margins[0] - 6, r.height()), Qt.AlignRight | Qt.AlignVCenter, '幅值')


class TfrView(QWidget):
    """单个窗口的时频图（64x64 颜色索引，jet 着色，平滑放大）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 160)
        self.img, self.caption = None, ''

    def set_image(self, idx, caption=''):
        if idx is None:
            self.img = None
        else:
            rgb = (JET[np.asarray(idx)] * 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            h, w, _ = rgb.shape
            self.img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.caption = caption
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect()).adjusted(30, 4, -8, -22)
        if self.img is None:
            p.setPen(LBL)
            p.drawText(self.rect(), Qt.AlignCenter, '时频图')
            return
        p.drawImage(r, self.img)
        p.setPen(QPen(AX, 1))
        p.drawRect(r)
        p.setFont(_small(self))
        p.setPen(LBL)
        p.drawText(QRectF(r.left(), r.bottom() + 3, r.width(), 18), Qt.AlignCenter, f'时间 →    {self.caption}')
        p.save()
        p.translate(r.left() - 8, r.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-r.height() / 2, -14, r.height(), 16), Qt.AlignCenter, '频率 →')
        p.restore()


class IndexChart(_Plot):
    """每个窗口的异常指数（或分类置信度），虚线为告警阈值；点击选择窗口。"""
    windowClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.t, self.v, self.alarm, self.thr, self.current, self.ylabel = [], [], [], 1.0, -1, '异常指数'

    def set_data(self, t, v, alarm, thr=1.0, ylabel='异常指数'):
        self.t, self.v, self.alarm, self.thr, self.ylabel = list(t), list(v), list(alarm), thr, ylabel
        self.update()

    def set_current(self, i):
        self.current = i
        self.update()

    def _ymax(self):
        return max([self.thr * 1.6] + [x * 1.1 for x in self.v])

    def _sx(self, r, t):
        T = max(self.t[-1], 1e-9) if self.t else 1
        return r.left() + (t / T if len(self.t) > 1 else 0.5) * r.width()

    def mousePressEvent(self, e):
        if not self.t:
            return
        r = self.plot_rect()
        xs = [self._sx(r, t) for t in self.t]
        i = int(np.argmin([abs(x - e.position().x()) for x in xs]))
        self.windowClicked.emit(i)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.t:
            self._empty(p, '异常指数曲线')
            return
        r = self.plot_rect()
        ym = self._ymax()
        sy = lambda v: r.bottom() - min(v, ym) / ym * r.height()
        pts = [QPointF(self._sx(r, t), sy(v)) for t, v in zip(self.t, self.v)]
        g = QLinearGradient(0, r.top(), 0, r.bottom())
        g.setColorAt(0, QColor(63, 208, 255, 90))
        g.setColorAt(1, QColor(63, 208, 255, 0))
        if len(pts) > 1:
            poly = QPolygonF([QPointF(pts[0].x(), r.bottom())] + pts + [QPointF(pts[-1].x(), r.bottom())])
            p.setPen(Qt.NoPen)
            p.setBrush(g)
            p.drawPolygon(poly)
            p.setPen(QPen(QColor(CYAN), 2))
            p.drawPolyline(QPolygonF(pts))
        for i, (pt, al) in enumerate(zip(pts, self.alarm)):
            p.setPen(QPen(QColor('#ffffff'), 2) if i == self.current else Qt.NoPen)
            p.setBrush(QColor(WARN if al else OK))
            rad = 6 if i == self.current else 4
            p.drawEllipse(pt, rad, rad)
        p.setPen(QPen(QColor(WARN), 1.5, Qt.DashLine))
        p.drawLine(QPointF(r.left(), sy(self.thr)), QPointF(r.right(), sy(self.thr)))
        p.setPen(QPen(AX, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)
        p.setFont(_small(self))
        p.setPen(LBL)
        for k in range(5):
            v = ym * k / 4
            p.drawText(QRectF(0, sy(v) - 8, self.margins[0] - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f'{v:.2f}')
        p.setPen(QColor(WARN))
        p.drawText(QRectF(r.right() - 90, sy(self.thr) - 18, 88, 16), Qt.AlignRight, f'告警阈值 {self.thr:g}')
        p.setPen(LBL)
        T = self.t[-1] if self.t else 0
        for k in range(6):
            x = r.left() + r.width() * k / 5
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 18), Qt.AlignHCenter, f'{T * k / 5:.0f} s')


class StatCard(QFrame):
    """右侧数值卡片：标题 + 大号数值 + 说明 + 底部进度条（对应设计图中的百分比卡片）。"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title, self.value, self.note, self.frac, self.color = title, '--', '', 0.0, QColor(CYAN)
        self.setMinimumHeight(92)

    def set(self, value, note='', frac=None, color=CYAN):
        self.value, self.note, self.color = value, note, QColor(color)
        if frac is not None:
            self.frac = float(np.clip(frac, 0, 1))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        g = QLinearGradient(r.left(), 0, r.right(), 0)
        c = QColor(self.color)
        c0 = QColor(c); c0.setAlpha(70)
        c1 = QColor(c); c1.setAlpha(15)
        g.setColorAt(0, c0)
        g.setColorAt(1, c1)
        p.setPen(QPen(QColor(c.red(), c.green(), c.blue(), 150), 1))
        p.setBrush(g)
        path_r = r.adjusted(0, 0, 0, 0)
        p.drawRoundedRect(path_r, 4, 4)
        p.fillRect(QRectF(r.left(), r.top() + 8, 3, r.height() - 16), c)
        p.setFont(self.font())
        p.setPen(QColor(TEXT))
        p.drawText(QRectF(r.left() + 14, r.top() + 6, r.width() - 20, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setFont(scaled_font(self.font(), 2.0, True))
        p.setPen(c)
        p.drawText(QRectF(r.left() + 14, r.top() + 26, r.width() - 20, 36), Qt.AlignLeft | Qt.AlignVCenter, self.value)
        p.setFont(_small(self))
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(r.left() + 14, r.top() + 62, r.width() - 20, 16), Qt.AlignLeft | Qt.AlignVCenter, self.note)
        bar = QRectF(r.left() + 14, r.bottom() - 9, r.width() - 28, 4)
        p.fillRect(bar, QColor(255, 255, 255, 25))
        p.fillRect(QRectF(bar.left(), bar.top(), bar.width() * self.frac, bar.height()), c)


class Clock(QLabel):
    def __init__(self, fmt='yyyy-MM-dd  HH:mm:ss', parent=None):
        super().__init__(parent)
        self.fmt = fmt
        self._tick()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)

    def _tick(self):
        self.setText(QDateTime.currentDateTime().toString(self.fmt))
