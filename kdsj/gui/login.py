"""登录页。"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGridLayout
from .style import Backdrop, Panel, glow_label, MUTED, WARN, CYAN
from .widgets import Clock
from . import auth
from .. import __version__


class LoginWindow(Backdrop):
    loggedIn = Signal(str)

    def __init__(self):
        super().__init__(overlay=0.55)
        self.setWindowTitle('矿盾·声鉴 — 登录')
        self.resize(1280, 760)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 18, 30, 18)

        top = QHBoxLayout()
        brand = QLabel('KDSJ  ·  Mine Equipment Acoustic Monitoring')
        brand.setStyleSheet(f'color: {MUTED}; letter-spacing: 1px;')
        clock = Clock()
        clock.setStyleSheet(f'color: {CYAN}; font-size: 18px; font-weight: bold;')
        top.addWidget(brand)
        top.addStretch()
        top.addWidget(clock)
        outer.addLayout(top)
        outer.addStretch(2)

        title = glow_label('“矿盾 · 声鉴”', 40)
        title.setAlignment(Qt.AlignCenter)
        sub = glow_label('矿山设备声音智能监测系统', 20, '#bfeaff', False)
        sub.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)
        outer.addWidget(sub)
        outer.addSpacing(30)

        row = QHBoxLayout()
        row.addStretch()
        panel = Panel('用户登录')
        panel.setFixedWidth(420)
        g = QGridLayout()
        g.setVerticalSpacing(14)
        g.setHorizontalSpacing(12)
        self.user = QLineEdit()
        self.user.setPlaceholderText('请输入账号')
        self.pwd = QLineEdit()
        self.pwd.setPlaceholderText('请输入密码')
        self.pwd.setEchoMode(QLineEdit.Password)
        for i, (lab, w) in enumerate((('账  号', self.user), ('密  码', self.pwd))):
            l = QLabel(lab)
            l.setStyleSheet('color: #ffcf6b; font-weight: bold; font-size: 14px;')
            g.addWidget(l, i, 0)
            g.addWidget(w, i, 1)
        panel.body.addLayout(g)
        self.msg = QLabel('')
        self.msg.setStyleSheet(f'color: {WARN};')
        self.msg.setAlignment(Qt.AlignCenter)
        panel.body.addWidget(self.msg)
        self.btn = QPushButton('登  录')
        self.btn.setObjectName('primary')
        self.btn.setCursor(Qt.PointingHandCursor)
        panel.body.addWidget(self.btn)
        row.addWidget(panel)
        row.addStretch()
        outer.addLayout(row)
        outer.addStretch(3)

        foot = QLabel(f'矿盾声鉴  v{__version__}    声学感知 · 智能识别 · 实时告警')
        foot.setStyleSheet(f'color: {MUTED};')
        foot.setAlignment(Qt.AlignCenter)
        outer.addWidget(foot)

        self.btn.clicked.connect(self._login)
        self.pwd.returnPressed.connect(self._login)
        self.user.returnPressed.connect(self.pwd.setFocus)

    def _login(self):
        if auth.check(self.user.text(), self.pwd.text()):
            self.msg.setText('')
            self.loggedIn.emit(self.user.text().strip() or '用户')
        else:
            self.msg.setText('密码错误，请重新输入')
            self.pwd.selectAll()
            self.pwd.setFocus()
