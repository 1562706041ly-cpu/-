"""图形界面（PySide6）。启动：python -m kdsj gui

需要额外安装：pip install PySide6
背景图：把图片放到 kdsj/assets/background.jpg（没有时使用程序生成的深蓝背景）。
"""
import os
import sys


def main():
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(v, '4')
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.exit('缺少 PySide6，请先运行：python -m pip install PySide6')
    from .style import QSS, setup_fonts
    from .login import LoginWindow
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName('矿盾声鉴')
    setup_fonts(app)
    app.setStyleSheet(QSS)
    state = {}

    def show_login():
        w = LoginWindow()
        w.loggedIn.connect(lambda user: show_main(user, w))
        state['login'] = w
        w.show()

    def show_main(user, login):
        m = MainWindow(user)
        m.loggedOut.connect(lambda: (m.close(), show_login()))
        state['main'] = m
        m.showMaximized()
        login.close()

    show_login()
    sys.exit(app.exec())
