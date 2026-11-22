"""不打开窗口，直接把登录页和主页渲染成图片（用于检查界面和制作说明书截图）。

用法：python tools/gui_preview.py 正常录音文件夹 测试录音.wav 输出目录
需要 PySide6。没有显示器的电脑上先设置环境变量 QT_QPA_PLATFORM=offscreen。
"""
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main(normal_dir, test_wav, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.environ['KDSJ_DATA'] = tempfile.mkdtemp(prefix='kdsj_preview_')
    from PySide6.QtWidgets import QApplication
    from kdsj.gui.style import QSS, setup_fonts
    from kdsj.gui.login import LoginWindow
    from kdsj.gui.main_window import MainWindow
    from kdsj.gui.store import DeviceStore
    from kdsj.gui.workers import train_job, analyze_job
    app = QApplication.instance() or QApplication(sys.argv)
    print('字体：', setup_fonts(app))
    app.setStyleSheet(QSS)

    login = LoginWindow()
    login.resize(1500, 900)
    login.user.setText('admin')
    login.show(); app.processEvents()
    login.grab().save(os.path.join(out_dir, 'login.png'))

    store = DeviceStore()
    files = sorted(glob.glob(os.path.join(normal_dir, '*.wav')))
    model, info = train_job(files, 'gauss', {'kind': 'mel', 'win_s': 10.0, 'fs': 16000}, 0.95,
                            progress=lambda *a: None)
    store.add_model('3 号阀门', model, info)
    store.add_model('1 号主通风机', model, info)
    win = MainWindow('admin', store)
    win.resize(1500, 900)
    win.select_device('3 号阀门')
    r = analyze_job(win.model(), test_wav, win.hop.value(), 0.5, progress=lambda *a: None)
    win._analyzed('3 号阀门', r)
    win.show(); app.processEvents()
    win.grab().save(os.path.join(out_dir, 'main.png'))
    win.resize(1280, 720); app.processEvents()
    win.grab().save(os.path.join(out_dir, 'main_1280.png'))
    print('已保存', os.path.join(out_dir, 'login.png'), os.path.join(out_dir, 'main.png'))


if __name__ == '__main__':
    main(*sys.argv[1:4])
