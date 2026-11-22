"""生成用户手册用的界面截图（不打开窗口，直接渲染）。

用法：python tools/make_screenshots.py 正常录音文件夹 测试录音.wav 输出目录
"""
import glob
import os
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['KDSJ_DATA'] = tempfile.mkdtemp(prefix='kdsj_shots_')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main(normal_dir, test_wav, out):
    os.makedirs(out, exist_ok=True)
    from PySide6.QtWidgets import QApplication
    from kdsj.gui.style import QSS, setup_fonts
    from kdsj.gui.login import LoginWindow
    from kdsj.gui.main_window import MainWindow, NewDeviceDialog
    from kdsj.gui.workers import train_job, analyze_job
    from kdsj.monitor import write_report
    app = QApplication.instance() or QApplication(sys.argv)
    setup_fonts(app)
    app.setStyleSheet(QSS)
    W, H = 1500, 900

    def shot(widget, name):
        widget.show()
        app.processEvents()
        widget.grab().save(os.path.join(out, name))

    lw = LoginWindow(); lw.resize(W, H); lw.user.setText('admin'); lw.pwd.setText('123456')
    shot(lw, '01_login.png')
    lw.pwd.setText('000'); lw._login(); app.processEvents()
    lw.msg.parentWidget().grab().save(os.path.join(out, '02_login_wrong.png')); lw.close()

    win = MainWindow('admin'); win.resize(W, H)
    shot(win, '03_main_empty.png')

    dlg = NewDeviceDialog(win)
    dlg.name.setText('3 号阀门'); dlg.folder.setText(normal_dir)
    p = dlg.params()
    dlg.folder.setText('D:\\矿盾声鉴\\正常录音\\3号阀门')        # 截图里显示 Windows 路径
    dlg.count.setText(f'找到 {len(p["files"])} 个 wav 文件')
    shot(dlg, '04_new_device.png'); dlg.close()

    model, info = train_job(p['files'], p['method'], p['feature'], p['q'], progress=lambda *a: None)
    path = win.store.add_model(p['name'], model, info)
    win.refresh_devices(select=p['name'])
    win.log(f'设备“{p["name"]}”建模完成：{info["n_files"]} 段正常录音，告警阈值 {model.threshold:.4g}', 'ok')
    win.status.setText(f'建模完成：{p["name"]}，阈值 {model.threshold:.4g}')
    shot(win, '05_main_trained.png')

    win.log(f'开始分析“{p["name"]}”的录音 {os.path.basename(test_wav)}')
    r = analyze_job(win.model(), test_wav, win.hop.value(), 0.5, progress=lambda *a: None)
    win._analyzed(p['name'], r)
    win._finished()
    shot(win, '06_main_result.png')

    panels = {'07_wave.png': win.wave, '08_tfr.png': win.tfr, '09_index.png': win.chart,
              '11_events.png': win.events, '12_alarms.png': win.alarm_table, '13_devices.png': win.dev_list}
    for name, w in panels.items():
        w.parentWidget().grab().save(os.path.join(out, name))
    win.card_state.parentWidget().grab().save(os.path.join(out, '10_status.png'))
    win.select_window(8)
    app.processEvents()
    win.tfr.parentWidget().grab().save(os.path.join(out, '14_tfr_alarm.png'))

    rep = os.path.join(out, 'report_example.csv')
    write_report(rep, r['results'], r['events'], r['stats'], os.path.basename(test_wav))
    print('截图已保存到', out)


if __name__ == '__main__':
    main(*sys.argv[1:4])
