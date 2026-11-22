"""后台线程：建模、分析录音。界面线程只负责显示，不会卡住。"""
import traceback
from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    progress = Signal(int, int, str)        # 当前, 总数, 说明
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw
        self.result = None

    def run(self):
        try:
            self.result = self.fn(*self.args, progress=lambda i, n, s='': self.progress.emit(i, n, s), **self.kw)
            self.done.emit(self.result)
        except Exception as e:                                         # 把错误显示给用户，而不是让程序闪退
            traceback.print_exc()
            self.failed.emit(f'{type(e).__name__}: {e}')


def train_job(files, method, feature, q, progress):
    from ..anomaly import train_from_files
    return train_from_files(files, method, feature, q, progress=lambda i, n: progress(i, n, f'读取正常录音 {i}/{n}'))


def analyze_job(model, path, hop_s, threshold, progress):
    from ..audio import load_signal
    from ..monitor import analyze
    progress(0, 1, '读取录音…')
    x, fs = load_signal(path)
    fs = fs or model.feature['fs']
    res, events, imgs, stats = analyze(model, x, fs, hop_s=hop_s, threshold=threshold,
                                       progress=lambda i, n: progress(i, n, f'生成时频图 {i}/{n}'))
    return {'path': path, 'x': x, 'fs': fs, 'results': res, 'events': events, 'imgs': imgs, 'stats': stats}
