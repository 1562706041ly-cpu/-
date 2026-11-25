"""监测主页：设备列表、录音波形、时频图、异常指数曲线、运行状态卡片、告警记录。"""
import glob
import os
import time
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton, QListWidget,
                               QListWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
                               QFileDialog, QMessageBox, QDialog, QLineEdit, QComboBox, QDoubleSpinBox,
                               QDialogButtonBox, QInputDialog, QAbstractItemView, QSizePolicy, QTextBrowser)
from PySide6.QtGui import QColor
from .style import Backdrop, Panel, glow_label, CYAN, MUTED, OK, WARN, GOLD
from .widgets import WaveformView, TfrView, IndexChart, StatCard, Clock
from .store import DeviceStore
from .workers import Worker, train_job, analyze_job
from .. import SOFTWARE_NAME, VERSION_LABEL, SUBTITLE


class NewDeviceDialog(QDialog):
    """新建设备：选正常录音文件夹 -> 只用正常声音训练异常检测模型。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('新建设备')
        self.setMinimumWidth(520)
        g = QGridLayout(self)
        self.name = QLineEdit()
        self.name.setPlaceholderText('例如：1 号主通风机')
        self.folder = QLineEdit()
        self.folder.setPlaceholderText('选择该设备正常运行时的录音文件夹')
        browse = QPushButton('浏览…')
        browse.clicked.connect(self._browse)
        self.count = QLabel('')
        self.count.setStyleSheet(f'color: {MUTED};')
        self.feature = QComboBox()
        self.feature.addItem('对数梅尔谱', 'mel')
        self.feature.addItem('同步提取变换图', 'set')
        self.method = QComboBox()
        self.method.addItem('频带统计 + 马氏距离', 'gauss')
        self.method.addItem('自编码器', 'ae')
        self.win = QDoubleSpinBox()
        self.win.setRange(1.0, 30.0)
        self.win.setValue(10.0)
        self.win.setSuffix(' 秒')
        self.win.setToolTip('每次判断用多长的声音。实验验证的设置是 10 秒')
        self.q = QDoubleSpinBox()
        self.q.setRange(0.80, 0.999)
        self.q.setDecimals(3)
        self.q.setSingleStep(0.01)
        self.q.setValue(0.95)
        self.q.setToolTip('阈值取正常录音异常分数的这个分位数。0.95 对应目标误报率约 5%')
        rows = [('设备名称', self.name), ('正常录音', self.folder), ('', self.count), ('时频特征', self.feature),
                ('检测方法', self.method), ('窗长', self.win), ('阈值分位数', self.q)]
        for i, (lab, w) in enumerate(rows):
            g.addWidget(QLabel(lab), i, 0)
            g.addWidget(w, i, 1)
        g.addWidget(browse, 1, 2)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('开始建模')
        bb.button(QDialogButtonBox.Cancel).setText('取消')
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        g.addWidget(bb, len(rows), 0, 1, 3)
        self.folder.textChanged.connect(self._count)

    def files(self):
        d = self.folder.text().strip()
        return sorted(glob.glob(os.path.join(d, '**', '*.wav'), recursive=True)) if d and os.path.isdir(d) else []

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, '选择正常录音文件夹')
        if d:
            self.folder.setText(d)

    def _count(self):
        n = len(self.files())
        self.count.setText(f'找到 {n} 个 wav 文件' + ('' if n >= 2 else '，至少需要 2 个'))

    def _ok(self):
        if not self.name.text().strip():
            QMessageBox.warning(self, '提示', '请填写设备名称')
        elif len(self.files()) < 2:
            QMessageBox.warning(self, '提示', '正常录音文件夹里至少需要 2 个 wav 文件')
        else:
            self.accept()

    def params(self):
        return dict(name=self.name.text().strip(), files=self.files(), method=self.method.currentData(),
                    feature={'kind': self.feature.currentData(), 'win_s': self.win.value(), 'fs': 16000},
                    q=self.q.value())


class MainWindow(Backdrop):
    loggedOut = Signal()

    def __init__(self, user='用户', store=None):
        super().__init__(overlay=0.78)
        self.setWindowTitle(f'{SOFTWARE_NAME} {VERSION_LABEL}')
        self.resize(1500, 900)
        self.store = store or DeviceStore()
        self.models, self.current, self.worker, self.n_alarms = {}, None, None, 0

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 12)
        root.setSpacing(10)
        root.addLayout(self._header(user))

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._left(), 0)
        body.addLayout(self._center(), 1)
        body.addWidget(self._right(), 0)
        root.addLayout(body, 1)
        self.refresh_devices()
        self.log(f'用户 {user} 登录系统，已载入 {self.dev_list.count()} 台设备')

    # ---------------- 布局 ----------------
    def _header(self, user):
        h = QHBoxLayout()
        left = QLabel(f'KDSJ  {VERSION_LABEL}')
        left.setStyleSheet(f'color: {MUTED};')
        title = glow_label(f'“矿盾 · 声鉴” —— {SUBTITLE}', 22)
        title.setAlignment(Qt.AlignCenter)
        clock = Clock()
        clock.setStyleSheet(f'color: {CYAN}; font-size: 16px; font-weight: bold;')
        who = QLabel(f'当前用户：{user}')
        who.setStyleSheet(f'color: {MUTED};')
        out = QPushButton('退出登录')
        out.clicked.connect(self.loggedOut.emit)
        h.addWidget(left)
        h.addStretch()
        h.addWidget(title)
        h.addStretch()
        h.addWidget(clock)
        h.addSpacing(16)
        h.addWidget(who)
        h.addWidget(out)
        return h

    def _left(self):
        p = Panel('设备列表')
        p.setFixedWidth(270)
        self.dev_list = QListWidget()
        self.dev_list.currentItemChanged.connect(self._device_changed)
        p.body.addWidget(self.dev_list, 1)
        self.btn_new = QPushButton('＋ 新建设备')
        self.btn_import = QPushButton('导入已有模型')
        self.btn_del = QPushButton('删除所选设备')
        self.btn_new.clicked.connect(self.new_device)
        self.btn_import.clicked.connect(self.import_model)
        self.btn_del.clicked.connect(self.delete_device)
        for b in (self.btn_new, self.btn_import, self.btn_del):
            p.body.addWidget(b)
        self.model_info = QLabel('')
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet(f'color: {MUTED}; font-size: 12px;')
        p.body.addWidget(self.model_info)
        return p

    def _center(self):
        v = QVBoxLayout()
        v.setSpacing(10)
        wp = Panel('录音波形')
        bar = QHBoxLayout()
        self.btn_analyze = QPushButton('▶ 分析录音')
        self.btn_analyze.setObjectName('primary')
        self.btn_analyze.clicked.connect(self.analyze_file)
        self.hop = QDoubleSpinBox()
        self.hop.setRange(0.5, 30)
        self.hop.setValue(10.0)
        self.hop.setToolTip('相邻两次判断的间隔。等于窗长时窗口不重叠；选中设备时自动设为该设备的窗长')
        self.hop.setPrefix('窗口步长 ')
        self.hop.setSuffix(' 秒')
        self.btn_report = QPushButton('导出报告')
        self.btn_report.clicked.connect(self.export_report)
        self.btn_report.setEnabled(False)
        self.prog = QProgressBar()
        self.prog.setValue(0)
        self.status = QLabel('请选择设备')
        self.status.setStyleSheet(f'color: {MUTED};')
        bar.addWidget(self.btn_analyze)
        bar.addWidget(self.hop)
        bar.addWidget(self.btn_report)
        bar.addWidget(self.prog, 1)
        bar.addWidget(self.status, 1)
        wp.body.addLayout(bar)
        self.wave = WaveformView()
        self.wave.windowClicked.connect(self.select_window)
        wp.body.addWidget(self.wave, 1)
        v.addWidget(wp, 3)

        mid = QHBoxLayout()
        mid.setSpacing(10)
        tp = Panel('当前窗口时频图')
        self.tfr = TfrView()
        tp.body.addWidget(self.tfr)
        tp.setMinimumWidth(320)
        ip = Panel('异常指数')
        self.chart = IndexChart()
        self.chart.windowClicked.connect(self.select_window)
        ip.body.addWidget(self.chart)
        mid.addWidget(tp, 2)
        mid.addWidget(ip, 5)
        v.addLayout(mid, 4)

        ap = Panel('告警记录', accent=WARN)
        self.alarm_table = QTableWidget(0, 6)
        self.alarm_table.setHorizontalHeaderLabels(['分析时间', '设备', '录音文件', '时段', '类别', '异常指数/置信度'])
        self.alarm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.alarm_table.verticalHeader().setVisible(False)
        self.alarm_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.alarm_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.alarm_table.setAlternatingRowColors(True)
        self.alarm_table.cellClicked.connect(self._alarm_clicked)
        ap.body.addWidget(self.alarm_table)
        v.addWidget(ap, 3)
        return v

    def _right(self):
        col = QWidget()
        col.setFixedWidth(300)
        v = QVBoxLayout(col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        p = Panel('运行状态')
        self.card_state = StatCard('设备状态')
        self.card_max = StatCard('最高异常指数')
        self.card_alarm = StatCard('本次告警事件')
        self.card_speed = StatCard('处理速度')
        for c in (self.card_state, self.card_max, self.card_alarm, self.card_speed):
            c.setMinimumHeight(92)
            p.body.addWidget(c)
        v.addWidget(p, 0)
        ep = Panel('事件详情')
        self.events = QTextBrowser()
        self.events.setOpenLinks(False)
        self.events.setStyleSheet('QTextBrowser { background: rgba(3,14,32,170); border: 1px solid #1d5a80; }')
        self.events.anchorClicked.connect(self._event_clicked)
        ep.body.addWidget(self.events)
        self._event_html, self._event_targets = [], {}
        v.addWidget(ep, 1)
        return col

    def log(self, text, level='info', window=None):
        """在“事件详情”最上面加一条记录。level：info | ok | warn；window：点击后跳转的窗口编号。"""
        import html
        color = {'info': '#bfe6ff', 'ok': OK, 'warn': WARN}[level]
        body = html.escape(text)
        if window is not None and self.current:
            key = str(len(self._event_targets))
            self._event_targets[key] = (self.current['path'], window)
            body = f'<a href="ev:{key}" style="color:{color}; text-decoration:none;">{body}</a>'
        self._event_html.insert(0, f'<div style="color:{color}; margin-bottom:8px;">'
                                   f'<span style="color:#6f93ab;">{time.strftime("%H:%M:%S")}</span>&nbsp; {body}</div>')
        del self._event_html[200:]
        self.events.setHtml(''.join(self._event_html))

    def _event_clicked(self, url):
        t = self._event_targets.get(url.toString().split(':', 1)[-1])
        if t and self.current and self.current['path'] == t[0]:
            self.select_window(t[1])

    # ---------------- 设备 ----------------
    def refresh_devices(self, select=None):
        self.dev_list.blockSignals(True)
        self.dev_list.clear()
        for m in self.store.list():
            kind = '异常检测' if m['kind'] == 'anomaly' else '分类识别'
            it = QListWidgetItem(f'●  {m["name"]}\n     {kind} · {m["feature"]["kind"]} · {m["feature"]["win_s"]:g} s')
            it.setData(Qt.UserRole, m)
            it.setForeground(QColor('#dff6ff'))
            self.dev_list.addItem(it)
            if m['name'] == select:
                self.dev_list.setCurrentItem(it)
        self.dev_list.blockSignals(False)
        if self.dev_list.count() and self.dev_list.currentItem() is None:
            self.dev_list.setCurrentRow(0)
        self._device_changed(self.dev_list.currentItem(), None)

    def device(self):
        it = self.dev_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def model(self):
        m = self.device()
        if m is None:
            return None
        if m['path'] not in self.models:
            from ..anomaly import load_model
            self.models[m['path']] = load_model(m['path'])
        return self.models[m['path']]

    def select_device(self, name):
        for i in range(self.dev_list.count()):
            if self.dev_list.item(i).data(Qt.UserRole)['name'] == name:
                self.dev_list.setCurrentRow(i)

    def _device_changed(self, cur, prev):
        m = cur.data(Qt.UserRole) if cur else None
        self.btn_analyze.setEnabled(m is not None and self.worker is None)
        self.btn_del.setEnabled(m is not None)
        if m is None:
            self.model_info.setText('还没有设备。点击“新建设备”，选择设备正常运行时的录音即可建模。')
            self.card_state.set('--', '未选择设备', 0, MUTED)
            return
        info = m.get('info', {})
        if m['kind'] == 'anomaly':
            n, nc = info.get('n_files', 0), info.get('n_files_cal', 0)
            txt = (f'类型：异常检测\n方法：{"马氏距离" if m["method"] == "gauss" else "自编码器"}\n'
                   f'正常录音：{n} 个，建模 {n - nc} 个，定阈值 {nc} 个\n'
                   f'阈值：{m["threshold"]:.4g}\n创建：{m["created"]}')
        else:
            txt = f'类型：分类识别\n类别：{"、".join(m.get("classes", []))}\n网络：{m.get("model")}\n创建：{m["created"]}'
        self.model_info.setText(txt)
        self.status.setText(f'已选择：{m["name"]}')
        self.card_state.set('待分析', m['name'], 0, CYAN)
        self.hop.setValue(float(m['feature']['win_s']))

    def new_device(self):
        dlg = NewDeviceDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        p = dlg.params()
        if self.store.exists(p['name']) and QMessageBox.question(
                self, '确认', f'设备“{p["name"]}”已存在，是否覆盖？') != QMessageBox.Yes:
            return
        self._start(Worker(train_job, p['files'], p['method'], p['feature'], p['q']),
                    lambda r: self._trained(p['name'], r), f'正在为“{p["name"]}”建模…')

    def _trained(self, name, r):
        model, info = r
        path = self.store.add_model(name, model, info)
        self.models.pop(path, None)
        self.refresh_devices(select=name)
        self.status.setText(f'建模完成：{name}，阈值 {model.threshold:.4g}')
        self.log(f'设备“{name}”建模完成：{info["n_files"]} 段正常录音，告警阈值 {model.threshold:.4g}', 'ok')
        QMessageBox.information(self, '建模完成',
                                f'设备“{name}”建模完成。\n建模用 {info["n_files"] - info["n_files_cal"]} 个录音，'
                                f'定阈值用 {info["n_files_cal"]} 个录音。\n现在可以点击“分析录音”。')

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择模型文件', '', '模型文件 (*.pt)')
        if not path:
            return
        try:
            from ..anomaly import load_model
            model = load_model(path)
        except Exception as e:
            QMessageBox.critical(self, '无法读取', f'这个文件不是矿盾声鉴的模型：\n{e}')
            return
        name, ok = QInputDialog.getText(self, '设备名称', '给这个模型起个名字：',
                                        text=os.path.splitext(os.path.basename(path))[0])
        if ok and name.strip():
            self.store.import_file(name.strip(), path, model)
            self.refresh_devices(select=name.strip())
            self.log(f'导入模型：{name.strip()}')

    def delete_device(self):
        m = self.device()
        if m and QMessageBox.question(self, '确认删除', f'删除设备“{m["name"]}”及其模型？') == QMessageBox.Yes:
            self.models.pop(m['path'], None)
            self.store.remove(m['name'])
            self.refresh_devices()
            self.log(f'删除设备：{m["name"]}')

    # ---------------- 分析 ----------------
    def analyze_file(self, path=None):
        model = self.model()
        if model is None:
            QMessageBox.information(self, '提示', '请先在左侧选择或新建设备')
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, '选择要分析的录音', '', '录音 (*.wav)')
        if not path:
            return
        name = self.device()['name']
        self.log(f'开始分析“{name}”的录音 {os.path.basename(path)}')
        self._start(Worker(analyze_job, model, path, self.hop.value(), 0.5),
                    lambda r: self._analyzed(name, r), f'正在分析 {os.path.basename(path)}…')

    def _start(self, worker, on_done, text):
        self.worker = worker
        for b in (self.btn_new, self.btn_import, self.btn_del, self.btn_analyze, self.btn_report):
            b.setEnabled(False)
        self.status.setText(text)
        self.prog.setValue(0)
        worker.progress.connect(self._progress)
        worker.done.connect(on_done)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        worker.start()

    def _progress(self, i, n, s):
        self.prog.setMaximum(max(n, 1))
        self.prog.setValue(i)
        if s:
            self.status.setText(s)

    def _failed(self, msg):
        self.log('出错：' + msg, 'warn')
        QMessageBox.critical(self, '出错了', msg)
        self.status.setText('出错：' + msg)

    def _finished(self):
        self.worker = None
        for b in (self.btn_new, self.btn_import):
            b.setEnabled(True)
        has = self.device() is not None
        self.btn_analyze.setEnabled(has)
        self.btn_del.setEnabled(has)
        self.btn_report.setEnabled(self.current is not None)

    def _analyzed(self, name, r):
        r['device'] = name
        r['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.current = r
        res, ev, st = r['results'], r['events'], r['stats']
        model = self.model()
        anomaly = getattr(model, 'is_anomaly', False)
        self.wave.set_signal(r['x'], r['fs'])
        self.wave.set_windows([(w.start_s, w.end_s, w.alarm) for w in res])
        self.chart.set_data([(w.start_s + w.end_s) / 2 for w in res], [w.prob for w in res], [w.alarm for w in res],
                            1.0 if anomaly else 0.5, '异常指数' if anomaly else '置信度')
        mx = max((w.prob for w in res), default=0)
        if ev:
            self.card_state.set('告警', f'{name}：{len(ev)} 个异常时段', 1, WARN)
        else:
            self.card_state.set('正常', f'{name}：未发现异常', 1, OK)
        self.card_max.set(f'{mx:.2f}', '异常指数 = 分数 / 阈值，> 1 告警' if anomaly else '最高置信度',
                          mx / 2 if anomaly else mx, WARN if (anomaly and mx > 1) else GOLD if anomaly and mx > 0.8 else CYAN)
        self.card_alarm.set(str(len(ev)), f'共 {len(res)} 个窗口，{sum(w.alarm for w in res)} 个告警',
                            sum(w.alarm for w in res) / max(len(res), 1), WARN if ev else OK)
        self.card_speed.set(f'{st["realtime_factor"]:.0f} ×', f'{st["audio_s"]:.0f} s 录音用时 {st["time_total_s"]:.2f} s',
                            min(1, st['realtime_factor'] / 100), CYAN)
        fname = os.path.basename(r['path'])
        for e in ev:
            self.n_alarms += 1
            row = self.alarm_table.rowCount()
            self.alarm_table.insertRow(row)
            vals = [r['time'], name, fname, f'{e.start_s:.1f} – {e.end_s:.1f} s', e.label, f'{e.max_prob:.2f}']
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(Qt.UserRole, (r['path'], e.start_s))
                if c >= 4:
                    it.setForeground(QColor(WARN))
                self.alarm_table.setItem(row, c, it)
        self.alarm_table.scrollToBottom()
        self._log_analysis(name, fname, res, ev, st, r['imgs'], model, anomaly)
        first = next((i for i, w in enumerate(res) if w.alarm), 0)
        self.select_window(first)
        self.status.setText(f'{fname}：{len(res)} 个窗口，{len(ev)} 个告警时段')
        self.prog.setValue(self.prog.maximum())

    @staticmethod
    def _severity(v):
        return '严重' if v >= 1.5 else '中度' if v >= 1.2 else '轻度'

    def _log_analysis(self, name, fname, res, ev, st, imgs, model, anomaly):
        if not ev:
            self.log(f'“{name}” {fname}：{st["audio_s"]:.0f} s 录音分析完成，{len(res)} 个窗口全部正常', 'ok')
            return
        # 先加明细，再加总结，这样总结显示在最上面
        for e in reversed(ev):
            idx = [k for k, w in enumerate(res) if w.alarm and e.start_s <= w.start_s < e.end_s]
            k = max(idx, key=lambda j: res[j].prob) if idx else None
            if anomaly:
                text = f'⚠ {e.start_s:.0f}–{e.end_s:.0f} s 声音异常，异常指数 {e.max_prob:.2f}，{self._severity(e.max_prob)}'
                ex = model.explain(imgs[k]) if k is not None and hasattr(model, 'explain') else None
                if ex:
                    text += f'；主要差异在 {ex["f_lo"]:.0f}–{ex["f_hi"]:.0f} Hz，能量{ex["direction"]}'
            else:
                text = f'⚠ {e.start_s:.0f}–{e.end_s:.0f} s 识别为“{e.label}”，置信度 {e.max_prob:.2f}'
            self.log(text, 'warn', k)
        worst = max(ev, key=lambda e: e.max_prob)
        self.log(f'“{name}” {fname}：发现 {len(ev)} 段异常，最严重在 {worst.start_s:.0f}–{worst.end_s:.0f} s，'
                 f'建议到现场检查该设备', 'warn', next((k for k, w in enumerate(res) if w.start_s >= worst.start_s), None))

    def select_window(self, i):
        r = self.current
        if r is None or not (0 <= i < len(r['results'])):
            return
        w = r['results'][i]
        self.tfr.set_image(r['imgs'][i], f'{w.start_s:.1f}–{w.end_s:.1f} s  {w.label}  {w.prob:.2f}')
        self.wave.set_current(i)
        self.chart.set_current(i)

    def _alarm_clicked(self, row, col):
        path, t = self.alarm_table.item(row, 0).data(Qt.UserRole)
        if self.current and self.current['path'] == path:
            res = self.current['results']
            self.select_window(min(range(len(res)), key=lambda k: abs(res[k].start_s - t)))

    def export_report(self):
        r = self.current
        if r is None:
            return
        from ..monitor import write_report
        base = os.path.splitext(os.path.basename(r['path']))[0]
        default = str(self.store.reports_dir() / f'{r["device"]}_{base}_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        path, _ = QFileDialog.getSaveFileName(self, '导出报告', default, 'CSV 文件 (*.csv)')
        if path:
            write_report(path, r['results'], r['events'], r['stats'], r['path'])
            self.log(f'报告已导出：{os.path.basename(path)}')
            self.status.setText(f'报告已保存：{path}')
