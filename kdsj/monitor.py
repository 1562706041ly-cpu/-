"""整段音频的滑窗识别与告警。

把任意长度的音频按模型训练时的窗长切成窗口（相邻窗口重叠 hop_s），逐窗生成时频图并识别。
支持两种模型：
  分类模型（Recognizer）：连续判为"告警类别"且置信度 ≥ threshold 的窗口合并成一条告警事件；
  异常检测模型（AnomalyModel）：异常分数超过模型自带的阈值即告警。prob 字段为"异常指数" = 分数 / 阈值，
  大于 1 表示超过阈值。
"""
import csv
import time
from dataclasses import dataclass, asdict
import numpy as np
from .audio import load_signal, frames
from .tfr import make_image


@dataclass
class WindowResult:
    start_s: float
    end_s: float
    label: str
    prob: float
    alarm: bool


@dataclass
class AlarmEvent:
    start_s: float
    end_s: float
    label: str
    max_prob: float
    n_windows: int


def resample_linear(x, fs, fs_new):
    """线性插值重采样（仅在录音采样率与模型不一致时使用）。"""
    if fs == fs_new:
        return x
    n = int(round(len(x) * fs_new / fs))
    return np.interp(np.arange(n) * fs / fs_new, np.arange(len(x)), x)


def analyze(recognizer, x, fs, hop_s=None, threshold=0.5, progress=None):
    """返回 (逐窗结果列表, 告警事件列表, 每个窗口的时频图, 统计信息)。"""
    feat = recognizer.feature
    x = resample_linear(np.asarray(x, np.float64), fs, feat['fs'])
    fs = feat['fs']
    win = int(round(feat['win_s'] * fs))
    hop = int(round((hop_s or feat['win_s'] / 2) * fs))
    W, starts = frames(x, win, hop)
    t0 = time.time()
    imgs = []
    for i, w in enumerate(W):
        imgs.append(make_image(w, fs, feat['kind']))
        if progress:
            progress(i + 1, len(W))
    imgs = np.stack(imgs)
    t_img = time.time() - t0
    res = []
    if getattr(recognizer, 'is_anomaly', False):
        ratio = recognizer.score(imgs) / recognizer.threshold
        t_all = time.time() - t0
        for s, r in zip(starts, ratio):
            res.append(WindowResult(s / fs, (s + win) / fs, '异常' if r > 1 else '正常', float(r), bool(r > 1)))
    else:
        pred, P = recognizer.predict(imgs)
        t_all = time.time() - t0
        names = recognizer.class_names
        for s, k, p in zip(starts, pred, P):
            lab = names[k] if k >= 0 else '无法判别'
            prob = float(p[k]) if k >= 0 else 0.0
            res.append(WindowResult(s / fs, (s + win) / fs, lab, prob,
                                    lab in recognizer.alarm_classes and prob >= threshold))
    events, cur = [], None
    for r in res:
        if r.alarm and cur and cur.label == r.label and r.start_s <= cur.end_s:
            cur.end_s, cur.max_prob, cur.n_windows = r.end_s, max(cur.max_prob, r.prob), cur.n_windows + 1
        elif r.alarm:
            cur = AlarmEvent(r.start_s, r.end_s, r.label, r.prob, 1)
            events.append(cur)
        else:
            cur = None
    stats = {'n_windows': len(W), 'audio_s': len(x) / fs, 'time_image_s': t_img, 'time_total_s': t_all,
             'realtime_factor': (len(x) / fs) / max(t_all, 1e-9)}
    return res, events, imgs, stats


def analyze_file(recognizer, path, **kw):
    x, fs = load_signal(path)
    return analyze(recognizer, x, fs or recognizer.feature['fs'], **kw)


def signal_windows(x, fs, feature):
    """把信号按模型的窗长切窗（不重叠），返回时频图数组。"""
    x = resample_linear(np.asarray(x, np.float64), fs, feature['fs'])
    win = int(round(feature['win_s'] * feature['fs']))
    W, _ = frames(x, win, win)
    return np.stack([make_image(w, feature['fs'], feature['kind']) for w in W])


def write_report(path, results, events, stats, source=''):
    """导出 CSV 报告：先是统计信息和告警事件，再是逐窗结果。"""
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['文件', source])
        for k, v in stats.items():
            w.writerow([k, f'{v:.4g}' if isinstance(v, float) else v])
        w.writerow([])
        w.writerow(['告警事件', '开始(s)', '结束(s)', '类别', '最高置信度/异常指数', '窗口数'])
        for i, e in enumerate(events, 1):
            w.writerow([i, f'{e.start_s:.2f}', f'{e.end_s:.2f}', e.label, f'{e.max_prob:.3f}', e.n_windows])
        w.writerow([])
        w.writerow(['窗口', '开始(s)', '结束(s)', '类别', '置信度/异常指数', '告警'])
        for i, r in enumerate(results, 1):
            w.writerow([i, f'{r.start_s:.2f}', f'{r.end_s:.2f}', r.label, f'{r.prob:.3f}', '是' if r.alarm else ''])
