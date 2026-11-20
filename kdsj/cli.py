"""命令行入口：python -m kdsj <命令> [参数]

  selftest                          自检
  vmd     信号文件 [--optimizer ttao] VMD 参数寻优并分解，保存模态和收敛曲线
  train-anomaly 正常录音目录 --out 模型.pt
                                    只用一台设备的正常录音训练异常检测模型（自动定告警阈值）
  analyze 模型.pt 音频.wav [...]    整段音频滑窗识别（分类模型或异常检测模型），输出告警事件和 CSV 报告
  gui                               打开图形界面
"""
import argparse
import json
import os
import sys
import time


def _limit_threads(n):
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(v, str(n))


def cmd_vmd(a):
    import numpy as np
    from . import vmd, optim
    from .audio import load_signal
    x, fs = load_signal(a.signal)
    fs = a.fs or fs or 1
    ar, kr = (a.alpha_min, a.alpha_max), (a.k_min, a.k_max)
    ev = optim.Evaluator(lambda z: vmd.fitness(z, x, a.metric, a.aggregate, ar, kr),
                         keyfun=lambda z: vmd.key(z, ar, kr), max_evals=a.max_evals)
    res = optim.optimize(a.optimizer, ev, 2, a.pop, a.iters, seed=a.seed)
    alpha, K = res['info']['alpha'], res['info']['K']
    u, omega, _ = vmd.vmd(x, alpha, K)
    os.makedirs(a.out, exist_ok=True)
    np.savetxt(os.path.join(a.out, 'imfs.csv'), u.T, delimiter=',')
    np.savetxt(os.path.join(a.out, 'convergence.csv'), res['trace'], delimiter=',')
    info = {'optimizer': res['name'], 'alpha': alpha, 'K': K, 'fitness': res['best_fit'],
            'center_freq_hz': [float(o * fs) for o in omega], 'n_calls': res['n_calls'],
            'n_unique': res['n_unique'], 'time_s': res['time_s']}
    json.dump(info, open(os.path.join(a.out, 'result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'[{res["name"]}] alpha={alpha}, K={K}, 适应度 {res["best_fit"]:.4f}，'
          f'调用 {res["n_calls"]} 次（实际计算 {res["n_unique"]} 次），用时 {res["time_s"]:.1f} s')
    print(f'结果已保存到 {a.out}')


def cmd_train_anomaly(a):
    import glob
    import numpy as np
    from .anomaly import AnomalyModel
    from .audio import load_signal
    from .monitor import signal_windows
    files = sorted(glob.glob(os.path.join(a.normal_dir, '**', '*.wav'), recursive=True))
    if len(files) < 2:
        sys.exit(f'目录 {a.normal_dir} 里至少需要 2 个 wav 文件')
    feature = {'kind': a.feature, 'win_s': a.win, 'fs': a.fs}
    imgs = []
    for i, f in enumerate(files, 1):
        x, fs = load_signal(f)
        imgs.append(signal_windows(x, fs or a.fs, feature))
        if i % 20 == 0 or i == len(files):
            print(f'读取并生成时频图 {i}/{len(files)}', flush=True)
    n_cal = max(1, int(round(len(files) * a.cal_ratio)))          # 按文件划分：最后一部分文件只用于定阈值
    X_tr, X_cal = np.concatenate(imgs[:-n_cal]), np.concatenate(imgs[-n_cal:])
    m = AnomalyModel(a.method, feature, a.q).fit(X_tr, X_cal)
    m.save(a.out)
    print(f'训练 {len(X_tr)} 个窗口（{len(files) - n_cal} 个文件），定阈值 {len(X_cal)} 个窗口（{n_cal} 个文件），'
          f'阈值 {m.threshold:.4g}（正常样本 {100 * a.q:.0f}% 分位数）')
    print(f'模型已保存到 {a.out}')


def cmd_analyze(a):
    from .anomaly import load_model
    from .monitor import analyze_file, write_report
    rec = load_model(a.model, a.device)
    for path in a.audio:
        res, events, _, stats = analyze_file(rec, path, hop_s=a.hop, threshold=a.threshold)
        print(f'{path}：{stats["n_windows"]} 个窗口，{len(events)} 条告警，'
              f'处理速度为实时的 {stats["realtime_factor"]:.1f} 倍')
        what = '异常指数' if getattr(rec, 'is_anomaly', False) else '置信度'
        for e in events:
            print(f'  告警 {e.start_s:.1f}–{e.end_s:.1f} s  {e.label}  {what} {e.max_prob:.2f}')
        if a.report:
            out = a.report if len(a.audio) == 1 else os.path.splitext(path)[0] + '_report.csv'
            write_report(out, res, events, stats, path)
            print(f'  报告：{out}')


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m kdsj', description='矿盾声鉴：设备声音识别与告警')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest', help='自检')
    v = sub.add_parser('vmd', help='VMD 参数寻优与分解')
    v.add_argument('signal')
    v.add_argument('--fs', type=float, default=None, help='采样率（wav 会自动读取）')
    v.add_argument('--optimizer', default='ttao', choices=['ttao', 'bka', 'random'])
    v.add_argument('--pop', type=int, default=12)
    v.add_argument('--iters', type=int, default=20)
    v.add_argument('--max_evals', type=int, default=None)
    v.add_argument('--metric', default='envelope', choices=['envelope', 'permutation', 'neg_kurtosis'])
    v.add_argument('--aggregate', default='min', choices=['min', 'mean'])
    v.add_argument('--alpha_min', type=int, default=100)
    v.add_argument('--alpha_max', type=int, default=2500)
    v.add_argument('--k_min', type=int, default=3)
    v.add_argument('--k_max', type=int, default=10)
    v.add_argument('--seed', type=int, default=42)
    v.add_argument('--out', default=os.path.join('results', time.strftime('vmd_%Y%m%d_%H%M%S')))
    an = sub.add_parser('analyze', help='整段音频识别与告警')
    an.add_argument('model')
    an.add_argument('audio', nargs='+')
    an.add_argument('--hop', type=float, default=None, help='窗口步长（秒），默认窗长的一半')
    an.add_argument('--threshold', type=float, default=0.5, help='告警置信度阈值')
    an.add_argument('--report', default='', help='CSV 报告路径')
    an.add_argument('--device', default=None)
    ta = sub.add_parser('train-anomaly', help='只用正常录音训练异常检测模型')
    ta.add_argument('normal_dir')
    ta.add_argument('--out', default='anomaly_model.pt')
    ta.add_argument('--method', default='gauss', choices=['gauss', 'ae'])
    ta.add_argument('--feature', default='mel', choices=['mel', 'set'])
    ta.add_argument('--win', type=float, default=10.0, help='窗长（秒）')
    ta.add_argument('--fs', type=int, default=16000, help='模型采样率，录音不同时自动重采样')
    ta.add_argument('--q', type=float, default=0.95, help='阈值分位数，0.95 对应目标误报率约 5%%')
    ta.add_argument('--cal_ratio', type=float, default=0.2, help='用于定阈值的文件比例')
    sub.add_parser('gui', help='打开图形界面')
    a = p.parse_args(argv)
    _limit_threads(4)
    if a.cmd == 'selftest':
        from .selftest import run
        sys.exit(1 if run() else 0)
    elif a.cmd == 'vmd':
        cmd_vmd(a)
    elif a.cmd == 'train-anomaly':
        cmd_train_anomaly(a)
    elif a.cmd == 'analyze':
        cmd_analyze(a)
    elif a.cmd == 'gui':
        from .gui import main as gui_main
        gui_main()
