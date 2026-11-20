"""命令行入口：python -m kdsj <命令> [参数]

  selftest                          自检
  vmd     信号文件 [--optimizer ttao] VMD 参数寻优并分解，保存模态和收敛曲线
  analyze 模型.pt 音频.wav [...]    整段音频滑窗识别，输出告警事件和 CSV 报告
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


def cmd_analyze(a):
    from .recognizer import Recognizer
    from .monitor import analyze_file, write_report
    rec = Recognizer.load(a.model, a.device)
    for path in a.audio:
        res, events, _, stats = analyze_file(rec, path, hop_s=a.hop, threshold=a.threshold)
        print(f'{path}：{stats["n_windows"]} 个窗口，{len(events)} 条告警，'
              f'处理速度为实时的 {stats["realtime_factor"]:.1f} 倍')
        for e in events:
            print(f'  告警 {e.start_s:.1f}–{e.end_s:.1f} s  {e.label}  置信度 {e.max_prob:.2f}')
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
    sub.add_parser('gui', help='打开图形界面')
    a = p.parse_args(argv)
    _limit_threads(4)
    if a.cmd == 'selftest':
        from .selftest import run
        sys.exit(1 if run() else 0)
    elif a.cmd == 'vmd':
        cmd_vmd(a)
    elif a.cmd == 'analyze':
        cmd_analyze(a)
    elif a.cmd == 'gui':
        from .gui import main as gui_main
        gui_main()
