"""从 MIMII 压缩包中只取第 1 通道，存成每台设备一个 int16 数组（不解压整个 zip，省磁盘）。

MIMII：Hitachi, Ltd., CC BY-SA 4.0，https://zenodo.org/records/3384388
每个 wav：10 s、16 kHz、16 bit、8 通道（环形麦克风阵列），已混入真实工厂背景噪声。

输出：<out>/<snr>_<machine>.npz
  audio  : n_files x 160000 int16（第 1 通道）
  id     : 设备编号（0/2/4/6）
  state  : 0 = normal，1 = abnormal
  name   : 文件名（按编号排序）

用法：python mimii_extract.py --zip_dir mimii --out mimii_ch1 --snr 0_dB
"""
import argparse, os, re, zipfile
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from kdsj.audio import read_wav

ap = argparse.ArgumentParser()
ap.add_argument('--zip_dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--snr', default='0_dB')
ap.add_argument('--machines', default='fan,pump,slider,valve')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

for m in a.machines.split(','):
    out = os.path.join(a.out, f'{a.snr}_{m}.npz')
    if os.path.exists(out):
        print(out, '已存在'); continue
    z = zipfile.ZipFile(os.path.join(a.zip_dir, f'{a.snr}_{m}.zip'))
    names = [n for n in z.namelist() if n.endswith('.wav')]
    key = lambda n: (int(re.search(r'id_(\d+)', n).group(1)), 'abnormal' in n, n)
    names.sort(key=key)
    audio = np.zeros((len(names), 160000), np.int16)
    ids = np.zeros(len(names), np.int8); state = np.zeros(len(names), np.int8)
    for i, n in enumerate(names):
        X, fs = read_wav(z.read(n))
        assert fs == 16000
        x = np.round(X[:, 0] * 32768).astype(np.int16)
        L = min(len(x), 160000); audio[i, :L] = x[:L]
        ids[i] = int(re.search(r'id_(\d+)', n).group(1)); state[i] = 'abnormal' in n
    np.savez(out, audio=audio, id=ids, state=state, name=np.array(names))
    print(out, audio.shape, '异常', int(state.sum()), flush=True)
