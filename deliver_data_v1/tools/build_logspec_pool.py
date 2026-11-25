"""Precompute the 4-channel log magnitude spectrum of all pool windows (un-augmented view).

Usage:  python build_logspec_pool.py --pool <dir>/pool_motor2_100_ch1245.npz [--out <dir>] [--no-torch-check]

Recipe (identical to code_keep/m2_paderborn_kr.py with --norm window: zn() then log_spectrum()):
  z = ((X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-8)).astype(np.float32)
  S = log(|rfft(z * hann_periodic(2048))| + 1e-6)        -> (2720, 4, 1025) float32
  bin k = k * 20000 / 2048 Hz (k = 0 .. 1024), resolution 9.765625 Hz.
In training the existing encoder computes the spectrum from the (augmented) waveform; this array is the
spectrum of the un-augmented (weak) view only, before the per-channel, per-bin standardisation of SpecEnc.
Outputs: pool_motor2_100_ch1245_logspec.npy and pool_motor2_100_ch1245_logspec.json in --out
(default: the folder of --pool).
"""
import argparse
import hashlib
import json
import os

import numpy as np

EPS_STD = 1e-8
EPS_LOG = 1e-6


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def zn(X):
    return ((X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), EPS_STD)).astype(np.float32)


def hann_periodic(L):
    n = np.arange(L, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / L)


def log_spectrum_np(z):
    """z float32 (B, C, L). Window product, FFT and log in float64, result rounded once to float32."""
    w = hann_periodic(z.shape[-1])
    return np.log(np.abs(np.fft.rfft(z.astype(np.float64) * w, axis=-1)) + EPS_LOG).astype(np.float32)


def torch_reference(z, dtype):
    """log_spectrum of code_keep/m2_paderborn_kr.py (verbatim body), input zn(X) cast to dtype."""
    import torch
    x = torch.from_numpy(z).to(dtype)
    w = torch.hann_window(x.shape[-1], periodic=True, device=x.device, dtype=x.dtype)
    return torch.log(torch.fft.rfft(x * w).abs() + 1e-6).numpy().astype(np.float32), torch.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-torch-check', action='store_true')
    a = ap.parse_args()
    out_dir = a.out or os.path.dirname(os.path.abspath(a.pool))
    man = json.load(open(a.pool[:-4] + '.json', encoding='utf-8'))
    pool_sha = sha256_file(a.pool)
    if pool_sha != man['sha256']:
        raise SystemExit(f'pool sha256 mismatch ({pool_sha}); re-assemble the pool file')
    P = np.load(a.pool)
    X = P['X']
    fs, L = float(P['sample_rate_hz']), int(P['window_size'])
    assert X.shape[-1] == L and X.dtype == np.float32
    z = zn(X)
    S = np.empty(X.shape[:-1] + (L // 2 + 1,), np.float32)
    for i in range(0, len(z), 256):
        S[i:i + 256] = log_spectrum_np(z[i:i + 256])
    assert np.isfinite(S).all()

    check = {'done': False}
    if not a.no_torch_check:
        try:
            import torch
            d64, d32, ver = 0.0, np.empty(S.shape, np.float32), None
            for i in range(0, len(z), 256):
                R, ver = torch_reference(z[i:i + 256], torch.float64)
                d64 = max(d64, float(np.abs(R - S[i:i + 256]).max()))
                R, _ = torch_reference(z[i:i + 256], torch.float32)
                d32[i:i + 256] = np.abs(R - S[i:i + 256])
            check = {'done': True, 'torch_version': ver, 'windows_checked': int(len(z)),
                     'reference': 'code_keep/m2_paderborn_kr.py log_spectrum: torch.log(torch.fft.rfft(x * '
                                  'torch.hann_window(L, periodic=True, dtype=x.dtype)).abs() + 1e-6), x = zn(X)',
                     'max_abs_diff_vs_torch_float64': d64,
                     'max_abs_diff_vs_torch_float32': float(d32.max()),
                     'fraction_of_values_with_diff_gt_1e-4_vs_torch_float32': float((d32 > 1e-4).mean()),
                     'quantile_0.9999_diff_vs_torch_float32': float(np.quantile(d32, 0.9999)),
                     'note': 'the float32 torch FFT has rounding error up to about 1e-2 at bins with very small '
                             'magnitude (log value near -7 or lower); against a float64 reference the torch '
                             'float32 result deviates by that much, this array (float64 computation, rounded '
                             'once to float32) does not. Pass criterion: diff vs torch float64 <= 1e-4.'}
            if d64 > 1e-4:
                raise SystemExit(f'numpy vs torch float64 max abs diff {d64} > 1e-4')
        except ImportError:
            check = {'done': False, 'reason': 'torch not installed'}

    name = os.path.basename(a.pool)[:-4] + '_logspec'
    npy = os.path.join(out_dir, name + '.npy')
    os.makedirs(out_dir, exist_ok=True)
    np.save(npy, S)
    doc = {'file': name + '.npy', 'sha256': sha256_file(npy), 'size_bytes': os.path.getsize(npy),
           'shape': list(S.shape), 'dtype': 'float32',
           'pool': os.path.basename(a.pool), 'pool_sha256': pool_sha,
           'channels': [int(c) for c in P['channels']], 'sample_rate_hz': fs, 'window_size': L,
           'n_bins': L // 2 + 1, 'bin_resolution_hz': fs / L,
           'bin_frequency': f'bin k (0..{L // 2}) = k * {fs:g} / {L} Hz',
           'recipe': ['z = ((X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-8))'
                      '.astype(np.float32)   (per window, per channel; zn of m2_paderborn_kr.py, --norm window)',
                      'w = periodic Hann window of length 2048: 0.5 - 0.5*cos(2*pi*n/2048), n = 0..2047 '
                      '(= torch.hann_window(2048, periodic=True))',
                      'S = log(|rfft(z * w)| + 1e-6): z cast to float64, product, rfft and log in float64, '
                      'result rounded to float32'],
           'row_order': 'same as pool X (pool index)',
           'note': 'spectrum of the un-augmented (weak) view only; the encoder in training computes the spectrum '
                   'from the augmented waveform and then standardises it per channel and bin with training '
                   'statistics, which is not applied here',
           'torch_check': check}
    with open(os.path.join(out_dir, name + '.json'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(doc, indent=1) + '\n')
    print(f'{npy}: shape {S.shape} float32, sha256 {doc["sha256"]}, torch check {check}')


if __name__ == '__main__':
    main()
