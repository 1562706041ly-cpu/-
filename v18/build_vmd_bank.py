"""Build the VMD bank (+ manifest) for one or more split NPZ files, compatible with m2_paderborn_kr.py.

Same VMD settings as the Paderborn bank: K=8, alpha=300, tau=0, DC=0, init=1, tol=1e-7 (vmdpy).
The bank decomposes the PER-WINDOW Z-SCORED training windows, because the training script adds the modes
to z-scored windows (the Paderborn NPZ was already z-scored, the centrifugal NPZ is raw acceleration).

Usage:
    pip install vmdpy
    python build_vmd_bank.py centrifugal\\splits\\motor2_100_ch4_C1_fold1.npz [more.npz ...] [--workers 8]
Writes <npz stem>_vmd.npy and <npz stem>_vmd.json next to every NPZ. Existing, valid banks are skipped.
"""
import argparse, hashlib, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

K, ALPHA, TAU, DC, INIT, TOL = 8, 300.0, 0.0, 0, 1, 1e-7


def _vmd_one(x):
    from vmdpy import VMD
    u, _, omega = VMD(x, ALPHA, TAU, K, DC, INIT, TOL)
    order = np.argsort(omega[-1])                      # modes sorted by centre frequency (low -> high)
    u = u[order]
    if u.shape[1] != len(x):                           # vmdpy drops one sample for odd lengths
        u = np.pad(u, ((0, 0), (0, len(x) - u.shape[1])))
    return u.astype(np.float32)


def build(npz, workers):
    stem = npz[:-4]; out_npy, out_json = stem + '_vmd.npy', stem + '_vmd.json'
    d = np.load(npz, allow_pickle=False)
    X = np.asarray(d['X_train'], dtype=np.float32)
    train_sha = hashlib.sha256(np.ascontiguousarray(X).view(np.uint8)).hexdigest()
    if os.path.exists(out_json) and os.path.exists(out_npy):
        m = json.load(open(out_json))
        if m.get('train_sha256') == train_sha:
            print(f'skip {npz} (bank exists)'); return
    Z = ((X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-8)).astype(np.float64)
    t = time.time()
    with ProcessPoolExecutor(workers) as ex:
        bank = np.stack(list(ex.map(_vmd_one, Z, chunksize=16)))
    np.save(out_npy, bank)
    bank_sha = hashlib.sha256(open(out_npy, 'rb').read()).hexdigest()
    rel = float(np.linalg.norm(bank.sum(1) - Z) / np.linalg.norm(Z))
    json.dump(dict(dataset=os.path.basename(npz), shape=[int(X.shape[0]), int(X.shape[1])], vmd_shape=list(bank.shape),
                   train_sha256=train_sha, bank_sha256=bank_sha, K=K, alpha=ALPHA, tau=TAU, DC=DC, init=INIT, tol=TOL,
                   dtype='float32', input='per-window z-scored X_train', reconstruction_rel_error=round(rel, 4)),
              open(out_json, 'w'), indent=1)
    print(f'{npz}: bank {bank.shape} in {time.time() - t:.0f} s, reconstruction error {rel:.3f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('npz', nargs='+'); ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    for f in a.npz: build(f, a.workers)
