"""VMD bank for the shared multi-channel pool (one decomposition per installation window and channel).

Same VMD settings as every other bank: K=8, alpha=300, tau=0, DC=0, init=1, tol=1e-7 (vmdpy), modes sorted by
centre frequency. Input = per-window, per-channel z-scored pool windows (= what m2_paderborn_kr.py trains on
with --norm window). Stored as float16: bank (N_pool, C, 8, 2048).
Usage: python build_vmd_pool.py [--pool pool/pool_motor2_100_ch1245.npz] [--workers 4]
The training script checks on load that the modes reconstruct its training windows (relative error < 0.1).
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

K, ALPHA, TAU, DC, INIT, TOL = 8, 300.0, 0.0, 0, 1, 1e-7


def _vmd_one(x):
    from vmdpy import VMD
    u, _, omega = VMD(x, ALPHA, TAU, K, DC, INIT, TOL)
    u = u[np.argsort(omega[-1])]
    if u.shape[1] != len(x):
        u = np.pad(u, ((0, 0), (0, len(x) - u.shape[1])))
    return u.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=os.path.join('pool', 'pool_motor2_100_ch1245.npz'))
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    out = a.pool[:-4] + '_vmd.npy'
    try:
        import vmdpy  # noqa: F401
    except ImportError:
        sys.exit('vmdpy is not installed: python -m pip install vmdpy')
    X = np.load(a.pool)['X'].astype(np.float64)                                 # (N, C, L)
    Z = (X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-8)
    N, C, L = Z.shape
    flat = Z.reshape(N * C, L)
    t = time.time(); U = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, u in enumerate(ex.map(_vmd_one, flat, chunksize=16)):
            U.append(u)
            if (i + 1) % 544 == 0:
                done = (i + 1) / len(flat)
                print(f'  {100 * done:5.1f} %  ({(time.time() - t) / 60:.1f} min, about {(time.time() - t) / done / 60 * (1 - done):.0f} min left)', flush=True)
    bank = np.stack(U).reshape(N, C, K, L)
    rel = float(np.linalg.norm(bank.sum(2) - Z) / np.linalg.norm(Z))
    np.save(out, bank.astype(np.float16))
    sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    pool_sha = hashlib.sha256(open(a.pool, 'rb').read()).hexdigest()
    json.dump(dict(pool=os.path.basename(a.pool), pool_sha256=pool_sha, vmd_shape=list(bank.shape), dtype='float16',
                   bank_sha256=sha, K=K, alpha=ALPHA, tau=TAU, DC=DC, init=INIT, tol=TOL,
                   input='per-window per-channel z-scored pool X', reconstruction_rel_error=round(rel, 4)),
              open(out[:-4] + '.json', 'w'), indent=1)
    print(f'{out}: {bank.shape} in {(time.time() - t) / 60:.1f} min, reconstruction error {rel:.3f} (sandbox bank: 0.039)')


if __name__ == '__main__':
    main()
