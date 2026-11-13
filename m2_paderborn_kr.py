"""Paderborn single-condition open-set pipeline with K/R-controlled augmentation and
reliable-relation contrastive learning.

Derived from the uploaded `m2_paderborn` script. Old modes (generic / fixed / adaptive_energy /
shuffled_energy) and the old plain contrastive loss are kept so that baselines stay replayable.

New:
  --mode adaptive_kr | shuffled_kr   K(x)+R(x) controlled VMD budget (strength + number of modes)
  --contrast reliable                reliable positive/negative relation contrastive loss
  --refresh-every N                  re-estimate K,R and relations with the updated encoder
  --n-unknown-clusters M             number of unknown centres (default: len(--unknown))
Unknown-class labels are never used for fitting, mapping, restart selection or thresholds.
They appear only in the explicitly evaluation-only blocks (`kr_diag`, `relation_diag`, cluster_acc).
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
import argparse, os, hashlib, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score

_parser = argparse.ArgumentParser()
_parser.add_argument('--condition', type=int, default=1, help='legacy metadata field; not used for Paderborn')
_parser.add_argument('--data', default=None)
_parser.add_argument('--vmd-bank', default=None, help="path to the VMD bank, or 'none' (only with --mode generic)")
_parser.add_argument('--mode', choices=['generic', 'fixed', 'adaptive_energy', 'shuffled_energy',
                                        'adaptive_kr', 'shuffled_kr', 'adaptive_kr2', 'shuffled_kr2'], required=True)
_parser.add_argument('--contrast', choices=['plain', 'reliable', 'reliable2'], default='plain')
# v16 Chapter 3 (adaptive_kr2): nuisance-calibrated K/R-controlled VMD augmentation
_parser.add_argument('--record-key', default=None, help='NPZ train metadata field with the record/installation id (used for LABELLED windows only)')
_parser.add_argument('--kr2-kmin', type=float, default=0.25, help='budget of uncertain samples (fraction of record spread)')
_parser.add_argument('--kr2-rho', type=float, default=0.5, help='relative budget of reliable-unknown vs reliable-known samples')
_parser.add_argument('--kr2-no-mode', action='store_true', help='ablation: same scale for every VMD mode (drop mode-level calibration)')
_parser.add_argument('--kr2-no-sample', action='store_true', help='ablation: same kappa for every sample (drop K/R control)')
_parser.add_argument('--kr2-no-fidelity', action='store_true', help='ablation: no log-spectral fidelity check')
# v18 Chapter 3: energy participates in the augmentation (use with --mode adaptive_kr2)
_parser.add_argument('--e1-lam', type=float, default=0.0,
                     help='scheme 1: weight of the energy-margin loss on VMD pseudo-unknowns (class-discriminative modes pushed); 0=off')
_parser.add_argument('--e2-fidelity', action='store_true',
                     help='scheme 2: energy semantic fidelity - a known-routed view must keep its predicted class and its energy')
_parser.add_argument('--e3-sens', action='store_true',
                     help='scheme 3: per-sample mode budget from the energy sensitivity of each VMD mode')
# v20 Chapter 3: the energy decides the DIRECTION of the VMD mode perturbation (magnitude bound unchanged)
_parser.add_argument('--edir', choices=['none', 'dist', 'head', 'random'], default='none',
                     help="direction of the mode log-amplitude change (adaptive_kr2): 'dist' = gradient sign of the distance "
                          "energy, 'head' = of the classifier free energy (known-routed: energy up, unknown-routed: down), "
                          "'random' = random sign with the same magnitudes (control); 'none' = old random Gaussian u")
_parser.add_argument('--guard-group-key', default=None,
                     help='NPZ train metadata field (e.g. train_installation): calibrate the known-guard threshold leave-one-GROUP-out')
_parser.add_argument('--kr2-gain', type=float, default=1.0, help='multiplier of the mode budget sigma_rec*g (1 = natural record spread)')
_parser.add_argument('--edir-lam', type=float, default=0.5, help='u = lam * 2 * direction + (1 - lam) * random  (|u| <= 2)')
_parser.add_argument('--delta', type=float, default=0.1)
_parser.add_argument('--warmup', type=int, default=300)
_parser.add_argument('--delta-known', type=float, default=0.05)
_parser.add_argument('--delta-unknown', type=float, default=0.2)
_parser.add_argument('--delta-weak', type=float, default=0.02, help='strength for unreliable samples (adaptive_kr)')
_parser.add_argument('--no-vary-modes', action='store_true', help='adaptive_kr: always perturb all modes')
_parser.add_argument('--alpha', type=float, default=0.5)
_parser.add_argument('--r-thr', type=float, default=0.5, help='R threshold for the evidence zones (diagnostics + relations)')
_parser.add_argument('--energy-T', type=float, default=0.05, help="temperature of the distance-based energy (--k-score denergy)")
_parser.add_argument('--k-score-train', choices=['fused', 'proto', 'knn', 'denergy', 'gated'], default=None,
                     help='K(x) used DURING training (Chapter 3 budget, R, Chapter 4 relations); default = --k-score')
_parser.add_argument('--gate-gamma', type=float, default=2.0, help="exponent of the distance gate for --k-score(-train) gated")
_parser.add_argument('--k-score', choices=['fused', 'proto', 'knn', 'denergy', 'gated'], default='fused',
                     help="K(x): 'fused' energy+prototype (v1-v5), 'proto' prototype only, 'knn' nearest labelled window (LOO-calibrated)")
_parser.add_argument('--label-balance-key', default=None,
                     help='NPZ train metadata field (e.g. bearing id): spread the labelled windows evenly over its values per known class')
_parser.add_argument('--label-record-key', default=None,
                     help='optional NPZ train metadata field (record/file id): prefer distinct records inside each group')
_parser.add_argument('--known-subcenters', type=int, default=1,
                     help='fixed centres per known class (k-means inside labelled [+expanded] windows); 1 = class mean (old)')
_parser.add_argument('--known-expand', type=float, default=0.0,
                     help='fraction of unlabeled windows (reliable known by K/R) added to the fixed known centres; 0=off')
_parser.add_argument('--refresh-every', type=int, default=0, help='0 = only at end of warmup')
# reliable-relation contrastive learning
_parser.add_argument('--knn', type=int, default=5)
_parser.add_argument('--w-unc', type=float, default=0.3, help='negative weight of uncertain pairs')
_parser.add_argument('--fn-frac', type=float, default=0.1, help='per-row fraction treated as suspected false negatives (0=off)')
_parser.add_argument('--hard-beta', type=float, default=0.5, help='hard-negative emphasis on RELIABLE negatives (0=off)')
_parser.add_argument('--hn-cap', type=float, default=3.0)
_parser.add_argument('--pos-thr', type=float, default=0.5)
_parser.add_argument('--rel-no-label', action='store_true', help='ablation: drop label-anchor evidence')
_parser.add_argument('--rel-no-nbr', action='store_true', help='ablation: drop neighbourhood evidence')
_parser.add_argument('--rel-no-cons', action='store_true', help='ablation: drop cross-view consistency')
_parser.add_argument('--splits', default='PADERBORN')
_parser.add_argument('--known', default='0,1')
_parser.add_argument('--unknown', default='2')
_parser.add_argument('--n-unknown-clusters', type=int, default=0,
                     help='0 = len(--unknown); -1 = choose 1..--auto-unk-max by silhouette on unlabelled training features')
_parser.add_argument('--auto-unk-max', type=int, default=3)
_parser.add_argument('--guard-expand', type=float, default=0.0,
                     help='fraction of unlabelled windows (reliable known by K/R) added to the known-guard reference set only')
_parser.add_argument('--known-guard', type=float, default=0.0,
                     help='0=off. Quantile q of labelled leave-one-out nearest-labelled similarity; unknown-routed windows '
                          'at least this similar to a labelled window are sent back to the nearest known class')
_parser.add_argument('--unk-init', choices=['random', 'kr'], default='random',
                     help="init of the unknown k-means centres: random unlabeled samples (old) or the K/R 'reliable unknown' zone")
_parser.add_argument('--seeds', default='0')
_parser.add_argument('--device', default='cuda')
_parser.add_argument('--steps', type=int, default=1500)
_parser.add_argument('--sup-steps', type=int, default=600)
_parser.add_argument('--tag', required=True)
# v19: multi-channel input (X = (N, C, L) in the NPZ) and amplitude-preserving normalisation
_parser.add_argument('--input', choices=['wave', 'spec'], default='wave',
                     help="'wave' = 1-D CNN on the waveform (old); 'spec' = log-spectrum front-end inside the encoder")
_parser.add_argument('--norm', choices=['window', 'global'], default='window',
                     help="'window' = per-window, per-channel z-score (old; removes the vibration level); "
                          "'global' = per-channel mean/std of the TRAINING windows (keeps the level differences)")
_parser.add_argument('--use_optimized_kmeans', action='store_true', help='kept for CLI compatibility; silhouette restarts are the default')
_parser.add_argument('--only_fixed_center', action='store_true', help='select restart by inertia instead of silhouette')
_args = _parser.parse_args()

_run_tag = _args.tag
_out = PROJECT_ROOT / 'runs_paderborn' / _run_tag
os.environ.update(COND=str(_args.condition), OUT=str(_out), SEEDS=_args.seeds, SPLITS=_args.splits,
                  PUMP_DEVICE=_args.device, LOCAL_STEPS_SUP=str(_args.sup_steps), LOCAL_STEPS_GCD=str(_args.steps))

from gpu_runtime import DEVICE, FAST_AUG, feats_gpu, vectorized_aug, save_checkpoint
import kr_modules as KR

COND = int(os.environ.get('COND', '1'))
OUT = os.environ.get('OUT')
_known = tuple(int(v) for v in _args.known.split(',') if v.strip())
_unknown = tuple(int(v) for v in _args.unknown.split(',') if v.strip())
if not _known or not _unknown or set(_known) & set(_unknown):
    raise ValueError('known/unknown must be non-empty and disjoint')
SPLITS = {'PADERBORN': (_known, _unknown)}
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
LABEL_RATIO, STEPS_SUP, STEPS_GCD, BS, N_LAB_BS, LAM, TAU = 0.1, 600, 1500, 128, 32, 0.35, 0.1
STEPS_SUP = int(os.environ.get('LOCAL_STEPS_SUP', STEPS_SUP))
STEPS_GCD = int(os.environ.get('LOCAL_STEPS_GCD', STEPS_GCD))
VMD_DELTA, AUG_MODE, ALPHA = _args.delta, _args.mode, _args.alpha
if _args.warmup < 1 or _args.warmup >= _args.steps:
    raise ValueError('--warmup must be >=1 and < --steps')
if _args.edir != 'none' and _args.mode != 'adaptive_kr2':
    raise ValueError('--edir is a Chapter 3 v2 option: use it with --mode adaptive_kr2')
REL_CFG = dict(use_label=not _args.rel_no_label, use_nbr=not _args.rel_no_nbr, use_cons=not _args.rel_no_cons,
               w_unc=_args.w_unc, fn_frac=_args.fn_frac, hard_beta=_args.hard_beta, hn_cap=_args.hn_cap,
               pos_thr=_args.pos_thr, zone_compat=True)
WEAK_P = dict(shift=64, scale=0.05, noise=0.02, mask=0.0)   # weak views used only to estimate R(x)

# ----------------------------------------------------------------------------- data
data_file = Path(_args.data) if _args.data else PROJECT_ROOT / 'data' / 'paderborn_filegroup_v2.npz'
if not data_file.exists():
    raise FileNotFoundError(f'找不到Paderborn预处理数据: {data_file}')
d = np.load(str(data_file), allow_pickle=False)
for key in ('X_train', 'y_train', 'X_val', 'y_val'):
    if key not in d.files:
        raise KeyError(f'{data_file} 缺少NPZ字段 {key}; 实际字段={d.files}')
XTR_raw = np.asarray(d['X_train'], dtype=np.float32)
XV_raw = np.asarray(d['X_val'], dtype=np.float32)
YTR = np.asarray(d['y_train'], dtype=np.int64).reshape(-1)
YV = np.asarray(d['y_val'], dtype=np.int64).reshape(-1)
if (XTR_raw.ndim not in (2, 3) or XV_raw.ndim != XTR_raw.ndim or len(XTR_raw) != len(YTR) or len(XV_raw) != len(YV)
        or XTR_raw.shape[1:] != XV_raw.shape[1:] or len(XV_raw) == 0):
    raise ValueError('NPZ signal/label shapes mismatch; expected X=(N,L) or (N,C,L), y=(N,)')
N_CH = 1 if XTR_raw.ndim == 2 else XTR_raw.shape[1]
SIG_LEN = XTR_raw.shape[-1]
# Optional per-window metadata (e.g. bearing / record ids) found in the NPZ: any extra 1-D array whose
# length equals the train or val size. Used ONLY for evaluation-only group diagnostics.
META_TR, META_V = {}, {}
for _k in d.files:
    if _k in ('X_train', 'y_train', 'X_val', 'y_val'): continue
    _v = d[_k]
    if _v.ndim != 1: continue
    if len(_v) == len(YTR): META_TR[_k] = np.asarray(_v).astype(str)
    elif len(_v) == len(YV): META_V[_k] = np.asarray(_v).astype(str)
if META_TR or META_V:
    print(f'[meta] train keys={list(META_TR)}, val keys={list(META_V)}', flush=True)
if _args.norm == 'window':
    zn = lambda X: ((X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-8)).astype(np.float32)
else:
    # per-channel statistics of the training windows only (labelled + unlabelled training data, no eval data);
    # the per-window mean (DC) is still removed, the vibration level is kept
    _c = XTR_raw - XTR_raw.mean(-1, keepdims=True)
    _sd = np.maximum(_c.std(axis=(0, -1), keepdims=True) if _c.ndim == 3 else _c.std(keepdims=True), 1e-8)[0]
    zn = lambda X: ((X - X.mean(-1, keepdims=True)) / _sd).astype(np.float32)
    print(f'[norm] global per-channel std of training windows: {np.ravel(_sd).round(4).tolist()}', flush=True)
XTR, XV = zn(XTR_raw), zn(XV_raw)
to3 = lambda a: a if a.ndim == 3 else a[:, None]          # (N,L) -> (N,1,L); (N,C,L) unchanged


NO_VMD = str(_args.vmd_bank).lower() == 'none'
if NO_VMD:
    # VMD is only used by the --mode fixed / adaptive_* / shuffled_* augmentations
    if _args.mode != 'generic':
        raise ValueError('--vmd-bank none is only allowed with --mode generic')
    vmd_modes_all, N_MODES = None, 8
elif _args.norm != 'window':
    raise NotImplementedError('VMD banks decompose per-window z-scored windows: use --norm window with a VMD bank')
elif N_CH > 1:
    # v19 multi-channel: ONE bank for the shared installation pool, (N_pool, C, M, L) float16; the split NPZ
    # carries train_pool_index. Only the Chapter 3 v2 augmentation (adaptive_kr2 / shuffled_kr2) supports it.
    if _args.mode not in ('generic', 'adaptive_kr2', 'shuffled_kr2') or _args.e1_lam > 0 or _args.e2_fidelity or _args.e3_sens:
        raise NotImplementedError('multi-channel VMD supports --mode generic / adaptive_kr2 / shuffled_kr2 without E1-E3')
    cache_path = Path(_args.vmd_bank)
    manifest = json.loads(cache_path.with_suffix('.json').read_text(encoding='utf-8'))
    bank_hasher = hashlib.sha256()
    with cache_path.open('rb') as bank_file:
        for chunk in iter(lambda: bank_file.read(1024 * 1024), b''):
            bank_hasher.update(chunk)
    if bank_hasher.hexdigest() != manifest.get('bank_sha256'):
        raise RuntimeError('VMD bank hash does not match manifest (re-assemble the bank parts).')
    if 'train_pool_index' not in d.files:
        raise KeyError('multi-channel VMD needs train_pool_index in the split NPZ (make_splits_from_pool.py)')
    vmd_np = np.load(cache_path, mmap_mode='r')
    _pi = np.asarray(d['train_pool_index'], dtype=np.int64)
    if vmd_np.ndim != 4 or vmd_np.shape[1] != N_CH or vmd_np.shape[3] != SIG_LEN or len(_pi) != len(XTR_raw):
        raise ValueError(f'Bad multi-channel VMD bank shape {vmd_np.shape} for X_train {XTR_raw.shape}')
    vmd_modes_all = torch.tensor(np.asarray(vmd_np[_pi], dtype=np.float32))                 # (N, C, M, L)
    # content check: the modes must reconstruct THESE z-scored training windows (catches a wrong pool/bank)
    _chk = np.linspace(0, len(XTR) - 1, min(64, len(XTR))).astype(int)
    _rel = float(np.linalg.norm(vmd_modes_all[_chk].sum(2).numpy() - XTR[_chk]) / np.linalg.norm(XTR[_chk]))
    if _rel > 0.1:
        raise RuntimeError(f'VMD bank does not reconstruct the training windows (relative error {_rel:.3f})')
    N_MODES = vmd_modes_all.shape[2]
    print(f'[vmd] multi-channel pool bank, reconstruction error on training windows {_rel:.3f}', flush=True)
else:
    cache_path = Path(_args.vmd_bank) if _args.vmd_bank else PROJECT_ROOT / 'data' / 'paderborn_filegroup_v2_vmd.npy'
    manifest_path = cache_path.with_suffix('.json')
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f'缺少Paderborn VMD缓存或清单: {cache_path}, {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    train_digest = hashlib.sha256(np.ascontiguousarray(XTR_raw).view(np.uint8)).hexdigest()
    if manifest.get('train_sha256') != train_digest:
        raise RuntimeError('VMD cache does not match this NPZ training array; rebuild the cache.')
    expected_bank_digest = manifest.get('bank_sha256')
    if not expected_bank_digest:
        raise RuntimeError('VMD manifest is missing bank_sha256.')
    bank_hasher = hashlib.sha256()
    with cache_path.open('rb') as bank_file:
        for chunk in iter(lambda: bank_file.read(1024 * 1024), b''):
            bank_hasher.update(chunk)
    if bank_hasher.hexdigest() != expected_bank_digest:
        raise RuntimeError('VMD bank hash does not match manifest.')
    if manifest.get('shape') != [int(len(XTR_raw)), int(XTR_raw.shape[1])]:
        raise RuntimeError('VMD manifest shape does not match NPZ.')
    vmd_np = np.load(cache_path, mmap_mode='r')
    if vmd_np.ndim != 3 or vmd_np.shape[0] != len(XTR_raw) or vmd_np.shape[2] != XTR_raw.shape[1]:
        raise ValueError(f'Bad VMD cache shape: {vmd_np.shape}')
    vmd_modes_all = torch.tensor(np.asarray(vmd_np), dtype=torch.float32)
    N_MODES = vmd_modes_all.shape[1]
print(f'[data] channels={N_CH}, norm={_args.norm}, train={XTR.shape}, val={XV.shape}, labels={sorted(np.unique(YTR).tolist())}; VMD={None if NO_VMD else tuple(vmd_modes_all.shape)}', flush=True)


# ----------------------------------------------------------------------------- model
def block(i, o, k, s=1): return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
class Enc(nn.Module):
    def __init__(s, in_ch=1):
        super().__init__(); s.f = nn.Sequential(block(in_ch, 16, 64, 8), block(16, 32, 3), block(32, 64, 3), block(64, 64, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten())
    def forward(s, x): return s.f(x)


def log_spectrum(x):
    """(B, C, L) waveform -> (B, C, L//2+1) log magnitude spectrum (periodic Hann window). Works for torch tensors."""
    w = torch.hann_window(x.shape[-1], periodic=True, device=x.device, dtype=x.dtype)
    return torch.log(torch.fft.rfft(x * w).abs() + 1e-6)


class SpecEnc(nn.Module):
    """v19 frequency-domain encoder: the WAVEFORM is still the input (all augmentations, incl. VMD, act on the
    waveform); the log spectrum is computed inside, standardised per channel and bin with fixed statistics of the
    training windows, then a 1-D CNN over frequency. Magnitude spectra are insensitive to the time shift of the
    window, so the encoder cannot use the window position, and fault lines stay at fixed bins."""
    def __init__(s, in_ch, mu, sd):
        super().__init__()
        s.register_buffer('mu', torch.as_tensor(mu, dtype=torch.float32)); s.register_buffer('sd', torch.as_tensor(sd, dtype=torch.float32))
        s.f = nn.Sequential(block(in_ch, 32, 15, 2), block(32, 64, 5), block(64, 64, 3), block(64, 64, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten())
    def forward(s, x): return s.f((log_spectrum(x) - s.mu) / s.sd)


def make_encoder():
    if _args.input == 'spec':
        return SpecEnc(N_CH, SPEC_MU, SPEC_SD)
    return Enc(N_CH)


SPEC_MU = SPEC_SD = None
if _args.input == 'spec':
    # per-channel, per-bin statistics of the log spectrum of the TRAINING windows (no eval data)
    with torch.no_grad():
        _S = torch.cat([log_spectrum(torch.as_tensor(to3(XTR[i:i + 512]))) for i in range(0, len(XTR), 512)])
    SPEC_MU = _S.mean(0, keepdim=True).numpy(); SPEC_SD = np.maximum(_S.std(0, keepdim=True).numpy(), 1e-3)
    del _S
    print(f'[input] log-spectrum front-end, bins={SPEC_MU.shape[-1]}', flush=True)


def aug_vmd_live(x, idx, delta, generator, n_modes=None):
    """VMD-mode amplitude perturbation with fidelity check.

    delta: scalar | (B,) tensor/array.  n_modes: optional (B,) long tensor = number of
    randomly chosen modes that are perturbed for each sample.
    """
    modes = vmd_modes_all[idx].to(x.device)
    B, K, L = modes.shape
    delta = torch.as_tensor(delta, dtype=x.dtype, device=x.device).reshape(-1, 1, 1)
    direction = torch.rand(B, K, 1, device=x.device, generator=generator) * 2 - 1
    if n_modes is not None:
        direction = direction * KR.mode_mask(n_modes.to(x.device), K, generator)
    x_aug = x + (direction * delta * modes).sum(dim=1, keepdim=True)
    cos = lambda z: F.cosine_similarity(x.flatten(1), z.flatten(1), dim=1)
    for _ in range(3):
        mask = cos(x_aug) < 0.85
        if not mask.any(): break
        x_aug[mask] = 0.5 * x[mask] + 0.5 * x_aug[mask]
    fallback = (cos(x_aug) < 0.85) | (~torch.isfinite(x_aug).flatten(1).all(1))
    if fallback.any(): x_aug[fallback] = x[fallback]
    return x_aug, dict(vmd_applied=True, fallback=float(fallback.float().mean()))


@torch.no_grad()
def get_features(enc, X):
    enc.eval()
    dev = next(enc.parameters()).device
    chunks = []
    for i in range(0, len(X), 256):
        x = torch.as_tensor(to3(X[i:i + 256]), dtype=torch.float32, device=dev)
        chunks.append(enc(x).cpu())
    return torch.cat(chunks).numpy()


# ----------------------------------------------------------------------------- clustering / metrics
def cluster_acc(y, p, known, fixed_known_centers=False):
    """POST-HOC cluster accuracy (unknown clusters matched with true unknown labels).
    Report this only as a legacy metric; use KR.openset_metrics for deployable numbers."""
    y = np.asarray(y).reshape(-1)
    raw_cluster_id = np.asarray(p).reshape(-1)
    known = [int(x) for x in known]
    cls, cl = np.unique(y), np.unique(raw_cluster_id)
    mp = {}
    if fixed_known_centers:
        mp.update({i: known[i] for i in range(len(known))})
        unknown_cl = [int(v) for v in cl if int(v) not in mp]
        unknown_cls = [int(t) for t in cls if int(t) not in known]
        w_unknown = np.zeros((len(unknown_cl), len(unknown_cls)), dtype=np.int64)
        for i, cluster_id in enumerate(unknown_cl):
            for j, true_label in enumerate(unknown_cls):
                w_unknown[i, j] = np.sum((raw_cluster_id == cluster_id) & (y == true_label))
        if w_unknown.size:
            r, c = linear_sum_assignment(-w_unknown)
            for i, j in zip(r, c):
                mp[unknown_cl[i]] = unknown_cls[j]
    else:
        w = np.zeros((len(cl), len(cls)), dtype=np.int64)
        for i, cluster_id in enumerate(cl):
            for j, true_label in enumerate(cls):
                w[i, j] = np.sum((raw_cluster_id == cluster_id) & (y == true_label))
        r, c = linear_sum_assignment(-w)
        mp.update({int(cl[i]): int(cls[j]) for i, j in zip(r, c)})
    pred_label = np.array([mp.get(int(v), -1) for v in raw_cluster_id])
    ok = pred_label == y
    old = np.isin(y, known)
    metrics = {"all": float(ok.mean()),
               "old": float(ok[old].mean()) if old.any() else float("nan"),
               "new": float(ok[~old].mean()) if (~old).any() else float("nan"),
               "nmi": float(nmi(y, raw_cluster_id))}
    return metrics, raw_cluster_id, pred_label, mp


def kr_unknown_init(FU, K, R, n_clusters, r_thr, seed, yu, known):
    """Initialise the unknown centres from the 'reliable unknown' zone (K<0.5 and R>=r_thr).

    Uses only K/R, which are computed from labelled known data + unlabeled features.
    `yu` is used ONLY to log the (evaluation-only) purity of the zone, never to build centres.
    Falls back to the 10% lowest-K unlabeled samples if the zone is too small.
    """
    zone = np.where((K < 0.5) & (R >= r_thr))[0]
    src = 'reliable_unknown'
    if len(zone) < max(5 * n_clusters, 10):
        zone = np.argsort(K)[:max(int(0.1 * len(K)), 5 * n_clusters)]
        src = 'lowest_K_10pct'
    if n_clusters == 1:
        C = FU[zone].mean(0, keepdims=True)
    else:
        C = KMeans(n_clusters, n_init=5, random_state=seed).fit(FU[zone]).cluster_centers_
    info = dict(source=src, n_zone=int(len(zone)),
                zone_unknown_frac_EVAL_ONLY=float((~np.isin(yu[zone], known)).mean()))
    return C, info


def known_expand_select(featA, n_lab, yl_idx, K, R, conf_head, pred_head, frac, r_thr, kind, yu, known):
    """Pick reliable-known UNLABELED windows to enlarge the (fixed) known centres.

    Candidates: K>=0.5 and R>=r_thr (decided from known labels + features only).
    Ranked by K*R*conf; the top `frac * n_unlabeled` are kept.
    Pseudo class: label of the nearest labelled window (kind='knn') or the head argmax (otherwise).
    `yu` is used ONLY for the evaluation-only purity log.
    Returns positions into the UNLABELED block and their class indices (0..C-1).
    """
    fn = F.normalize(featA, dim=1)
    FLn, FUn = fn[:n_lab], fn[n_lab:]
    if kind in ('knn', 'denergy', 'gated'):
        pseudo = yl_idx[(FUn @ FLn.T).argmax(1)]
        conf = torch.ones_like(K)
    else:
        pseudo, conf = pred_head, conf_head
    cand = (K >= 0.5) & (R >= r_thr)
    score = torch.where(cand, K * R * conf, torch.full_like(K, -1.0))
    n_take = min(int(round(frac * len(K))), int(cand.sum()))
    sel = score.argsort(descending=True)[:n_take]
    sel_np, cls_np = sel.cpu().numpy(), pseudo[sel].cpu().numpy()
    true = yu[sel_np]
    info = dict(n_candidates=int(cand.sum()), n_selected=int(n_take),
                selected_known_frac_EVAL_ONLY=float(np.isin(true, known).mean()) if n_take else float('nan'),
                selected_class_correct_EVAL_ONLY=float((true == np.asarray(known)[cls_np]).mean()) if n_take else float('nan'))
    return sel_np, cls_np, info


def ss_kmeans(FL, yl, FU, k, known, seed, iters=100, select='silhouette', restarts=10, init_unknown=None,
              known_centers=None, center_owner=None):
    """Semi-supervised k-means: known centres = labelled means (fixed), unknown centres learned on FU.
    Restart selection uses only unlabeled features (silhouette) or inertia. No unknown labels.
    init_unknown: optional (k-n_known, D) start for the unknown centres (single deterministic run).
    known_centers: optional (n_known, D) override (e.g. labelled + reliable pseudo-known means)."""
    rng = np.random.default_rng(seed)
    n_unk = k - len(known)                         # number of unknown clusters
    if known_centers is None:
        known_centers = np.array([FL[yl == c].mean(0) for c in known])
    n_fixed = len(known_centers)                   # may be > len(known) with --known-subcenters
    k_all = n_fixed + n_unk
    best = None
    for _ in range(1 if init_unknown is not None else restarts):
        if init_unknown is not None:
            unknown_centers = np.asarray(init_unknown, dtype=FU.dtype).copy()
        else:
            unknown_centers = np.array([FU[rng.choice(len(FU))] for _ in range(n_unk)])
        C = np.vstack([known_centers, unknown_centers])
        for _ in range(iters):
            pu = ((FU[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C_unknown = [FU[pu == n_fixed + j].mean(0) if (pu == n_fixed + j).any() else C[n_fixed + j]
                         for j in range(n_unk)]
            C2 = np.vstack([known_centers, np.stack(C_unknown)])
            if np.allclose(C2, C): break
            C = C2
        pu = ((FU[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        inertia = float(sum(((FU[pu == j] - C[j]) ** 2).sum() for j in range(k_all) if (pu == j).any()))
        pu_cls = remap_clusters(pu, center_owner, len(known), n_fixed) if center_owner is not None else pu
        if select == 'silhouette' and 2 <= len(np.unique(pu_cls)) < len(FU):
            score = silhouette_score(FU, pu_cls)
        else:
            score = -inertia
        if best is None or score > best[0]:
            best = (score, C.copy(), pu.copy())
    return best[1], best[2]


def remap_clusters(raw, center_owner, n_known, n_fixed):
    """Raw centre ids -> class-level cluster ids: known sub-centre j -> its class index,
    unknown centre n_fixed+u -> n_known+u. Identity when there are no sub-centres."""
    raw = np.asarray(raw)
    out = raw - n_fixed + n_known
    is_fixed = raw < n_fixed
    out[is_fixed] = np.asarray(center_owner)[raw[is_fixed]]
    return out


def build_known_subcenters(pools, m, seed):
    """pools: list (per known class) of (n_c, D) arrays (labelled + optional pseudo-known windows).
    Returns (sum_c m_c, D) centres and the owner class index of each centre.
    m_c = min(m, n_c // 3) so that each sub-centre has at least ~3 windows; m_c >= 1."""
    centers, owner, used = [], [], []
    for c, P in enumerate(pools):
        mc = max(1, min(m, len(P) // 3))
        if mc == 1:
            centers.append(P.mean(0, keepdims=True))
        else:
            centers.append(KMeans(mc, n_init=10, random_state=seed).fit(P).cluster_centers_)
        owner += [c] * mc
        used.append(mc)
    return np.vstack(centers), np.array(owner), used


nearest = lambda Fx, C: ((Fx[:, None] - C[None]) ** 2).sum(-1).argmin(1)


def supcon(z, y, tau):
    n = z.shape[0] // 2; sim = z @ z.T / tau; sim.fill_diagonal_(-1e9)
    ids = torch.arange(n, device=z.device).repeat(2) if y is None else y.repeat(2)
    pos = (ids[:, None] == ids[None]).float(); pos.fill_diagonal_(0)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -((pos * logp).sum(1) / pos.sum(1).clamp(min=1)).mean()


# ----------------------------------------------------------------------------- legacy K-only teacher
def prepare_frozen_energy(enc, cls_head, XL, XU, yl_np, global_idx, seed, split, alpha=0.5):
    features = get_features(enc, np.concatenate([XL, XU]))
    features_t = torch.tensor(features, device=DEVICE)
    energy_score = torch.logsumexp(cls_head(features_t), dim=1).detach().cpu().numpy()
    num_L = len(XL)
    features_L = features_t[:num_L]
    known_classes = np.unique(yl_np)
    prototypes = [features_L[torch.as_tensor(yl_np == c, device=DEVICE)].mean(dim=0) for c in known_classes if (yl_np == c).sum() > 0]
    prototypes = F.normalize(torch.stack(prototypes), dim=1)
    proto_sim = torch.max(torch.matmul(F.normalize(features_t, dim=1), prototypes.T), dim=1)[0].detach().cpu().numpy()

    def calibrate_score(scores):
        labeled_scores = scores[:num_L]
        low = float(np.quantile(labeled_scores, 0.05)); mid = float(np.median(labeled_scores))
        scale = max((mid - low) / np.log(19.0), 1e-6)
        return torch.sigmoid(torch.as_tensor((scores - low) / scale)).numpy()

    q_E, q_D = calibrate_score(energy_score), calibrate_score(proto_sim)
    q_fused = alpha * q_E + (1 - alpha) * q_D
    delta = (_args.delta_known + (_args.delta_unknown - _args.delta_known) * (1 - q_fused)).astype(np.float32)
    shuffled = delta.copy()
    rng = np.random.default_rng(seed + 30000)
    for ids in (np.arange(num_L), np.arange(num_L, len(delta))):
        shuffled[ids] = delta[rng.permutation(ids)]
    calibration = dict(alpha=alpha, strength_mean=float(delta.mean()))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(Path(OUT) / f'energy_teacher_{split}_seed{seed}_alpha{alpha}.npz',
                        global_indices=global_idx, is_labeled=np.arange(len(global_idx)) < num_L,
                        energy_score=energy_score, proto_sim=proto_sim, q_E=q_E, q_D=q_D, q_fused=q_fused,
                        delta=delta, delta_shuffled=shuffled)
    return delta, shuffled, calibration


# ----------------------------------------------------------------------------- K/R state
def refresh_kr_state(enc, cls_head, XA, n_lab, yl_idx_t, seed, step, split, gen):
    """Re-estimate K, R, the VMD budget and the relation cache with the CURRENT encoder."""
    FA = KR.encode(enc, XA)
    sc = KR.fit_scorer(FA[:n_lab], yl_idx_t, cls_head, alpha=ALPHA, kind=_args.k_score_train or _args.k_score,
                       energy_T=_args.energy_T, gamma=_args.gate_gamma)
    weak = lambda X: vectorized_aug(X, WEAK_P)
    kr = KR.compute_kr(enc, cls_head, sc, XA, weak, n_lab)
    delta, nm = KR.budget(kr['K'], kr['R'], _args.delta_known, _args.delta_unknown, _args.delta_weak,
                          N_MODES, vary_modes=not _args.no_vary_modes)
    delta[:n_lab] = _args.delta_known            # labelled anchors: reliable known
    cache = KR.build_relation_cache(kr, n_lab, yl_idx_t, knn=_args.knn, r_thr=_args.r_thr)
    shuf_rng = np.random.default_rng(seed + 40000)
    perm = np.arange(len(delta))
    for ids in (np.arange(n_lab), np.arange(n_lab, len(delta))):
        perm[ids] = shuf_rng.permutation(ids)
    perm_t = torch.as_tensor(perm, device=XA.device)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(Path(OUT) / f'kr_teacher_{split}_seed{seed}_step{step}.npz',
                        K=kr['K'].cpu().numpy(), R=kr['R'].cpu().numpy(), delta=delta.cpu().numpy(),
                        n_modes=nm.cpu().numpy(), stab=kr['stab'].cpu().numpy(), dec=kr['dec'].cpu().numpy())
    extra = {}
    if _args.e3_sens and NUIS:
        # Chapter 3 scheme 3: energy sensitivity of every VMD mode, per sample, with the CURRENT encoder/head.
        # Modes whose amplitude change moves the classifier energy (class confidence) are protected.
        was = enc.training; enc.eval(); sens = []
        with torch.no_grad():
            gi = torch.as_tensor(GLOBAL_IDX)
            e0 = -torch.logsumexp(cls_head(KR.encode(enc, XA)), 1)
            for m in range(N_MODES):
                dm = vmd_modes_all[gi, m:m + 1].to(XA.device)
                ep = -torch.logsumexp(cls_head(KR.encode(enc, XA + 0.5 * dm)), 1)
                en = -torch.logsumexp(cls_head(KR.encode(enc, XA - 0.5 * dm)), 1)
                sens.append(0.5 * ((ep - e0).abs() + (en - e0).abs()))
        enc.train(was)
        S = torch.stack(sens, 1)                                           # (N, M)
        Sn = S / S.mean(1, keepdim=True).clamp(min=1e-8)                   # relative sensitivity per sample
        g_x = 1.0 / (1.0 + Sn)                                             # in (0, 1): sensitive modes -> small
        g_x = g_x / g_x.mean(1, keepdim=True) * float(NUIS['g'].mean())     # same average budget as the class-level g
        extra['mode_scale_x'] = torch.tensor(NUIS['sigma_rec'], device=XA.device).reshape(1, -1) * g_x
        print('[E3] mean relative energy sensitivity per mode: ' + ' '.join(f'{v:.2f}' for v in Sn.mean(0).tolist()), flush=True)
    if _args.kr2_no_sample:
        kappa = torch.full_like(kr['K'], float(KR.budget_v2(kr['K'], kr['R'], _args.kr2_kmin, _args.kr2_rho).mean()))
    else:
        kappa = KR.budget_v2(kr['K'], kr['R'], _args.kr2_kmin, _args.kr2_rho)
        kappa[:n_lab] = 1.0                       # labelled anchors are reliable known
    return dict(sc=sc, kr=kr, cache=cache, delta=delta, nmodes=nm,
                delta_shuf=delta[perm_t], nmodes_shuf=nm[perm_t], kappa=kappa, kappa_shuf=kappa[perm_t], **extra)


NUIS = {}
ENERGY = {}
GLOBAL_IDX = None


def energy_direction_u(enc, cls_head, x, modes, K_b, sc, gen):
    """v20: u for KR.aug_vmd_nuisance whose SIGN per (channel, mode) follows the energy gradient.

    dE/da_m at a = 0 equals <dE/dx, mode_m> (x' = x + sum_m (exp(a_m) - 1) mode_m), so one backward pass through the
    encoder gives the energy slope of every mode log-amplitude. Under the box bound |a_m| <= c * s_m the steepest
    change is a_m = c * s_m * sign(slope) (L-inf steepest ascent). Known-routed samples (K >= 0.5) move towards
    HIGHER energy (less known), unknown-routed samples towards LOWER energy: both towards the known/unknown boundary,
    i.e. the hardest natural-looking variation; the contrastive loss then makes the encoder invariant to it.
    Energy (high = unknown): 'dist' E = -T logsumexp_j cos(f(x), l_j)/T over the labelled windows (same as K),
                             'head' E = -logsumexp f_head(x).
    u = lam * 2 * sign + (1 - lam) * N(0,1) clipped to [-2, 2] (lam = 1: pure steepest direction)."""
    shape = modes.shape[:-1] + (1,)                                            # (B, [C,] M, 1)
    u_rand = torch.randn(shape, device=x.device, generator=gen).clamp(-2.0, 2.0)
    info = {}
    if _args.edir == 'random':
        sgn = torch.randint(0, 2, shape, device=x.device, generator=gen).float() * 2 - 1
    else:
        was = enc.training; enc.eval()
        xg = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            f = enc(xg)
            if _args.edir == 'head':
                E = -torch.logsumexp(cls_head(f), 1)
            else:
                E = -_args.energy_T * torch.logsumexp(F.normalize(f, dim=1) @ sc['FLn'].detach().T / _args.energy_T, 1)
            gx, = torch.autograd.grad(E.sum(), xg)
        enc.train(was)
        ga = (gx[:, :, None, :] * modes).sum(-1) if modes.dim() == 4 else (gx[:, 0, None, :] * modes).sum(-1)   # (B,[C,]M)
        side = torch.where(K_b >= 0.5, 1.0, -1.0).reshape(-1, *[1] * (ga.dim() - 1))
        sgn = torch.sign(side * ga)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn).unsqueeze(-1)
        info = dict(edir_known_frac=float((K_b >= 0.5).float().mean()))
    return _args.edir_lam * 2.0 * sgn + (1.0 - _args.edir_lam) * u_rand, info


def prepare_nuisance(lab, yl, seed):
    """Chapter 3 v2 statistics from LABELLED known windows only (raw VMD bank + optional record ids)."""
    rec = None
    if _args.record_key:
        if _args.record_key not in META_TR:
            raise KeyError(f'--record-key {_args.record_key} not in NPZ train metadata {list(META_TR)}')
        rec = META_TR[_args.record_key][lab]
    ml = vmd_modes_all[torch.as_tensor(lab)].numpy()
    if ml.ndim == 3:
        st = KR.nuisance_mode_stats(ml, XTR[lab], yl, rec)
    else:   # multi-channel: statistics per channel, stacked -> (C, M) mode arrays and (C,) fidelity bounds
        per = [KR.nuisance_mode_stats(ml[:, c], XTR[lab][:, c], yl, rec) for c in range(ml.shape[1])]
        st = {k: np.stack([p[k] for p in per]) for k in ('sigma_rec', 'sigma_cls', 'g')}
        st.update(tau_lsd=np.array([p['tau_lsd'] for p in per], np.float32), n_pairs=per[0]['n_pairs'])
    scale = st['sigma_rec'] * st['g'] * _args.kr2_gain
    if _args.kr2_no_mode:
        scale = np.full_like(scale, scale.mean())
    if scale.ndim == 1:
        perm = np.random.default_rng(seed + 50000).permutation(len(scale))
        scale_shuf = scale[perm]
    else:   # control: shuffle the (channel, mode) budgets jointly
        perm = np.random.default_rng(seed + 50000).permutation(scale.size)
        scale_shuf = scale.ravel()[perm].reshape(scale.shape)
    NUIS.clear()
    NUIS.update(st, mode_scale=torch.tensor(scale, device=DEVICE), mode_scale_shuf=torch.tensor(scale_shuf, device=DEVICE), perm=perm,
                disc=torch.tensor((1.0 - st['g']) * st['sigma_cls'], device=DEVICE))
    fmt = lambda a: ' | '.join(' '.join(f'{v:.2f}' for v in row) for row in np.atleast_2d(a))
    tau_s = ' '.join(f'{v:.3f}' for v in np.atleast_1d(st['tau_lsd']))
    print('[NUISANCE] sigma_rec=' + fmt(st['sigma_rec']) + ' || g=' + fmt(st['g'])
          + f" || tau_lsd={tau_s} (pairs={st['n_pairs']}) | record ids: {'yes' if rec is not None else 'no'}", flush=True)


def balanced_labels(rng, known, group_key, record_key=None):
    """Same label budget as the random protocol (LABEL_RATIO per known class), but spread evenly over the
    groups (e.g. bearings) of each KNOWN class, and - if record_key is given - over distinct records.
    Uses only known-class metadata (which bearing an annotated window came from); no unknown labels."""
    if group_key not in META_TR:
        raise KeyError(f'--label-balance-key {group_key} not in NPZ train metadata {list(META_TR)}')
    if record_key and record_key not in META_TR:
        raise KeyError(f'--label-record-key {record_key} not in NPZ train metadata {list(META_TR)}')
    G = META_TR[group_key]
    Rk = META_TR[record_key] if record_key else None
    out = []
    for c in known:
        idx_c = np.where(YTR == c)[0]
        budget = int(LABEL_RATIO * len(idx_c))
        groups = list(np.unique(G[idx_c]))
        rng.shuffle(groups)
        quota = {g: budget // len(groups) + (1 if i < budget % len(groups) else 0) for i, g in enumerate(groups)}
        for g, q in quota.items():
            pool = idx_c[G[idx_c] == g]
            if Rk is None:
                out.append(rng.choice(pool, min(q, len(pool)), replace=False))
                continue
            recs = list(np.unique(Rk[pool]))
            rng.shuffle(recs)
            picked = []
            while len(picked) < min(q, len(pool)):          # round-robin over records: distinct records first
                for r in recs:
                    left = np.setdiff1d(pool[Rk[pool] == r], picked)
                    if len(left) and len(picked) < q:
                        picked.append(int(rng.choice(left)))
            out.append(np.array(picked))
    return np.concatenate(out).astype(int)


def run(split, seed):
    known, unknown = SPLITS[split]
    n_unk_clusters = _args.n_unknown_clusters if _args.n_unknown_clusters > 0 else len(unknown)  # -1 = auto (chosen later)
    k = len(known) + n_unk_clusters
    if set(np.unique(YTR).tolist()) != set(known + unknown) or set(np.unique(YV).tolist()) != set(known + unknown):
        raise ValueError(f'Paderborn labels mismatch: train={sorted(np.unique(YTR).tolist())}, val={sorted(np.unique(YV).tolist())}, configured={sorted(known + unknown)}')
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tr = np.where(np.isin(YTR, known + unknown))[0]
    if _args.label_balance_key:
        lab = balanced_labels(rng, known, _args.label_balance_key, _args.label_record_key)
    else:
        lab = np.concatenate([rng.choice(np.where(YTR == c)[0], int(LABEL_RATIO * (YTR == c).sum()), replace=False) for c in known])
    unl = np.setdiff1d(tr, lab)
    XL, yl, XU, yu = XTR[lab], YTR[lab], XTR[unl], YTR[unl]
    XV_, yv_ = XV, YV
    res = dict(n_labeled=len(lab), n_unlabeled=len(unl), n_val=len(yv_), labeled_idx=lab.tolist(), args=vars(_args))

    spec = lambda X: np.log1p(np.abs(np.fft.rfft(X, axis=-1))).reshape(len(X), -1)
    pca = PCA(64, random_state=seed).fit(spec(XU)); km = KMeans(k, n_init=10, random_state=seed).fit(pca.transform(spec(XU)))
    res['M0_频谱+KMeans'] = dict(U=cluster_acc(yu, km.labels_, known)[0], V=cluster_acc(yv_, km.predict(pca.transform(spec(XV_))), known)[0])

    enc = make_encoder().to(DEVICE); head = nn.Linear(64, len(known)).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), 1e-3, weight_decay=1e-4)
    yl_idx = np.array([known.index(c) for c in yl])
    ylt = torch.tensor(yl_idx, dtype=torch.long, device=DEVICE); XLt = torch.tensor(to3(XL), device=DEVICE)
    sup_aug = dict(shift=256, scale=0.2, noise=0.1, mask=0.10)
    for _ in range(STEPS_SUP):
        enc.train()
        b = torch.randint(0, len(XL), (64,), device=DEVICE)
        opt.zero_grad()
        sup_loss = F.cross_entropy(head(enc(vectorized_aug(XLt[b], sup_aug))), ylt[b].long())
        sup_loss.backward()
        opt.step()

    proj = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).to(DEVICE)
    cls_head = head                       # keep the pre-trained head
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()) + list(cls_head.parameters()), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS_GCD)
    XA = torch.tensor(to3(np.concatenate([XL, XU])), device=DEVICE)
    n_lab = len(XL)

    global_idx = np.concatenate([lab, unl])
    global GLOBAL_IDX
    GLOBAL_IDX = global_idx
    batch_rng = torch.Generator(device=DEVICE).manual_seed(seed + 10000)
    vmd_rng = torch.Generator(device=DEVICE).manual_seed(seed + 20000)
    frozen_delta = frozen_shuffled = calibration = None
    kr_state = None
    p = dict(shift=256, scale=0.2, noise=0.1, mask=0.10)
    log = []
    need_kr = AUG_MODE in ('adaptive_kr', 'shuffled_kr', 'adaptive_kr2', 'shuffled_kr2') or _args.contrast in ('reliable', 'reliable2')
    aug_stats = {}
    if AUG_MODE in ('adaptive_kr2', 'shuffled_kr2'):
        prepare_nuisance(lab, yl, seed)
        res['nuisance_stats'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in NUIS.items() if k not in ('perm', 'disc', 'mode_scale', 'mode_scale_shuf')}
    rel_stats = {}

    for step in range(STEPS_GCD):
        enc.train()
        bl = torch.randint(0, n_lab, (N_LAB_BS,), device=DEVICE, generator=batch_rng)
        bu = torch.randint(n_lab, len(XA), (BS - N_LAB_BS,), device=DEVICE, generator=batch_rng)
        bpos = torch.cat([bl, bu])
        x = XA[bpos]
        idx_tensor = torch.as_tensor(global_idx[bpos.cpu().numpy()], dtype=torch.long)

        warm = step < _args.warmup
        x_aug1 = vectorized_aug(x, p)
        if warm or AUG_MODE == 'generic':
            x_aug2 = vectorized_aug(x, p)
        else:
            if AUG_MODE == 'fixed': current_delta, nmod = VMD_DELTA, None
            elif AUG_MODE == 'adaptive_energy': current_delta, nmod = frozen_delta[bpos.cpu().numpy()], None
            elif AUG_MODE == 'shuffled_energy': current_delta, nmod = frozen_shuffled[bpos.cpu().numpy()], None
            elif AUG_MODE == 'adaptive_kr': current_delta, nmod = kr_state['delta'][bpos], kr_state['nmodes'][bpos]
            elif AUG_MODE == 'shuffled_kr': current_delta, nmod = kr_state['delta_shuf'][bpos], kr_state['nmodes_shuf'][bpos]
            if AUG_MODE in ('adaptive_kr2', 'shuffled_kr2'):
                sh = AUG_MODE == 'shuffled_kr2'
                kap = (kr_state['kappa_shuf'] if sh else kr_state['kappa'])[bpos]
                msc = NUIS['mode_scale_shuf'] if sh else NUIS['mode_scale']
                if _args.e3_sens and not sh:
                    msc = kr_state['mode_scale_x'][bpos]                      # (B, M) per-sample mode budget
                efn = None
                if _args.e2_fidelity:
                    known_side = kr_state['kr']['K'][bpos] >= 0.5
                    def efn(cand, _x=x, _ks=known_side):
                        was = enc.training; enc.eval()
                        with torch.no_grad():
                            lo, lc = cls_head(enc(_x)), cls_head(enc(cand))
                        enc.train(was)
                        keep = (lc.argmax(1) == lo.argmax(1)) & (-torch.logsumexp(lc, 1) <= -torch.logsumexp(lo, 1) + ENERGY['tol'])
                        return keep | ~_ks
                modes_b = vmd_modes_all[idx_tensor].to(x.device)
                u_dir = None
                if _args.edir != 'none' and not sh:
                    u_dir, st_dir = energy_direction_u(enc, cls_head, x, modes_b, kr_state['kr']['K'][bpos], kr_state['sc'], vmd_rng)
                    aug_stats.update(st_dir)
                x_vmd, st_aug = KR.aug_vmd_nuisance(x, modes_b, kap, msc, vmd_rng,
                                                    float('inf') if _args.kr2_no_fidelity else NUIS['tau_lsd'], extra_check=efn, u=u_dir)
                aug_stats.update(st_aug)
                if (step + 1) % 300 == 0:
                    # mechanism check (no labels): mean distance-energy change produced by the augmentation, per routing side
                    was = enc.training; enc.eval()
                    with torch.no_grad():
                        Ed = lambda z: -_args.energy_T * torch.logsumexp(F.normalize(enc(z), dim=1) @ kr_state['sc']['FLn'].detach().T / _args.energy_T, 1)
                        dE = Ed(x_vmd) - Ed(x)
                    enc.train(was)
                    kb = kr_state['kr']['K'][bpos] >= 0.5
                    aug_stats['dE_known'] = float(dE[kb].mean()) if kb.any() else float('nan')
                    aug_stats['dE_unk'] = float(dE[~kb].mean()) if (~kb).any() else float('nan')
            else:
                x_vmd, _ = aug_vmd_live(x, idx_tensor, current_delta, vmd_rng, nmod)
            x_aug2 = vectorized_aug(x_vmd, p)

        f_all = enc(torch.cat([x_aug1, x_aug2]))
        z = F.normalize(proj(f_all), dim=1)
        zl = torch.cat([z[:N_LAB_BS], z[BS:BS + N_LAB_BS]])

        if _args.contrast == 'reliable' and not warm:
            loss_ssl, rel_stats = KR.relation_contrastive_loss(z[:BS], z[BS:], bpos, kr_state['cache'], TAU, REL_CFG)
        elif _args.contrast == 'reliable2' and not warm:
            loss_ssl, rel_stats = KR.relation_contrastive_loss_v2(z[:BS], z[BS:], bpos, kr_state['cache'], TAU, REL_CFG)
        else:
            loss_ssl = supcon(z, None, TAU)
        loss_sup = supcon(zl, ylt[bl], TAU)
        loss = (1 - LAM) * loss_ssl + LAM * loss_sup + 0.5 * F.cross_entropy(cls_head(f_all[:N_LAB_BS]), ylt[bl])
        if _args.e1_lam > 0 and not warm:
            # Chapter 3 scheme 1: the same VMD bank produces class-DESTROYING views of labelled windows (pseudo-unknowns);
            # an energy margin keeps labelled views at low energy and pseudo-unknowns at high energy (Liu et al. 2020 form).
            xl = x[:N_LAB_BS]; ml = vmd_modes_all[idx_tensor[:N_LAB_BS]].to(x.device)
            sgn = (torch.randint(0, 2, (N_LAB_BS, ml.shape[1], 1), device=x.device, generator=vmd_rng) * 2 - 1).float()
            mag = 2.0 + torch.rand(N_LAB_BS, ml.shape[1], 1, device=x.device, generator=vmd_rng)
            x_pu = xl + ((torch.exp(0.5 * sgn * mag * 3.0 * NUIS['disc'].reshape(1, -1, 1)) - 1.0) * ml).sum(1, keepdim=True)
            x_pu = (x_pu - x_pu.mean(-1, keepdim=True)) / x_pu.std(-1, keepdim=True).clamp(min=1e-8)
            e_in = -torch.logsumexp(cls_head(f_all[:N_LAB_BS]), 1)
            e_out = -torch.logsumexp(cls_head(enc(vectorized_aug(x_pu, p))), 1)
            loss_e = F.relu(e_in - ENERGY['m_in']).pow(2).mean() + F.relu(ENERGY['m_out'] - e_out).pow(2).mean()
            loss = loss + _args.e1_lam * loss_e
            aug_stats['E_in'] = float(e_in.detach().mean()); aug_stats['E_out'] = float(e_out.detach().mean())
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        refresh_now = (step + 1 == _args.warmup) or (_args.refresh_every > 0 and step + 1 > _args.warmup
                                                      and (step + 1 - _args.warmup) % _args.refresh_every == 0)
        if step + 1 == _args.warmup and (_args.e1_lam > 0 or _args.e2_fidelity or _args.e3_sens):
            with torch.no_grad():
                e_lab = -torch.logsumexp(cls_head(KR.encode(enc, XA[:n_lab])), 1)
            ENERGY.update(m_in=float(e_lab.median()), m_out=float(e_lab.median()) + max(2.0, 2.0 * float(e_lab.std())),
                          tol=float(e_lab.std()))
            print(f"[ENERGY] labelled energy median {ENERGY['m_in']:.3f}, margin out {ENERGY['m_out']:.3f}, tol {ENERGY['tol']:.3f}", flush=True)
        if step + 1 == _args.warmup:
            frozen_delta, frozen_shuffled, calibration = prepare_frozen_energy(enc, cls_head, XL, XU, yl, global_idx, seed, split, alpha=ALPHA)
            save_checkpoint(OUT, f'm2_{split}_seed{seed}_warmup.pt', enc, proj, cls_head,
                            dict(experiment=2, stage='shared_warmup', condition=COND, split=split, seed=seed, known=known, unknown=unknown))
        if refresh_now and need_kr:
            kr_state = refresh_kr_state(enc, cls_head, XA, n_lab, ylt, seed, step + 1, split, vmd_rng)
            # evaluation-only diagnostics of this refresh (true labels never flow back)
            ytrue_all = np.concatenate([yl, yu])
            dg = KR.kr_diag(kr_state['kr']['K'][n_lab:].cpu().numpy(), kr_state['kr']['R'][n_lab:].cpu().numpy(), yu, known, _args.r_thr)
            dg.update(KR.relation_diag(kr_state['cache'], ytrue_all)); dg['step'] = step + 1
            dg['delta_mean'] = float(kr_state['delta'].mean()); dg['nmodes_mean'] = float(kr_state['nmodes'].float().mean())
            log.append(dg)
            print(f'[KR step{step + 1}] ' + ' '.join(f'{a}={b:.3f}' if isinstance(b, float) else f'{a}={b}' for a, b in dg.items()), flush=True)
        if (step + 1) % 300 == 0:
            print(f'  step {step + 1}/{STEPS_GCD} loss={loss.item():.3f} ssl={loss_ssl.item():.3f} sup={loss_sup.item():.3f}'
                  + (' rel=' + ','.join(f'{a}:{b:.2f}' for a, b in rel_stats.items()) if rel_stats else '')
                  + (' aug=' + ','.join(f'{a}:{b:.2f}' for a, b in aug_stats.items()) if aug_stats else ''), flush=True)

    FL_raw, FU_raw = get_features(enc, XL), get_features(enc, XU)
    all_features = np.concatenate([FL_raw, FU_raw], axis=0)
    mean_f = all_features.mean(axis=0, keepdims=True)
    std_f = all_features.std(axis=0, keepdims=True) + 1e-8
    FL_norm, FU_norm = (FL_raw - mean_f) / std_f, (FU_raw - mean_f) / std_f

    with torch.no_grad():
        energy_U = -torch.logsumexp(cls_head(torch.tensor(FU_raw, device=DEVICE)), dim=1).cpu().numpy()

    init_unknown, init_info, known_centers = None, None, None
    if _args.unk_init == 'kr' or _args.known_expand > 0:
        with torch.no_grad():
            sc_f = KR.fit_scorer(KR.encode(enc, XA[:n_lab]), ylt, cls_head, alpha=ALPHA, kind=_args.k_score, energy_T=_args.energy_T, gamma=_args.gate_gamma)
            kr_f = KR.compute_kr(enc, cls_head, sc_f, XA, lambda X: vectorized_aug(X, WEAK_P), n_lab)
        K_u, R_u = kr_f['K'][n_lab:], kr_f['R'][n_lab:]
        res['final_kr_diag_train_EVAL_ONLY'] = KR.kr_diag(K_u.cpu().numpy(), R_u.cpu().numpy(), yu, known, _args.r_thr)
        print('[FINAL-K train]', {a: (round(b, 3) if isinstance(b, float) else b)
                                  for a, b in res['final_kr_diag_train_EVAL_ONLY'].items()}, flush=True)
    if _args.known_expand > 0:
        prob_u = kr_f['prob'][n_lab:]
        sel, sel_cls, exp_info = known_expand_select(kr_f['feat'], n_lab, ylt, K_u, R_u, prob_u.max(1).values,
                                                     prob_u.argmax(1), _args.known_expand, _args.r_thr,
                                                     _args.k_score, yu, known)
        exp_info['per_class_added'] = [int((sel_cls == c).sum()) for c in range(len(known))]
        print('[KNOWN-EXPAND]', exp_info, flush=True)
        res['known_expand'] = exp_info
    else:
        sel, sel_cls = np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    gsel = np.zeros(0, dtype=int)
    if _args.guard_expand > 0:
        # reliable pseudo-known windows used ONLY as extra reference points for the known guard (centres unchanged)
        prob_u = kr_f['prob'][n_lab:]
        gsel, _, ginfo = known_expand_select(kr_f['feat'], n_lab, ylt, K_u, R_u, prob_u.max(1).values,
                                             prob_u.argmax(1), _args.guard_expand, _args.r_thr, _args.k_score, yu, known)
        res['guard_expand'] = ginfo
        print('[GUARD-EXPAND]', ginfo, flush=True)
    pools = [np.concatenate([FL_norm[yl_idx == c], FU_norm[sel[sel_cls == c]]]) for c in range(len(known))]
    center_owner = None
    if _args.known_subcenters > 1:
        known_centers, center_owner, used = build_known_subcenters(pools, _args.known_subcenters, seed)
        print(f'[KNOWN-SUB] sub-centres per class={used}, pool sizes={[len(p) for p in pools]}', flush=True)
        res['known_subcenters'] = dict(per_class=used, pool_sizes=[len(p) for p in pools])
    elif _args.known_expand > 0:
        known_centers = np.array([p.mean(0) for p in pools])
    n_fixed = len(known) if known_centers is None else len(known_centers)
    remap = (lambda r: remap_clusters(r, center_owner, len(known), n_fixed)) if center_owner is not None else (lambda r: r)

    def fit_unknown(m):
        """Fit the semi-supervised k-means with m unknown centres (unlabelled data + known labels only)."""
        init_u, info = None, None
        if _args.unk_init == 'kr':
            init_u, info = kr_unknown_init(FU_norm, K_u.cpu().numpy(), R_u.cpu().numpy(), m, _args.r_thr, seed, yu, known)
        C_m, pu_raw_m = ss_kmeans(FL_norm, yl, FU_norm, len(known) + m, known, seed,
                                  select='inertia' if _args.only_fixed_center else 'silhouette',
                                  init_unknown=init_u, known_centers=known_centers, center_owner=center_owner)
        return C_m, pu_raw_m, info

    if _args.n_unknown_clusters < 0:
        # choose the number of unknown clusters by silhouette on UNLABELLED training features (no unknown labels)
        cand = {}
        for m in range(1, _args.auto_unk_max + 1):
            C_m, pu_raw_m, info_m = fit_unknown(m)
            pu_m = remap(pu_raw_m)
            sil = float(silhouette_score(FU_norm, pu_m)) if 2 <= len(np.unique(pu_m)) < len(FU_norm) else -1.0
            cand[m] = (sil, C_m, pu_raw_m, info_m)
        n_unk_clusters = max(cand, key=lambda m: cand[m][0])
        C, pu_raw, init_info = cand[n_unk_clusters][1:]
        res['auto_unknown'] = dict(chosen=n_unk_clusters, silhouette={m: v[0] for m, v in cand.items()})
        print('[AUTO-UNK]', res['auto_unknown'], flush=True)
    else:
        C, pu_raw, init_info = fit_unknown(n_unk_clusters)
    if init_info is not None:
        print('[UNK-INIT]', init_info, flush=True)
        res['unk_init'] = init_info
    pu = remap(pu_raw)                              # class-level ids: 0..n_known-1 known, >= n_known unknown
    FV_raw = get_features(enc, XV_)
    FV_norm = (FV_raw - mean_f) / std_f
    raw_V_ = remap(nearest(FV_norm, C))

    if _args.known_guard > 0:
        # Known-class guard: a window routed to an unknown cluster is sent back to the nearest known class
        # if its cosine similarity to the nearest LABELLED window is at least tau. tau = the `known_guard`
        # quantile of the leave-one-out nearest-labelled similarities of the labelled windows. Known labels only.
        nrm = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        Ln = nrm(np.concatenate([FL_norm, FU_norm[gsel]]))   # labelled (+ pseudo-known) reference set
        S_LL = nrm(FL_norm) @ Ln.T
        S_LL[np.arange(len(FL_norm)), np.arange(len(FL_norm))] = -np.inf   # leave-one-out for labelled rows
        if _args.guard_group_key:
            # v20: leave-one-GROUP-out calibration (e.g. installation / bearing): a labelled window may only be matched
            # to reference windows of OTHER groups, so tau reflects "a window of a new installation" (eval scenario)
            # instead of "another window of the same installation". Group ids of known-class windows only.
            if _args.guard_group_key not in META_TR:
                raise KeyError(f'--guard-group-key {_args.guard_group_key} not in NPZ train metadata {list(META_TR)}')
            g_ref = np.concatenate([META_TR[_args.guard_group_key][lab], META_TR[_args.guard_group_key][unl][gsel]])
            g_lab = META_TR[_args.guard_group_key][lab]
            S_LL[g_lab[:, None] == g_ref[None, :]] = -np.inf
            if not np.isfinite(S_LL.max(1)).all():
                raise RuntimeError('--guard-group-key: some labelled windows have no reference window of another group')
        tau = float(np.quantile(S_LL.max(1), _args.known_guard))
        kc = np.array([FL_norm[yl == c].mean(0) for c in known]) if known_centers is None else known_centers
        owner = np.arange(len(known)) if center_owner is None else np.asarray(center_owner)

        def guard(Fx, pred):
            sim = (nrm(Fx) @ Ln.T).max(1)
            back = (pred >= len(known)) & (sim >= tau)
            out = pred.copy()
            out[back] = owner[((Fx[back][:, None] - kc[None]) ** 2).sum(-1).argmin(1)]
            return out, int(back.sum())

        pu, n_back_u = guard(FU_norm, np.asarray(pu))
        raw_V_, n_back_v = guard(FV_norm, np.asarray(raw_V_))
        res['known_guard'] = dict(q=_args.known_guard, tau=tau, reassigned_train=n_back_u, reassigned_val=n_back_v)
        print('[KNOWN-GUARD]', res['known_guard'], flush=True)
    metrics_U, raw_U, pred_U, mp_U = cluster_acc(yu, pu, known, True)
    metrics_V, raw_V, pred_V, mp_V = cluster_acc(yv_, raw_V_, known, True)
    print(f'[VAL] {metrics_V}', flush=True)

    res['M2_GCD+半监督KMeans'] = dict(U=metrics_U, V=metrics_V)
    res['M2_GCD+半监督KMeans']['energy_mean_new'] = float(np.mean(energy_U[~np.isin(yu, known)]))
    res['M2_GCD+半监督KMeans']['energy_mean_old'] = float(np.mean(energy_U[np.isin(yu, known)]))
    # deployable open-set metrics (known labels only)
    res['openset'] = dict(U=KR.openset_metrics(yu, raw_U, known), V=KR.openset_metrics(yv_, raw_V, known))
    print('[OPEN-V]', {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res['openset']['V'].items()}, flush=True)

    # evaluation-only group diagnostics (which bearings/records are labelled, which ones raise false unknowns)
    gd = {}
    for key, vals in META_TR.items():
        gd[f'labelled_by_{key}'] = {str(g): int(n) for g, n in zip(*np.unique(vals[lab], return_counts=True))}
    raw_V_np = np.asarray(raw_V)
    for key, vals in META_V.items():
        per = {}
        for g in np.unique(vals):
            m = vals == g
            is_k = np.isin(yv_[m], known)
            pu_g = raw_V_np[m] >= len(known)
            per[str(g)] = dict(n=int(m.sum()), cls=int(np.bincount(yv_[m]).argmax()),
                               pred_unknown_frac=float(pu_g.mean()),
                               known=bool(is_k.all()))
        gd[f'val_by_{key}'] = per
    if gd:
        res['group_diag_EVAL_ONLY'] = gd
        for name, v in gd.items():
            if name.startswith('labelled_by_'):
                print(f'[GROUP] {name}: {v}', flush=True)
            else:
                print(f'[GROUP] {name} (pred-unknown fraction): ' +
                      ', '.join(f'{g}({"K" if s["known"] else "U"}{s["cls"]},n={s["n"]})={s["pred_unknown_frac"]:.2f}'
                                for g, s in v.items()), flush=True)

    # final K/R diagnostics on the validation windows (evaluation only)
    if True:  # always log final K/R diagnostics (evaluation only)
        with torch.no_grad():
            XVt = torch.tensor(to3(XV_), device=DEVICE)
            sc_final = KR.fit_scorer(KR.encode(enc, torch.cat([XA[:n_lab]])), ylt, cls_head, alpha=ALPHA, kind=_args.k_score, energy_T=_args.energy_T, gamma=_args.gate_gamma)
            kr_v = KR.compute_kr(enc, cls_head, sc_final, XVt, lambda X: vectorized_aug(X, WEAK_P), 0)
        res['kr_diag_val'] = KR.kr_diag(kr_v['K'].cpu().numpy(), kr_v['R'].cpu().numpy(), yv_, known, _args.r_thr)
    res['kr_log'] = log

    save_checkpoint(OUT, f'm2_{split}_seed{seed}.pt', enc, proj, cls_head,
                    dict(experiment=2, split=split, seed=seed, known=known, unknown=unknown, steps=STEPS_GCD, label_ratio=LABEL_RATIO))
    np.savez_compressed(f'{OUT}/pred_{split}_seed{seed}.npz',
                        yv=yv_, raw_V=raw_V, pred_V=pred_V, yu=yu, raw_U=raw_U, pred_U=pred_U,
                        mp_V=np.array([mp_V], dtype=object), mp_U=np.array([mp_U], dtype=object))
    if seed == 0: np.savez_compressed(f'{OUT}/feat_gcd_{split}_seed0.npz', FU=FU_norm, yu=yu, pu=pu, FV=FV_norm, yv=yv_)
    return res


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f'{OUT}/results_exp2.json'):
        raise FileExistsError('Results already exist for this tag; choose a new --tag.')
    allres = {}
    for split in _args.splits.split(','):
        for seed in SEEDS:
            key = f'{split}_seed{seed}'
            if key in allres: continue
            t = time.time(); allres[key] = run(split, seed)
            json.dump(allres, open(f'{OUT}/results_exp2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1,
                      default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
            summary = {m: {'U': [round(v['U'][q], 3) for q in ('all', 'old', 'new')], 'V': [round(v['V'][q], 3) for q in ('all', 'old', 'new')], 'NMI_V': round(v['V']['nmi'], 3)}
                       for m, v in allres[key].items() if m.startswith('M')}
            print(key, f'{time.time() - t:.0f}s', summary, flush=True)
