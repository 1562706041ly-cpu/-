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
_parser.add_argument('--vmd-bank', default=None)
_parser.add_argument('--mode', choices=['generic', 'fixed', 'adaptive_energy', 'shuffled_energy',
                                        'adaptive_kr', 'shuffled_kr'], required=True)
_parser.add_argument('--contrast', choices=['plain', 'reliable'], default='plain')
_parser.add_argument('--delta', type=float, default=0.1)
_parser.add_argument('--warmup', type=int, default=300)
_parser.add_argument('--delta-known', type=float, default=0.05)
_parser.add_argument('--delta-unknown', type=float, default=0.2)
_parser.add_argument('--delta-weak', type=float, default=0.02, help='strength for unreliable samples (adaptive_kr)')
_parser.add_argument('--no-vary-modes', action='store_true', help='adaptive_kr: always perturb all modes')
_parser.add_argument('--alpha', type=float, default=0.5)
_parser.add_argument('--r-thr', type=float, default=0.5, help='R threshold for the evidence zones (diagnostics + relations)')
_parser.add_argument('--k-score', choices=['fused', 'proto', 'knn'], default='fused',
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
_parser.add_argument('--n-unknown-clusters', type=int, default=0, help='0 = len(--unknown)')
_parser.add_argument('--unk-init', choices=['random', 'kr'], default='random',
                     help="init of the unknown k-means centres: random unlabeled samples (old) or the K/R 'reliable unknown' zone")
_parser.add_argument('--seeds', default='0')
_parser.add_argument('--device', default='cuda')
_parser.add_argument('--steps', type=int, default=1500)
_parser.add_argument('--sup-steps', type=int, default=600)
_parser.add_argument('--tag', required=True)
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
REL_CFG = dict(use_label=not _args.rel_no_label, use_nbr=not _args.rel_no_nbr, use_cons=not _args.rel_no_cons,
               w_unc=_args.w_unc, fn_frac=_args.fn_frac, hard_beta=_args.hard_beta, hn_cap=_args.hn_cap,
               pos_thr=_args.pos_thr)
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
if XTR_raw.ndim != 2 or XV_raw.ndim != 2 or len(XTR_raw) != len(YTR) or len(XV_raw) != len(YV) or XTR_raw.shape[1] != XV_raw.shape[1] or len(XV_raw) == 0:
    raise ValueError('NPZ signal/label shapes mismatch; expected X=(N,L), y=(N,)')
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
zn = lambda X: ((X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-8)).astype(np.float32)
XTR, XV = zn(XTR_raw), zn(XV_raw)

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
print(f'[data] train={XTR.shape}, val={XV.shape}, labels={sorted(np.unique(YTR).tolist())}; VMD={tuple(vmd_modes_all.shape)}', flush=True)


# ----------------------------------------------------------------------------- model
def block(i, o, k, s=1): return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(block(1, 16, 64, 8), block(16, 32, 3), block(32, 64, 3), block(64, 64, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten())
    def forward(s, x): return s.f(x)


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
        x = torch.as_tensor(X[i:i + 256], dtype=torch.float32, device=dev)[:, None]
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
    if kind == 'knn':
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
    sc = KR.fit_scorer(FA[:n_lab], yl_idx_t, cls_head, alpha=ALPHA, kind=_args.k_score)
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
    return dict(sc=sc, kr=kr, cache=cache, delta=delta, nmodes=nm,
                delta_shuf=delta[perm_t], nmodes_shuf=nm[perm_t])


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
    n_unk_clusters = _args.n_unknown_clusters or len(unknown)
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

    spec = lambda X: np.log1p(np.abs(np.fft.rfft(X, axis=1)))
    pca = PCA(64, random_state=seed).fit(spec(XU)); km = KMeans(k, n_init=10, random_state=seed).fit(pca.transform(spec(XU)))
    res['M0_频谱+KMeans'] = dict(U=cluster_acc(yu, km.labels_, known)[0], V=cluster_acc(yv_, km.predict(pca.transform(spec(XV_))), known)[0])

    enc = Enc().to(DEVICE); head = nn.Linear(64, len(known)).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), 1e-3, weight_decay=1e-4)
    yl_idx = np.array([known.index(c) for c in yl])
    ylt = torch.tensor(yl_idx, dtype=torch.long, device=DEVICE); XLt = torch.tensor(XL, device=DEVICE)[:, None]
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
    XA = torch.tensor(np.concatenate([XL, XU]), device=DEVICE)[:, None]
    n_lab = len(XL)

    global_idx = np.concatenate([lab, unl])
    batch_rng = torch.Generator(device=DEVICE).manual_seed(seed + 10000)
    vmd_rng = torch.Generator(device=DEVICE).manual_seed(seed + 20000)
    frozen_delta = frozen_shuffled = calibration = None
    kr_state = None
    p = dict(shift=256, scale=0.2, noise=0.1, mask=0.10)
    log = []
    need_kr = AUG_MODE in ('adaptive_kr', 'shuffled_kr') or _args.contrast == 'reliable'
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
            else: current_delta, nmod = kr_state['delta_shuf'][bpos], kr_state['nmodes_shuf'][bpos]
            x_vmd, _ = aug_vmd_live(x, idx_tensor, current_delta, vmd_rng, nmod)
            x_aug2 = vectorized_aug(x_vmd, p)

        f_all = enc(torch.cat([x_aug1, x_aug2]))
        z = F.normalize(proj(f_all), dim=1)
        zl = torch.cat([z[:N_LAB_BS], z[BS:BS + N_LAB_BS]])

        if _args.contrast == 'reliable' and not warm:
            loss_ssl, rel_stats = KR.relation_contrastive_loss(z[:BS], z[BS:], bpos, kr_state['cache'], TAU, REL_CFG)
        else:
            loss_ssl = supcon(z, None, TAU)
        loss_sup = supcon(zl, ylt[bl], TAU)
        loss = (1 - LAM) * loss_ssl + LAM * loss_sup + 0.5 * F.cross_entropy(cls_head(f_all[:N_LAB_BS]), ylt[bl])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        refresh_now = (step + 1 == _args.warmup) or (_args.refresh_every > 0 and step + 1 > _args.warmup
                                                      and (step + 1 - _args.warmup) % _args.refresh_every == 0)
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
                  + (' rel=' + ','.join(f'{a}:{b:.2f}' for a, b in rel_stats.items()) if rel_stats else ''), flush=True)

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
            sc_f = KR.fit_scorer(KR.encode(enc, XA[:n_lab]), ylt, cls_head, alpha=ALPHA, kind=_args.k_score)
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
    pools = [np.concatenate([FL_norm[yl_idx == c], FU_norm[sel[sel_cls == c]]]) for c in range(len(known))]
    center_owner = None
    if _args.known_subcenters > 1:
        known_centers, center_owner, used = build_known_subcenters(pools, _args.known_subcenters, seed)
        print(f'[KNOWN-SUB] sub-centres per class={used}, pool sizes={[len(p) for p in pools]}', flush=True)
        res['known_subcenters'] = dict(per_class=used, pool_sizes=[len(p) for p in pools])
    elif _args.known_expand > 0:
        known_centers = np.array([p.mean(0) for p in pools])
    n_fixed = len(known) if known_centers is None else len(known_centers)
    if _args.unk_init == 'kr':
        init_unknown, init_info = kr_unknown_init(FU_norm, K_u.cpu().numpy(), R_u.cpu().numpy(),
                                                  n_unk_clusters, _args.r_thr, seed, yu, known)
        print('[UNK-INIT]', init_info, flush=True)
        res['unk_init'] = init_info
    C, pu_raw = ss_kmeans(FL_norm, yl, FU_norm, k, known, seed, select='inertia' if _args.only_fixed_center else 'silhouette',
                          init_unknown=init_unknown, known_centers=known_centers, center_owner=center_owner)
    remap = (lambda r: remap_clusters(r, center_owner, len(known), n_fixed)) if center_owner is not None else (lambda r: r)
    pu = remap(pu_raw)                              # class-level ids: 0..n_known-1 known, >= n_known unknown
    metrics_U, raw_U, pred_U, mp_U = cluster_acc(yu, pu, known, True)
    FV_raw = get_features(enc, XV_)
    FV_norm = (FV_raw - mean_f) / std_f
    raw_V_ = remap(nearest(FV_norm, C))
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
            XVt = torch.tensor(XV_, device=DEVICE)[:, None]
            sc_final = KR.fit_scorer(KR.encode(enc, torch.cat([XA[:n_lab]])), ylt, cls_head, alpha=ALPHA, kind=_args.k_score)
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
