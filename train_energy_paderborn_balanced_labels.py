"""Paderborn adaptation of uploaded training code.

Bearing-ID grouped pilot: train K002/KA03/KI03; test K001/KA01/KI01.
Labels 0=healthy, 1=outer race, 2=inner race. Known=0,1; unknown=2.
Inner-race training examples remain unlabeled in model fitting. The reported ACC is
post-hoc clustering ACC and not an online class-recognition accuracy.
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
import argparse, os, hashlib, platform, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score
# gpu_runtime import is deferred until after the CLI device is exported

_parser = argparse.ArgumentParser()
_parser.add_argument('--condition', type=int, default=1, help='legacy metadata field; not used for Paderborn')
_parser.add_argument('--data', default=None, help='default: ./data/paderborn_gcd_grouped.npz')
_parser.add_argument('--vmd-bank', default=None, help='default: ./data/paderborn_vmd_modes.npy')
_parser.add_argument('--mode', choices=['generic','fixed','adaptive_energy','shuffled_energy'], required=True)
_parser.add_argument('--delta', type=float, default=0.1)
_parser.add_argument('--warmup', type=int, default=300)
_parser.add_argument('--delta-known', type=float, default=0.05)
_parser.add_argument('--delta-unknown', type=float, default=0.2)
_parser.add_argument('--alpha', type=float, default=0.5)
_parser.add_argument('--balanced-labels', action='store_true', help='select 25 known labels/class balanced by bearing and condition')
_parser.add_argument('--splits', default='PADERBORN')
_parser.add_argument('--known', default='0,1', help='0=healthy, 1=outer-race')
_parser.add_argument('--unknown', default='2', help='2=inner-race')
_parser.add_argument('--seeds', default='0')
_parser.add_argument('--device', default='cuda')
_parser.add_argument('--steps', type=int, default=1500)
_parser.add_argument('--sup-steps', type=int, default=600)
_parser.add_argument('--tag', required=True)
_parser.add_argument('--use_optimized_kmeans', action='store_true', help='是否使用固定中心+轮廓系数优化')
_parser.add_argument('--only_fixed_center', action='store_true', help='仅固定已知中心，不开启轮廓系数选优')
_args = _parser.parse_args()

_run_tag = _args.tag
_out = PROJECT_ROOT/'runs_paderborn'/_run_tag
os.environ.update(COND=str(_args.condition), OUT=str(_out), SEEDS=_args.seeds, SPLITS=_args.splits,
                  PUMP_DEVICE=_args.device, LOCAL_STEPS_SUP=str(_args.sup_steps), LOCAL_STEPS_GCD=str(_args.steps))

# gpu_runtime reads PUMP_DEVICE at import time, so import only after setting it.
from gpu_runtime import DEVICE, FAST_AUG, feats_gpu, vectorized_aug, save_checkpoint

COND = int(os.environ.get('COND', '1'))
OUT = os.environ.get('OUT', str(PROJECT_ROOT / f'runs_gpu/gpu_v1/condition_{COND}/exp2'))
_known = tuple(int(v) for v in _args.known.split(',') if v.strip())
_unknown = tuple(int(v) for v in _args.unknown.split(',') if v.strip())
if not _known or not _unknown or set(_known) & set(_unknown):
    raise ValueError('known/unknown must be non-empty and disjoint')
SPLITS = {'PADERBORN': (_known, _unknown)}
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
LABEL_RATIO, STEPS_SUP, STEPS_GCD, BS, N_LAB_BS, LAM, TAU = 0.1, 600, 1500, 128, 32, 0.35, 0.1
STEPS_SUP = int(os.environ.get('LOCAL_STEPS_SUP', STEPS_SUP))
STEPS_GCD = int(os.environ.get('LOCAL_STEPS_GCD', STEPS_GCD))

VMD_DELTA = _args.delta
AUG_MODE = _args.mode
ALPHA = _args.alpha
if _args.warmup < 1 or _args.warmup >= _args.steps:
    raise ValueError('--warmup must be >=1 and < --steps')

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
zn = lambda X: ((X-X.mean(1,keepdims=True))/np.maximum(X.std(1,keepdims=True),1e-8)).astype(np.float32)
XTR, XV = zn(XTR_raw), zn(XV_raw)

cache_path = Path(_args.vmd_bank) if _args.vmd_bank else PROJECT_ROOT / 'data' / 'paderborn_filegroup_v2_vmd.npy'
manifest_path = cache_path.with_suffix('.json')
if not cache_path.exists() or not manifest_path.exists():
    raise FileNotFoundError(f'缺少Paderborn VMD缓存或清单: {cache_path}, {manifest_path}')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
import hashlib
train_digest = hashlib.sha256(np.ascontiguousarray(XTR_raw).view(np.uint8)).hexdigest()
# Verify training data and VMD cache integrity.
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
print(f'[data] train={XTR.shape}, val={XV.shape}, labels={sorted(np.unique(YTR).tolist())}; VMD={tuple(vmd_modes_all.shape)}', flush=True)

def block(i, o, k, s=1): return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(block(1, 16, 64, 8), block(16, 32, 3), block(32, 64, 3), block(64, 64, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten())
    def forward(s, x): return s.f(x)

def aug_vmd_live(x, idx, delta, generator):
    modes=vmd_modes_all[idx].to(x.device)
    B,K,L=modes.shape
    delta=torch.as_tensor(delta,dtype=x.dtype,device=x.device).reshape(-1,1,1)
    direction=torch.rand(B,K,1,device=x.device,generator=generator)*2-1
    x_aug=x+(direction*delta*modes).sum(dim=1,keepdim=True)
    cos=lambda z:F.cosine_similarity(x.flatten(1),z.flatten(1),dim=1)
    shrunk=torch.zeros(B,dtype=torch.bool,device=x.device)
    for _ in range(3):
        mask=cos(x_aug)<0.85
        if not mask.any(): break
        shrunk |= mask
        x_aug[mask]=0.5*x[mask]+0.5*x_aug[mask]
    fallback=(cos(x_aug)<0.85) | (~torch.isfinite(x_aug).flatten(1).all(1))
    if fallback.any(): x_aug[fallback]=x[fallback]
    return x_aug, dict(vmd_applied=True)

@torch.no_grad()
def get_features(enc, X):
    enc.eval()
    dev = next(enc.parameters()).device
    chunks = []
    for i in range(0, len(X), 256):
        x = torch.as_tensor(X[i:i+256], dtype=torch.float32, device=dev)[:, None]
        chunks.append(enc(x).cpu())
    return torch.cat(chunks).numpy()

# === 修改：让 cluster_acc 返回详细数组供核对 ===
def cluster_acc(y, p, known, fixed_known_centers=False):
    y = np.asarray(y).reshape(-1)
    raw_cluster_id = np.asarray(p).reshape(-1)
    known = [int(x) for x in known]

    cls = np.unique(y)
    cl = np.unique(raw_cluster_id)
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

    metrics = {
        "all": float(ok.mean()),
        "old": float(ok[old].mean()) if old.any() else float("nan"),
        "new": float(ok[~old].mean()) if (~old).any() else float("nan"),
        "nmi": float(nmi(y, raw_cluster_id)),
    }
    return metrics, raw_cluster_id, pred_label, mp

def ss_kmeans_optimized(FL, yl, FU, k, known, seed, iters=100):
    rng = np.random.default_rng(seed)
    best_sil = -1
    best_centers = None
    best_pu = None
    n_known = len(known)
    
    for restart in range(10):
        known_centers = np.array([FL[yl == c].mean(0) for c in known])
        unknown_centers = [FU[rng.choice(len(FU))] for _ in range(k - n_known)]
        C = np.vstack([known_centers, np.array(unknown_centers)])
        
        for _ in range(iters):
            pu = ((FU[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C_known = np.array([FL[yl == c].mean(0) for c in known])
            C_unknown = []
            for j in range(k - n_known):
                cluster_id = n_known + j
                mask = pu == cluster_id
                if mask.any(): C_unknown.append(FU[mask].mean(0))
                else: C_unknown.append(C[cluster_id])
            C2 = np.vstack([C_known, np.stack(C_unknown)])
            if np.allclose(C2, C): break
            C = C2
            
        pu = ((FU[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        n_cluster = len(np.unique(pu))
        if 2 <= n_cluster < len(FU):
            sil = silhouette_score(FU, pu)
            if sil > best_sil:
                best_sil = sil
                best_centers = C.copy()
                best_pu = pu.copy()
                
    return best_centers, best_pu

def ss_kmeans_fixed_center_only(FL, yl, FU, k, known, seed, iters=100):
    rng = np.random.default_rng(seed)
    n_known = len(known)
    best_inertia = float('inf')
    best_centers = None
    best_pu = None
    
    for restart in range(10):
        known_centers = np.array([FL[yl == c].mean(0) for c in known])
        unknown_centers = [FU[rng.choice(len(FU))] for _ in range(k - n_known)]
        C = np.vstack([known_centers, np.array(unknown_centers)])
        
        for _ in range(iters):
            pu = ((FU[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C_known = np.array([FL[yl == c].mean(0) for c in known])
            C_unknown = []
            for j in range(k - n_known):
                cluster_id = n_known + j
                mask = pu == cluster_id
                if mask.any(): C_unknown.append(FU[mask].mean(0))
                else: C_unknown.append(C[cluster_id])
            C2 = np.vstack([C_known, np.stack(C_unknown)])
            if np.allclose(C2, C): break
            C = C2
            
        pu = ((FU[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        
        inertia = sum(((FU[pu == j] - C[j]) ** 2).sum() for j in range(k) if (pu == j).any())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = C.copy()
            best_pu = pu.copy()
            
    return best_centers, best_pu

nearest = lambda Fx, C: ((Fx[:, None] - C[None]) ** 2).sum(-1).argmin(1)

def supcon(z, y, tau):
    n = z.shape[0] // 2; sim = z @ z.T / tau; sim.fill_diagonal_(-1e9)
    ids = torch.arange(n, device=z.device).repeat(2) if y is None else y.repeat(2)
    pos = (ids[:, None] == ids[None]).float(); pos.fill_diagonal_(0)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -((pos * logp).sum(1) / pos.sum(1).clamp(min=1)).mean()

def prepare_frozen_energy(enc, cls_head, XL, XU, yl_np, global_idx, seed, split, alpha=0.5):
    features = get_features(enc, np.concatenate([XL, XU]))
    features_t = torch.tensor(features, device=DEVICE)
    
    energy_score = torch.logsumexp(cls_head(features_t), dim=1).detach().cpu().numpy()
    
    num_L = len(XL)
    features_L = features_t[:num_L]
    known_classes = np.unique(yl_np)
    
    prototypes = []
    for c in known_classes:
        mask = (yl_np == c)
        if mask.sum() > 0: prototypes.append(features_L[mask].mean(dim=0))
    prototypes = F.normalize(torch.stack(prototypes), dim=1)
    
    features_norm = F.normalize(features_t, dim=1)
    proto_sim = torch.max(torch.matmul(features_norm, prototypes.T), dim=1)[0].detach().cpu().numpy()

    def calibrate_score(scores):
        labeled_scores = scores[:num_L]
        low = float(np.quantile(labeled_scores, 0.05))
        mid = float(np.median(labeled_scores))
        scale = max((mid - low) / np.log(19.0), 1e-6)
        return torch.sigmoid(torch.as_tensor((scores - low) / scale)).numpy()

    q_E = calibrate_score(energy_score)
    q_D = calibrate_score(proto_sim)
    
    q_fused = alpha * q_E + (1 - alpha) * q_D
    delta = (_args.delta_known + (_args.delta_unknown - _args.delta_known) * (1 - q_fused)).astype(np.float32)
    
    shuffled = delta.copy()
    rng = np.random.default_rng(seed + 30000)
    for ids in (np.arange(num_L), np.arange(num_L, len(delta))):
        shuffled[ids] = delta[rng.permutation(ids)]
        
    calibration = dict(alpha=alpha, strength_mean=float(delta.mean()))
    os.makedirs(OUT, exist_ok=True)
    
    np.savez_compressed(Path(OUT)/f'energy_teacher_{split}_seed{seed}_alpha{alpha}.npz',
                        global_indices=global_idx, is_labeled=np.arange(len(global_idx))<num_L,
                        energy_score=energy_score, proto_sim=proto_sim, q_E=q_E, q_D=q_D, q_fused=q_fused,
                        delta=delta, delta_shuffled=shuffled)
    
    return delta, shuffled, calibration

def select_balanced_known_labels(y, bearing, condition, source, known, seed):
    """Select 25 known-label windows/class, balanced across training strata.

    This is an oracle-simulation of a class-aware annotation budget, as in the
    original script. No unlabeled ground-truth is used for model fitting.
    """
    expected_bearings = {0: {'K002', 'K004'}, 1: {'KA03', 'KA06'}}
    expected_conditions = {'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10'}
    if tuple(known) != (0, 1):
        raise ValueError('Balanced protocol requires known labels (0,1)')
    if any(a.ndim != 1 or len(a) != len(y) for a in (y, bearing, condition, source)):
        raise ValueError('Train metadata shape mismatch')
    sampler = np.random.default_rng(seed + 50000)
    result = []
    for label in known:
        eligible = np.flatnonzero(y == label)
        if set(bearing[eligible].tolist()) != expected_bearings[label]:
            raise ValueError(f'Unexpected training bearing IDs for known class {label}')
        strata = [(bid, cond) for bid in sorted(expected_bearings[label])
                  for cond in sorted(expected_conditions)]
        # 8 x 3 + 1 = 25 labeled windows for each known class.
        extra = int(sampler.integers(len(strata)))
        for i, (bid, cond) in enumerate(strata):
            candidates = eligible[(bearing[eligible] == bid) & (condition[eligible] == cond)]
            recordings = np.unique(source[candidates])
            count = 3 + int(i == extra)
            if len(recordings) < count:
                raise ValueError(f'Not enough distinct recordings for {label}/{bid}/{cond}')
            picked_sources = sampler.choice(recordings, size=count, replace=False)
            for rec in picked_sources:
                within = candidates[source[candidates] == rec]
                result.append(int(sampler.choice(within)))
    lab = np.asarray(result, dtype=np.int64)
    if len(lab) != 50 or len(np.unique(lab)) != 50:
        raise RuntimeError('Expected exactly 50 unique labeled windows')
    if len(np.unique(source[lab])) != 50:
        raise RuntimeError('Expected one labeled window per distinct recording')
    return lab


def run(split, seed):
    known, unknown = SPLITS[split]; k = len(known) + len(unknown)
    if set(np.unique(YTR).tolist()) != set(known + unknown) or set(np.unique(YV).tolist()) != set(known + unknown):
        raise ValueError(f'Paderborn labels mismatch: train={sorted(np.unique(YTR).tolist())}, val={sorted(np.unique(YV).tolist())}, configured={sorted(known+unknown)}')
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tr = np.where(np.isin(YTR, known + unknown))[0]
    lab = np.concatenate([rng.choice(np.where(YTR == c)[0], int(LABEL_RATIO * (YTR == c).sum()), replace=False) for c in known])
    if _args.balanced_labels:
        # Preserve the original RNG draws above; use a separate seeded sampler.
        lab = select_balanced_known_labels(
            YTR, np.asarray(d['train_bearing_id']), np.asarray(d['train_condition']),
            np.asarray(d['train_source_file']), known, seed)
        print(f'[LABELS] balanced=True seed={seed} labeled={len(lab)}', flush=True)

    unl = np.setdiff1d(tr, lab)
    XL, yl, XU, yu = XTR[lab], YTR[lab], XTR[unl], YTR[unl]
    XV_, yv_ = XV, YV
    res = dict(n_labeled=len(lab), n_unlabeled=len(unl), n_val=len(yv_))
    if _args.balanced_labels:
        res['label_sampling'] = 'balanced_known_25_per_class_one_per_recording'
    
    spec = lambda X: np.log1p(np.abs(np.fft.rfft(X, axis=1)))
    pca = PCA(64, random_state=seed).fit(spec(XU)); km = KMeans(k, n_init=10, random_state=seed).fit(pca.transform(spec(XU)))
    res['M0_频谱+KMeans'] = dict(U=cluster_acc(yu, km.labels_, known)[0], V=cluster_acc(yv_, km.predict(pca.transform(spec(XV_))), known)[0])
    
    enc = Enc().to(DEVICE); head = nn.Linear(64, len(known)).to(DEVICE); opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), 1e-3, weight_decay=1e-4)
    ylt = torch.tensor([known.index(c) for c in yl], device=DEVICE); XLt = torch.tensor(XL, device=DEVICE)[:, None]
    for _ in range(STEPS_SUP):
        enc.train(); b = torch.randint(0, len(XL), (64,), device=DEVICE); opt.zero_grad(); F.cross_entropy(head(enc(vectorized_aug(XLt[b], dict(shift=256, scale=0.2, noise=0.1, mask=0.10)))), ylt[b]).backward(); opt.step()
    
    enc = enc  # 保留之前预训练好的 Enc
    proj = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).to(DEVICE)
    cls_head = head  # 关键修复：保留预训练好的分类头，避免被重新初始化
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()) + list(cls_head.parameters()), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS_GCD); XA = torch.tensor(np.concatenate([XL, XU]), device=DEVICE)[:, None]
    
    global_idx = np.concatenate([lab, unl])
    batch_rng=torch.Generator(device=DEVICE).manual_seed(seed+10000)
    vmd_rng=torch.Generator(device=DEVICE).manual_seed(seed+20000)
    frozen_delta=frozen_shuffled=calibration=None
    p=dict(shift=256,scale=0.2,noise=0.1,mask=0.10)
    
    for step in range(STEPS_GCD):
        enc.train(); bl = torch.randint(0, len(XL), (N_LAB_BS,), device=DEVICE, generator=batch_rng); bu = torch.randint(len(XL), len(XA), (BS - N_LAB_BS,), device=DEVICE, generator=batch_rng)
        x = torch.cat([XA[bl], XA[bu]])
        
        batch_positions = torch.cat([bl,bu]).cpu().numpy()
        idx_batch = global_idx[batch_positions]
        idx_tensor = torch.tensor(idx_batch, dtype=torch.long)
        
        phase='warmup' if step<_args.warmup else AUG_MODE
        x_aug1=vectorized_aug(x,p)
        if phase in ('warmup','generic'):
            x_aug2=vectorized_aug(x,p)
        else:
            if frozen_delta is None: raise RuntimeError('Missing warmup score cache')
            if AUG_MODE=='fixed': current_delta=VMD_DELTA
            elif AUG_MODE=='adaptive_energy': current_delta=frozen_delta[batch_positions]
            else: current_delta=frozen_shuffled[batch_positions]
            x_vmd,stats=aug_vmd_live(x,idx_tensor,current_delta,vmd_rng)
            x_aug2=vectorized_aug(x_vmd,p)
        
        f_all = enc(torch.cat([x_aug1, x_aug2]))
        z = F.normalize(proj(f_all), dim=1)
        zl = torch.cat([z[:N_LAB_BS], z[BS:BS + N_LAB_BS]])
        
        loss_ssl=supcon(z,None,TAU); loss_sup=supcon(zl,ylt[bl],TAU)
        loss_con = (1 - LAM) * loss_ssl + LAM * loss_sup
        loss_ce = F.cross_entropy(cls_head(f_all[:N_LAB_BS]), ylt[bl])
        loss = loss_con + 0.5 * loss_ce
        
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        
        if step+1==_args.warmup:
            frozen_delta,frozen_shuffled,calibration=prepare_frozen_energy(enc,cls_head,XL,XU,yl,global_idx,seed,split,alpha=ALPHA)
            save_checkpoint(OUT,f'm2_{split}_seed{seed}_warmup.pt',enc,proj,cls_head,dict(experiment=2,stage='shared_warmup',condition=COND,split=split,seed=seed,known=known,unknown=unknown))
        
    FL_raw, FU_raw = get_features(enc, XL), get_features(enc, XU)
    
    # === 修复 1：改用 Z-score 维度标准化，防止特征被压扁 ===
    all_features = np.concatenate([FL_raw, FU_raw], axis=0)
    mean_f = all_features.mean(axis=0, keepdims=True)
    std_f = all_features.std(axis=0, keepdims=True) + 1e-8
    FL_norm = (FL_raw - mean_f) / std_f
    FU_norm = (FU_raw - mean_f) / std_f
    # ====================================================
    
    with torch.no_grad():
        logits_U = cls_head(torch.tensor(FU_raw, device=DEVICE))
        energy_U = -torch.logsumexp(logits_U, dim=1).cpu().numpy()
        
    if _args.only_fixed_center:
        C, pu = ss_kmeans_fixed_center_only(FL_norm, yl, FU_norm, k, known, seed)
        is_fixed = True
    else:
        # 默认强制使用高级优化 KMeans
        C, pu = ss_kmeans_optimized(FL_norm, yl, FU_norm, k, known, seed)
        is_fixed = True
    # 原有的 else 分支已废弃
        # === 修复 2：强制默认走高级优化 KMeans，不再走 basic ===
        C, pu = ss_kmeans_optimized(FL_norm, yl, FU_norm, k, known, seed)
        is_fixed = True
        
    # === 修改：把返回的数组接住 ===
    metrics_U, raw_U, pred_U, mp_U = cluster_acc(yu, pu, known, is_fixed)
    FV_raw = get_features(enc, XV_)
    FV_norm = (FV_raw - mean_f) / std_f
    metrics_V, raw_V, pred_V, mp_V = cluster_acc(yv_, nearest(FV_norm, C), known, is_fixed)
    print(f'[VAL] {metrics_V}', flush=True)
    
    res['M2_GCD+半监督KMeans'] = dict(U=metrics_U, V=metrics_V)
    res['M2_GCD+半监督KMeans']['energy_mean_new'] = float(np.mean(energy_U[~np.isin(yu, known)]))
    res['M2_GCD+半监督KMeans']['energy_mean_old'] = float(np.mean(energy_U[np.isin(yu, known)]))
    
    save_checkpoint(OUT, f'm2_{split}_seed{seed}.pt', enc, proj, cls_head, dict(experiment=2, split=split, seed=seed, known=known, unknown=unknown, steps=STEPS_GCD, label_ratio=LABEL_RATIO))
    
    # === 新增：保存真实预测数组和映射，用于后期全量核验 ===
    np.savez_compressed(f'{OUT}/pred_{split}_seed{seed}.npz',
                        yv=yv_, raw_V=raw_V, pred_V=pred_V, yu=yu, raw_U=raw_U, pred_U=pred_U,
                        mp_V=np.array([mp_V], dtype=object), mp_U=np.array([mp_U], dtype=object))
    
    if seed == 0: np.savez_compressed(f'{OUT}/feat_gcd_{split}_seed0.npz', FU=FU_norm, yu=yu, pu=pu, FV=FV_norm, yv=yv_)
    return res

if __name__ == '__main__':
    if os.path.exists(f'{OUT}/results_exp2.json'):
        raise FileExistsError('Results already exist for this tag; choose a new --tag.')
    allres = {}
    for split in _args.splits.split(','):
        for seed in SEEDS:
            key = f'{split}_seed{seed}'
            if key in allres: continue
            t = time.time(); allres[key] = run(split, seed)
            json.dump(allres, open(f'{OUT}/results_exp2.json', 'w',encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
            summary = {m: {'U': [round(v['U'][q], 3) for q in ('all','old','new')], 'V': [round(v['V'][q], 3) for q in ('all','old','new')], 'NMI_V': round(v['V']['nmi'], 3)} for m, v in allres[key].items() if m.startswith('M')}
            print(key, f'{time.time()-t:.0f}s', summary, flush=True)
