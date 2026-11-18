"""v27 mechanism check: independent vs cross-channel coordinated band-gain perturbation (criteria: results_v24/v27_crosschannel_PREREGISTERED.txt).
Run from scratch/v24p with PYTHONPATH=. Resumable: results are appended to v27_models.jsonl and skipped on restart."""
import json, os, sys, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import kr_modules as KR
from gpu_runtime import vectorized_aug
torch.set_num_threads(4)
SPL = '/projects/sandbox/centrifugal/splits_ch1245/motor2_100_ch1245_%s_fold%s.npz'
OUTJ = '/projects/sandbox/results_v24/v27_models.jsonl'
WEAK = dict(shift=64, scale=0.05, noise=0.02, mask=0.0); STD = dict(shift=256, scale=0.2, noise=0.1, mask=0.10)
GAINS = [1.0, 2.0, 4.0]; PRIMARY = 2.0; NB = 32; NREP = 3; NPOS = NNEG = 100000
PAIRS = list(itertools.combinations(range(4), 2))

def block(i, o, k, s=1): return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
def log_spectrum(x):
    w = torch.hann_window(x.shape[-1], periodic=True, device=x.device, dtype=x.dtype)
    return torch.log(torch.fft.rfft(x * w).abs() + 1e-6)
class SpecEnc(nn.Module):
    def __init__(s, in_ch, mu, sd):
        super().__init__()
        s.register_buffer('mu', torch.as_tensor(mu, dtype=torch.float32)); s.register_buffer('sd', torch.as_tensor(sd, dtype=torch.float32))
        s.f = nn.Sequential(block(in_ch, 32, 15, 2), block(32, 64, 5), block(64, 64, 3), block(64, 64, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten())
    def forward(s, x): return s.f((log_spectrum(x) - s.mu) / s.sd)
zn = lambda X: ((X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-8)).astype(np.float32)

L = 2048; NBIN = L // 2 + 1
EDGES = np.linspace(0, NBIN, NB + 1).astype(int)                       # same partition as KR.log_band_energy / torch_lsd
BIDX = torch.as_tensor(np.repeat(np.arange(NB), np.diff(EDGES)))       # bin -> band, includes DC (band 0) and Nyquist (last band)
assert len(BIDX) == NBIN

def band_view(x, kappa, s, u, tau):
    """x (B,C,L); kappa (B,); s (C,NB) budget (already times gain); u (B,C,NB). Band gain exp(0.5 u kappa s h); the whole-window exponent is halved up to 3 times when
    the per-channel log-spectral distance exceeds tau, then the window falls back to x. Returns view, fallback flag, number of halvings."""
    B, C, _ = x.shape; Xf = torch.fft.rfft(x, dim=-1)
    out = x.clone(); ok = torch.zeros(B, dtype=torch.bool); shr = torch.zeros(B); expo = 0.5 * u * kappa.reshape(-1, 1, 1) * s.reshape(1, C, NB)
    for k in range(4):
        gain = torch.exp(expo)[..., BIDX]
        cand = torch.fft.irfft(Xf * gain, n=L, dim=-1)
        lsd = KR.torch_lsd(cand.reshape(B * C, L), x.reshape(B * C, L)).reshape(B, C)
        good = (lsd <= tau[None]).all(1) & torch.isfinite(cand).flatten(1).all(1) & ~ok
        out[good] = cand[good]; shr[good] = k; ok |= good
        if ok.all(): break
        expo = expo * 0.5
    return out, ~ok, shr

def band_E(x):                                                          # (N,C,L) tensor -> (N,C,NB) numpy
    xn = x.numpy().astype(np.float64); return np.stack([KR.log_band_energy(xn[:, c]) for c in range(xn.shape[1])], 1)
def rel_of(E): return np.stack([E[:, a] - E[:, b] for a, b in PAIRS], 1)    # (N,6,NB)

def pairs_fast(cls, meas, rng):
    cells = {}
    for i, (c, m) in enumerate(zip(cls, meas)): cells.setdefault((c, m), []).append(i)
    keys = list(cells); arr = [np.array(cells[k]) for k in keys]; kc = np.array([k[0] for k in keys])
    pos_c = [np.array([q for q in range(len(keys)) if kc[q] == kc[j] and q != j]) for j in range(len(keys))]
    neg_c = [np.array([q for q in range(len(keys)) if kc[q] != kc[j]]) for j in range(len(keys))]
    def draw(n, cand):
        j = rng.integers(0, len(keys), n); a = np.empty(n, int); b = np.empty(n, int)
        for k in range(len(keys)):
            sel = np.where(j == k)[0]
            if len(sel) == 0: continue
            a[sel] = arr[k][rng.integers(0, len(arr[k]), len(sel))]; q = cand[k][rng.integers(0, len(cand[k]), len(sel))]
            for qq in np.unique(q):
                s2 = sel[q == qq]; b[s2] = arr[qq][rng.integers(0, len(arr[qq]), len(s2))]
        return a, b
    pa, pb = draw(NPOS, pos_c); na, nb = draw(NNEG, neg_c)
    return np.concatenate([pa, na]), np.concatenate([pb, nb]), np.concatenate([np.ones(NPOS), np.zeros(NNEG)])
def pair_auroc(Fz, ia, ib, lab):
    Fn = F.normalize(Fz, dim=1); return float(roc_auc_score(lab, (Fn[ia] * Fn[ib]).sum(1).numpy()))

dcos = lambda a, b: (1 - F.cosine_similarity(a, b, dim=1)).numpy()

import time
GAINS2=[0.25,0.5,0.75,1.0,2.0,3.0,4.0,6.0,8.0,12.0]
rows=[]
t0=time.time()
for c in ('C1','C2','C3'):
    for f in (1,2,3):
        d = np.load(SPL % (c, f), allow_pickle=False)
        X = zn(np.asarray(d['X_train'], np.float32)); y = d['y_train']; pi = np.asarray(d['train_pool_index'], np.int64)
        inst = np.asarray(d['train_installation']).astype(str); meas = np.char.add(np.char.add(inst, '|'), np.asarray(d['train_measurement']).astype(str)); known = d['known_labels'].tolist()
        BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r')
        seed=0
        rd = f'runs_paderborn/P_{c}_f{f}_seed{seed}'
        r = list(json.load(open(f'{rd}/results_exp2.json', encoding='utf-8')).values())[0]
        lab = np.array(r['labeled_idx']); unl = np.setdiff1d(np.arange(len(y)), lab); order = np.concatenate([lab, unl]); n_lab = len(lab); N = len(order)
        ck = torch.load(f'{rd}/m2_PADERBORN_seed{seed}_warmup.pt', map_location='cpu', weights_only=False)
        enc = SpecEnc(X.shape[1], ck['encoder']['mu'], ck['encoder']['sd']); enc.load_state_dict(ck['encoder']); enc.eval()
        head = nn.Linear(64, len(known)); head.load_state_dict(ck['cls_head'])
        XA = torch.tensor(X[order]); ylab = y[lab]; ylt = torch.tensor([known.index(v) for v in ylab])
        ml = torch.tensor(np.asarray(BANK[pi[lab]], np.float32)).numpy()
        tau = torch.tensor([KR.nuisance_mode_stats(ml[:, ch], X[lab][:, ch], ylab, inst[lab])['tau_lsd'] for ch in range(4)], dtype=torch.float32)
        E_all = band_E(XA); E_lab = E_all[:n_lab]; rec_lab = meas[lab]; sig = []
        for cl in known:
            m_ = ylab == cl; rr = np.unique(rec_lab[m_])
            sig.append(np.stack([E_lab[m_ & (rec_lab == q)].mean(0) for q in rr]).std(0) if len(rr) >= 2 else E_lab[m_].std(0))
        s_base = torch.tensor(np.mean(sig, 0), dtype=torch.float32)
        with torch.no_grad():
            FA = KR.encode(enc, XA); torch.manual_seed(1000 + seed)
            sc = KR.fit_scorer(FA[:n_lab], ylt, head, kind='knn', energy_T=0.05, gamma=2.0)
            kr = KR.compute_kr(enc, head, sc, XA, lambda Z: vectorized_aug(Z, WEAK), n_lab)
            kap = KR.budget_v2(kr['K'], kr['R'], 0.25, 0.5); kap[:n_lab] = 1.0
            torch.manual_seed(7); VA = KR.encode(enc, vectorized_aug(XA, STD))
            gen = torch.Generator().manual_seed(seed + 20000); uI = torch.randn(N, 4, NB, generator=gen).clamp(-2.0, 2.0); uC = uI[:, 0:1, :].expand(N, 4, NB).contiguous()
            row=dict(combo=c,fold=f,gains={})
            for g in GAINS2:
                gd={}
                for nm,u in (('VI',uI),('VC',uC)):
                    v,fb,shr=band_view(XA,kap,s_base*g,u,tau)
                    torch.manual_seed(7); fa=KR.encode(enc,vectorized_aug(v,STD))
                    gd[nm]=dict(feat=float(np.median(dcos(VA,fa)[n_lab:])),sig=float(np.median(((v-XA).norm(dim=2)/XA.norm(dim=2)).numpy()[n_lab:].mean(1))),shrunk=float((shr[n_lab:]>0).float().mean()),fallback=float(fb[n_lab:].float().mean()))
                row['gains'][str(g)]=gd
            rows.append(row); print(c,f,'t=%.0fs'%(time.time()-t0),'g=2 feat VI %.4f VC %.4f | g=12 VC feat %.4f shrunk %.2f fallback %.2f'%(row['gains']['2.0']['VI']['feat'],row['gains']['2.0']['VC']['feat'],row['gains']['12.0']['VC']['feat'],row['gains']['12.0']['VC']['shrunk'],row['gains']['12.0']['VC']['fallback']),flush=True)
json.dump(rows,open('/projects/sandbox/results_v24/v27b_response_curve.json','w'),indent=1)
print('\nmean over 9 models (one seed per split): gain | VI feat sig shrunk fallback | VC feat sig shrunk fallback')
for g in GAINS2:
    k=str(g); m=lambda a,b: np.mean([r['gains'][k][a][b] for r in rows])
    print('%5.2f | %.4f %.3f %.2f %.2f | %.4f %.3f %.2f %.2f'%(g,m('VI','feat'),m('VI','sig'),m('VI','shrunk'),m('VI','fallback'),m('VC','feat'),m('VC','sig'),m('VC','shrunk'),m('VC','fallback')))
# gain at which VI reaches the VC(g=2) feature distance, and VC gain needed to reach VI(g=2)
import numpy as _np
def interp_gain(arm,target):
    out=[]
    for r in rows:
        xs=[_np.log(g) for g in GAINS2]; ys=[r['gains'][str(g)][arm]['feat'] for g in GAINS2]
        out.append(float(_np.exp(_np.interp(target,ys,xs))) if min(ys)<=target<=max(ys) else float('nan'))
    return out
tv=[r['gains']['2.0']['VC']['feat'] for r in rows]; tc=[r['gains']['2.0']['VI']['feat'] for r in rows]
gi=[interp_gain('VI',t) for t in tv]; gi=[interp_gain_i for interp_gain_i in [ _np.exp(_np.interp(t,[r['gains'][str(g)]['VI']['feat'] for g in GAINS2],[_np.log(g) for g in GAINS2])) if r['gains'][str(GAINS2[0])]['VI']['feat']<=t<=r['gains'][str(GAINS2[-1])]['VI']['feat'] else float('nan') for r,t in zip(rows,tv)]]
gc=[ _np.exp(_np.interp(t,[r['gains'][str(g)]['VC']['feat'] for g in GAINS2],[_np.log(g) for g in GAINS2])) if r['gains'][str(GAINS2[0])]['VC']['feat']<=t<=r['gains'][str(GAINS2[-1])]['VC']['feat'] else float('nan') for r,t in zip(rows,tc)]
print('independent gain whose feature distance equals the coordinated one at gain 2 (per model):',_np.round(gi,2).tolist())
print('coordinated gain needed to reach the independent feature distance at gain 2 (nan = not reachable up to gain 12):',_np.round(gc,2).tolist())
