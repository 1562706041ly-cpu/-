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
GRID=[round(0.5+0.05*i,2) for i in range(21)]
GC=2.0; GI2=2.0
rows=[]; t0=time.time(); mi=0
for c in ('C1','C2','C3'):
    for f in (1,2,3):
        d = np.load(SPL % (c, f), allow_pickle=False)
        X = zn(np.asarray(d['X_train'], np.float32)); y = d['y_train']; pi = np.asarray(d['train_pool_index'], np.int64)
        inst = np.asarray(d['train_installation']).astype(str); meas = np.char.add(np.char.add(inst, '|'), np.asarray(d['train_measurement']).astype(str)); known = d['known_labels'].tolist()
        BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r')
        recs = np.unique(meas); half = {r_: (i % 2) for i, r_ in enumerate(sorted(recs))}
        for seed in (0,1,2):
            rd = f'runs_paderborn/P_{c}_f{f}_seed{seed}'
            r = list(json.load(open(f'{rd}/results_exp2.json', encoding='utf-8')).values())[0]
            lab = np.array(r['labeled_idx']); unl = np.setdiff1d(np.arange(len(y)), lab); order = np.concatenate([lab, unl]); n_lab = len(lab); N = len(order)
            ck = torch.load(f'{rd}/m2_PADERBORN_seed{seed}_warmup.pt', map_location='cpu', weights_only=False)
            enc = SpecEnc(X.shape[1], ck['encoder']['mu'], ck['encoder']['sd']); enc.load_state_dict(ck['encoder']); enc.eval()
            XA = torch.tensor(X[order]); ylab = y[lab]
            ml = torch.tensor(np.asarray(BANK[pi[lab]], np.float32)).numpy()
            tau = torch.tensor([KR.nuisance_mode_stats(ml[:, ch], X[lab][:, ch], ylab, inst[lab])['tau_lsd'] for ch in range(4)], dtype=torch.float32)
            E_all = band_E(XA); E_lab = E_all[:n_lab]; rec_lab = meas[lab]; sig = []
            for cl in known:
                m_ = ylab == cl; rr = np.unique(rec_lab[m_])
                sig.append(np.stack([E_lab[m_ & (rec_lab == q)].mean(0) for q in rr]).std(0) if len(rr) >= 2 else E_lab[m_].std(0))
            s_base = torch.tensor(np.mean(sig, 0), dtype=torch.float32)
            hm = np.array([half[m_] for m_ in meas[order]])                       # 0 = selection half, 1 = check half
            ones = torch.ones(N)
            with torch.no_grad():
                torch.manual_seed(7); fA = KR.encode(enc, vectorized_aug(XA, STD)); aug_state=None
                gen = torch.Generator().manual_seed(seed + 20000); uI = torch.randn(N, 4, NB, generator=gen).clamp(-2.0, 2.0); uC = uI[:, 0:1, :].expand(N, 4, NB).contiguous()
                def stats(u, g):
                    v, fb, shr = band_view(XA, ones, s_base * g, u, tau)
                    torch.manual_seed(7); fa = KR.encode(enc, vectorized_aug(v, STD)); dd = dcos(fA, fa); sg = ((v - XA).norm(dim=2) / XA.norm(dim=2)).numpy().mean(1); sh = (shr > 0).float().numpy()
                    return {h: dict(feat=float(np.median(dd[hm == h])), sig=float(np.median(sg[hm == h])), shrunk=float(sh[hm == h].mean())) for h in (0, 1)}
                row = dict(combo=c, fold=f, seed=seed, VC=stats(uC, GC), VI2=stats(uI, GI2), VI={str(g): stats(uI, g) for g in GRID})
            rows.append(row); mi += 1
            print(c, f, seed, 't=%.0fs' % (time.time() - t0), 'sel-half feat VC2 %.4f VI2 %.4f | VI(0.85) %.4f' % (row['VC'][0]['feat'], row['VI2'][0]['feat'], row['VI']['0.85'][0]['feat']), flush=True)
json.dump(rows, open('/projects/sandbox/results_v24/v28_gain_scan.json', 'w'))
score = {g: float(np.mean([abs(np.log(r['VI'][str(g)][0]['feat'] / r['VC'][0]['feat'])) for r in rows])) for g in GRID}
gstar = min(score, key=score.get)
print('\nselection score (mean |ln ratio| on the selection half) by grid value:'); print(' '.join(f'{g}:{score[g]:.3f}' for g in GRID))
print('SELECTED independent gain g_I* =', gstar)
best = [min(GRID, key=lambda g: abs(np.log(r['VI'][str(g)][0]['feat'] / r['VC'][0]['feat']))) for r in rows]
print('per-model best grid value (selection half): mean %.3f min %.2f max %.2f' % (np.mean(best), min(best), max(best)))
for h, nm in ((0, 'SELECTION'), (1, 'CHECK')):
    rr_ = [abs(np.log(r['VI'][str(gstar)][h]['feat'] / r['VC'][h]['feat'])) for r in rows]
    print(f'{nm} half at g_I*: |ln ratio| median %.3f max %.3f ; within ln(1.25) in %d/27 models ; mean ratio feat_VI/feat_VC %.3f' % (np.median(rr_), max(rr_), sum(x <= np.log(1.25) for x in rr_), np.mean([r['VI'][str(gstar)][h]['feat'] / r['VC'][h]['feat'] for r in rows])))
    for nm2, get in (('VC(2.0)', lambda r: r['VC'][h]), ('VI(g*)', lambda r: r['VI'][str(gstar)][h]), ('VI(2.0)', lambda r: r['VI2'][h])):
        print(f'   {nm2:8s} feat %.4f sig %.3f shrunk %.3f' % tuple(np.mean([get(r)[k] for r in rows]) for k in ('feat', 'sig', 'shrunk')))
per=[(rows[3*i]['combo'],rows[3*i]['fold'],round(float(np.mean([abs(np.log(r['VI'][str(gstar)][1]['feat']/r['VC'][1]['feat'])) for r in rows[3*i:3*i+3]])),3)) for i in range(9)]
print('per-split CHECK-half |ln ratio| at g_I*:',per)
