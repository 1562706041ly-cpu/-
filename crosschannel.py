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
done = set()
if os.path.exists(OUTJ):
    for ln in open(OUTJ):
        r0 = json.loads(ln); done.add((r0['combo'], r0['fold'], r0['seed']))
mi = 0
for c in ('C1', 'C2', 'C3'):
    for f in (1, 2, 3):
        d = np.load(SPL % (c, f), allow_pickle=False)
        X = zn(np.asarray(d['X_train'], np.float32)); y = d['y_train']; pi = np.asarray(d['train_pool_index'], np.int64)
        inst = np.asarray(d['train_installation']).astype(str); meas = np.char.add(np.char.add(inst, '|'), np.asarray(d['train_measurement']).astype(str)); known = d['known_labels'].tolist()
        BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r')
        for seed in (0, 1, 2):
            mi += 1
            if (c, f, seed) in done: continue
            rd = f'runs_paderborn/P_{c}_f{f}_seed{seed}'
            r = list(json.load(open(f'{rd}/results_exp2.json', encoding='utf-8')).values())[0]
            lab = np.array(r['labeled_idx']); unl = np.setdiff1d(np.arange(len(y)), lab); order = np.concatenate([lab, unl]); n_lab = len(lab); N = len(order)
            ck = torch.load(f'{rd}/m2_PADERBORN_seed{seed}_warmup.pt', map_location='cpu', weights_only=False)
            enc = SpecEnc(X.shape[1], ck['encoder']['mu'], ck['encoder']['sd']); enc.load_state_dict(ck['encoder']); enc.eval()
            head = nn.Linear(64, len(known)); head.load_state_dict(ck['cls_head'])
            XA = torch.tensor(X[order]); ylab = y[lab]; ylt = torch.tensor([known.index(v) for v in ylab])
            # tau_lsd exactly as in training (labelled windows, record key = installation); the modes only enter the unused sigma statistics
            ml = torch.tensor(np.asarray(BANK[pi[lab]], np.float32)).numpy()
            tau = torch.tensor([KR.nuisance_mode_stats(ml[:, ch], X[lab][:, ch], ylab, inst[lab])['tau_lsd'] for ch in range(4)], dtype=torch.float32)
            # band budget from LABELLED known windows only, recording-level (installation|measurement)
            E_all = band_E(XA); E_lab = E_all[:n_lab]; rec_lab = meas[lab]; sig = []; n_fb = 0
            for cl in known:
                m_ = ylab == cl; rr = np.unique(rec_lab[m_])
                if len(rr) >= 2: sig.append(np.stack([E_lab[m_ & (rec_lab == q)].mean(0) for q in rr]).std(0))
                else: sig.append(E_lab[m_].std(0)); n_fb += 1
            s_base = torch.tensor(np.mean(sig, 0), dtype=torch.float32)                       # (C, NB)
            out = dict(combo=c, fold=f, seed=seed, n_classes_budget_fallback=n_fb, s_mean_by_channel=s_base.mean(1).tolist())
            # natural recording-level variation of the cross-channel relation (labelled known windows)
            R_lab = rel_of(E_lab); dn, dw, ncls = [], [], 0
            for cl in known:
                m_ = ylab == cl; rr = [q for q in np.unique(rec_lab[m_])]
                if len(rr) < 2: continue
                rm = [R_lab[m_ & (rec_lab == q)].mean(0) for q in rr]
                dn.append(np.mean([np.abs(rm[i] - rm[j]).mean() for i in range(len(rr)) for j in range(i + 1, len(rr))]))
                wv = []
                for q in rr:
                    ww = R_lab[m_ & (rec_lab == q)]
                    if len(ww) >= 2: wv.append(np.mean([np.abs(ww[i] - ww[j]).mean() for i in range(len(ww)) for j in range(i + 1, len(ww))]))
                if wv: dw.append(np.mean(wv))
            Dnat = float(np.mean(dn)) if dn else float('nan'); out['Dnat'] = Dnat; out['Dwithin_record'] = float(np.mean(dw)) if dw else float('nan'); out['evaluable'] = bool(dn) and Dnat >= 1e-3
            mean_E = E_lab.mean(0)                                                             # (C, NB)
            low = np.zeros((4, NB), bool); low.flat[np.argsort(mean_E.ravel())[:int(round(0.1 * 4 * NB))]] = True
            low_pair = np.stack([low[a] | low[b] for a, b in PAIRS], 0)                        # (6, NB)
            R_x = rel_of(E_all)
            with torch.no_grad():
                FA = KR.encode(enc, XA); torch.manual_seed(1000 + seed)
                sc = KR.fit_scorer(FA[:n_lab], ylt, head, kind='knn', energy_T=0.05, gamma=2.0)
                kr = KR.compute_kr(enc, head, sc, XA, lambda Z: vectorized_aug(Z, WEAK), n_lab); K0 = kr['K']
                kap = KR.budget_v2(kr['K'], kr['R'], 0.25, 0.5); kap[:n_lab] = 1.0
                torch.manual_seed(7); VA = KR.encode(enc, vectorized_aug(XA, STD))
                unk_idx = np.where(~np.isin(y[order], known))[0]
                if True:
                    rng = np.random.default_rng(555 + 31 * mi); ia, ib, plab = pairs_fast(y[order][unk_idx], meas[order][unk_idx], rng)
                    out['P1_VA'] = pair_auroc(VA[unk_idx], ia, ib, plab)
                out['gains'] = {}
                for g in GAINS:
                    sg = s_base * g; gd = {}
                    def make(seed_u):
                        gen = torch.Generator().manual_seed(seed_u)
                        uI = torch.randn(N, 4, NB, generator=gen).clamp(-2.0, 2.0); uC = uI[:, 0:1, :].expand(N, 4, NB).contiguous()
                        return uI, uC
                    uI, uC = make(seed + 20000)
                    views = {}
                    for nm, u in (('VI', uI), ('VC', uC)): views[nm] = band_view(XA, kap, sg, u, tau)
                    for nm, (v, fb, shr) in views.items():
                        sel = torch.arange(N) >= n_lab
                        rel_sig = ((v - XA).norm(dim=2) / XA.norm(dim=2)).numpy()           # (N, C)
                        lsdr = (KR.torch_lsd(v.reshape(N * 4, L), XA.reshape(N * 4, L)).reshape(N, 4) / tau[None]).numpy()
                        Ev = band_E(v); Rv = rel_of(Ev); dvw = np.abs(Rv - R_x).mean((1, 2))
                        contrib = np.abs(Rv - R_x)[n_lab:].mean(0)                           # (6, NB)
                        fv = KR.encode(enc, v)
                        gd[nm] = dict(sig_change_med=np.median(rel_sig[n_lab:], 0).tolist(), lsd_over_tau_med=np.median(lsdr[n_lab:], 0).tolist(),
                                      frac_shrunk=float((shr[n_lab:] > 0).float().mean()), frac_fallback=float(fb[n_lab:].float().mean()),
                                      Dv_unl=float(dvw[n_lab:].mean()), Dv_lab=float(dvw[:n_lab].mean()), low_energy_share=float(contrib[low_pair].sum() / max(contrib.sum(), 1e-12)),
                                      feat_raw_med=float(np.median(dcos(FA, fv)[n_lab:])), route_flip=float(((KR.known_score(sc, fv) >= 0.5) != (K0 >= 0.5))[n_lab:].float().mean()))
                        torch.manual_seed(7); fa = KR.encode(enc, vectorized_aug(v, STD)); views[nm] = views[nm] + (fa,)
                        gd[nm]['feat_aug_med'] = float(np.median(dcos(VA, fa)[n_lab:]))
                    gd['feat_aug_VI_vs_VC_med'] = float(np.median(dcos(views['VI'][3], views['VC'][3])[n_lab:]))
                    if g == PRIMARY:
                        rep = {'VI': [], 'VC': []}
                        for k_ in range(1, NREP + 1):
                            uIr, uCr = make(seed + 20000 + 7919 * k_)
                            for nm, u in (('VI', uIr), ('VC', uCr)):
                                v, fb, shr = band_view(XA, kap, sg, u, tau); torch.manual_seed(7); fa = KR.encode(enc, vectorized_aug(v, STD)); rep[nm].append(float(np.median(dcos(VA, fa)[n_lab:])))
                        gd['feat_aug_repeats'] = rep
                        out['P1_VIA'] = pair_auroc(views['VI'][3][unk_idx], ia, ib, plab); out['P1_VCA'] = pair_auroc(views['VC'][3][unk_idx], ia, ib, plab)
                    out['gains'][str(g)] = gd
            with open(OUTJ, 'a') as fh: fh.write(json.dumps(out) + '\n')
            g2 = out['gains']['2.0']
            print(c, f, seed, 'Dnat %.3f | Dv VI %.3f VC %.3f | sig ch1 VI %.3f VC %.3f | feat VI %.4f VC %.4f' % (Dnat, g2['VI']['Dv_unl'], g2['VC']['Dv_unl'], g2['VI']['sig_change_med'][0], g2['VC']['sig_change_med'][0], g2['VI']['feat_aug_med'], g2['VC']['feat_aug_med']), flush=True)
print('script finished; models in file:', sum(1 for _ in open(OUTJ)))
