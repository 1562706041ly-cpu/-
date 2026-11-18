"""v24d stage 1: dose check of --kr2-gain on fixed warm-up models (see results_v24/v24d_dose_PREREGISTERED.txt). Run from scratch/v24p with PYTHONPATH=."""
import json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import kr_modules as KR
from gpu_runtime import vectorized_aug
torch.set_num_threads(4)
SPL = '/projects/sandbox/centrifugal/splits_ch1245/motor2_100_ch1245_%s_fold%s.npz'
BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r')
STD = dict(shift=256, scale=0.2, noise=0.1, mask=0.10); WEAK = dict(shift=64, scale=0.05, noise=0.02, mask=0.0)
GAINS = [0.0, 1.0, 2.0, 4.0, 8.0]

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

def vmd_view(x, modes, kappa, scale, tau, u):
    """Same arithmetic as KR._aug_vmd_nuisance_mc; also returns realised per-channel LSD, shrink count and fallback."""
    B, C, M, L = modes.shape
    expo = 0.5 * u * kappa.reshape(-1, 1, 1, 1) * scale.reshape(1, C, M, 1)
    nominal = expo.abs().mean(dim=(2, 3))                      # (B, C)
    out = x.clone(); shrink = torch.zeros(B); ok = torch.zeros(B, dtype=torch.bool); lsd_real = torch.zeros(B, C)
    for s in range(4):
        cand = x + ((torch.exp(expo) - 1.0) * modes).sum(2)
        lsd = KR.torch_lsd(cand.reshape(B * C, L), x.reshape(B * C, L)).reshape(B, C)
        good = (lsd <= tau[None]).all(1) & torch.isfinite(cand).flatten(1).all(1) & ~ok
        out[good] = cand[good]; shrink[good] = s; lsd_real[good] = lsd[good]; ok |= good
        if ok.all(): break
        expo = expo * 0.5
    return out, shrink, ~ok, lsd_real, nominal

cos1 = lambda a, b: 1 - F.cosine_similarity(a, b, dim=1)
pct = lambda v, p: float(np.percentile(v, p))
rows = []
for c in ('C1', 'C2', 'C3'):
    for f in (1, 2, 3):
        d = np.load(SPL % (c, f), allow_pickle=False)
        X = zn(np.asarray(d['X_train'], np.float32)); y = d['y_train']; pi = np.asarray(d['train_pool_index'], np.int64)
        modes_all = torch.tensor(np.asarray(BANK[pi], np.float32)); Xt = torch.tensor(X)
        rec = np.asarray(d['train_installation']).astype(str); known = d['known_labels'].tolist()
        for seed in (0, 1, 2):
            rd = f'runs_paderborn/P_{c}_f{f}_seed{seed}'
            r = list(json.load(open(f'{rd}/results_exp2.json', encoding='utf-8')).values())[0]
            lab = np.array(r['labeled_idx']); unl = np.setdiff1d(np.arange(len(y)), lab); order = np.concatenate([lab, unl]); n_lab = len(lab)
            ck = torch.load(f'{rd}/m2_PADERBORN_seed{seed}_warmup.pt', map_location='cpu', weights_only=False)
            enc = SpecEnc(X.shape[1], ck['encoder']['mu'], ck['encoder']['sd']); enc.load_state_dict(ck['encoder']); enc.eval()
            head = nn.Linear(64, len(known)); head.load_state_dict(ck['cls_head'])
            XA = Xt[order]; MA = modes_all[order]; ylt = torch.tensor([known.index(v) for v in y[lab]])
            ml = modes_all[torch.as_tensor(lab)].numpy()
            per = [KR.nuisance_mode_stats(ml[:, ch], X[lab][:, ch], y[lab], rec[lab]) for ch in range(ml.shape[1])]
            st = {k: np.stack([p[k] for p in per]) for k in ('sigma_rec', 'sigma_cls', 'g')}
            base_scale = torch.tensor(st['sigma_rec'] * st['g'], dtype=torch.float32); tau = torch.tensor([p['tau_lsd'] for p in per], dtype=torch.float32)
            out = dict(combo=c, fold=f, seed=seed, gains={})
            with torch.no_grad():
                FA = KR.encode(enc, XA)
                torch.manual_seed(1000 + seed)
                sc = KR.fit_scorer(FA[:n_lab], ylt, head, kind='knn', energy_T=0.05, gamma=2.0)
                kr = KR.compute_kr(enc, head, sc, XA, lambda Z: vectorized_aug(Z, WEAK), n_lab)
                kap = KR.budget_v2(kr['K'], kr['R'], 0.25, 0.5); kap[:n_lab] = 1.0
                B_, C_, M_, L_ = MA.shape
                u = torch.randn(B_, C_, M_, 1, generator=torch.Generator().manual_seed(seed + 20000)).clamp(-2.0, 2.0)
                torch.manual_seed(7); a3 = vectorized_aug(XA, STD); f3 = KR.encode(enc, a3)
                torch.manual_seed(8); f3p = KR.encode(enc, vectorized_aug(XA, STD))
                d_std = float(np.median(cos1(f3, f3p).numpy()))
                K0 = kr['K']; rt0 = K0 >= 0.5
                sel = np.arange(len(XA)) >= n_lab
                for g in GAINS:
                    xg, shr, fb, lsd, nom = vmd_view(XA, MA, kap, base_scale * g, tau, u)
                    rel = ((xg - XA).flatten(1).norm(dim=1) / XA.flatten(1).norm(dim=1)).numpy()
                    torch.manual_seed(7); f4 = KR.encode(enc, vectorized_aug(xg, STD))
                    dv = cos1(f3, f4).numpy()
                    fg = KR.encode(enc, xg); Kg = KR.known_score(sc, fg)
                    ratio_ch = (lsd / tau[None]).numpy()                       # (B, C)
                    ratio_max = ratio_ch.max(1)
                    out['gains'][str(g)] = dict(
                        max_abs_diff=float((xg - XA).abs().max()),
                        nominal_expo_mean_by_channel=nom[sel].mean(0).tolist(), nominal_expo_mean=float(nom[sel].mean()),
                        relchg=dict(med=pct(rel, 50), p90=pct(rel, 90), p99=pct(rel, 99), max=float(rel.max())),
                        lsd_over_tau=dict(med=pct(ratio_max, 50), p90=pct(ratio_max, 90), p99=pct(ratio_max, 99), max=float(ratio_max.max())),
                        lsd_over_tau_by_channel_med=[pct(ratio_ch[:, k], 50) for k in range(C_)],
                        frac_shrunk=float((shr > 0).float().mean()), frac_fallback=float(fb.float().mean()), mean_halvings=float(shr[~fb].mean()) if (~fb).any() else float('nan'),
                        feat_d_med=float(np.median(dv)), r=float(np.median(dv)) / max(d_std, 1e-12),
                        route_change=float((rt0 != (Kg >= 0.5))[n_lab:].float().mean()), mean_abs_dK=float((Kg - K0).abs()[n_lab:].mean()))
                out['d_std'] = d_std
            rows.append(out); g1 = out['gains']['1.0']; g4 = out['gains']['4.0']; g8 = out['gains']['8.0']
            print(c, f, seed, 'relchg g1 %.3f g4 %.3f | fb g4 %.3f g8 %.3f | r g1 %.3f g4 %.3f g8 %.3f' % (g1['relchg']['med'], g4['relchg']['med'], g4['frac_fallback'], g8['frac_fallback'], g1['r'], g4['r'], g8['r']), flush=True)
json.dump(rows, open('/projects/sandbox/results_v24/v24d_dose.json', 'w'), indent=1)

n = len(rows); G = lambda r, g: r['gains'][str(g)]
R0 = sum(all(G(r, 0.0)['max_abs_diff'] == 0.0 for _ in [0]) for r in rows)
mono = sum(all(G(r, a)['relchg']['med'] < G(r, b)['relchg']['med'] for a, b in ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0))) for r in rows)
mono_nom = sum(all(G(r, a)['nominal_expo_mean'] < G(r, b)['nominal_expo_mean'] for a, b in ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0))) for r in rows)
x2 = sum(G(r, 4.0)['relchg']['med'] >= 2 * G(r, 1.0)['relchg']['med'] for r in rows)
print('\nmodels', n)
print(f'R0 gain=0 view equals x exactly: {R0}/27 (needs 27)')
print(f'R1 strictly increasing: nominal {mono_nom}/27, realised median signal change {mono}/27; gain4 >= 2 x gain1: {x2}/27 (each needs >= 23)')
adm = {}
for g in (2.0, 4.0, 8.0):
    adm[g] = sum(G(r, g)['frac_fallback'] <= 0.05 and G(r, g)['frac_shrunk'] <= 0.30 for r in rows)
    canc = sum(G(r, g)['relchg']['med'] / max(G(r, 1.0)['relchg']['med'], 1e-12) < 0.5 * g for r in rows)
    print(f'R2 gain {g:g}: admissible in {adm[g]}/27 (needs >= 23)   R3 cancelled (<0.5*g of the gain-1 change) in {canc}/27 (>=14 = cancelled)')
ok_g = [g for g in (2.0, 4.0, 8.0) if adm[g] >= 23]
gstar = max(ok_g) if ok_g else None
print('largest admissible gain g* =', gstar)
if gstar:
    cr = sum(G(r, gstar)['r'] >= 0.30 for r in rows); print(f'R4 r at g*={gstar:g} >= 0.30 in {cr}/27 (needs >= 23)')
mean = lambda g, fn: float(np.mean([fn(G(r, g)) for r in rows]))
print('\nmean over 27 models by gain:')
print('gain  nominal|expo|  relchg(med/p99)   LSD/tau(med/p99/max)   shrunk  fallback  halvings  r      route_chg  |dK|')
for g in GAINS:
    print(f'{g:4g}  {mean(g, lambda x: x["nominal_expo_mean"]):9.3f}   {mean(g, lambda x: x["relchg"]["med"]):7.3f}/{mean(g, lambda x: x["relchg"]["p99"]):6.3f}   '
          f'{mean(g, lambda x: x["lsd_over_tau"]["med"]):6.3f}/{mean(g, lambda x: x["lsd_over_tau"]["p99"]):6.3f}/{mean(g, lambda x: x["lsd_over_tau"]["max"]):6.3f}   '
          f'{mean(g, lambda x: x["frac_shrunk"]):6.3f} {mean(g, lambda x: x["frac_fallback"]):8.3f}  {np.nanmean([G(r, g)["mean_halvings"] for r in rows]):7.3f}  {mean(g, lambda x: x["r"]):6.3f}  {mean(g, lambda x: x["route_change"]):8.3f}  {mean(g, lambda x: x["mean_abs_dK"]):6.3f}')
print('\nper-channel median LSD/tau (channels 1,2,4,5), mean over models:')
for g in GAINS: print(f'{g:4g}  ' + ' '.join(f'{np.mean([G(r, g)["lsd_over_tau_by_channel_med"][k] for r in rows]):6.3f}' for k in range(4)))
print('range of r by gain (min-max):', {g: (round(min(G(r, g)['r'] for r in rows), 3), round(max(G(r, g)['r'] for r in rows), 3)) for g in GAINS})
