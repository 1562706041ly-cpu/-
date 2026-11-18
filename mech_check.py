"""v24m mechanism check (see results_v24/v24m_mechanism_PREREGISTERED.txt). Run from scratch/v24p with PYTHONPATH=. (gpu_runtime.py, kr_modules.py there)."""
import json, math, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
import kr_modules as KR
from gpu_runtime import vectorized_aug
torch.set_num_threads(4)
SPL = '/projects/sandbox/centrifugal/splits_ch1245/motor2_100_ch1245_%s_fold%s.npz'
BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r')
STD = dict(shift=256, scale=0.2, noise=0.1, mask=0.10); WEAK = dict(shift=64, scale=0.05, noise=0.02, mask=0.0)
KMIN, RHO = 0.25, 0.5

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

def vmd_view(x, modes, kappa, scale, tau, gen):
    """Replicates KR._aug_vmd_nuisance_mc but also returns per-window shrink count, fallback flag and the realised LSD per channel."""
    B, C, M, L = modes.shape
    u = torch.randn(B, C, M, 1, generator=gen).clamp(-2.0, 2.0)
    expo = 0.5 * u * kappa.reshape(-1, 1, 1, 1) * scale.reshape(1, C, M, 1)
    out = x.clone(); shrink = torch.zeros(B); ok = torch.zeros(B, dtype=torch.bool); lsd_real = torch.zeros(B, C)
    for s in range(4):
        cand = x + ((torch.exp(expo) - 1.0) * modes).sum(2)
        lsd = KR.torch_lsd(cand.reshape(B * C, L), x.reshape(B * C, L)).reshape(B, C)
        good = (lsd <= tau[None]).all(1) & torch.isfinite(cand).flatten(1).all(1) & ~ok
        out[good] = cand[good]; shrink[good] = s; lsd_real[good] = lsd[good]; ok |= good
        if ok.all(): break
        expo = expo * 0.5
    return out, shrink, ~ok, lsd_real

cos1 = lambda a, b: 1 - F.cosine_similarity(a, b, dim=1)
q = lambda v, p: float(np.percentile(v, p))
rows = []
for c in ('C1', 'C2', 'C3'):
    for f in (1, 2, 3):
        d = np.load(SPL % (c, f), allow_pickle=False)
        X = zn(np.asarray(d['X_train'], np.float32)); y = d['y_train']; pi = np.asarray(d['train_pool_index'], np.int64)
        modes_all = torch.tensor(np.asarray(BANK[pi], np.float32)); Xt = torch.tensor(X)
        rel = float(np.linalg.norm(modes_all[::22].sum(2).numpy() - X[::22]) / np.linalg.norm(X[::22]))
        assert rel < 0.1, rel
        rec = np.asarray(d['train_installation']).astype(str); known = d['known_labels'].tolist()
        for seed in (0, 1, 2):
            rd = f'runs_paderborn/P_{c}_f{f}_seed{seed}'
            r = list(json.load(open(f'{rd}/results_exp2.json', encoding='utf-8')).values())[0]
            lab = np.array(r['labeled_idx']); unl = np.setdiff1d(np.arange(len(y)), lab); order = np.concatenate([lab, unl]); n_lab = len(lab)
            ck = torch.load(f'{rd}/m2_PADERBORN_seed{seed}_warmup.pt', map_location='cpu', weights_only=False)
            enc = SpecEnc(X.shape[1], ck['encoder']['mu'], ck['encoder']['sd']); enc.load_state_dict(ck['encoder']); enc.eval()
            head = nn.Linear(64, len(known)); head.load_state_dict(ck['cls_head'])
            XA = Xt[order]; MA = modes_all[order]; unk = ~np.isin(y[order], known)
            ylt = torch.tensor([known.index(v) for v in y[lab]])
            # nuisance statistics exactly as prepare_nuisance (labelled windows, per channel)
            ml = modes_all[torch.as_tensor(lab)].numpy()
            per = [KR.nuisance_mode_stats(ml[:, ch], X[lab][:, ch], y[lab], rec[lab]) for ch in range(ml.shape[1])]
            st = {k: np.stack([p[k] for p in per]) for k in ('sigma_rec', 'sigma_cls', 'g')}
            scale = torch.tensor(st['sigma_rec'] * st['g'], dtype=torch.float32); tau = torch.tensor([p['tau_lsd'] for p in per], dtype=torch.float32)
            out = dict(combo=c, fold=f, seed=seed, n=len(XA), tau=tau.tolist(), scale_mean_by_channel=scale.mean(1).tolist())
            with torch.no_grad():
                weak = lambda Z: vectorized_aug(Z, WEAK)
                FA = KR.encode(enc, XA)
                kk = {}
                for kind in ('knn', 'gated'):
                    torch.manual_seed(1000 + seed)
                    sc = KR.fit_scorer(FA[:n_lab], ylt, head, kind=kind, energy_T=0.05, gamma=2.0)
                    kr = KR.compute_kr(enc, head, sc, XA, weak, n_lab)
                    kap = KR.budget_v2(kr['K'], kr['R'], KMIN, RHO); kap[:n_lab] = 1.0
                    kk[kind] = (kr, kap, sc)
                kr, kap, sc = kk['knn']
                gen = torch.Generator().manual_seed(seed + 20000)
                x2, shr, fb, lsd = vmd_view(XA, MA, kap, scale, tau, gen)
                x2g, shrg, fbg, _ = vmd_view(XA, MA, kk['gated'][1], scale, tau, torch.Generator().manual_seed(seed + 20000))
                perm = torch.cat([torch.arange(n_lab), n_lab + torch.randperm(len(XA) - n_lab, generator=torch.Generator().manual_seed(seed + 40000))])
                x2s, _, _, _ = vmd_view(XA, MA, kap[perm], scale, tau, torch.Generator().manual_seed(seed + 20000))
                x1 = MA.sum(2)
                torch.manual_seed(7); a3 = vectorized_aug(XA, STD)
                torch.manual_seed(7); a4 = vectorized_aug(x2, STD)
                torch.manual_seed(8); a3p = vectorized_aug(XA, STD)
                torch.manual_seed(9); a5 = vectorized_aug(XA, WEAK)
                f0, f1, f2, f2s = KR.encode(enc, XA), KR.encode(enc, x1), KR.encode(enc, x2), KR.encode(enc, x2s)
                f3, f4, f3p, f5 = KR.encode(enc, a3), KR.encode(enc, a4), KR.encode(enc, a3p), KR.encode(enc, a5)
                relchg = lambda v: ((v - XA).flatten(1).norm(dim=1) / XA.flatten(1).norm(dim=1)).numpy()
                wcos = lambda v: F.cosine_similarity(v.reshape(len(v), v.shape[1], -1), XA.reshape(len(XA), XA.shape[1], -1), dim=2).mean(1).numpy()
                ratio = (lsd / tau[None]).max(1).values.numpy()
                ratio_all = ratio.copy(); changed = ~fb.numpy()
                m = lambda v, sel=None: dict(med=q(v if sel is None else v[sel], 50), p90=q(v if sel is None else v[sel], 90), p99=q(v if sel is None else v[sel], 99))
                out['S1_relchg_V1'] = m(relchg(x1)); out['S1_relchg_V2'] = m(relchg(x2)); out['S1_relchg_V2s'] = m(relchg(x2s)); out['S1_relchg_V2g'] = m(relchg(x2g))
                out['S1_wavecos_V2'] = m(wcos(x2)); out['S1_relchg_V3'] = m(relchg(a3)); out['S1_relchg_V5'] = m(relchg(a5))
                out['S2_lsd_over_tau_V2'] = m(ratio_all)
                out['S2_frac_shrunk'] = float((shr > 0).float().mean()); out['S2_frac_fallback'] = float(fb.float().mean())
                out['S2_frac_shrunk_or_fallback'] = float(((shr > 0) | fb).float().mean()); out['S2_frac_fallback_gated'] = float(fbg.float().mean())
                dd = lambda a, b: cos1(a, b).numpy()
                out['F1_V0_V1'] = m(dd(f0, f1)); out['F1_V0_V2'] = m(dd(f0, f2)); out['F1_V0_V2s'] = m(dd(f0, f2s))
                out['F1_V3_V4'] = m(dd(f3, f4)); out['F1_V3_V3p'] = m(dd(f3, f3p)); out['F1_V0_V5'] = m(dd(f0, f5)); out['F1_V0_V3'] = m(dd(f0, f3))
                out['F2_ratio_r'] = out['F1_V3_V4']['med'] / max(out['F1_V3_V3p']['med'], 1e-12)
                rt0 = (kr['K'] >= 0.5); K2 = KR.known_score(sc, f2); rt2 = K2 >= 0.5
                out['R1_route_change'] = float((rt0 != rt2)[n_lab:].float().mean()); out['R1_meanabsdK'] = float((K2 - kr['K']).abs()[n_lab:].mean())
                ku = kap[n_lab:].numpy(); ru = (kr['K'][n_lab:] >= 0.5).numpy(); kk_ = kap.numpy()
                out['B1_kappa_unl'] = dict(mean=float(ku.mean()), p10=q(ku, 10), p90=q(ku, 90), mean_routed_known=float(ku[ru].mean()) if ru.any() else None, mean_routed_unknown=float(ku[~ru].mean()) if (~ru).any() else None)
                sel = np.arange(len(XA)) >= n_lab
                out['B1_spearman_kappa_lsdratio'] = float(spearmanr(kk_[sel], ratio_all[sel])[0]); out['B1_spearman_kappa_featdist'] = float(spearmanr(kk_[sel], dd(f0, f2)[sel])[0])
                out['B2_rel_diff_median_relchg_V2_vs_V2s'] = abs(out['S1_relchg_V2']['med'] - out['S1_relchg_V2s']['med']) / max(out['S1_relchg_V2s']['med'], 1e-12)
                out['B2_featdist_median_V2_vs_V2s'] = (q(dd(f0, f2), 50), q(dd(f0, f2s), 50))
                out['frac_unknown_in_unl'] = float(unk[n_lab:].mean())
            rows.append(out); print(c, f, seed, 'r=%.3f' % out['F2_ratio_r'], 'lsd/tau med %.3f' % out['S2_lsd_over_tau_V2']['med'], 'shrunk/fb %.3f' % out['S2_frac_shrunk_or_fallback'], flush=True)
json.dump(rows, open('/projects/sandbox/results_v24/v24m_mechanism.json', 'w'), indent=1)

n = len(rows)
cnt = lambda fn: int(sum(fn(r) for r in rows))
verdict = dict(
    NEGLIGIBLE=cnt(lambda r: r['S2_lsd_over_tau_V2']['med'] < 0.25 and r['F2_ratio_r'] < 0.10),
    FIDELITY_BOUND=cnt(lambda r: r['S2_frac_shrunk_or_fallback'] > 0.20),
    COMPARABLE=cnt(lambda r: r['F2_ratio_r'] >= 0.50),
    FOLLOWS_BUDGET=cnt(lambda r: r['B1_spearman_kappa_lsdratio'] >= 0.5),
    ALLOCATION_RELEVANT=cnt(lambda r: r['B2_rel_diff_median_relchg_V2_vs_V2s'] >= 0.10))
need = dict(NEGLIGIBLE=23, FIDELITY_BOUND=14, COMPARABLE=23, FOLLOWS_BUDGET=23, ALLOCATION_RELEVANT=23)
print('\nmodels', n)
for k, v in verdict.items(): print(f'{k:20s} {v}/27   rule needs >= {need[k]}   -> {"HOLDS" if v >= need[k] else "no"}')
mean = lambda g: float(np.mean([g(r) for r in rows]))
for name, g in (('median rel signal change V2 (VMD view)', lambda r: r['S1_relchg_V2']['med']), ('median rel signal change V1 (recon only)', lambda r: r['S1_relchg_V1']['med']),
                ('median rel signal change V3 (standard aug)', lambda r: r['S1_relchg_V3']['med']), ('median LSD/tau V2', lambda r: r['S2_lsd_over_tau_V2']['med']),
                ('p99 LSD/tau V2', lambda r: r['S2_lsd_over_tau_V2']['p99']), ('frac shrunk', lambda r: r['S2_frac_shrunk']), ('frac fallback', lambda r: r['S2_frac_fallback']),
                ('F1 median d(V0,V1)', lambda r: r['F1_V0_V1']['med']), ('F1 median d(V0,V2)', lambda r: r['F1_V0_V2']['med']), ('F1 median d(V0,V3) std aug vs original', lambda r: r['F1_V0_V3']['med']),
                ('F1 median d(V3,V4) VMD under same std aug', lambda r: r['F1_V3_V4']['med']), ('F1 median d(V3,V3p) two std views', lambda r: r['F1_V3_V3p']['med']),
                ('ratio r', lambda r: r['F2_ratio_r']), ('route change (unl)', lambda r: r['R1_route_change']), ('mean |dK| (unl)', lambda r: r['R1_meanabsdK']),
                ('kappa mean (unl)', lambda r: r['B1_kappa_unl']['mean']), ('kappa p10', lambda r: r['B1_kappa_unl']['p10']), ('kappa p90', lambda r: r['B1_kappa_unl']['p90']),
                ('Spearman kappa vs LSD/tau', lambda r: r['B1_spearman_kappa_lsdratio']), ('Spearman kappa vs feat dist', lambda r: r['B1_spearman_kappa_featdist']),
                ('B2 rel diff V2 vs V2s', lambda r: r['B2_rel_diff_median_relchg_V2_vs_V2s'])):
    v = np.array([g(r) for r in rows], float); print(f'{name:46s} mean {np.nanmean(v):8.4f}  min {np.nanmin(v):8.4f}  max {np.nanmax(v):8.4f}')
print('tau per channel (mean over models):', np.round(np.mean([r['tau'] for r in rows], 0), 3).tolist())
print('mode-scale mean per channel (sigma_rec*g):', np.round(np.mean([r['scale_mean_by_channel'] for r in rows], 0), 3).tolist())
