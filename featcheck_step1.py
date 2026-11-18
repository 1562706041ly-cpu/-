"""v25 step 1: feature-stability check of the VMD view on fixed warm-up models (criteria: results_v24/v25_featcheck_PREREGISTERED.txt). Run from scratch/v24p with PYTHONPATH=."""
import json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import kr_modules as KR
from gpu_runtime import vectorized_aug
torch.set_num_threads(4)
SPL = '/projects/sandbox/centrifugal/splits_ch1245/motor2_100_ch1245_%s_fold%s.npz'
BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r')
WEAK = dict(shift=64, scale=0.05, noise=0.02, mask=0.0)
GAINS = [2.0, 4.0]; Q = 0.95

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
    """Same arithmetic as KR._aug_vmd_nuisance_mc; returns the view, the log-spectral fallback flag and the number of halvings."""
    B, C, M, L = modes.shape
    expo = 0.5 * u * kappa.reshape(-1, 1, 1, 1) * scale.reshape(1, C, M, 1)
    out = x.clone(); ok = torch.zeros(B, dtype=torch.bool); shr = torch.zeros(B)
    for s in range(4):
        cand = x + ((torch.exp(expo) - 1.0) * modes).sum(2)
        lsd = KR.torch_lsd(cand.reshape(B * C, L), x.reshape(B * C, L)).reshape(B, C)
        good = (lsd <= tau[None]).all(1) & torch.isfinite(cand).flatten(1).all(1) & ~ok
        out[good] = cand[good]; shr[good] = s; ok |= good
        if ok.all(): break
        expo = expo * 0.5
    return out, ~ok, shr

relchg = lambda v, x: ((v - x).flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1)).numpy()
dcos = lambda a, b: (1 - F.cosine_similarity(a, b, dim=1)).numpy()
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
            lab = np.array(r['labeled_idx']); unl = np.setdiff1d(np.arange(len(y)), lab); order = np.concatenate([lab, unl]); n_lab = len(lab); N = len(order)
            ck = torch.load(f'{rd}/m2_PADERBORN_seed{seed}_warmup.pt', map_location='cpu', weights_only=False)
            enc = SpecEnc(X.shape[1], ck['encoder']['mu'], ck['encoder']['sd']); enc.load_state_dict(ck['encoder']); enc.eval()
            head = nn.Linear(64, len(known)); head.load_state_dict(ck['cls_head'])
            XA = Xt[order]; MA = modes_all[order]; ylt = torch.tensor([known.index(v) for v in y[lab]]); unk_all = ~np.isin(y[order], known)
            ml = modes_all[torch.as_tensor(lab)].numpy()
            per = [KR.nuisance_mode_stats(ml[:, ch], X[lab][:, ch], y[lab], rec[lab]) for ch in range(ml.shape[1])]
            st = {k: np.stack([p[k] for p in per]) for k in ('sigma_rec', 'sigma_cls', 'g')}
            base_scale = torch.tensor(st['sigma_rec'] * st['g'], dtype=torch.float32); tau_lsd = torch.tensor([p['tau_lsd'] for p in per], dtype=torch.float32)
            out = dict(combo=c, fold=f, seed=seed)
            with torch.no_grad():
                FA = KR.encode(enc, XA); torch.manual_seed(1000 + seed)
                sc = KR.fit_scorer(FA[:n_lab], ylt, head, kind='knn', energy_T=0.05, gamma=2.0)
                kr = KR.compute_kr(enc, head, sc, XA, lambda Z: vectorized_aug(Z, WEAK), n_lab)
                Kk, Rr = kr['K'], kr['R']; kap = KR.budget_v2(Kk, Rr, 0.25, 0.5); kap[:n_lab] = 1.0
                # calibration on LABELLED windows only
                f0l = FA[:n_lab]; dw = []
                for k in range(20):
                    torch.manual_seed(500 + k); dw.append(dcos(f0l, KR.encode(enc, vectorized_aug(XA[:n_lab], WEAK))))
                tau_f = float(np.quantile(np.concatenate(dw), Q))
                fn = F.normalize(f0l, dim=1); D = (1 - fn @ fn.T).numpy(); yl = y[lab]; il = rec[lab]
                same = (yl[:, None] == yl[None]) & (il[:, None] != il[None]); tau_c2 = float(np.median(D[same])) if same.any() else float('nan')
                out['tau_f_C1'] = tau_f; out['tau_f_C2'] = tau_c2
                B_, C_, M_, L_ = MA.shape; u = torch.randn(B_, C_, M_, 1, generator=torch.Generator().manual_seed(seed + 20000)).clamp(-2.0, 2.0)
                unl_m = np.arange(N) >= n_lab
                zone = dict(rel_known=(Kk.numpy() >= 0.5) & (Rr.numpy() >= 0.5), rel_unknown=(Kk.numpy() < 0.5) & (Rr.numpy() >= 0.5), uncertain=(Rr.numpy() < 0.5))
                kt = kap.numpy(); lo, hi = np.percentile(kt[unl_m], [100 / 3, 200 / 3])
                out['gains'] = {}
                for g in GAINS:
                    xv, fb, shr = vmd_view(XA, MA, kap, base_scale * g, tau_lsd, u); fbn = fb.numpy()
                    fv = KR.encode(enc, xv); df = dcos(FA, fv)
                    active = ~fbn
                    rej = active & (df > tau_f)                       # C1 rule
                    rej_c2 = active & (df > tau_c2) if tau_c2 == tau_c2 else np.zeros(N, bool)
                    xfinal = xv.clone(); xfinal[torch.as_tensor(rej)] = XA[torch.as_tensor(rej)]
                    # integrity (the rule is executed as written)
                    integ = bool(np.isfinite(df).all() and np.isfinite(tau_f)
                                 and torch.equal(xfinal[torch.as_tensor(rej)], XA[torch.as_tensor(rej)])
                                 and torch.equal(xfinal[torch.as_tensor(active & ~rej)], xv[torch.as_tensor(active & ~rej)])
                                 and torch.equal(xv[torch.as_tensor(fbn)], XA[torch.as_tensor(fbn)]) and not (rej & fbn).any())
                    ff = KR.encode(enc, xfinal); df_final = dcos(FA, ff)
                    Ku, Kf = KR.known_score(sc, fv), KR.known_score(sc, ff); K0 = Kk
                    flip_unchecked = ((Ku >= 0.5) != (K0 >= 0.5)).numpy(); flip_final = ((Kf >= 0.5) != (K0 >= 0.5)).numpy()
                    ra = relchg(xv, XA)
                    sel = unl_m & active
                    gd = dict(integrity=integ, frac_logspec_fallback_all=float(fbn.mean()), frac_logspec_fallback_unl=float(fbn[unl_m].mean()),
                              reject_rate_unl=float(rej[sel].mean()) if sel.any() else float('nan'), reject_rate_lab=float(rej[(~unl_m) & active].mean()) if ((~unl_m) & active).any() else float('nan'),
                              reject_rate_C2_unl=float(rej_c2[sel].mean()) if sel.any() else float('nan'),
                              by_kappa_tercile=[float(rej[sel & (kt <= lo)].mean()) if (sel & (kt <= lo)).any() else None, float(rej[sel & (kt > lo) & (kt < hi)].mean()) if (sel & (kt > lo) & (kt < hi)).any() else None, float(rej[sel & (kt >= hi)].mean()) if (sel & (kt >= hi)).any() else None],
                              by_zone={k: (float(rej[sel & v].mean()) if (sel & v).any() else None) for k, v in zone.items()},
                              by_truth=dict(known=float(rej[sel & ~unk_all].mean()) if (sel & ~unk_all).any() else None, unknown=float(rej[sel & unk_all].mean()) if (sel & unk_all).any() else None),
                              signal_change_rejected=float(np.median(ra[sel & rej])) if (sel & rej).any() else None, signal_change_kept=float(np.median(ra[sel & ~rej])) if (sel & ~rej).any() else None,
                              feat_change_rejected=float(np.median(df[sel & rej])) if (sel & rej).any() else None, feat_change_kept=float(np.median(df[sel & ~rej])) if (sel & ~rej).any() else None,
                              feat_change_unchecked_med=float(np.median(df[sel])), feat_change_final_med=float(np.median(df_final[sel])),
                              route_flip_unchecked=float(flip_unchecked[sel].mean()), route_flip_final=float(flip_final[sel].mean()), tau_f=tau_f)
                    out['gains'][str(g)] = gd
            rows.append(out)
            g2 = out['gains']['2.0']; g4 = out['gains']['4.0']
            print(c, f, seed, 'tau_f %.4f' % tau_f, '| reject g2 %.3f g4 %.3f | integrity %s %s | fb g2 %.3f' % (g2['reject_rate_unl'], g4['reject_rate_unl'], g2['integrity'], g4['integrity'], g2['frac_logspec_fallback_all']), flush=True)
json.dump(rows, open('/projects/sandbox/results_v24/v25_step1.json', 'w'), indent=1)

n = len(rows); G = lambda r, g: r['gains'][str(g)]
integ = sum(G(r, 2.0)['integrity'] and G(r, 4.0)['integrity'] for r in rows)
rr2 = np.array([G(r, 2.0)['reject_rate_unl'] for r in rows]); rr4 = np.array([G(r, 4.0)['reject_rate_unl'] for r in rows])
oper_m = int(((rr2 >= 0.05) & (rr2 <= 0.60)).sum()); inert = int((rr2 < 0.05).sum()); over = int((rr2 > 0.60).sum())
per_split = rr2.reshape(9, 3).mean(1); extreme = int(((per_split > 0.90) | (per_split < 0.01)).sum())
print(f'\nINTEGRITY {integ}/27 (needs 27)')
print(f'gain 2 rejection rate: within 5-60 % in {oper_m}/27; < 5 % in {inert}/27; > 60 % in {over}/27 (each rule needs >= 23)')
print('per-split mean rejection rate at gain 2:', np.round(per_split, 3).tolist(), ' extreme splits (>0.90 or <0.01):', extreme, '(needs <= 3)')
operates = oper_m >= 23 and extreme <= 3
verdict = 'INTEGRITY FAILED' if integ < 27 else ('OPERATES' if operates else ('INERT' if inert >= 23 else ('OVER-REJECT' if over >= 23 else 'INCONCLUSIVE')))
print('VERDICT (pre-registered):', verdict)
m = lambda fn: float(np.nanmean([fn(r) for r in rows])); rg = lambda fn: (round(float(np.nanmin([fn(r) for r in rows])), 3), round(float(np.nanmax([fn(r) for r in rows])), 3))
print('\nmean over 27 models (range)')
print('tau_f (C1) %.4f %s   C2 median %.4f %s' % (m(lambda r: r['tau_f_C1']), rg(lambda r: r['tau_f_C1']), m(lambda r: r['tau_f_C2']), rg(lambda r: r['tau_f_C2'])))
for g in GAINS:
    print(f'--- gain {g:g}')
    for k in ('frac_logspec_fallback_all', 'reject_rate_unl', 'reject_rate_lab', 'reject_rate_C2_unl', 'signal_change_rejected', 'signal_change_kept', 'feat_change_rejected', 'feat_change_kept',
              'feat_change_unchecked_med', 'feat_change_final_med', 'route_flip_unchecked', 'route_flip_final'):
        print(f'  {k:30s} {m(lambda r, k=k, g=g: G(r, g)[k] if G(r, g)[k] is not None else np.nan):8.4f} {rg(lambda r, k=k, g=g: G(r, g)[k] if G(r, g)[k] is not None else np.nan)}')
    for nm, ix in (('kappa tercile low', 0), ('kappa tercile mid', 1), ('kappa tercile high', 2)):
        print(f'  reject by {nm:20s} {m(lambda r, ix=ix, g=g: G(r, g)["by_kappa_tercile"][ix] if G(r, g)["by_kappa_tercile"][ix] is not None else np.nan):8.4f}')
    for z in ('rel_known', 'rel_unknown', 'uncertain'):
        print(f'  reject in zone {z:20s} {m(lambda r, z=z, g=g: G(r, g)["by_zone"][z] if G(r, g)["by_zone"][z] is not None else np.nan):8.4f}')
    for t in ('known', 'unknown'):
        print(f'  reject, truly {t:20s} {m(lambda r, t=t, g=g: G(r, g)["by_truth"][t] if G(r, g)["by_truth"][t] is not None else np.nan):8.4f}')
print('\nper-split mean rejection rate at gain 4:', np.round(rr4.reshape(9, 3).mean(1), 3).tolist())
