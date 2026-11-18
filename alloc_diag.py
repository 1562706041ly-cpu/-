"""v24a allocation diagnostic, part W (see results_v24/v24a_allocation_PREREGISTERED.txt). Run from scratch/v24p with PYTHONPATH=."""
import json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import kr_modules as KR
from gpu_runtime import vectorized_aug
torch.set_num_threads(4)
SPL = '/projects/sandbox/centrifugal/splits_ch1245/motor2_100_ch1245_%s_fold%s.npz'
BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r')
WEAK = dict(shift=64, scale=0.05, noise=0.02, mask=0.0)
GAINS = [2.0, 4.0]; NDRAW = 20

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
    B, C, M, L = modes.shape
    expo = 0.5 * u * kappa.reshape(-1, 1, 1, 1) * scale.reshape(1, C, M, 1)
    out = x.clone(); ok = torch.zeros(B, dtype=torch.bool); shrunk = torch.zeros(B, dtype=torch.bool)
    for s in range(4):
        cand = x + ((torch.exp(expo) - 1.0) * modes).sum(2)
        lsd = KR.torch_lsd(cand.reshape(B * C, L), x.reshape(B * C, L)).reshape(B, C)
        good = (lsd <= tau[None]).all(1) & torch.isfinite(cand).flatten(1).all(1) & ~ok
        out[good] = cand[good]; ok |= good; shrunk[good & (s > 0)] = True
        if ok.all(): break
        expo = expo * 0.5
    return out, shrunk

relchg = lambda v, x: ((v - x).flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1)).numpy()
cosd = lambda a, b: (1 - F.cosine_similarity(a, b, dim=1)).numpy()
def tercile_ratio(kap, val):
    lo, hi = np.percentile(kap, [100 / 3, 200 / 3]); b = kap <= lo; t = kap >= hi
    return float(np.median(val[t]) / max(np.median(val[b]), 1e-12))
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
            base_scale = torch.tensor(st['sigma_rec'] * st['g'], dtype=torch.float32); tau = torch.tensor([p['tau_lsd'] for p in per], dtype=torch.float32)
            out = dict(combo=c, fold=f, seed=seed)
            with torch.no_grad():
                FA = KR.encode(enc, XA); torch.manual_seed(1000 + seed)
                sc = KR.fit_scorer(FA[:n_lab], ylt, head, kind='knn', energy_T=0.05, gamma=2.0)
                kr = KR.compute_kr(enc, head, sc, XA, lambda Z: vectorized_aug(Z, WEAK), n_lab)
                Kk, Rr = kr['K'], kr['R']
                kap = KR.budget_v2(Kk, Rr, 0.25, 0.5); kap[:n_lab] = 1.0
                # shuffled kappa exactly as refresh_kr_state
                shuf_rng = np.random.default_rng(seed + 40000); perm = np.arange(N)
                for ids in (np.arange(n_lab), np.arange(n_lab, N)): perm[ids] = shuf_rng.permutation(ids)
                kaps = kap[torch.as_tensor(perm)]
                ku = kap[n_lab:].numpy(); ksu = kaps[n_lab:].numpy(); Ku = Kk[n_lab:].numpy(); Ru = Rr[n_lab:].numpy(); unk = unk_all[n_lab:]
                zone = dict(rel_known=(Ku >= 0.5) & (Ru >= 0.5), rel_unknown=(Ku < 0.5) & (Ru >= 0.5), uncertain=(Ru < 0.5))
                out['W1'] = dict(zone_frac={k: float(v.mean()) for k, v in zone.items()}, zone_mean_kappa={k: (float(ku[v].mean()) if v.any() else None) for k, v in zone.items()},
                                 kappa_p10=float(np.percentile(ku, 10)), kappa_p50=float(np.percentile(ku, 50)), kappa_p90=float(np.percentile(ku, 90)),
                                 p90_over_p10=float(np.percentile(ku, 90) / max(np.percentile(ku, 10), 1e-9)),
                                 plateau_share={str(lv): float((np.abs(ku - lv) <= 0.05).mean()) for lv in (1.0, 0.625, 0.25)})
                out['W2'] = dict(spearman=float(spearmanr(ku, ksu)[0]), mean_abs_diff=float(np.abs(ku - ksu).mean()), frac_ge_0p1=float((np.abs(ku - ksu) >= 0.1).mean()))
                out['W5'] = dict(mean_kappa_known=float(ku[~unk].mean()), mean_kappa_unknown=float(ku[unk].mean()), auroc_kappa_known_vs_unknown=float(roc_auc_score(~unk, ku)),
                                 unknown_share_in_zone={k: float(unk[v].mean()) if v.any() else None for k, v in zone.items()})
                f0 = KR.encode(enc, XA)
                B_, C_, M_, L_ = MA.shape; u = torch.randn(B_, C_, M_, 1, generator=torch.Generator().manual_seed(seed + 20000)).clamp(-2.0, 2.0)
                sub = np.arange(n_lab, N)[::4]
                out['gains'] = {}
                for g in GAINS:
                    sca = base_scale * g
                    xa, sha = vmd_view(XA, MA, kap, sca, tau, u); xs, shs = vmd_view(XA, MA, kaps, sca, tau, u)
                    ra, rs = relchg(xa, XA)[n_lab:], relchg(xs, XA)[n_lab:]
                    da, ds = cosd(f0, KR.encode(enc, xa))[n_lab:], cosd(f0, KR.encode(enc, xs))[n_lab:]
                    # per-window noise: independent u draws, adaptive kappa
                    logs = []
                    for k in range(NDRAW):
                        uk = torch.randn(len(sub), C_, M_, 1, generator=torch.Generator().manual_seed(900000 + 97 * k + seed)).clamp(-2.0, 2.0)
                        xv, _ = vmd_view(XA[sub], MA[sub], kap[sub], sca, tau, uk); logs.append(np.log(np.maximum(relchg(xv, XA[sub]), 1e-9)))
                    noise_sd = float(np.median(np.std(np.stack(logs), axis=0, ddof=1)))
                    tr_sig = tercile_ratio(ku, ra); tr_sig_s = tercile_ratio(ksu, rs); tr_f = tercile_ratio(ku, da); tr_f_s = tercile_ratio(ksu, ds)
                    out['gains'][str(g)] = dict(
                        spearman_kappa_signal_adaptive=float(spearmanr(ku, ra)[0]), spearman_kappa_signal_shuffled=float(spearmanr(ksu, rs)[0]),
                        spearman_kappaAdaptive_vs_signal_in_shuffled_arm=float(spearmanr(ku, rs)[0]),
                        tercile_ratio_signal_adaptive=tr_sig, tercile_ratio_signal_shuffled=tr_sig_s, median_signal_adaptive=float(np.median(ra)), median_signal_shuffled=float(np.median(rs)),
                        noise_sd_log_signal=noise_sd, budget_snr=float(abs(np.log(tr_sig)) / max(noise_sd, 1e-9)),
                        spearman_kappa_feature_adaptive=float(spearmanr(ku, da)[0]), spearman_kappa_feature_shuffled=float(spearmanr(ksu, ds)[0]),
                        tercile_ratio_feature_adaptive=tr_f, tercile_ratio_feature_shuffled=tr_f_s, frac_shrunk_adaptive=float(sha[n_lab:].float().mean()), frac_shrunk_shuffled=float(shs[n_lab:].float().mean()))
            rows.append(out); print(c, f, seed, 'p90/p10 %.2f' % out['W1']['p90_over_p10'], 'mean|dk| %.3f' % out['W2']['mean_abs_diff'],
                                    'terc sig g2 %.2f g4 %.2f' % (out['gains']['2.0']['tercile_ratio_signal_adaptive'], out['gains']['4.0']['tercile_ratio_signal_adaptive']), flush=True)
json.dump(rows, open('/projects/sandbox/results_v24/v24a_allocation_W.json', 'w'), indent=1)

n = len(rows); cnt = lambda fn: int(sum(fn(r) for r in rows)); G = lambda r, g: r['gains'][str(g)]
spread = cnt(lambda r: r['W1']['p90_over_p10'] >= 1.5 and max(r['W1']['plateau_share'].values()) <= 0.60)
differs = cnt(lambda r: r['W2']['mean_abs_diff'] >= 0.10)
dec = {g: cnt(lambda r: abs(G(r, g)['spearman_kappaAdaptive_vs_signal_in_shuffled_arm']) < 0.15) for g in GAINS}
ts = {g: cnt(lambda r: G(r, g)['tercile_ratio_signal_adaptive'] >= 1.5) for g in GAINS}
tf = {g: cnt(lambda r: G(r, g)['tercile_ratio_feature_adaptive'] >= 1.5) for g in GAINS}
print('\nmodels', n)
print(f'SPREAD {spread}/27   DIFFERS {differs}/27   (each needs >= 23)')
for g in GAINS: print(f'gain {g:g}: DECOUPLED {dec[g]}/27  TRANSMITTED_SIGNAL {ts[g]}/27  TRANSMITTED_FEATURE {tf[g]}/27')
mean = lambda fn: float(np.nanmean([fn(r) for r in rows])); rng_ = lambda fn: (round(float(np.nanmin([fn(r) for r in rows])), 3), round(float(np.nanmax([fn(r) for r in rows])), 3))
def show(name, fn): print(f'{name:58s} mean {mean(fn):8.3f}  range {rng_(fn)}')
show('W1 kappa p10', lambda r: r['W1']['kappa_p10']); show('W1 kappa p50', lambda r: r['W1']['kappa_p50']); show('W1 kappa p90', lambda r: r['W1']['kappa_p90']); show('W1 p90/p10', lambda r: r['W1']['p90_over_p10'])
for z in ('rel_known', 'rel_unknown', 'uncertain'): show(f'W1 zone fraction {z}', lambda r, z=z: r['W1']['zone_frac'][z])
for z in ('rel_known', 'rel_unknown', 'uncertain'): show(f'W1 mean kappa in zone {z}', lambda r, z=z: r['W1']['zone_mean_kappa'][z] if r['W1']['zone_mean_kappa'][z] is not None else np.nan)
for lv in ('1.0', '0.625', '0.25'): show(f'W1 share within 0.05 of {lv}', lambda r, lv=lv: r['W1']['plateau_share'][lv])
show('W2 Spearman(kappa, kappa_shuf)', lambda r: r['W2']['spearman']); show('W2 mean |kappa - kappa_shuf|', lambda r: r['W2']['mean_abs_diff']); show('W2 frac |diff| >= 0.1', lambda r: r['W2']['frac_ge_0p1'])
show('W5 mean kappa known', lambda r: r['W5']['mean_kappa_known']); show('W5 mean kappa unknown', lambda r: r['W5']['mean_kappa_unknown']); show('W5 AUROC kappa known-vs-unknown', lambda r: r['W5']['auroc_kappa_known_vs_unknown'])
for z in ('rel_known', 'rel_unknown', 'uncertain'): show(f'W5 unknown share in zone {z}', lambda r, z=z: r['W5']['unknown_share_in_zone'][z] if r['W5']['unknown_share_in_zone'][z] is not None else np.nan)
for g in GAINS:
    print(f'--- gain {g:g}')
    for k in ('spearman_kappa_signal_adaptive', 'spearman_kappa_signal_shuffled', 'spearman_kappaAdaptive_vs_signal_in_shuffled_arm', 'tercile_ratio_signal_adaptive', 'tercile_ratio_signal_shuffled',
              'median_signal_adaptive', 'median_signal_shuffled', 'noise_sd_log_signal', 'budget_snr', 'spearman_kappa_feature_adaptive', 'spearman_kappa_feature_shuffled',
              'tercile_ratio_feature_adaptive', 'tercile_ratio_feature_shuffled', 'frac_shrunk_adaptive', 'frac_shrunk_shuffled'):
        show('W3/4 ' + k, lambda r, k=k, g=g: G(r, g)[k])
