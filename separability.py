"""v26: unknown-class separability under the VMD view (criteria: results_v24/v26_unknown_separability_PREREGISTERED.txt). Run from scratch/v24p with PYTHONPATH=."""
import json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import kr_modules as KR
from gpu_runtime import vectorized_aug
torch.set_num_threads(4)
SPL = '/projects/sandbox/centrifugal/splits_ch1245/motor2_100_ch1245_%s_fold%s.npz'
BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r')
WEAK = dict(shift=64, scale=0.05, noise=0.02, mask=0.0); STD = dict(shift=256, scale=0.2, noise=0.1, mask=0.10)
GAINS = [1.0, 2.0, 4.0]; NPOS = NNEG = 100000

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
    out = x.clone(); ok = torch.zeros(B, dtype=torch.bool)
    for s in range(4):
        cand = x + ((torch.exp(expo) - 1.0) * modes).sum(2)
        lsd = KR.torch_lsd(cand.reshape(B * C, L), x.reshape(B * C, L)).reshape(B, C)
        good = (lsd <= tau[None]).all(1) & torch.isfinite(cand).flatten(1).all(1) & ~ok
        out[good] = cand[good]; ok |= good
        if ok.all(): break
        expo = expo * 0.5
    return out

def sample_pairs(cls, meas, rng):
    cells = {}
    for i, (c, m) in enumerate(zip(cls, meas)): cells.setdefault((c, m), []).append(i)
    keys = list(cells); arr = [np.array(cells[k]) for k in keys]; kc = np.array([k[0] for k in keys])
    by_class_other = {c: [j for j, k in enumerate(keys) if k[0] == c] for c in np.unique(cls)}
    def pick(n, positive):
        a = np.empty(n, int); b = np.empty(n, int); c1 = rng.integers(0, len(keys), n)
        for t in range(n):
            j = c1[t]; a[t] = arr[j][rng.integers(0, len(arr[j]))]
            if positive: cand = [q for q in by_class_other[kc[j]] if q != j]
            else: cand = [q for q in range(len(keys)) if kc[q] != kc[j]]
            q = cand[rng.integers(0, len(cand))]; b[t] = arr[q][rng.integers(0, len(arr[q]))]
        return a, b
    # precompute candidate lists per cell for speed
    return pick

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
            a[sel] = arr[k][rng.integers(0, len(arr[k]), len(sel))]
            q = cand[k][rng.integers(0, len(cand[k]), len(sel))]
            for qq in np.unique(q):
                s2 = sel[q == qq]; b[s2] = arr[qq][rng.integers(0, len(arr[qq]), len(s2))]
        return a, b
    pa, pb = draw(NPOS, pos_c); na, nb = draw(NNEG, neg_c)
    return np.concatenate([pa, na]), np.concatenate([pb, nb]), np.concatenate([np.ones(NPOS), np.zeros(NNEG)])

def metrics(Fz, cls, meas, ia, ib, lab):
    Fn = F.normalize(Fz, dim=1)
    p1 = float(roc_auc_score(lab, (Fn[ia] * Fn[ib]).sum(1).numpy()))
    S = (Fn @ Fn.T).numpy(); same_m = meas[:, None] == meas[None]; S[same_m] = -9.0
    nn_idx = np.argsort(-S, axis=1)[:, :10]; pur = (cls[nn_idx] == cls[:, None]).mean(1)
    p2 = float(np.mean([pur[cls == c].mean() for c in np.unique(cls)]))
    cents = {c: F.normalize(Fn[torch.as_tensor(cls == c)].mean(0), dim=0) for c in np.unique(cls)}
    within = float(np.mean([(1 - (Fn[torch.as_tensor(cls == c)] @ cents[c]).numpy()).mean() for c in cents]))
    cl = list(cents); between = float(np.mean([1 - float(cents[a] @ cents[b]) for i, a in enumerate(cl) for b in cl[i + 1:]]))
    return p1, p2, within, between, within / max(between, 1e-12)

def group_stats(E, y, rec, ddof):
    cm, sr = [], []
    for c in np.unique(y):
        m = y == c; cm.append(E[m].mean(0))
        if len(np.unique(rec[m])) >= 2: sr.append(np.stack([E[m & (rec == r)].mean(0) for r in np.unique(rec[m])]).std(0))
        else: sr.append(E[m].std(0))
    return np.mean(sr, 0), np.stack(cm).std(0, ddof=ddof), sum(len(np.unique(rec[y == c])) < 2 for c in np.unique(y))

rows = []; mi = 0
for c in ('C1', 'C2', 'C3'):
    for f in (1, 2, 3):
        d = np.load(SPL % (c, f), allow_pickle=False)
        X = zn(np.asarray(d['X_train'], np.float32)); y = d['y_train']; pi = np.asarray(d['train_pool_index'], np.int64)
        modes_all = torch.tensor(np.asarray(BANK[pi], np.float32)); Xt = torch.tensor(X)
        inst = np.asarray(d['train_installation']).astype(str); measall = np.char.add(np.char.add(inst, '|'), np.asarray(d['train_measurement']).astype(str)); known = d['known_labels'].tolist()
        for seed in (0, 1, 2):
            rd = f'runs_paderborn/P_{c}_f{f}_seed{seed}'
            r = list(json.load(open(f'{rd}/results_exp2.json', encoding='utf-8')).values())[0]
            lab = np.array(r['labeled_idx']); unl = np.setdiff1d(np.arange(len(y)), lab); order = np.concatenate([lab, unl]); n_lab = len(lab); N = len(order)
            ck = torch.load(f'{rd}/m2_PADERBORN_seed{seed}_warmup.pt', map_location='cpu', weights_only=False)
            enc = SpecEnc(X.shape[1], ck['encoder']['mu'], ck['encoder']['sd']); enc.load_state_dict(ck['encoder']); enc.eval()
            head = nn.Linear(64, len(known)); head.load_state_dict(ck['cls_head'])
            XA = Xt[order]; MA = modes_all[order]; ylt = torch.tensor([known.index(v) for v in y[lab]])
            ml = modes_all[torch.as_tensor(lab)].numpy()
            per = [KR.nuisance_mode_stats(ml[:, ch], X[lab][:, ch], y[lab], inst[lab]) for ch in range(ml.shape[1])]
            st = {k: np.stack([p[k] for p in per]) for k in ('sigma_rec', 'sigma_cls', 'g')}
            base_scale = torch.tensor(st['sigma_rec'] * st['g'], dtype=torch.float32); tau_lsd = torch.tensor([p['tau_lsd'] for p in per], dtype=torch.float32)
            out = dict(combo=c, fold=f, seed=seed)
            with torch.no_grad():
                FA = KR.encode(enc, XA); torch.manual_seed(1000 + seed)
                sc = KR.fit_scorer(FA[:n_lab], ylt, head, kind='knn', energy_T=0.05, gamma=2.0)
                kr = KR.compute_kr(enc, head, sc, XA, lambda Z: vectorized_aug(Z, WEAK), n_lab)
                kap = KR.budget_v2(kr['K'], kr['R'], 0.25, 0.5); kap[:n_lab] = 1.0
                B_, C_, M_, L_ = MA.shape; u = torch.randn(B_, C_, M_, 1, generator=torch.Generator().manual_seed(seed + 20000)).clamp(-2.0, 2.0)
                uidx = np.where(~np.isin(y[order], known))[0]                    # unknown windows (labels for analysis only)
                cls_u = y[order][uidx]; meas_u = measall[order][uidx]; inst_u = inst[order][uidx]
                Xu, Mu, ku, uu = XA[uidx], MA[uidx], kap[uidx], u[uidx]
                rng = np.random.default_rng(777 + 31 * mi); ia, ib, plab = pairs_fast(cls_u, meas_u, rng)
                assert (meas_u[ia] != meas_u[ib]).all() and ((cls_u[ia] == cls_u[ib]) == (plab == 1)).all()
                out['recordings_per_unknown_class'] = {int(cc): int(len(np.unique(meas_u[cls_u == cc]))) for cc in np.unique(cls_u)}
                enc_ = lambda Z: KR.encode(enc, Z)
                torch.manual_seed(7); VA = enc_(vectorized_aug(Xu, STD))
                nz = []
                mA = metrics(VA, cls_u, meas_u, ia, ib, plab)
                for s_ in (8, 9, 10):
                    torch.manual_seed(s_); nz.append(metrics(enc_(vectorized_aug(Xu, STD)), cls_u, meas_u, ia, ib, plab))
                out['VA'] = dict(P1=mA[0], P2=mA[1], within=mA[2], between=mA[3], ratio=mA[4]); out['V0'] = dict(P1=metrics(enc_(Xu), cls_u, meas_u, ia, ib, plab)[0])
                out['noise_P1'] = [m_[0] - mA[0] for m_ in nz]; out['noise_mag'] = float(np.mean(np.abs(out['noise_P1'])))
                out['noise_P2'] = [m_[1] - mA[1] for m_ in nz]
                out['gains'] = {}
                for g in GAINS:
                    xv = vmd_view(Xu, Mu, ku, base_scale * g, tau_lsd, uu)
                    torch.manual_seed(7); mV = metrics(enc_(vectorized_aug(xv, STD)), cls_u, meas_u, ia, ib, plab)
                    gd = dict(P1=mV[0], P2=mV[1], within=mV[2], between=mV[3], ratio=mV[4], dP1=mV[0] - mA[0], dP2=mV[1] - mA[1], dratio=mV[4] - mA[4])
                    if g == 2.0:
                        gd['per_channel_dP1'] = []
                        for ch in range(C_):
                            xc = Xu.clone(); xc[:, ch] = xv[:, ch]
                            torch.manual_seed(7); gd['per_channel_dP1'].append(metrics(enc_(vectorized_aug(xc, STD)), cls_u, meas_u, ia, ib, plab)[0] - mA[0])
                    out['gains'][str(g)] = gd
                # installation-different secondary metric (classes with >= 2 installations in the training split)
                ok_cls = [cc for cc in np.unique(cls_u) if len(np.unique(inst_u[cls_u == cc])) >= 2]
                out['classes_with_2_installations'] = [int(v) for v in ok_cls]; out['classes_total'] = [int(v) for v in np.unique(cls_u)]
                if ok_cls:
                    sel = np.isin(cls_u, ok_cls) & True; ii = np.where(np.isin(cls_u[ia], ok_cls) & np.isin(cls_u[ib], ok_cls) & (inst_u[ia] != inst_u[ib]) & ((plab == 0) | (cls_u[ia] == cls_u[ib])))[0]
                    if (plab[ii] == 1).any() and (plab[ii] == 0).any():
                        xv2 = vmd_view(Xu, Mu, ku, base_scale * 2.0, tau_lsd, uu)
                        torch.manual_seed(7); fa2 = F.normalize(enc_(vectorized_aug(Xu, STD)), dim=1); torch.manual_seed(7); fv2 = F.normalize(enc_(vectorized_aug(xv2, STD)), dim=1)
                        aa = roc_auc_score(plab[ii], (fa2[ia[ii]] * fa2[ib[ii]]).sum(1).numpy()); av = roc_auc_score(plab[ii], (fv2[ia[ii]] * fv2[ib[ii]]).sum(1).numpy())
                        out['dP1_diff_installation_g2'] = float(av - aa); out['n_pairs_diff_installation'] = int(len(ii))
            # part B: exposure (labels for analysis only)
            Eall = np.stack([KR.log_mode_energy(modes_all[:, ch].numpy()) for ch in range(C_)], 0)       # (C, N, M)
            kn_m = np.isin(y, known); un_m = ~kn_m
            Dk, Du = np.zeros((C_, M_)), np.zeros((C_, M_)); nrec_fallback = [0, 0]
            for ch in range(C_):
                sr_k, sc_k, nf_k = group_stats(Eall[ch][kn_m], y[kn_m], inst[kn_m], 1); sr_u, sc_u, nf_u = group_stats(Eall[ch][un_m], y[un_m], inst[un_m], 1)
                Dk[ch] = sc_k ** 2 / (sr_k ** 2 + sc_k ** 2 + 1e-12); Du[ch] = sc_u ** 2 / (sr_u ** 2 + sc_u ** 2 + 1e-12); nrec_fallback[0] += nf_k; nrec_fallback[1] += nf_u
            scale = base_scale.numpy(); Dcode = 1.0 - st['g']
            denom = float(scale.sum()); nk, nu, nc = float((Dk * scale).sum()), float((Du * scale).sum()), float((Dcode * scale).sum())
            out['B'] = dict(exposure_known=nk / denom, exposure_unknown=nu / denom, exposure_known_code=nc / denom, num_known=nk, num_unknown=nu, denominator=denom,
                            spearman_scale_Dunk=float(spearmanr(scale.ravel(), Du.ravel())[0]), spearman_scale_Dknown=float(spearmanr(scale.ravel(), Dk.ravel())[0]),
                            classes_with_record_fallback_known_unknown=nrec_fallback, Dknown_full=Dk.tolist(), Dunk=Du.tolist(), scale=scale.tolist())
            rows.append(out); mi += 1
            g2 = out['gains']['2.0']
            print(c, f, seed, 'P1 VA %.4f | d(g2) %+.4f noise_mag %.4f | exposure unk %.3f known %.3f' % (out['VA']['P1'], g2['dP1'], out['noise_mag'], out['B']['exposure_unknown'], out['B']['exposure_known']), flush=True)
json.dump(rows, open('/projects/sandbox/results_v24/v26_separability.json', 'w'), indent=1)

n = len(rows); G = lambda r, g: r['gains'][str(g)]
dg = np.array([G(r, 2.0)['dP1'] for r in rows]); nm = np.array([r['noise_mag'] for r in rows])
degr = int((dg <= -(nm + 0.01)).sum()); impr = int((dg >= (nm + 0.01)).sum()); nochg = int((np.abs(dg) < 0.01).sum())
ek = np.array([r['B']['exposure_known'] for r in rows]); eu = np.array([r['B']['exposure_unknown'] for r in rows])
expo = int(((eu >= 1.5 * ek) & ((eu - ek) >= 0.05)).sum())
print(f'\nmodels {n}')
print(f'DEGRADES {degr}/27   IMPROVES {impr}/27   NO-CHANGE {nochg}/27   (each needs >= 23)   EXPOSURE {expo}/27 (needs >= 23)')
verdictP = 'DEGRADES' if degr >= 23 else ('IMPROVES' if impr >= 23 else ('NO-CHANGE' if nochg >= 23 else 'INCONCLUSIVE'))
print('P1 verdict:', verdictP, '| exposure:', 'EXPOSURE' if expo >= 23 else 'not EXPOSURE')
per_split = dg.reshape(9, 3).mean(1); print('per-split mean dP1 at gain 2:', np.round(per_split, 4).tolist())
m = lambda fn: float(np.nanmean([fn(r) for r in rows])); rg = lambda fn: (round(float(np.nanmin([fn(r) for r in rows])), 4), round(float(np.nanmax([fn(r) for r in rows])), 4))
print('\nmean over 27 models (range)')
print('P1 VA %.4f %s | V0 %.4f | noise_mag %.4f %s' % (m(lambda r: r['VA']['P1']), rg(lambda r: r['VA']['P1']), m(lambda r: r['V0']['P1']), m(lambda r: r['noise_mag']), rg(lambda r: r['noise_mag'])))
for g in GAINS:
    print(f'gain {g:g}: dP1 {m(lambda r: G(r, g)["dP1"]):+.4f} {rg(lambda r: G(r, g)["dP1"])}  dP2 {m(lambda r: G(r, g)["dP2"]):+.4f}  within {m(lambda r: G(r, g)["within"]):.4f} between {m(lambda r: G(r, g)["between"]):.4f} ratio {m(lambda r: G(r, g)["ratio"]):.4f} (VA ratio {m(lambda r: r["VA"]["ratio"]):.4f})')
print('per-channel dP1 at gain 2 (channels 1,2,4,5):', np.round(np.mean([G(r, 2.0)['per_channel_dP1'] for r in rows], 0), 4).tolist())
print('part B: exposure_unknown %.4f %s | exposure_known(full) %.4f %s | exposure_known(code) %.4f | ratio mean %.3f | Spearman(scale,Dunk) %.3f | Spearman(scale,Dknown) %.3f' % (
      m(lambda r: r['B']['exposure_unknown']), rg(lambda r: r['B']['exposure_unknown']), m(lambda r: r['B']['exposure_known']), rg(lambda r: r['B']['exposure_known']), m(lambda r: r['B']['exposure_known_code']),
      np.mean(eu / ek), m(lambda r: r['B']['spearman_scale_Dunk']), m(lambda r: r['B']['spearman_scale_Dknown'])))
print('recordings per unknown class: min %d max %d' % (min(min(r['recordings_per_unknown_class'].values()) for r in rows), max(max(r['recordings_per_unknown_class'].values()) for r in rows)))
print('classes with < 2 records (known, unknown), mean per model:', np.round(np.mean([r['B']['classes_with_record_fallback_known_unknown'] for r in rows], 0), 2).tolist())
dd = [r['dP1_diff_installation_g2'] for r in rows if 'dP1_diff_installation_g2' in r]
print('different-installation secondary (gain 2): models with data %d/27, mean dP1 %.4f' % (len(dd), np.mean(dd) if dd else float('nan')), '| models where some unknown class has < 2 installations: %d' % sum(len(r['classes_with_2_installations']) < len(r['classes_total']) for r in rows))
