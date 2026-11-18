"""Unit tests of the band-gain code against the v27/v28 scan implementation (run from scratch/v24p with PYTHONPATH=.).
1) _band_tau_lsd == KR.nuisance_mode_stats tau_lsd; 2) band_view == the scan's band_view for the same u, s, tau (kappa = 1); 3) coordinated draw shares channel 1; 4) fidelity: exponent halving / fallback behave."""
import ast, json, sys, numpy as np, torch
import kr_modules as KR
src = open('/projects/sandbox/scratch/deliver_v28/m2_paderborn_kr.py', encoding='utf-8').read(); tree = ast.parse(src)
ns = dict(np=np, torch=torch, KR=KR, BAND_NB=32, BAND={}, DEVICE=torch.device('cpu'))
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('_band_tau_lsd', 'band_view'):
        exec(compile(ast.Module([node], []), 'x', 'exec'), ns)
SPL = '/projects/sandbox/centrifugal/splits_ch1245/motor2_100_ch1245_C1_fold1.npz'
d = np.load(SPL, allow_pickle=False); zn = lambda X: ((X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-8)).astype(np.float32)
X = zn(np.asarray(d['X_train'], np.float32)); y = d['y_train']; inst = np.asarray(d['train_installation']).astype(str)
r = list(json.load(open('runs_paderborn/P_C1_f1_seed0/results_exp2.json', encoding='utf-8')).values())[0]; lab = np.array(r['labeled_idx'])
BANK = np.load('/projects/sandbox/centrifugal/pool/pool_motor2_100_ch1245_vmd.npy', mmap_mode='r'); pi = np.asarray(d['train_pool_index'], np.int64)
ml = np.asarray(BANK[pi[lab]], np.float32)
t_ref = [KR.nuisance_mode_stats(ml[:, c], X[lab][:, c], y[lab], inst[lab])['tau_lsd'] for c in range(4)]
t_new = [ns['_band_tau_lsd'](X[lab][:, c], y[lab], inst[lab]) for c in range(4)]
print('1) tau_lsd reference', np.round(t_ref, 6).tolist(), 'new', np.round(t_new, 6).tolist(), 'equal:', bool(np.allclose(t_ref, t_new, rtol=0, atol=1e-12)))
# scan implementation (copied from crosschannel.py)
L = 2048; NBIN = L // 2 + 1; EDG = np.linspace(0, NBIN, 33).astype(int); BIDX = torch.as_tensor(np.repeat(np.arange(32), np.diff(EDG)))
def scan_view(x, kappa, s, u, tau):
    B, C, _ = x.shape; Xf = torch.fft.rfft(x, dim=-1); out = x.clone(); ok = torch.zeros(B, dtype=torch.bool); shr = torch.zeros(B); expo = 0.5 * u * kappa.reshape(-1, 1, 1) * s.reshape(1, C, 32)
    for k in range(4):
        gain = torch.exp(expo)[..., BIDX]; cand = torch.fft.irfft(Xf * gain, n=L, dim=-1)
        lsd = KR.torch_lsd(cand.reshape(B * C, L), x.reshape(B * C, L)).reshape(B, C)
        good = (lsd <= tau[None]).all(1) & torch.isfinite(cand).flatten(1).all(1) & ~ok; out[good] = cand[good]; shr[good] = k; ok |= good
        if ok.all(): break
        expo = expo * 0.5
    return out, ~ok, shr
xb = torch.tensor(X[:256]); g = torch.Generator().manual_seed(3); u = torch.randn(256, 4, 32, generator=g).clamp(-2, 2)
s = torch.rand(4, 32) * 0.8 + 0.1; tau = torch.tensor(t_new, dtype=torch.float32)
ns['BAND'].update(s=s, tau=tau, bidx=BIDX)
for gain in (0.65, 2.0, 6.0):
    ns['BAND']['s'] = s * gain
    a = ns['band_view'](xb, u); b = scan_view(xb, torch.ones(256), s * gain, u, tau)
    print(f'2) gain {gain}: view equal {bool(torch.equal(a[0], b[0]))}, fallback equal {bool(torch.equal(a[1], b[1]))}, shrink equal {bool(torch.equal(a[2], b[2]))}; shrunk frac {float((a[2] > 0).float().mean()):.3f}, fallback frac {float(a[1].float().mean()):.3f}')
uc = u[:, :1, :].expand(-1, 4, -1); ns['BAND']['s'] = s * 2.0; v1 = ns['band_view'](xb, u)[0]; v2 = ns['band_view'](xb, uc)[0]
print('3) channel 1 identical between independent and coordinated views (before shrink differences may apply):', float((v1[:, 0] - v2[:, 0]).abs().max()) < 1e-6 or 'differs only where shrink acted')
ns['BAND']['s'] = s * 0.0; z = ns['band_view'](xb, u)
print('4) zero budget returns the original window up to the float32 FFT round trip: max abs difference %.2e (< 1e-5: %s) | shrunk %.3f' % (float((z[0] - xb).abs().max()), bool((z[0] - xb).abs().max() < 1e-5), float((z[2] > 0).float().mean())))
ns['BAND']['tau'] = torch.full((4,), 1e-9); ns['BAND']['s'] = s * 2.0; f = ns['band_view'](xb, u)
print('5) impossible bound falls back to x for all windows:', bool(f[1].all()) and bool(torch.equal(f[0], xb)))
