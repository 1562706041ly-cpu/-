"""Design information only (labelled windows, no unknown labels, no rejection counts): scale of candidate feature-stability bounds."""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import kr_modules as KR
from gpu_runtime import vectorized_aug
torch.set_num_threads(4)

SPL = '/projects/sandbox/centrifugal/splits_ch1245/motor2_100_ch1245_%s_fold%s.npz'
WEAK = dict(shift=64, scale=0.05, noise=0.02, mask=0.0)
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
out = []
for c in ('C1', 'C2', 'C3'):
    for f in (1, 2, 3):
        d = np.load(SPL % (c, f), allow_pickle=False); X = zn(np.asarray(d['X_train'], np.float32)); y = d['y_train']
        inst = np.asarray(d['train_installation']).astype(str)
        for seed in (0,):
            rd = f'runs_paderborn/P_{c}_f{f}_seed{seed}'; r = list(json.load(open(f'{rd}/results_exp2.json', encoding='utf-8')).values())[0]
            lab = np.array(r['labeled_idx']); ck = torch.load(f'{rd}/m2_PADERBORN_seed{seed}_warmup.pt', map_location='cpu', weights_only=False)
            enc = SpecEnc(X.shape[1], ck['encoder']['mu'], ck['encoder']['sd']); enc.load_state_dict(ck['encoder']); enc.eval()
            XL = torch.tensor(X[lab])
            with torch.no_grad():
                f0 = KR.encode(enc, XL); dw = []
                for k in range(20):
                    torch.manual_seed(500 + k); dw.append((1 - F.cosine_similarity(f0, KR.encode(enc, vectorized_aug(XL, WEAK)), dim=1)).numpy())
                dw = np.concatenate(dw)
                fn = F.normalize(f0, dim=1); D = (1 - fn @ fn.T).numpy(); yl = y[lab]; il = inst[lab]
                same = (yl[:, None] == yl[None]) & (il[:, None] != il[None])
                dc = D[same]
            out.append((np.percentile(dw, 50), np.percentile(dw, 95), np.median(dc) if len(dc) else np.nan, np.percentile(dc, 10) if len(dc) else np.nan))
            print(c, f, seed, 'weak-view d: med %.4f p95 %.4f | same-class cross-installation d: median %.4f p10 %.4f' % out[-1], flush=True)
a = np.array(out); print('mean over 9 splits: weak med %.4f p95 %.4f | cross-installation median %.4f p10 %.4f' % tuple(np.nanmean(a, 0)))
