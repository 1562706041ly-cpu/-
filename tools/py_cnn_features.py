"""用 PyTorch 复现 run_step3 中的 CNN（同结构、同划分、验证集早停），导出 fc1 特征。
仅用于在没有 MATLAB 的环境下验证 LSSVM 阶段；正式运行请用 run_step3_cnn_lssvm.m。
用法: python py_cnn_features.py imgs.npz out_prefix seed"""
import sys, numpy as np, torch, torch.nn as nn
src, outp, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(seed); np.random.seed(seed)
d = np.load(src); X = torch.tensor(d['X']).permute(0, 3, 1, 2).float(); y = d['y']
def split(y, r=(0.6, 0.2, 0.2), gap=2):
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; n = len(idx); a = int(r[0]*n); b = int(r[1]*n)
        tr += list(idx[:a-gap]); va += list(idx[a:a+b-gap]); te += list(idx[a+b:])
    return map(np.array, (tr, va, te))
itr, iva, ite = split(y)
mu = X[itr].mean((0, 2, 3), keepdim=True); sd = X[itr].std((0, 2, 3), keepdim=True) + 1e-6
Xn = (X - mu) / sd; Y = torch.tensor(y - 1)
nC = len(np.unique(y))
def blk(i, o): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
feat = nn.Sequential(*blk(3, 16), nn.MaxPool2d(2), *blk(16, 32), nn.MaxPool2d(2), *blk(32, 64),
                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 64))
head = nn.Sequential(nn.ReLU(), nn.Linear(64, nC))
net = nn.Sequential(feat, head); opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.CrossEntropyLoss()
best, bad, state = 1e9, 0, None
for ep in range(60):
    net.train(); perm = np.random.permutation(itr)
    for k in range(0, len(perm), 32):
        b = perm[k:k+32]; opt.zero_grad(); l = lossf(net(Xn[b]), Y[b]); l.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    with torch.no_grad(): vl = lossf(net(Xn[iva]), Y[iva]).item()
    if vl < best: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= 8: break
net.load_state_dict(state); net.eval()
with torch.no_grad():
    F = feat(Xn).numpy(); p = net(Xn[ite]).argmax(1).numpy() + 1
acc = (p == y[ite]).mean()
print(f"seed {seed}: epochs {ep+1}, CNN test acc {acc*100:.2f}%")
np.savetxt(outp + '_F.csv', F, delimiter=','); np.savetxt(outp + '_y.csv', y, fmt='%d')
np.savetxt(outp + '_split.csv', np.r_[np.c_[itr, 0*itr+1], np.c_[iva, 0*iva+2], np.c_[ite, 0*ite+3]] + [1, 0], fmt='%d', delimiter=',')
open(outp + '_cnnacc.txt', 'w').write(str(acc))
