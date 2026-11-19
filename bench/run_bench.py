"""基准测试：在加噪声 / 少样本 / 跨工况场景下比较 CNN、ResNet、ECA-ResNet（各自 + BKA-LSSVM）。

场景名格式：<图库>[_few<n>][+ord][+mix]
  图库：clean | snr10 | snr5 | snr0 | snr5_n2 | snr5_pink ...（对应 banks/20_0_<图库>.npz 和 30_2_<图库>.npz）
        训练、验证、测试都在同一噪声水平
  few<n>：每类只有 n 个带标签样本（75% 训练、25% 验证，中间留 2 个样本的间隔）
  +ord ：30-2 工况改用阶次归一化后的图库（30_2_<图库>_ord.npz，见 make_banks.py --order_ref）
  +mix ：多工况训练，20-0 和 30-2 两个工况各按时间顺序划分后合在一起训练（需要目标工况的标签）
  +adabn：无监督域自适应，测试前用 30-2 的无标签样本重新统计 BatchNorm（不需要目标工况标签）
  +dann ：无监督域自适应，训练时加入 30-2 的无标签样本和工况判别器（梯度反转，不需要目标工况标签）
每个场景都报告两个测试集：
  test  : 20-0 工况，每类时间上最后 40 个样本（与 v2 相同）
  cross : 30-2 工况。默认是全部 200x5 个样本（训练时完全没见过）；
  xtail : 30-2 工况每类时间上最后 40 个样本（共 200 个）。+mix/+adabn/+dann 用到了 30-2 前面的样本，
          所以比较跨工况效果时统一看 xtail

结果逐行追加到 CSV，中断后重新运行会自动跳过已完成的 (场景, 模型, 种子)。

本机（RTX 2060）用法：
  python run_bench.py --banks banks --device cuda
  python run_bench.py --banks banks --device cuda --scenarios snr0 --models cnn,eca_resnet --seeds 1-3
"""
import argparse, csv, os, time

def _threads(n):
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = str(n)

ap = argparse.ArgumentParser()
ap.add_argument('--banks', default='banks')
ap.add_argument('--out', default='results/bench.csv')
ap.add_argument('--device', default='cuda')
ap.add_argument('--threads', type=int, default=4, help='CPU 线程上限（防止笔记本卡顿）')
ap.add_argument('--scenarios', default='clean,snr10,snr0,snr-5,clean_few20,clean_few50')
ap.add_argument('--models', default='cnn,resnet,eca_resnet')
ap.add_argument('--seeds', default='1-10')
ap.add_argument('--no_lssvm', action='store_true')
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--patience', type=int, default=8)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--lr', type=float, default=1e-3)
args = ap.parse_args()
_threads(args.threads)

import numpy as np
import torch
import torch.nn as nn
from models import build, n_params
from lssvm import lssvm_stage
from tfr import jet

torch.set_num_threads(args.threads)
DEV = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
if DEV.type == 'cuda':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
JET = jet(256).astype(np.float32)
FIELDS = ['scenario', 'model', 'seed', 'n_params', 'epochs', 'train_s',
          'cnn_test_acc', 'cnn_test_f1', 'cnn_cross_acc', 'cnn_cross_f1',
          'lssvm_test_acc', 'lssvm_test_f1', 'lssvm_cross_acc', 'lssvm_cross_f1',
          'gam', 'sig2', 'cv_error', 'invalid', 'cnn_xtail_acc', 'lssvm_xtail_acc']


def parse_seeds(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def load_bank(level, cond):
    d = np.load(os.path.join(args.banks, f'{cond}_{level}.npz'))
    X = torch.from_numpy(JET[d['X']]).permute(0, 3, 1, 2).contiguous()   # N x 3 x 64 x 64
    return X, d['y'].astype(np.int64)


def split(y, few):
    """每类按时间顺序划分。完整：118/38/40（同 split_indices，边界留 2）；少样本：n 个带标签样本。"""
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if few:
            t = int(np.floor(0.75 * few + 0.5))
            tr += list(idx[:t]); va += list(idx[t + 2: few + 2])
        else:
            tr += list(idx[:118]); va += list(idx[120:158])
        te += list(idx[160:200])
    return map(np.array, (tr, va, te))


def macro_f1(y, p, nC):
    f = []
    for c in range(1, nC + 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


@torch.no_grad()
def infer(net, X, mu, sd):
    net.eval(); logits, feats = [], []
    for k in range(0, len(X), 256):
        xb = ((X[k:k + 256] - mu) / sd).to(DEV)
        f = net.features(xb); feats.append(f.cpu()); logits.append(net.head(f).cpu())
    return torch.cat(logits).numpy(), torch.cat(feats).numpy().astype(np.float64)


def parse_scn(scn):
    base, *mods = scn.split('+')
    level, _, few = base.partition('_few')
    return level, (int(few) if few else 0), set(mods)


def bank_keys(scn):
    level, _, mods = parse_scn(scn)
    return level, level + ('_ord' if 'ord' in mods else '')


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


@torch.no_grad()
def adabn(net, Xt, mu, sd):
    """AdaBN（Li et al. 2018）：不改权重，只用目标工况的无标签数据重新统计 BatchNorm 的均值和方差。"""
    import copy
    tnet = copy.deepcopy(net).eval()
    bns = [m for m in tnet.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()   # momentum=None：累计平均
    for k in range(0, len(Xt), 128):
        tnet.features(((Xt[k:k + 128] - mu) / sd).to(DEV))
    return tnet.eval()


def run_one(scn, model, seed, data):
    level, few, mods = parse_scn(scn)
    lv20, lv30 = bank_keys(scn)
    X, y = data[('20_0', lv20)]; Xc, yc = data[('30_2', lv30)]
    itr, iva, ite = split(y, few)
    ctr, cva, cte = split(yc, 0)
    tgt_unl = np.r_[ctr, cva]          # 目标工况的无标签样本（只在 +dann / +adabn 时使用，标签不参与）
    xtail = cte                        # 目标工况每类时间上最后 40 个样本，所有场景都在这上面报告 xtail
    if 'mix' in mods:                  # 多工况训练：30-2 的样本接在 20-0 后面，同样按时间顺序划分
        ctr_f, cva_f, _ = split(yc, few)
        n0 = len(y)
        X = torch.cat([X, Xc]); y = np.r_[y, yc]
        itr = np.r_[itr, ctr_f + n0]; iva = np.r_[iva, cva_f + n0]
    nC = int(y.max())
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu = X[itr].mean((0, 2, 3), keepdim=True); sd = X[itr].std((0, 2, 3), keepdim=True) + 1e-6
    Xtr = ((X[itr] - mu) / sd).to(DEV); Ytr = torch.from_numpy(y[itr] - 1).to(DEV)
    Xva = ((X[iva] - mu) / sd).to(DEV); Yva = torch.from_numpy(y[iva] - 1).to(DEV)
    net = build(model, nC).to(DEV)
    params = list(net.parameters())
    dann = 'dann' in mods
    if dann:                           # DANN（Ganin et al. 2016）：fc1 特征 -> 梯度反转 -> 工况判别器
        Xtg = ((Xc[tgt_unl] - mu) / sd).to(DEV)
        disc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEV)
        params += list(disc.parameters()); bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(params, args.lr); lossf = nn.CrossEntropyLoss()
    best, bad, state, t0 = float('inf'), 0, None, time.time()
    n_batches = int(np.ceil(len(itr) / args.batch))
    for ep in range(args.epochs):
        net.train(); perm = torch.from_numpy(rng.permutation(len(itr))).to(DEV)
        for bi, k in enumerate(range(0, len(perm), args.batch)):
            b = perm[k:k + args.batch]
            if len(b) < 2:            # BatchNorm 不能处理只有 1 个样本的批次
                continue
            opt.zero_grad()
            if dann:
                p = (ep + bi / n_batches) / args.epochs
                lam = 2 / (1 + np.exp(-10 * p)) - 1          # 论文中的权重日程，固定不调
                tb = torch.from_numpy(rng.integers(0, len(Xtg), len(b))).to(DEV)
                f = net.features(torch.cat([Xtr[b], Xtg[tb]]))
                ns = len(b)
                d = disc(_GradReverse.apply(f, lam)).squeeze(1)
                dl = torch.cat([torch.zeros(ns), torch.ones(ns)]).to(DEV)
                l = lossf(net.head(f[:ns]), Ytr[b]) + bce(d, dl)
            else:
                l = lossf(net(Xtr[b]), Ytr[b])
            l.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xva), Yva).item()      # 早停只看源工况验证集，不用目标工况标签
        if vl < best:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(state)
    train_s = time.time() - t0
    L, F = infer(net, X, mu, sd)
    if 'adabn' in mods:
        tmu = Xc[tgt_unl].mean((0, 2, 3), keepdim=True); tsd = Xc[tgt_unl].std((0, 2, 3), keepdim=True) + 1e-6
        Lc, Fc = infer(adabn(net, Xc[tgt_unl], tmu, tsd), Xc, tmu, tsd)
    else:
        Lc, Fc = infer(net, Xc, mu, sd)
    p_te = L[ite].argmax(1) + 1; p_cr = Lc.argmax(1) + 1
    row = dict(scenario=scn, model=model, seed=seed, n_params=n_params(net), epochs=ep + 1, train_s=round(train_s, 1),
               cnn_test_acc=np.mean(p_te == y[ite]), cnn_test_f1=macro_f1(y[ite], p_te, nC),
               cnn_cross_acc=np.mean(p_cr == yc), cnn_cross_f1=macro_f1(yc, p_cr, nC),
               cnn_xtail_acc=np.mean(p_cr[xtail] == yc[xtail]))
    if not args.no_lssvm:
        idev = np.r_[itr, iva]
        st = lssvm_stage(F[idev], y[idev], {'test': (F[ite], y[ite]), 'cross': (Fc, yc)}, nC, rng)
        pt, pc = st['pred']['test'], st['pred']['cross']
        row.update(lssvm_test_acc=np.mean(pt == y[ite]), lssvm_test_f1=macro_f1(y[ite], pt, nC),
                   lssvm_cross_acc=np.mean(pc == yc), lssvm_cross_f1=macro_f1(yc, pc, nC),
                   lssvm_xtail_acc=np.mean(pc[xtail] == yc[xtail]),
                   gam=st['gam'], sig2=st['sig2'], cv_error=st['cv_error'], invalid=st['invalid'])
    return row


def main():
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline='') as fh:
            done = {(r['scenario'], r['model'], int(r['seed'])) for r in csv.DictReader(fh)}
    scns = args.scenarios.split(','); models = args.models.split(','); seeds = parse_seeds(args.seeds)
    data = {}
    for scn in scns:
        lv20, lv30 = bank_keys(scn)
        for key in [('20_0', lv20), ('30_2', lv30)]:
            if key not in data:
                data[key] = load_bank(key[1], key[0])
    print(f'设备 {DEV}，CPU 线程 {args.threads}，共 {len(scns) * len(models) * len(seeds)} 组，已完成 {len(done)} 组')
    for scn in scns:
        for seed in seeds:
            for model in models:
                if (scn, model, seed) in done:
                    continue
                t0 = time.time()
                row = run_one(scn, model, seed, data)
                new = not os.path.exists(args.out)
                with open(args.out, 'a', newline='') as fh:
                    w = csv.DictWriter(fh, FIELDS, extrasaction='ignore')
                    if new:
                        w.writeheader()
                    w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in row.items()})
                msg = f"[{scn:>14s}] {model:>10s} seed {seed:2d}: 网络 {100*row['cnn_test_acc']:.1f}% / 跨工况 {100*row['cnn_xtail_acc']:.1f}%"
                if 'lssvm_test_acc' in row:
                    msg += f" | +BKA-LSSVM {100*row['lssvm_test_acc']:.1f}% / {100*row['lssvm_xtail_acc']:.1f}%"
                print(msg + f"  ({row['epochs']} 轮, {time.time() - t0:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
