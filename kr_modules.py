"""Modules for the 'knownness / reliability' roadmap (Chapter 3 + Chapter 4).

Nothing in this file uses unknown-class labels for fitting or selection.
Functions whose names end in `_diag` take true labels and are EVALUATION ONLY.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------------
# Open-set metrics (match the protocol in the progress summary)
# ----------------------------------------------------------------------------
def _h(a, b):
    return float('nan') if (a != a or b != b or a + b == 0) else 2 * a * b / (a + b)


def openset_metrics(y, raw_cluster, known):
    """Cluster ids 0..n_known-1 are the fixed known centres, others are 'unknown'.

    known_acc   : known windows assigned to their own known class
    unk_recall  : true-unknown windows predicted unknown
    known_false_unk : known windows predicted unknown
    unk_precision   : predicted-unknown windows that are truly unknown (nan if none)
    No post-hoc Hungarian matching, so this is deployable (only known labels used).
    """
    known = [int(k) for k in known]
    n_known = len(known)
    y = np.asarray(y).reshape(-1)
    c = np.asarray(raw_cluster).reshape(-1)
    is_known = np.isin(y, known)
    pred_unk = c >= n_known
    pred_known_label = np.array([known[i] if i < n_known else -1 for i in c])
    out = dict(
        known_acc=float((pred_known_label[is_known] == y[is_known]).mean()) if is_known.any() else float('nan'),
        unk_recall=float(pred_unk[~is_known].mean()) if (~is_known).any() else float('nan'),
        known_false_unk=float(pred_unk[is_known].mean()) if is_known.any() else float('nan'),
        unk_precision=float((~is_known[pred_unk]).mean()) if pred_unk.any() else float('nan'),
        n_pred_unknown=int(pred_unk.sum()),
    )
    out['H'] = _h(out['known_acc'], out['unk_recall'])
    for u in np.unique(y[~is_known]):
        out[f'unk_recall_class{int(u)}'] = float(pred_unk[y == u].mean())
    for kc in known:
        if (y == kc).any():
            out[f'false_unk_class{kc}'] = float(pred_unk[y == kc].mean())
    return out


# ----------------------------------------------------------------------------
# Knownness K(x) and reliability R(x)
# ----------------------------------------------------------------------------
@torch.no_grad()
def fit_scorer(FL, yl_idx, cls_head, alpha=0.5, kind='fused', calib_q=0.05):
    """Fit the K(x) calibrator on LABELLED features only.

    FL: (nL, D) tensor, yl_idx: (nL,) class indices 0..C-1.
    kind:
      'fused' (v1-v5 default): alpha*sigmoid(energy) + (1-alpha)*sigmoid(max prototype cosine)
      'proto' : prototype cosine only (== fused with alpha=0)
      'knn'   : max cosine to the nearest LABELLED window. Handles multi-modal known classes
                (several bearings / records per class) that a single class mean cannot.
                Calibrated with LEAVE-ONE-OUT scores of the labelled windows, so the labelled
                windows are not scored against themselves (removes the optimistic bias).
    Calibration: labelled `calib_q` quantile -> K=0.5, labelled median -> K=0.95.
    """
    n_cls = int(yl_idx.max().item()) + 1
    protos = torch.stack([FL[yl_idx == c].mean(0) for c in range(n_cls)])
    protos = F.normalize(protos, dim=1)
    FLn = F.normalize(FL, dim=1)
    sc = dict(alpha=alpha, protos=protos, head=cls_head, kind=kind, FLn=FLn)

    def raw(Fx, loo=False):
        e = torch.logsumexp(cls_head(Fx), dim=1)
        Fxn = F.normalize(Fx, dim=1)
        s = (Fxn @ protos.T).max(1).values
        S = Fxn @ FLn.T
        if loo:                                    # Fx is FL itself: exclude self-match
            S = S - 3.0 * torch.eye(len(FLn), device=S.device)
        nn_sim = S.max(1).values
        return e, s, nn_sim

    eL, sL, nL = raw(FL, loo=True)
    for name, v in (('e', eL), ('s', sL), ('n', nL)):
        low = torch.quantile(v, calib_q)
        mid = v.median()
        sc[name + '_low'] = low
        sc[name + '_scale'] = ((mid - low) / math.log(19.0)).clamp(min=1e-6)
    sc['raw'] = raw
    return sc


@torch.no_grad()
def known_score(sc, Fx):
    e, s, n = sc['raw'](Fx)
    qs = torch.sigmoid((s - sc['s_low']) / sc['s_scale'])
    if sc['kind'] == 'proto':
        return qs
    if sc['kind'] == 'knn':
        return torch.sigmoid((n - sc['n_low']) / sc['n_scale'])
    qe = torch.sigmoid((e - sc['e_low']) / sc['e_scale'])
    return sc['alpha'] * qe + (1 - sc['alpha']) * qs


@torch.no_grad()
def encode(enc, X, bs=256):
    """X: (N,1,L) tensor on device -> (N,D)."""
    was_training = enc.training
    enc.eval()
    out = torch.cat([enc(X[i:i + bs]) for i in range(0, len(X), bs)])
    enc.train(was_training)
    return out


@torch.no_grad()
def compute_kr(enc, cls_head, sc, X, weak_aug, n_labeled, gen=None):
    """Return dict with K, R, class probabilities for every sample in X (labelled first).

    R(x) = stability * decisiveness
      stability   : 1 - |K(view1) - K(view2)|  for two WEAK stochastic views
      decisiveness: |2*mean(K1,K2) - 1|         (small when the known/unknown routing is near 0.5)
    Labelled samples are anchors: K = R = 1.
    """
    f0 = encode(enc, X)
    K0 = known_score(sc, f0)
    K1 = known_score(sc, encode(enc, weak_aug(X)))
    K2 = known_score(sc, encode(enc, weak_aug(X)))
    stab = (1 - (K1 - K2).abs()).clamp(0, 1)
    dec = (2 * (K1 + K2) / 2 - 1).abs().clamp(0, 1)
    R = stab * dec
    K = K0.clone()
    K[:n_labeled] = 1.0
    R[:n_labeled] = 1.0
    prob = F.softmax(cls_head(f0), dim=1)
    return dict(K=K, R=R, prob=prob, feat=f0, stab=stab, dec=dec)


def budget(K, R, d_known, d_unknown, d_weak, n_modes_total, vary_modes=True):
    """b(x) = f(K, R): perturbation strength and number of perturbed VMD modes.

    Reliable samples (high R) get the class-dependent strength
        d_cls(K) = d_known + (d_unknown - d_known) * (1 - K)       (less known -> more diversity)
    uncertain samples (low R) fall back to the weak strength d_weak.
    """
    d_cls = d_known + (d_unknown - d_known) * (1 - K)
    delta = R * d_cls + (1 - R) * d_weak
    if vary_modes:
        nm = (1 + R * (n_modes_total - 1)).round().clamp(1, n_modes_total).long()
    else:
        nm = torch.full_like(K, n_modes_total, dtype=torch.long)
    return delta, nm


def mode_mask(n_modes, K_total, gen):
    """Random subset mask (B, K_total, 1): sample b perturbs n_modes[b] random modes."""
    B = n_modes.shape[0]
    rank = torch.rand(B, K_total, device=n_modes.device, generator=gen).argsort(1).argsort(1)
    return (rank < n_modes[:, None]).float()[:, :, None]


# ----------------------------------------------------------------------------
# Reliable relation cache + relation-aware contrastive loss (Chapter 4)
# ----------------------------------------------------------------------------
@torch.no_grad()
def build_relation_cache(kr, n_labeled, yl_idx, knn=5, r_thr=0.5):
    """Per-sample evidence used by the relation estimator.

    c_known[i] : confidence that i is a RELIABLE known sample with class pred[i]
    u_unk[i]   : confidence that i is a RELIABLE unknown sample
    nbr[i,j]   : mutual k-nearest-neighbour relation in the current feature space
    """
    K, R, prob, feat = kr['K'], kr['R'], kr['prob'].clone(), kr['feat']
    N = len(K)
    prob[:n_labeled] = F.one_hot(yl_idx, prob.shape[1]).float()
    pred = prob.argmax(1)
    conf = prob.max(1).values
    c_known = K * R * conf
    c_known[:n_labeled] = 1.0
    u_unk = (1 - K) * R
    u_unk[:n_labeled] = 0.0

    fn = F.normalize(feat, dim=1)
    sim = fn @ fn.T
    sim.fill_diagonal_(-2)
    k = min(knn, N - 1)
    nn_idx = sim.topk(k, dim=1).indices
    A = torch.zeros(N, N, dtype=torch.bool, device=feat.device)
    A.scatter_(1, nn_idx, True)
    nbr = A & A.T
    return dict(c_known=c_known, u_unk=u_unk, pred=pred, nbr=nbr, R=R, r_thr=r_thr)


def static_relations(cache, idx):
    """Label-anchor and neighbour evidence for the samples at positions idx (long tensor)."""
    ci, ui, pr, R = cache['c_known'][idx], cache['u_unk'][idx], cache['pred'][idx], cache['R'][idx]
    same = pr[:, None] == pr[None]
    pos_lab = ci[:, None] * ci[None] * same.float()
    neg_lab = (ci[:, None] * ci[None] * (~same).float()
               + ci[:, None] * ui[None] + ui[:, None] * ci[None]).clamp(0, 1)
    pos_nbr = cache['nbr'][idx][:, idx].float() * R[:, None] * R[None]
    return pos_lab, neg_lab, pos_nbr


def relation_contrastive_loss(z1, z2, idx, cache, tau, cfg):
    """Weighted contrastive loss with reliable positives / reliable negatives.

    z1, z2: (B, d) L2-normalised projections of the two views, idx: (B,) positions into cache.
    cfg keys: use_label, use_nbr, use_cons, w_unc, fn_frac, hard_beta, hn_cap, pos_thr
    Returns (loss, stats).
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2])
    sim = z @ z.T / tau
    dev = z.device

    with torch.no_grad():
        pos_lab, neg_lab, pos_nbr = static_relations(cache, idx)
        s11 = (z1 @ z1.T).detach()
        if cfg['use_cons']:
            s12 = (z1 @ z2.T).detach()
            cons = (1 - (s11 - s12).abs()).clamp(0, 1)
            cons = torch.minimum(cons, cons.T)
        else:
            cons = torch.ones_like(s11)
        P = torch.zeros_like(s11)
        if cfg['use_label']:
            P = torch.maximum(P, pos_lab)
        else:
            neg_lab = torch.zeros_like(neg_lab)
        if cfg['use_nbr']:
            P = torch.maximum(P, pos_nbr * cons)
        P = P * (1 - neg_lab)                      # label evidence overrides neighbour evidence
        same_pos = idx[:, None] == idx[None]       # duplicated draws of the same window
        P = torch.where(same_pos, torch.ones_like(P), P)
        neg_lab = neg_lab * (~same_pos).float()
        rel_pos = P >= cfg['pos_thr']

        # suspected false negatives: the most similar non-reliable-negative pairs in each row
        sus = torch.zeros_like(rel_pos)
        if cfg['fn_frac'] > 0:
            cand = (neg_lab < 0.5) & ~rel_pos
            cand.fill_diagonal_(False)
            masked = s11.masked_fill(~cand, -9.0)
            kk = max(1, int(math.ceil(cfg['fn_frac'] * (B - 1))))
            thr = masked.topk(kk, dim=1).values[:, -1:]
            sus = cand & (s11 >= thr)

        # hard-negative emphasis restricted to RELIABLE negatives
        rel_neg = (neg_lab >= 0.5) & ~rel_pos
        if cfg['hard_beta'] > 0:
            h = torch.exp(cfg['hard_beta'] * s11 / tau).masked_fill(~rel_neg, 0.0)
            h = h / (h.sum(1, keepdim=True) / rel_neg.sum(1, keepdim=True).clamp(min=1)).clamp(min=1e-8)
            hard = h.clamp(max=cfg['hn_cap']).masked_fill(~rel_neg, 1.0)
        else:
            hard = torch.ones_like(s11)

        W = torch.where(rel_neg, neg_lab * hard,
                        (1 - P) * cfg['w_unc'] * (~sus).float())
        W = torch.where(rel_pos, torch.ones_like(W), W)
        Wp = P * rel_pos.float()                   # numerator (positive) weights

        # tile (B,B) -> (2B,2B); instance positives (a, a+-B) always weight 1
        W2, P2 = W.repeat(2, 2), Wp.repeat(2, 2)
        ar = torch.arange(B, device=dev)
        inst = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=dev)
        inst[ar, ar + B] = True
        inst[ar + B, ar] = True
        eye = torch.eye(2 * B, dtype=torch.bool, device=dev)
        W2 = torch.where(inst, torch.ones_like(W2), W2).masked_fill(eye, 0.0)
        P2 = torch.where(inst, torch.ones_like(P2), P2).masked_fill(eye, 0.0)
        logW = torch.log(W2.clamp(min=1e-6)).masked_fill(W2 <= 0, -1e9)

    log_den = torch.logsumexp(sim + logW, dim=1, keepdim=True)
    logp = sim - log_den
    loss = -((P2 * logp).sum(1) / P2.sum(1).clamp(min=1e-6)).mean()
    off = ~torch.eye(B, dtype=torch.bool, device=dev)
    stats = dict(rel_pos=float((rel_pos & off).float().sum() / off.sum()),
                 rel_neg=float((rel_neg & off).float().sum() / off.sum()),
                 sus_fn=float((sus & off).float().sum() / off.sum()),
                 mean_w_neg=float(W[off].mean()))
    return loss, stats


# ----------------------------------------------------------------------------
# Evaluation-only diagnostics (use true labels, never feed back into training)
# ----------------------------------------------------------------------------
def kr_diag(K, R, y_true, known, r_thr=0.5):
    """Zone purity and K-AUROC for the 'sample evidence partition'."""
    K, R = np.asarray(K), np.asarray(R)
    unk = ~np.isin(y_true, known)
    out = dict(n=int(len(K)), n_unknown=int(unk.sum()))
    if unk.any() and (~unk).any():
        out['auroc_K_unknown_vs_known'] = float(roc_auc_score(unk, 1 - K))
        out['auroc_R_correct_routing'] = float(roc_auc_score(((K < 0.5) == unk), R))
    zones = dict(reliable_known=(K >= 0.5) & (R >= r_thr),
                 reliable_unknown=(K < 0.5) & (R >= r_thr),
                 uncertain=R < r_thr)
    for name, m in zones.items():
        out[f'zone_{name}_n'] = int(m.sum())
        out[f'zone_{name}_unknown_frac'] = float(unk[m].mean()) if m.any() else float('nan')
    return out


@torch.no_grad()
def relation_diag(cache, y_true):
    """Pair precision of the static relation evidence on the whole training set."""
    N = len(y_true)
    idx = torch.arange(N, device=cache['pred'].device)
    pos_lab, neg_lab, pos_nbr = static_relations(cache, idx)
    y = torch.as_tensor(np.asarray(y_true), device=idx.device)
    same = (y[:, None] == y[None])
    off = ~torch.eye(N, dtype=torch.bool, device=idx.device)

    def prec(mask, want_same):
        m = mask & off
        if m.sum() == 0:
            return float('nan'), 0
        ok = (same[m] if want_same else ~same[m]).float().mean().item()
        return ok, int(m.sum())

    out = {}
    out['plain_false_negative_rate'] = float(same[off].float().mean())  # every pair is a negative in plain SimCLR
    out['label_pos_precision'], out['label_pos_n'] = prec(pos_lab >= 0.5, True)
    out['label_neg_precision'], out['label_neg_n'] = prec(neg_lab >= 0.5, False)
    out['nbr_pos_precision'], out['nbr_pos_n'] = prec(pos_nbr >= 0.25, True)
    return out
