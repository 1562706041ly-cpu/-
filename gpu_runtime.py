"""Device helpers for the preliminary experiments; not a thesis-method module."""
from pathlib import Path
import atexit
import os
import torch
import torch.nn.functional as F
import numpy as np

requested = os.environ.get('PUMP_DEVICE', 'cpu')
if requested == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested but unavailable. Check CUDA-enabled PyTorch and the NVIDIA driver.')
DEVICE = torch.device(requested)
FAST_AUG = os.environ.get('PUMP_FAST_AUG', '0') == '1'
FEATURE_BATCH = int(os.environ.get('PUMP_FEATURE_BATCH', '256'))
if FEATURE_BATCH < 1:
    raise ValueError('PUMP_FEATURE_BATCH must be positive.')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
print(f'[device] {DEVICE}; augmentation={"vectorized" if FAST_AUG else "legacy"}; FP32; feature_batch={FEATURE_BATCH}', flush=True)


@torch.no_grad()
def feats_gpu(enc, X):
    enc.eval()
    dev = next(enc.parameters()).device
    chunks = []
    for i in range(0, len(X), FEATURE_BATCH):
        x = torch.as_tensor(X[i:i + FEATURE_BATCH], dtype=torch.float32, device=dev)[:, None]
        chunks.append(F.normalize(enc(x), dim=1).cpu())
    return torch.cat(chunks).numpy()


def vectorized_aug(x, p):
    """Same augmentation families/ranges, but a different RNG consumption order."""
    B, C, L = x.shape
    dev = x.device
    t = torch.arange(L, device=dev)[None, :]
    shifts = torch.randint(-p['shift'], p['shift'] + 1, (B, 1), device=dev)
    index = ((t - shifts) % L)[:, None, :].expand(B, C, L)
    out = x.gather(-1, index)
    scale = 1 - p['scale'] + 2 * p['scale'] * torch.rand(B, 1, 1, device=dev)
    out = out * scale + p['noise'] * torch.randn_like(out)
    widths = torch.randint(0, max(1, int(L * p['mask'])), (B, 1), device=dev)
    starts = torch.floor(torch.rand(B, 1, device=dev) * (L - widths + 1)).long()
    keep = (t < starts) | (t >= starts + widths)
    return out * keep[:, None, :]


def save_checkpoint(outdir, filename, enc, proj, cls_head, metadata):
    """Portable final weights only; not a resumable optimizer/training checkpoint."""
    path = Path(outdir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 2,
        'encoder': {k: v.detach().cpu() for k, v in enc.state_dict().items()},
        'projection': {k: v.detach().cpu() for k, v in proj.state_dict().items()},
        'cls_head': {k: v.detach().cpu() for k, v in cls_head.state_dict().items()},
        'metadata': dict(metadata, device=str(DEVICE), fast_aug=FAST_AUG,
                         torch_version=str(torch.__version__),
                         cuda_runtime=torch.version.cuda,
                         precision='float32',
                         purpose='final_weights_not_optimizer_resume'),
    }
    torch.save(payload, path)


@atexit.register
def _report_memory():
    if DEVICE.type == 'cuda':
        try:
            torch.cuda.synchronize()
            print(f'[GPU] peak allocated: {torch.cuda.max_memory_allocated() / 1048576:.1f} MiB', flush=True)
        except Exception:
            pass


def vectorized_vmd_aug(x, modes, delta=0.1, fidelity_threshold=0.85):
    B, K, L = modes.shape
    dev = x.device
    e = (torch.rand(B, K, 1, device=dev) * 2 - 1) * delta
    x_aug = ((1 + e) * modes).sum(dim=1, keepdim=True)
    residual = x - modes.sum(dim=1, keepdim=True)
    x_aug = x_aug + residual

    x_flat = x.view(B, -1)
    x_aug_flat = x_aug.view(B, -1)
    cos_sim = F.cosine_similarity(x_flat, x_aug_flat, dim=1)

    mask = cos_sim < fidelity_threshold
    if mask.any():
        x_aug[mask] = x[mask]

    return x_aug
