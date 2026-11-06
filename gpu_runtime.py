"""CPU stub of the user's gpu_runtime, ONLY for smoke-testing the pipeline."""
import os, torch
DEVICE = torch.device(os.environ.get('PUMP_DEVICE', 'cpu'))
FAST_AUG = True
def feats_gpu(*a, **k): raise NotImplementedError
def vectorized_aug(x, p):
    B, C, L = x.shape
    out = x.clone()
    if p.get('shift', 0):
        s = int(torch.randint(-int(p['shift']), int(p['shift']) + 1, (1,)))
        out = torch.roll(out, s, dims=2)
    if p.get('scale', 0):
        out = out * (1 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * p['scale'])
    if p.get('noise', 0):
        out = out + torch.randn_like(out) * p['noise']
    if p.get('mask', 0):
        m = int(L * p['mask']); st = int(torch.randint(0, L - m, (1,))); out[:, :, st:st + m] = 0
    return out
def save_checkpoint(out, name, enc, proj, head, meta):
    os.makedirs(out, exist_ok=True)
    torch.save(dict(enc=enc.state_dict(), proj=proj.state_dict(), head=head.state_dict(), meta=meta), os.path.join(out, name))
