"""Reconstruct saved T1 models and verify existing validation predictions only.

No fitting on validation data, no model training, no access to test arrays.
The feature standardization and semi-supervised clustering are fit from training
samples only, exactly as in the saved validation-only training script.
"""
from pathlib import Path
import runpy
import sys
import numpy as np
import torch

root = Path(__file__).resolve().parent
script = root / 'train_energy_paderborn_val_only.py'
data = root / 'data' / 'paderborn_filegroup_v2.npz'
bank = root / 'data' / 'paderborn_filegroup_v2_vmd.npy'
runs = root / 'runs_paderborn'
for p in (script, data, bank, bank.with_suffix('.json')):
    if not p.is_file():
        raise FileNotFoundError(p)

# This imports the validation-only script but does not execute its main loop.
sys.path.insert(0, str(root))
sys.argv = [str(script), '--mode', 'generic', '--tag', 'validation_replay_preflight',
            '--data', str(data), '--vmd-bank', str(bank), '--device', 'cuda']
g = runpy.run_path(str(script), run_name='validation_replay_only')
if tuple(g['_known']) != (0, 1) or tuple(g['_unknown']) != (2,):
    raise RuntimeError('This replay requires known labels (0,1) and one unknown label (2).')
xtr, ytr, xv, yv = g['XTR'], g['YTR'], g['XV'], g['YV']
if set(np.unique(yv).tolist()) != {0, 1, 2}:
    raise ValueError('Validation labels must contain exactly 0, 1 and 2.')

modes = ('generic', 'fixed', 'adaptive_energy', 'shuffled_energy')
print('\n=== TRAIN-ONLY RECONSTRUCTION / VALIDATION REPLAY ===', flush=True)
for mode in modes:
    for seed in range(6):
        folder = runs / f'fg_v2_{mode}_seed{seed}'
        teacher_path = folder / f'energy_teacher_PADERBORN_seed{seed}_alpha0.5.npz'
        checkpoint_path = folder / f'm2_PADERBORN_seed{seed}.pt'
        pred_path = folder / f'pred_PADERBORN_seed{seed}.npz'
        for path in (teacher_path, checkpoint_path, pred_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        with np.load(teacher_path, allow_pickle=False) as teacher:
            idx = teacher['global_indices'].astype(np.int64)
            labeled = teacher['is_labeled'].astype(bool)
        n_lab = int(labeled.sum())
        if (len(idx) != len(xtr) or len(labeled) != len(idx) or n_lab == 0
                or not labeled[:n_lab].all() or labeled[n_lab:].any()
                or len(np.unique(idx)) != len(idx)
                or idx.min() < 0 or idx.max() >= len(xtr)):
            raise RuntimeError(f'Invalid recorded training indices: {folder}')
        # Only the training labels for the labeled known examples are accessed.
        yl = ytr[idx[:n_lab]]
        if set(np.unique(yl).tolist()) != {0, 1}:
            raise RuntimeError(f'Known labeled training classes are incorrect: {folder}')

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        meta = checkpoint.get('metadata', {})
        if meta.get('seed') != seed or meta.get('split') != 'PADERBORN':
            raise RuntimeError(f'Checkpoint seed/split metadata mismatch: {folder}')
        enc = g['Enc']().to(g['DEVICE'])
        enc.load_state_dict(checkpoint['encoder'], strict=True)
        fl = g['get_features'](enc, xtr[idx[:n_lab]])
        fu = g['get_features'](enc, xtr[idx[n_lab:]])
        fv = g['get_features'](enc, xv)
        features = np.concatenate([fl, fu], axis=0)
        mean_f = features.mean(axis=0, keepdims=True)
        std_f = features.std(axis=0, keepdims=True) + 1e-8
        fln, fun, fvn = (fl - mean_f) / std_f, (fu - mean_f) / std_f, (fv - mean_f) / std_f

        centers, raw_u = g['ss_kmeans_optimized'](fln, yl, fun, 3, (0, 1), seed)
        if centers is None or raw_u is None:
            raise RuntimeError(f'Clustering returned no centers: {folder}')
        raw_v = g['nearest'](fvn, centers)
        # For known labels 0 and 1 and exactly one unknown class, the fixed
        # deployment mapping is center 0 -> 0, center 1 -> 1, center 2 -> 2.
        pred_v = raw_v.copy()

        with np.load(pred_path, allow_pickle=False) as saved:
            expected_yv = saved['yv']
            expected_u = saved['raw_U']
            expected_raw_v = saved['raw_V']
            expected_pred_v = saved['pred_V']
        checks = {
            'val_labels': np.array_equal(yv, expected_yv),
            'train_clusters': np.array_equal(raw_u, expected_u),
            'val_clusters': np.array_equal(raw_v, expected_raw_v),
            'val_predictions': np.array_equal(pred_v, expected_pred_v),
        }
        print(f'{mode} seed{seed}: ' + ', '.join(f'{key}={value}' for key, value in checks.items()), flush=True)
        if not all(checks.values()):
            raise RuntimeError(f'Validation replay mismatch; stop and inspect: {folder}')
        del enc, checkpoint

# Close the npz archive opened by the imported validation-only script.
g['d'].close()
print('\nREPLAY COMPLETE: all 24 validation predictions reproduced.')
print('No training or test evaluation was run.')
