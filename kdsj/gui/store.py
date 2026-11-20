"""设备库：每台设备一个模型文件（.pt）和一个说明文件（.json），默认放在 用户目录/kdsj_data/devices。
可用环境变量 KDSJ_DATA 指定其他目录。"""
import json
import os
import re
import shutil
import time
from pathlib import Path


def data_dir():
    d = Path(os.environ.get('KDSJ_DATA', Path.home() / 'kdsj_data'))
    (d / 'devices').mkdir(parents=True, exist_ok=True)
    (d / 'reports').mkdir(parents=True, exist_ok=True)
    return d


def _safe(name):
    s = re.sub(r'[\\/:*?"<>|\s]+', '_', name.strip())
    return s or 'device'


class DeviceStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else data_dir()
        self.dev = self.root / 'devices'
        self.dev.mkdir(parents=True, exist_ok=True)

    def list(self):
        out = []
        for meta in sorted(self.dev.glob('*.json')):
            try:
                m = json.loads(meta.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            pt = meta.with_suffix('.pt')
            if pt.exists():
                m['path'] = str(pt)
                out.append(m)
        return out

    def exists(self, name):
        return (self.dev / f'{_safe(name)}.pt').exists()

    def add_model(self, name, model, info=None):
        base = self.dev / _safe(name)
        model.save(str(base.with_suffix('.pt')))
        self._write_meta(base, name, model, info)
        return str(base.with_suffix('.pt'))

    def import_file(self, name, src, model):
        base = self.dev / _safe(name)
        shutil.copyfile(src, base.with_suffix('.pt'))
        self._write_meta(base, name, model, {'imported_from': os.path.basename(src)})
        return str(base.with_suffix('.pt'))

    def _write_meta(self, base, name, model, info):
        kind = 'anomaly' if getattr(model, 'is_anomaly', False) else 'classifier'
        meta = {'name': name, 'kind': kind, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'feature': model.feature, 'info': info or {}}
        if kind == 'anomaly':
            meta['method'] = model.method
            meta['threshold'] = model.threshold
        else:
            meta['classes'] = model.class_names
            meta['model'] = model.model_name
        base.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    def remove(self, name):
        base = self.dev / _safe(name)
        for suf in ('.pt', '.json'):
            p = base.with_suffix(suf)
            if p.exists():
                p.unlink()

    def reports_dir(self):
        d = self.root / 'reports'
        d.mkdir(exist_ok=True)
        return d
