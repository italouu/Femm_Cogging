"""
Reconstrução de uma run a partir do log (config.json + best.pth) e predição.

Origem: tests/resultados/common.py (2026-10-13), sem mudança de lógica; a raiz
da bateria passa a ser argumento de find_runs/load_set.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.neural_op.archs import ARCH_REGISTRY
from src.neural_op.archs.interp import interpolate_grid_to_nodes
from src.neural_op.normalization import Normalizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INTERP_MODE = 'cell_centered_border'   # mesmo modo grade -> nós dos modelos da bateria smooth

ARCHS     = ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN')
RUN_ORDER = [(a, l) for a in ARCHS for l in ('mse', 'mae')]


def run_key(arch, loss):
    return f'{arch}_{loss}'


def load_model(run_dir: Path):
    """best.pth + Normalizer de uma run -> (model, normalizer, cfg, epoch)."""
    cfg   = json.loads((run_dir / 'config.json').read_text())
    entry = ARCH_REGISTRY[cfg['arch']]
    if hasattr(entry.cfg_cls, 'from_dict'):
        arch_cfg = entry.cfg_cls.from_dict(cfg['arch_cfg'])
    else:
        arch_cfg = entry.cfg_cls(**cfg['arch_cfg'])
    model = entry.make_model(arch_cfg)
    ckpt  = torch.load(run_dir / 'checkpoints' / 'best.pth', map_location='cpu', weights_only=False)
    sd    = {k: v for k, v in ckpt['model_state_dict'].items() if k != '_metadata'}
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    normalizer = (Normalizer.from_dict(cfg['norm_stats'])
                  if cfg.get('normalize') and cfg.get('norm_stats') else None)
    model.normalizer = normalizer
    return model, normalizer, cfg, ckpt['epoch']


def find_runs(root: Path):
    """{(arch, loss): run_dir} da bateria -- 1 run por par (falha se houver repetição)."""
    out = {}
    for arch, loss in RUN_ORDER:
        hits = [d for d in sorted((root / arch).glob('run_*'))
                if (d / 'checkpoints' / 'best.pth').exists()
                and json.loads((d / 'config.json').read_text())['loss'] == loss]
        assert len(hits) == 1, f'{root}/{arch} loss={loss}: {hits}'
        out[(arch, loss)] = hits[0]
    return out


def load_set(root: Path, losses=('mse', 'mae')):
    """{run_key: dict(arch, loss, run, run_dir, model, normalizer, epoch, interp_mode)}."""
    out = {}
    for (arch, loss), d in find_runs(root).items():
        if loss not in losses:
            continue
        model, normalizer, cfg, epoch = load_model(d)
        out[run_key(arch, loss)] = dict(arch=arch, loss=loss, run=d.name, run_dir=str(d),
                                         model=model, normalizer=normalizer, epoch=int(epoch),
                                         interp_mode=getattr(model, 'interp_mode', None))
    return out


def _enc(normalizer, t, key):
    return normalizer.encode(t, key) if normalizer is not None else t


def _dec(normalizer, t, key):
    return normalizer.decode(t, key) if normalizer is not None else t


def _t(a, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t if dtype is None else t.to(dtype)


def to_torch_sample(s):
    """Layouts numpy de parse_smooth -> tensores CPU no formato de 1 amostra do chunk."""
    v1, bp = s['v1'], s['bip']
    return dict(
        v1=dict(x_hw=_t(v1['x_hw'])[None], y_hw=_t(v1['y_hw']), node_x=_t(v1['node_x']),
                node_y=_t(v1['node_y']), edge_index=_t(v1['edge_index']),
                edge_attr=_t(v1['edge_attr']), L=_t(v1['L'])),
        bip=dict(x_hw=_t(bp['x_hw'])[None], node_x=_t(bp['node_x']), elem_x=_t(bp['elem_x']),
                 edge_index=_t(bp['edge_index']), edge_attr=_t(bp['edge_attr']),
                 cross_edge_index=_t(bp['cross_edge_index']),
                 cross_edge_attr=_t(bp['cross_edge_attr']), L=_t(bp['L'])),
    )


@torch.inference_mode()
def predict(arch, model, normalizer, ts):
    """Retorna (out_hw [2,H,W] em T no device, y_nodes [S,2] em T no device)."""
    if arch in ('FNO_BipartiteGNN', 'FNO_BipartiteGNN_Elem'):
        d = ts['bip']
        out_hw, y_nodes = model(
            _enc(normalizer, d['x_hw'], 'x_hw').to(DEVICE),
            _enc(normalizer, d['node_x'], 'node_x').to(DEVICE),
            _enc(normalizer, d['elem_x'], 'elem_x').to(DEVICE),
            d['edge_index'].to(DEVICE), d['edge_attr'].to(DEVICE),
            d['cross_edge_index'].to(DEVICE), d['cross_edge_attr'].to(DEVICE),
            d['L'].to(DEVICE))
        return _dec(normalizer, out_hw, 'y_hw')[0], _dec(normalizer, y_nodes, 'node_y')
    d = ts['v1']
    x_in = _enc(normalizer, d['x_hw'], 'x_hw').to(DEVICE)
    if arch == 'FNO2d':
        out_hw = _dec(normalizer, model(x_in), 'y_hw')
        nx = d['node_x'].to(DEVICE)
        y_nodes = interpolate_grid_to_nodes(out_hw, nx[:, 3], nx[:, 4], d['L'].to(DEVICE),
                                            mode=getattr(model, 'interp_mode', 'legacy'))
        return out_hw[0], y_nodes
    out_hw, y_nodes = model(x_in, _enc(normalizer, d['node_x'], 'node_x').to(DEVICE),
                            d['edge_index'].to(DEVICE), d['edge_attr'].to(DEVICE),
                            d['L'].to(DEVICE))
    return _dec(normalizer, out_hw, 'y_hw')[0], _dec(normalizer, y_nodes, 'node_y')


def floor_nodes(ts):
    """Piso da grade: y_hw do gabarito interpolado nos nós (INTERP_MODE)."""
    yh = ts['v1']['y_hw'][None].to(DEVICE)
    nx = ts['v1']['node_x'].to(DEVICE)
    return interpolate_grid_to_nodes(yh, nx[:, 3], nx[:, 4], ts['v1']['L'].to(DEVICE),
                                     mode=INTERP_MODE).cpu()
