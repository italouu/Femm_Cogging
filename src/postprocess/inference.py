"""
Reconstrução de uma run a partir do log (config.json + best.pth) e predição.
A raiz da bateria é argumento de find_runs/load_set.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.data_gen.parsers.elem_structure import check_elem_layout
from src.neural_op.archs import ARCH_REGISTRY
from src.neural_op.archs.interp import interpolate_grid_to_nodes
from src.neural_op.normalization import Normalizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INTERP_MODE = 'cell_centered_circular_theta'   # rótulo da interpolação grade -> nós (única, archs/interp.py)

ARCHS     = ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN', 'FNO_BipartiteGNN_Elem')
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
    """{run_key: dict(arch, loss, run, run_dir, model, normalizer, epoch)}."""
    out = {}
    for (arch, loss), d in find_runs(root).items():
        if loss not in losses:
            continue
        model, normalizer, cfg, epoch = load_model(d)
        out[run_key(arch, loss)] = dict(arch=arch, loss=loss, run=d.name, run_dir=str(d),
                                         model=model, normalizer=normalizer, epoch=int(epoch))
    return out


def _enc(normalizer, t, key):
    """Normaliza. O nº de canais (dim 1) precisa bater com as stats da run
    (len(stats[key]['mean'])) -- senão ValueError (ex.: run treinada sem o canal J)."""
    if normalizer is None:
        return t
    st = normalizer.stats.get(key)
    if st is not None and t.shape[1] != len(st['mean']):
        raise ValueError(
            f"{key}: entrada com {t.shape[1]} canal(is), stats da run com {len(st['mean'])} "
            f"-- a run não foi treinada neste layout de dados")
    return normalizer.encode(t, key)


def _dec(normalizer, t, key):
    return normalizer.decode(t, key) if normalizer is not None else t


def _t(a, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t if dtype is None else t.to(dtype)


def _bip_like(d):
    """Layout bipartite ou Elem (mesmas chaves) -> tensores de 1 amostra."""
    return dict(x_hw=_t(d['x_hw'])[None], node_x=_t(d['node_x']), elem_x=_t(d['elem_x']),
                edge_index=_t(d['edge_index']), edge_attr=_t(d['edge_attr']),
                cross_edge_index=_t(d['cross_edge_index']),
                cross_edge_attr=_t(d['cross_edge_attr']), L=_t(d['L']))


def to_torch_sample(s):
    """Layouts numpy de parse_smooth -> tensores CPU no formato de 1 amostra do chunk
    (+ 'elem' se parse_smooth(..., with_elem=True))."""
    vx, bp = s['vertex'], s['bip']
    out = dict(
        vertex=dict(x_hw=_t(vx['x_hw'])[None], y_hw=_t(vx['y_hw']), node_x=_t(vx['node_x']),
                    node_y=_t(vx['node_y']), edge_index=_t(vx['edge_index']),
                    edge_attr=_t(vx['edge_attr']), L=_t(vx['L'])),
        bip=dict(x_hw=_t(bp['x_hw'])[None], node_x=_t(bp['node_x']), elem_x=_t(bp['elem_x']),
                 edge_index=_t(bp['edge_index']), edge_attr=_t(bp['edge_attr']),
                 cross_edge_index=_t(bp['cross_edge_index']),
                 cross_edge_attr=_t(bp['cross_edge_attr']), L=_t(bp['L'])),
    )
    if 'elem' in s:
        out['elem'] = _bip_like(s['elem'])
    return out


@torch.inference_mode()
def predict(arch, model, normalizer, ts):
    """Retorna (out_hw [2,H,W] em T no device, y_nodes [S,2] em T no device).
    FNO_BipartiteGNN_Elem: usa ts['elem'] e y_nodes é POR ELEMENTO [M,2]
    (levar aos nós com fields.elem_to_nodes; contrato do layout Elem: docstring de src/data_gen/parsers/elem_structure.py)."""
    if arch in ('FNO_BipartiteGNN', 'FNO_BipartiteGNN_Elem'):
        d = ts['elem'] if arch == 'FNO_BipartiteGNN_Elem' else ts['bip']
        if arch == 'FNO_BipartiteGNN_Elem':
            check_elem_layout(d)
        out_hw, y_nodes = model(
            _enc(normalizer, d['x_hw'], 'x_hw').to(DEVICE),
            _enc(normalizer, d['node_x'], 'node_x').to(DEVICE),
            _enc(normalizer, d['elem_x'], 'elem_x').to(DEVICE),
            d['edge_index'].to(DEVICE), _enc(normalizer, d['edge_attr'], 'edge_attr').to(DEVICE),
            d['cross_edge_index'].to(DEVICE),
            _enc(normalizer, d['cross_edge_attr'], 'cross_edge_attr').to(DEVICE),
            d['L'].to(DEVICE))
        return _dec(normalizer, out_hw, 'y_hw')[0], _dec(normalizer, y_nodes, 'node_y')
    d = ts['vertex']
    x_in = _enc(normalizer, d['x_hw'], 'x_hw').to(DEVICE)
    if arch == 'FNO2d':
        out_hw = _dec(normalizer, model(x_in), 'y_hw')
        nx = d['node_x'].to(DEVICE)
        y_nodes = interpolate_grid_to_nodes(out_hw, nx[:, 3], nx[:, 4], d['L'].to(DEVICE))
        return out_hw[0], y_nodes
    out_hw, y_nodes = model(x_in, _enc(normalizer, d['node_x'], 'node_x').to(DEVICE),
                            d['edge_index'].to(DEVICE),
                            _enc(normalizer, d['edge_attr'], 'edge_attr').to(DEVICE),
                            d['L'].to(DEVICE))
    return _dec(normalizer, out_hw, 'y_hw')[0], _dec(normalizer, y_nodes, 'node_y')


def floor_nodes(ts):
    """Piso da grade: y_hw do gabarito interpolado nos nós (INTERP_MODE)."""
    yh = ts['vertex']['y_hw'][None].to(DEVICE)
    nx = ts['vertex']['node_x'].to(DEVICE)
    return interpolate_grid_to_nodes(yh, nx[:, 3], nx[:, 4], ts['vertex']['L'].to(DEVICE)).cpu()
