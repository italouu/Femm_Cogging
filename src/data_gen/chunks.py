"""
chunks.py
---------
Montagem e gravação dos chunks de treino (`data_chunk_XXXX.pt`) a partir dos dicts por
amostra dos parsers smooth (`femm_mesh_smooth`, `femm_mesh_elem_smooth`) -- 2026-10-21.

Substitui o caminho antigo, em que os scripts de chunks importavam os `_flush` privados de
`scripts/build_data_chunks*.py` e trocavam a global de saída deles por monkeypatch. As
regras de concatenação são as mesmas daqueles `_flush`:
  - STACK_KEYS (x_hw, y_hw, a_hw)       -> np.stack, eixo de batch novo ([C,H,W] -> [B,C,H,W]);
  - INDEX_KEYS (edge_index, cross_edge_index) -> concat no eixo 1 ([2,E] -> [2,E_tot]);
  - demais chaves                        -> concat no eixo 0;
  - 'dim' = (H, W) da 1ª amostra (dim_H/dim_W do dict por amostra não entram no chunk).

Layouts (montagem dos buffers, com os offsets de índice de cada amostra no chunk):
  'grid'      FNO2d                         -- só grade, sem índices;
  'mesh_v1'   FNO_GNN, GNN_PostBase         -- grafo de vértices: edge_index + nº de nós
              acumulado (L); acrescenta E_L (nº de arestas por amostra);
  'bipartite' FNO_BipartiteGNN, FNO_BipartiteGNN_Elem -- dois grafos: edge_index + L
              acumulado; cross_edge_index linha 0 + elem_L acumulado, linha 1 + L acumulado
              (convenção "linha 0 = grafo auxiliar (elem_*), linha 1 = principal (node_*)",
              válida nos dois papéis -- ver parsers/femm_mesh_elem.py).
"""
from pathlib import Path

import numpy as np
import torch

STACK_KEYS = frozenset({'x_hw', 'y_hw', 'a_hw'})
INDEX_KEYS = frozenset({'edge_index', 'cross_edge_index'})

LAYOUT_BY_ARCH = {
    'FNO2d':                 'grid',
    'FNO_GNN':               'mesh_v1',
    'GNN_PostBase':          'mesh_v1',
    'FNO_BipartiteGNN':      'bipartite',
    'FNO_BipartiteGNN_Elem': 'bipartite',
}


def sample_idx(path: Path) -> int:
    """sample_000123.ans.gz -> 123."""
    return int(path.name.split('_')[1].split('.')[0])


def _dim(s: dict) -> tuple:
    return (int(s['dim_H']), int(s['dim_W']))


def _bufs_grid(samples: list) -> dict:
    return {k: [s[k] for s in samples] for k in samples[0] if k not in ('dim_H', 'dim_W')}


def _bufs_mesh_v1(samples: list) -> dict:
    keys = [k for k in samples[0] if k not in ('dim_H', 'dim_W')]
    bufs = {k: [] for k in keys}
    bufs['E_L'] = []
    node_offset = 0
    for s in samples:
        ei = s['edge_index'] + node_offset
        bufs['edge_index'].append(ei)
        bufs['E_L'].append(np.array([ei.shape[1]], dtype=np.int64))
        node_offset += int(s['L'].sum())
        for k in keys:
            if k != 'edge_index':
                bufs[k].append(s[k])
    return bufs


def _bufs_bipartite(samples: list) -> dict:
    keys = [k for k in samples[0] if k not in ('dim_H', 'dim_W')]
    bufs = {k: [] for k in keys}
    node_offset = elem_offset = 0
    for s in samples:
        bufs['edge_index'].append(s['edge_index'] + node_offset)
        cei = s['cross_edge_index'].copy()
        cei[0] += elem_offset   # linha 0: grafo auxiliar (elem_*)
        cei[1] += node_offset   # linha 1: grafo principal (node_*)
        bufs['cross_edge_index'].append(cei)
        for k in keys:
            if k not in ('edge_index', 'cross_edge_index'):
                bufs[k].append(s[k])
        node_offset += int(s['L'].sum())
        elem_offset += int(s['elem_L'].sum())
    return bufs


_BUFS = {'grid': _bufs_grid, 'mesh_v1': _bufs_mesh_v1, 'bipartite': _bufs_bipartite}


def merge(bufs: dict, dim: tuple) -> dict:
    """Buffers (listas de arrays por chave) -> dict de tensores do chunk."""
    merged = {}
    for k, v in bufs.items():
        if k in STACK_KEYS:
            merged[k] = torch.from_numpy(np.stack(v, axis=0))
        elif k in INDEX_KEYS:
            merged[k] = torch.from_numpy(np.concatenate(v, axis=1))
        else:
            merged[k] = torch.from_numpy(np.concatenate(v, axis=0))
    merged['dim'] = dim
    return merged


def write_chunk(out_dir: Path, chunk_idx: int, samples: list, layout: str) -> Path:
    """Monta e grava out_dir/data_chunk_XXXX.pt (escrita atômica: grava em out_dir/.tmp/
    e renomeia no fim -- chunk interrompido nunca fica com o nome final)."""
    out_dir = Path(out_dir)
    tmp_dir = out_dir / '.tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    name = f"data_chunk_{chunk_idx:04d}.pt"
    merged = merge(_BUFS[layout](samples), _dim(samples[0]))
    torch.save(merged, tmp_dir / name)
    (tmp_dir / name).replace(out_dir / name)

    counts = '  '.join(f"{k}={merged[k].shape[-1] if k in INDEX_KEYS else merged[k].shape[0]}"
                       for k in ('node_x', 'elem_x', 'edge_index', 'cross_edge_index')
                       if k in merged)
    print(f"  [chunk {chunk_idx:04d}] {merged['x_hw'].shape[0]} amostras  {counts}")
    return out_dir / name
