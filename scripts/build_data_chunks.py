"""
build_data_chunks.py
--------------------
Pipeline completo de construção do dataset.

1. Chama gen_npz_structures.run() para gerar (ou completar) os .npz intermediários.
2. Agrupa os .npz em data_chunk_*.pt finais:
     data/torch/data_chunks/<dataset>/data_chunk_XXXX.pt

Campos stream legados (x, y, depth, cells) são descartados — apenas
x_hw, y_hw e, quando presentes, campos de grafo (node_x, node_y,
edge_index, edge_attr, L, E_L) são salvos.

Execução (a partir da raiz do projeto):
    python -m scripts.build_data_chunks
"""

from pathlib import Path

import numpy as np
import torch

from scripts.gen_npz_structures import run as gen_run
from src.configs.datagen import DatagenConfig
_dg        = DatagenConfig()
CHUNK_SIZE = _dg.chunk_size
DATASET    = _dg.dataset

# None = processa todas as amostras disponíveis
MAX_SAMPLES = None

# ── Diretórios ────────────────────────────────────────────────────────────────
_NPZ_DIR  = Path("data/temp/samples_npz") / DATASET
_DATA_DIR = Path("data/torch/data_chunks") / DATASET

# Campos legados do formato stream — sempre descartados ao ler os .npz
_DROP_KEYS = frozenset({'x', 'y', 'depth', 'cells'})

# Chaves cujos arrays por amostra têm shape [C, H, W] — precisam de stack, não concat
_STACK_KEYS = frozenset({'x_hw', 'y_hw', 'x_hw_grid', 'y_hw_grid'})


def _load_npz(path: Path) -> dict:
    """Carrega um .npz, descarta campos legados e reconstrói 'dim' como tupla."""
    d = np.load(path)
    result = {'dim': (int(d['dim_H']), int(d['dim_W']))}
    for key in d.files:
        if key not in ('dim_H', 'dim_W') and key not in _DROP_KEYS:
            result[key] = d[key]
    return result


def _concat_arrays(key: str, arrays: list) -> np.ndarray:
    """Regra de concatenação por nome de chave."""
    if key == 'edge_index':
        return np.concatenate(arrays, axis=1)   # [2, E] → [2, E_tot]
    if key in _STACK_KEYS:
        return np.stack(arrays, axis=0)         # [C, H, W] → [B, C, H, W]
    return np.concatenate(arrays, axis=0)       # [S, feat] ou [B] → axis=0


def _flush(chunk_idx: int, bufs: dict, dim: tuple):
    """Concatena buffers e salva data_chunk_XXXX.pt."""
    merged = {k: torch.from_numpy(_concat_arrays(k, v)) for k, v in bufs.items()}
    merged['dim'] = dim
    torch.save(merged, _DATA_DIR / f"data_chunk_{chunk_idx:04d}.pt")

    n = merged['x_hw'].shape[0]
    extra = (f"S_tot={merged['node_x'].shape[0]}  E_tot={merged['edge_index'].shape[1]}"
             if 'node_x' in merged else "")
    print(f"  [chunk {chunk_idx:04d}] {n} amostras  {extra}")


def build(max_samples=MAX_SAMPLES, chunk_size=CHUNK_SIZE):
    """
    Agrupa os .npz em data_chunk_*.pt finais.

    Parâmetros
    ----------
    max_samples : int | None
        Limite de .npz a converter. None = todos os disponíveis.
    chunk_size  : int
        Amostras por chunk (padrão: CHUNK_SIZE do config).
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(_NPZ_DIR.glob("sample_*.npz"),
                       key=lambda p: int(p.stem.split('_')[1]))
    if max_samples is not None:
        npz_paths = npz_paths[:max_samples]

    total = len(npz_paths)
    if total == 0:
        print("Nenhum .npz encontrado em", _NPZ_DIR)
        return 0

    _peek     = np.load(npz_paths[0])
    has_graph = 'edge_index' in _peek.files
    _peek.close()

    print(f"\n=== Construção de chunks ===")
    print(f"  .npz a processar : {total}  |  chunk_size : {chunk_size}")
    print(f"  graph: {has_graph}")

    chunk_idx   = 0
    node_offset = 0
    dim         = None
    bufs: dict  = {}

    for i, path in enumerate(npz_paths):
        expected_chunk = i // chunk_size
        data_pt = _DATA_DIR / f"data_chunk_{expected_chunk:04d}.pt"

        if data_pt.exists():
            if (i + 1) % chunk_size == 0 or i == total - 1:
                chunk_idx += 1
            continue

        s = _load_npz(path)
        if dim is None:
            dim = s['dim']

        if not bufs:
            bufs = {k: [] for k in s if k != 'dim'}
            if has_graph:
                bufs['E_L'] = []

        if has_graph:
            ei = s['edge_index'] + node_offset
            bufs['edge_index'].append(ei)
            bufs['E_L'].append(np.array([ei.shape[1]], dtype=np.int64))
            node_offset += int(s['L'].sum())

        for k in bufs:
            if k not in ('edge_index', 'E_L'):
                bufs[k].append(s[k])

        if len(bufs['x_hw']) == chunk_size:
            _flush(chunk_idx, bufs, dim)
            chunk_idx   += 1
            node_offset  = 0
            dim          = None
            bufs         = {}

    if bufs and bufs.get('x_hw'):
        _flush(chunk_idx, bufs, dim)
        chunk_idx += 1

    print(f"\n=== Resumo build_data_chunks ===")
    print(f"  Chunks salvos : {chunk_idx}")
    return chunk_idx


def main():
    print("=== Etapa 1: gen_npz_structures ===")
    gen_run(MAX_SAMPLES)
    build(MAX_SAMPLES)


if __name__ == "__main__":
    main()
