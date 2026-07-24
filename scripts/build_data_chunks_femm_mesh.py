"""
build_data_chunks_femm_mesh.py
-------------------------------
Agrupa os .npz de staging gerados por generate_data_femm_mesh.py
(data/temp/samples_mesh/<dataset>/sample_XXXXXX.npz) em data_chunk_XXXX.pt
finais, no mesmo espírito de build_data_chunks.py (offset de edge_index,
E_L calculado no agrupamento) — mas sem passar pela etapa
gen_npz_structures.py, já que o .npz de staging já sai no formato final
(ver src/data_gen/femm_mesh.generate_mesh_sample).

Saída: data/torch/data_chunks/<dataset>/data_chunk_XXXX.pt — mesmo diretório
usado pelo pipeline qtree/grid, como um dataset normal (o `dataset` deve ser
um nome próprio, não reaproveitado entre modos, senão os chunks se misturam).

Execução (a partir da raiz do projeto):
    python -m scripts.build_data_chunks_femm_mesh
"""
from pathlib import Path

import numpy as np
import torch

from src.configs.datagen import DatagenConfig
_dg        = DatagenConfig()
CHUNK_SIZE = _dg.chunk_size
DATASET    = _dg.dataset

# None = processa todas as amostras disponíveis
MAX_SAMPLES = None

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
_STAGING_DIR  = PROJECT_ROOT / "data" / "temp" / "samples_mesh" / DATASET
_OUT_DIR      = PROJECT_ROOT / "data" / "torch" / "data_chunks" / DATASET

# metadados de geometria por amostra que não entram no chunk final (não
# usados no treino; ficam só no .npz de staging para depuração/rastreabilidade)
_DROP_KEYS = frozenset({'r_in_mm', 'r_ext_mm', 'ang_1_deg', 'ang_2_deg'})

# chaves cujos arrays por amostra têm shape [C,H,W] ou [H,W] — precisam de
# stack (empilha um eixo de batch novo), não concat
_STACK_KEYS = frozenset({'x_hw', 'y_hw', 'a_hw_grid'})

# x_hw_grid/y_hw_grid -> x_hw/y_hw -- mesmo nome de chave usado pelos
# pipelines grid/qtree. Sem essa renomeação, ChunkStreamDataset (usado por
# build_loaders(mode='grid'), consumido por FNO2d/MaskedFNO2d) quebra com
# KeyError: 'x_hw' -- os chunks femm_mesh nunca tiveram essa chave (ver
# histórico 2026-07-24). Sem perda de informação: este pipeline não tem
# variante "suavizada" (média por área do qtree) -- x_hw_grid JÁ é a única
# grade existente aqui, o sufixo _grid só distinguia da variante que não
# existe neste modo. a_hw_grid fica como está (não consumido pelo loader
# 'grid', sem colisão de nome com nada).
_RENAME_KEYS = {'x_hw_grid': 'x_hw', 'y_hw_grid': 'y_hw'}


def _load_npz(path: Path) -> dict:
    """Carrega um .npz, descarta metadados de geometria, renomeia
    x_hw_grid/y_hw_grid -> x_hw/y_hw (ver _RENAME_KEYS) e reconstrói 'dim'
    como tupla — mesma convenção de build_data_chunks.py."""
    d = np.load(path)
    result = {'dim': (int(d['dim_H']), int(d['dim_W']))}
    for key in d.files:
        if key in ('dim_H', 'dim_W') or key in _DROP_KEYS:
            continue
        result[_RENAME_KEYS.get(key, key)] = d[key]
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
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(merged, _OUT_DIR / f"data_chunk_{chunk_idx:04d}.pt")

    n = merged['x_hw'].shape[0]
    print(f"  [chunk {chunk_idx:04d}] {n} amostras  "
          f"S_tot={merged['node_x'].shape[0]}  E_tot={merged['edge_index'].shape[1]}")


def build(max_samples=MAX_SAMPLES, chunk_size=CHUNK_SIZE):
    """
    Agrupa os .npz de staging em data_chunk_*.pt finais.

    Parâmetros
    ----------
    max_samples : int | None
        Limite de .npz a converter. None = todos os disponíveis.
    chunk_size  : int
        Amostras por chunk (padrão: CHUNK_SIZE do config).
    """
    npz_paths = sorted(_STAGING_DIR.glob("sample_*.npz"),
                        key=lambda p: int(p.stem.split('_')[1]))
    if max_samples is not None:
        npz_paths = npz_paths[:max_samples]

    total = len(npz_paths)
    if total == 0:
        print("Nenhum .npz encontrado em", _STAGING_DIR)
        return 0

    print(f"\n=== Construção de chunks (malha FEM) ===")
    print(f"  .npz a processar : {total}  |  chunk_size : {chunk_size}")

    chunk_idx   = 0
    node_offset = 0
    dim         = None
    bufs: dict  = {}

    for i, path in enumerate(npz_paths):
        expected_chunk = i // chunk_size
        data_pt = _OUT_DIR / f"data_chunk_{expected_chunk:04d}.pt"

        if data_pt.exists():
            if (i + 1) % chunk_size == 0 or i == total - 1:
                chunk_idx += 1
            continue

        s = _load_npz(path)
        if dim is None:
            dim = s['dim']

        if not bufs:
            bufs = {k: [] for k in s if k != 'dim'}
            bufs['E_L'] = []

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

    print(f"\n=== Resumo build_data_chunks_femm_mesh ===")
    print(f"  Chunks salvos : {chunk_idx}")
    return chunk_idx


def main():
    print("=== Etapa 1: generate_data_femm_mesh ===")
    from scripts.generate_data_femm_mesh import run as gen_run
    gen_run()
    print("\n=== Etapa 2: agrupamento em chunks ===")
    build()


if __name__ == "__main__":
    main()
