"""
build_data_chunks_femm_mesh_v2.py
-----------------------------------
Agrupa os .npz de data/temp/samples_npz/<dataset>/ (gerados pelo branch
mode='femm_mesh_v2' de gen_npz_structures.py -- grafo de vértices + grafo
de elementos + arestas cruzadas + grade H×W, ver src/data_gen/femm_mesh_v2.py)
em data_chunk_XXXX.pt finais.

Precisa cuidar de TRÊS espaços de índice (diferente do pipeline v1/femm_mesh,
que só tem nós+arestas vértice-vértice):
    - nós (vértices)                       -- offset acumulado por L
    - elementos                            -- offset acumulado por elem_L
    - edge_index (vértice-vértice)         -- as 2 linhas offsetadas por node_offset
    - cross_edge_index (elemento->vértice) -- linha 0 (elemento) offsetada por
                                               elem_offset, linha 1 (vértice)
                                               por node_offset (mesmo offset de edge_index)

L/elem_L/E_L/C_L (contagens por amostra) já vêm prontas do .npz (calculadas
em parse_ans_gzip_sample) -- só concatenadas, não recalculadas aqui.

Saída: data/torch/data_chunks/<dataset>/data_chunk_XXXX.pt -- sem sufixo de
parser (não existe variante de parser pra este formato, ver
gen_npz_structures.py::_run_femm_mesh_v2). <dataset> aqui é
DatagenConfig.femm_mesh_v2_dataset_name, não .dataset direto -- sufixado com
'_B' quando femm_mesh_v2_target_field='B' (2026-08-13), pra ler do staging
certo (ver mesma property).

Execução (a partir da raiz do projeto):
    python -m scripts.build_data_chunks_femm_mesh_v2
"""
from pathlib import Path

import numpy as np
import torch

from src.configs.datagen import DatagenConfig
_dg        = DatagenConfig()
CHUNK_SIZE = _dg.chunk_size
DATASET    = _dg.femm_mesh_v2_dataset_name

# None = processa todas as amostras disponíveis
MAX_SAMPLES = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STAGING_DIR = PROJECT_ROOT / "data" / "temp" / "samples_npz" / DATASET
_OUT_DIR     = PROJECT_ROOT / "data" / "torch" / "data_chunks" / DATASET

# chaves com shape [C,H,W] por amostra -- precisam de stack (eixo de batch novo)
_STACK_KEYS = frozenset({'x_hw', 'y_hw'})
# chaves de índice -- concat axis=1 + offset (tratadas à parte no loop)
_INDEX_KEYS = frozenset({'edge_index', 'cross_edge_index'})


def _load_npz(path: Path) -> dict:
    """Carrega um .npz e reconstrói 'dim' como tupla (dim_H/dim_W só existem
    separados por mecânica de serialização do .npz, sem tupla nativa)."""
    d = np.load(path)
    result = {'dim': (int(d['dim_H']), int(d['dim_W']))}
    for key in d.files:
        if key in ('dim_H', 'dim_W'):
            continue
        result[key] = d[key]
    return result


def _flush(chunk_idx: int, bufs: dict, dim: tuple):
    merged = {}
    for k, v in bufs.items():
        if k in _STACK_KEYS:
            merged[k] = torch.from_numpy(np.stack(v, axis=0))
        elif k in _INDEX_KEYS:
            merged[k] = torch.from_numpy(np.concatenate(v, axis=1))
        else:
            merged[k] = torch.from_numpy(np.concatenate(v, axis=0))
    merged['dim'] = dim
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(merged, _OUT_DIR / f"data_chunk_{chunk_idx:04d}.pt")

    n = merged['x_hw'].shape[0]
    print(f"  [chunk {chunk_idx:04d}] {n} amostras  "
          f"S_tot={merged['node_x'].shape[0]}  M_tot={merged['elem_x'].shape[0]}  "
          f"E_tot={merged['edge_index'].shape[1]}  C_tot={merged['cross_edge_index'].shape[1]}")


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

    print(f"\n=== Construção de chunks (femm_mesh_v2) ===")
    print(f"  .npz a processar : {total}  |  chunk_size : {chunk_size}")

    chunk_idx   = 0
    node_offset = 0
    elem_offset = 0
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

        ei = s['edge_index'] + node_offset
        bufs['edge_index'].append(ei)

        cei = s['cross_edge_index'].copy()
        cei[0] += elem_offset   # linha 0: índice do elemento
        cei[1] += node_offset   # linha 1: índice do vértice
        bufs['cross_edge_index'].append(cei)

        for k in bufs:
            if k not in _INDEX_KEYS:
                bufs[k].append(s[k])

        node_offset += int(s['L'].sum())
        elem_offset += int(s['elem_L'].sum())

        if len(bufs['x_hw']) == chunk_size:
            _flush(chunk_idx, bufs, dim)
            chunk_idx   += 1
            node_offset  = 0
            elem_offset  = 0
            dim          = None
            bufs         = {}

    if bufs and bufs.get('x_hw'):
        _flush(chunk_idx, bufs, dim)
        chunk_idx += 1

    print(f"\n=== Resumo build_data_chunks_femm_mesh_v2 ===")
    print(f"  Chunks salvos : {chunk_idx}")
    return chunk_idx


def main():
    print("=== Etapa 1: gen_npz_structures ===")
    from scripts.gen_npz_structures import run as gen_npz_run
    gen_npz_run()
    print("\n=== Etapa 2: agrupamento em chunks ===")
    build()


if __name__ == "__main__":
    main()
