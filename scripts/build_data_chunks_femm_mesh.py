"""
build_data_chunks_femm_mesh.py
-------------------------------
Agrupa os .npz JÁ FILTRADOS PELO PARSER (data/temp/samples_npz/
<dataset>_<npz_parser>/sample_XXXXXX.npz — ver o branch mode='femm_mesh'
de scripts/gen_npz_structures.py) em data_chunk_XXXX.pt finais, no mesmo
espírito de build_data_chunks.py (offset de edge_index, E_L calculado no
agrupamento).

Simetria com o pipeline qtree/grid (2026-07-27): mesmos 3 comandos, mesmas
pastas-base (data/raw/, data/temp/samples_npz/, data/torch/data_chunks/) —
    1. generate_data.py       -> dispatcha p/ generate_data_femm_mesh.py ->
                                  staging BRUTO em data/raw/<dataset>/
                                  (fonte única, não fica em data/temp/)
    2. gen_npz_structures.py  -> branch mode='femm_mesh' -> aplica parser,
                                  grava em data/temp/samples_npz/<dataset>_<npz_parser>/
    3. build_data_chunks.py   -> dispatcha p/ este módulo -> main() abaixo
                                  chama gen_npz_structures + build()

Saída: data/torch/data_chunks/<dataset>_<npz_parser>/data_chunk_XXXX.pt —
mesmo diretório-base do pipeline qtree/grid; o sufixo _<npz_parser> (mantido
também no npz intermediário, diferente do pipeline qtree/grid que não sufixa)
evita misturar chunks de parsers diferentes enquanto os chunks ainda são
apontados manualmente no treino (NnCfg.dataset).

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
# [REMOVIDO] lia direto do staging bruto (sem parser aplicado) -- desde a
# introdução do branch mode='femm_mesh' em gen_npz_structures.py, o
# agrupamento em chunks passa a ler do staging JÁ FILTRADO pelo parser
# (node_x/node_y/edge_attr com as colunas do PARSER_REGISTRY[npz_parser]),
# não mais do staging bruto de 9/4 colunas completas.
# _STAGING_DIR = PROJECT_ROOT / "data" / "temp" / "samples_mesh" / DATASET
# _OUT_DIR     = PROJECT_ROOT / "data" / "torch" / "data_chunks" / DATASET
# [REMOVIDO] _STAGING_DIR apontava para data/temp/samples_mesh_parsed/ (nome
# próprio do femm_mesh) -- unificado com data/temp/samples_npz/ (mesma pasta
# usada por gen_npz_structures.py no pipeline qtree/grid, ver docstring acima).
# _STAGING_DIR = PROJECT_ROOT / "data" / "temp" / "samples_mesh_parsed" / _dg.parsed_dataset_name
_STAGING_DIR  = PROJECT_ROOT / "data" / "temp" / "samples_npz" / _dg.parsed_dataset_name
_OUT_DIR      = PROJECT_ROOT / "data" / "torch" / "data_chunks" / _dg.parsed_dataset_name

# [REMOVIDO] _DROP_KEYS (r_in_mm/r_ext_mm/ang_1_deg/ang_2_deg) -- este módulo
# passa a ser concatenação PURA (sem decidir o que é dado final ou não). O
# descarte desses metadados de geometria (não usados no treino, só
# rastreabilidade do staging bruto) agora acontece em apply_parser_femm_mesh.py
# -- o .npz que chega aqui já É o formato final, chave por chave.
# _DROP_KEYS = frozenset({'r_in_mm', 'r_ext_mm', 'ang_1_deg', 'ang_2_deg'})

# chaves cujos arrays por amostra têm shape [C,H,W] ou [H,W] — precisam de
# stack (empilha um eixo de batch novo), não concat
_STACK_KEYS = frozenset({'x_hw', 'y_hw', 'a_hw'})


def _load_npz(path: Path) -> dict:
    """Carrega um .npz (já no formato final, sem chaves a descartar/renomear)
    e reconstrói 'dim' como tupla — mesma convenção de build_data_chunks.py
    (np.savez não tem tupla nativa, por isso dim_H/dim_W são salvos separados
    e remontados aqui; não é uma decisão de conteúdo, é mecânica de
    serialização, igual no pipeline qtree)."""
    d = np.load(path)
    result = {'dim': (int(d['dim_H']), int(d['dim_W']))}
    for key in d.files:
        if key in ('dim_H', 'dim_W'):
            continue
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
    # [REMOVIDO] etapa 1 chamava generate_data_femm_mesh.run() automaticamente
    # -- geração do staging bruto (1 solve FEM por amostra) agora é comando
    # próprio e explícito (python -m scripts.generate_data, que já dispatcha
    # pra generate_data_femm_mesh.py), igual ao pipeline qtree/grid onde
    # build_data_chunks.py NÃO chama generate_data.py -- espera o raw já
    # existir. Simetria com build_data_chunks.py::main() abaixo.
    # from scripts.generate_data_femm_mesh import run as gen_run
    # gen_run()
    print("=== Etapa 1: gen_npz_structures (aplica parser sobre staging bruto) ===")
    from scripts.gen_npz_structures import run as gen_npz_run
    gen_npz_run()
    print("\n=== Etapa 2: agrupamento em chunks ===")
    build()


if __name__ == "__main__":
    main()
