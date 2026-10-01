"""
build_unified_ans_chunks_direct.py
----------------------------------
Variante de scripts/build_unified_ans_datasets.py que gera os chunks dos 4
datasets unificados DIRETO do raw, sem a etapa intermediária de .npz por
amostra (economiza ~105 GB de staging e a escrita/leitura deles).

Raw de origem: data/raw/mesh_ans_138x276/ (sample_*.ans.gz + valid_designs.csv).
Saída: data/torch/data_chunks/mesh_ans_138x276_unified/<arch>/data_chunk_XXXX.pt
       -- mesmas pastas/conteúdo de build_unified_ans_datasets.py.

Fluxo por chunk (CHUNK_SIZE amostras consecutivas):
  1. parse em paralelo (parse_ans_gzip_sample_unified -- 1 .ans.gz -> 4 layouts);
     pool novo por chunk (reciclagem de worker -- vazamento de matplotlib.tri/
     scipy, ver CLAUDE.md);
  2. concatenação com offsets de índice (nó/aresta; + elemento/aresta-cruzada
     no bipartite), gravação via os _flush() dos builders existentes
     (mesmas regras de stack/concat, sem duplicar);
  3. escrita atômica: cada chunk vai pra <arch>/.tmp/ e é renomeado no fim --
     chunk interrompido nunca fica com nome final.
Retomável: chunk presente nas 4 pastas é pulado.

Execução (a partir da raiz do projeto):
    python -m scripts.build_unified_ans_chunks_direct
"""
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.configs.datagen import DatagenConfig
from src.data_gen.parsers.femm_mesh_unified import (
    parse_ans_gzip_sample_unified, UNIFIED_ARCHS,
)

_dg = DatagenConfig()

RAW_DATASET  = 'mesh_ans_138x276'
OUT_DATASET  = 'mesh_ans_138x276_unified'
N_R, N_A     = _dg.n_r, _dg.n_a
ANG_1, ANG_2 = _dg.ang_1, _dg.ang_2
CHUNK_SIZE   = _dg.chunk_size
MAX_WORKERS  = _dg.npz_max_workers

# None = todas as amostras do raw
MAX_SAMPLES = None

RAW_DIR     = Path("data/raw") / RAW_DATASET
CHUNKS_ROOT = Path("data/torch/data_chunks") / OUT_DATASET
TMP_PARSE   = Path("data/temp") / f"{OUT_DATASET}_parse"   # .ans descomprimido temporário


def _sample_idx(path: Path) -> int:
    return int(path.name.split('_')[1].split('.')[0])


def _parse_one(path: Path, r_in: float, r_ext: float, tmp_dir: Path) -> dict:
    """Worker: 1 .ans.gz -> {arch: dict de arrays}."""
    return parse_ans_gzip_sample_unified(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
                                          n_r=N_R, n_a=N_A, tmp_dir=tmp_dir)


def _dim(s: dict) -> tuple:
    return (int(s['dim_H']), int(s['dim_W']))


def _bufs_grid(samples: list) -> dict:
    # mesmo critério de build_data_chunks.build (sem grafo)
    return {k: [s[k] for s in samples] for k in samples[0] if k not in ('dim_H', 'dim_W')}


def _bufs_v1(samples: list) -> dict:
    # mesmo critério de build_data_chunks_femm_mesh.build
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


def _bufs_v2(samples: list) -> dict:
    # mesmo critério de build_data_chunks_femm_mesh_v2.build
    keys = [k for k in samples[0] if k not in ('dim_H', 'dim_W')]
    bufs = {k: [] for k in keys}
    node_offset = elem_offset = 0
    for s in samples:
        bufs['edge_index'].append(s['edge_index'] + node_offset)
        cei = s['cross_edge_index'].copy()
        cei[0] += elem_offset   # linha 0: elemento
        cei[1] += node_offset   # linha 1: vértice
        bufs['cross_edge_index'].append(cei)
        for k in keys:
            if k not in ('edge_index', 'cross_edge_index'):
                bufs[k].append(s[k])
        node_offset += int(s['L'].sum())
        elem_offset += int(s['elem_L'].sum())
    return bufs


def _builders():
    import scripts.build_data_chunks as b_grid
    import scripts.build_data_chunks_femm_mesh as b_v1
    import scripts.build_data_chunks_femm_mesh_v2 as b_v2
    # arch -> (montagem de bufs, módulo cujo _flush grava, nome da global de saída)
    return {
        'FNO2d':            (_bufs_grid, b_grid, '_DATA_DIR'),
        'FNO_GNN':          (_bufs_v1,   b_v1,   '_OUT_DIR'),
        'GNN_PostBase':     (_bufs_v1,   b_v1,   '_OUT_DIR'),
        'FNO_BipartiteGNN': (_bufs_v2,   b_v2,   '_OUT_DIR'),
    }


def _write_chunk(chunk_idx: int, arch: str, samples: list, builders: dict):
    make_bufs, mod, out_attr = builders[arch]
    final_dir = CHUNKS_ROOT / arch
    tmp_dir = final_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    name = f"data_chunk_{chunk_idx:04d}.pt"

    orig = getattr(mod, out_attr)
    setattr(mod, out_attr, tmp_dir)
    try:
        mod._flush(chunk_idx, make_bufs(samples), _dim(samples[0]))
    finally:
        setattr(mod, out_attr, orig)
    (tmp_dir / name).replace(final_dir / name)


def run(max_samples=MAX_SAMPLES, chunk_size=CHUNK_SIZE):
    TMP_PARSE.mkdir(parents=True, exist_ok=True)
    ans_paths = sorted(RAW_DIR.glob("sample_*.ans.gz"), key=_sample_idx)
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]

    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        rows = list(csv.DictReader(f))

    groups = [ans_paths[i:i + chunk_size] for i in range(0, len(ans_paths), chunk_size)]
    builders = _builders()

    print(f"\n=== Chunks unificados direto do raw ===")
    print(f"  origem  : {RAW_DIR}  ({len(ans_paths)} amostras)")
    print(f"  destino : {CHUNKS_ROOT}/{{{','.join(UNIFIED_ARCHS)}}}")
    print(f"  chunks  : {len(groups)} x {chunk_size}  |  workers: {MAX_WORKERS}")

    t0 = time.time()
    feitos = 0
    for chunk_idx, group in enumerate(groups):
        name = f"data_chunk_{chunk_idx:04d}.pt"
        if all((CHUNKS_ROOT / a / name).exists() for a in UNIFIED_ARCHS):
            continue

        t = time.time()
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(group))) as ex:
            futs = []
            for p in group:
                row = rows[_sample_idx(p)]
                futs.append(ex.submit(_parse_one, p,
                                      float(row['inner_diameter [mm]']) / 2,
                                      float(row['outer_diameter [mm]']) / 2,
                                      TMP_PARSE))
            # ordem preservada (mesma ordem de amostras em todas as archs)
            layouts = [f.result() for f in futs]

        for arch in UNIFIED_ARCHS:
            if not (CHUNKS_ROOT / arch / name).exists():
                _write_chunk(chunk_idx, arch, [l[arch] for l in layouts], builders)
        del layouts

        feitos += 1
        print(f"  chunk {chunk_idx + 1}/{len(groups)} ok  ({time.time() - t:.0f}s, "
              f"total {(time.time() - t0) / 60:.1f} min)", flush=True)

    print(f"\n=== Resumo: {feitos} chunk(s) gerado(s) nesta execução ===")


def main():
    run()


if __name__ == "__main__":
    main()
