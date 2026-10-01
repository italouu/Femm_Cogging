"""
build_unified_ans_datasets.py
-----------------------------
Constrói, a partir do raw `data/raw/mesh_ans_138x276/` (.ans.gz, mode=
'femm_mesh_v2'), os datasets das 4 arquiteturas comparadas -- FNO2d,
FNO_GNN, GNN_PostBase, FNO_BipartiteGNN -- com o MESMO gabarito B
(curl(A) exato por elemento + média simples por nó), pra re-rodar a
comparação em pé de igualdade. Ver CLAUDE.md "Datasets unificados a partir
do raw mesh_ans_138x276" (2026-10-01) e
src/data_gen/parsers/femm_mesh_unified.py.

Segue a filosofia raw -> npz -> chunks do projeto:

  1. npz    : data/temp/samples_npz/mesh_ans_138x276_unified/<arch>/sample_XXXXXX.npz
              (cada .ans.gz é parseado UMA vez e gera os 4 .npz; retomável --
              amostra com os 4 .npz já presentes é pulada)
  2. chunks : data/torch/data_chunks/mesh_ans_138x276_unified/<arch>/data_chunk_XXXX.pt
              concatenação pura, reaproveitando os build() existentes:
                FNO2d            -> scripts/build_data_chunks.py (grade)
                FNO_GNN/PostBase -> scripts/build_data_chunks_femm_mesh.py (v1)
                FNO_BipartiteGNN -> scripts/build_data_chunks_femm_mesh_v2.py
              (retomável por chunk -- chunk já existente é pulado)

Mesmo chunk_size (32) e mesma ordem de amostras nas 4 pastas -> mesmo split
treino/teste por chunk (split_seed) em todas as archs.

Treino: NnCfg(dataset='mesh_ans_138x276_unified/<arch>', ...).

Execução (a partir da raiz do projeto):
    python -m scripts.build_unified_ans_datasets
"""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
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
MAX_WORKERS        = _dg.npz_max_workers
SAMPLES_PER_WORKER = _dg.npz_samples_per_worker

# None = todas as amostras do raw
MAX_SAMPLES = None

RAW_DIR      = Path("data/raw") / RAW_DATASET
STAGING_ROOT = Path("data/temp/samples_npz") / OUT_DATASET
CHUNKS_ROOT  = Path("data/torch/data_chunks") / OUT_DATASET


def _sample_idx(path: Path) -> int:
    return int(path.name.split('_')[1].split('.')[0])


def _npz_path(arch: str, stem: str, staging_root: Path = None) -> Path:
    return Path(staging_root or STAGING_ROOT) / arch / f"{stem}.npz"


def _save_atomic(arrays: dict, out_path: Path):
    tmp_path = out_path.parent / f"{out_path.stem}.tmp"       # np.savez adiciona .npz
    tmp_npz  = out_path.parent / f"{out_path.stem}.tmp.npz"
    np.savez(tmp_path, **arrays)
    tmp_npz.replace(out_path)


def _parse_and_save_batch(items: list, r_maps: dict, staging_root: Path) -> tuple:
    """Worker: parseia cada .ans.gz uma vez e grava os 4 .npz (um por arch).
    staging_root vem por argumento (não da global) -- no Windows o worker é
    'spawn' e reimporta o módulo, então a global seria a do módulo, não a do
    processo pai."""
    ok, falhas = [], []
    for idx, path in items:
        try:
            r_in, r_ext = r_maps[idx]
            stem = path.name.removesuffix('.ans.gz')
            layouts = parse_ans_gzip_sample_unified(
                path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2, n_r=N_R, n_a=N_A,
                tmp_dir=staging_root)
            for arch in UNIFIED_ARCHS:
                out_path = _npz_path(arch, stem, staging_root)
                if not out_path.exists():
                    _save_atomic(layouts[arch], out_path)
            ok.append(idx)
        except Exception as e:
            falhas.append((idx, repr(e)))
    return ok, falhas


def run_npz(max_samples=MAX_SAMPLES):
    """Etapa 1: raw .ans.gz -> 4 .npz por amostra."""
    for arch in UNIFIED_ARCHS:
        (STAGING_ROOT / arch).mkdir(parents=True, exist_ok=True)

    ans_paths = sorted(RAW_DIR.glob("sample_*.ans.gz"), key=_sample_idx)
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]

    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        rows = list(csv.DictReader(f))
    r_maps = {}
    for p in ans_paths:
        row = rows[_sample_idx(p)]
        r_maps[_sample_idx(p)] = (float(row['inner_diameter [mm]']) / 2,
                                  float(row['outer_diameter [mm]']) / 2)

    pending = [(_sample_idx(p), p) for p in ans_paths
               if not all(_npz_path(a, p.name.removesuffix('.ans.gz')).exists()
                          for a in UNIFIED_ARCHS)]

    print(f"\n=== Etapa 1: raw -> npz (datasets unificados) ===")
    print(f"  origem   : {RAW_DIR}")
    print(f"  destino  : {STAGING_ROOT}/{{{','.join(UNIFIED_ARCHS)}}}")
    print(f"  total    : {len(ans_paths)}  |  já prontos: {len(ans_paths) - len(pending)}"
          f"  |  a processar: {len(pending)}")
    if not pending:
        return

    # Pool novo por RODADA de até MAX_WORKERS tasks -- mesma reciclagem de
    # worker de gen_npz_structures.py::multi_process_v2 (vazamento de
    # matplotlib.tri/scipy acumulado por processo; ver CLAUDE.md).
    batches = [pending[i:i + SAMPLES_PER_WORKER]
               for i in range(0, len(pending), SAMPLES_PER_WORKER)]
    n_ok, todas_falhas = 0, []
    for r in range(0, len(batches), MAX_WORKERS):
        rodada = batches[r:r + MAX_WORKERS]
        with ProcessPoolExecutor(max_workers=len(rodada)) as ex:
            futs = [ex.submit(_parse_and_save_batch, b, r_maps, STAGING_ROOT) for b in rodada]
            for fut in as_completed(futs):
                ok, falhas = fut.result()
                n_ok += len(ok)
                todas_falhas.extend(falhas)
        done = n_ok + len(todas_falhas)
        if done % 100 < SAMPLES_PER_WORKER * MAX_WORKERS or done == len(pending):
            print(f"  [{done}/{len(pending)}] ok={n_ok} falhas={len(todas_falhas)}", flush=True)

    if todas_falhas:
        print(f"  FALHAS ({len(todas_falhas)}):")
        for idx, err in todas_falhas[:20]:
            print(f"    sample_{idx:06d}: {err}")


def run_chunks(max_samples=MAX_SAMPLES):
    """Etapa 2: concatena os .npz de cada arch em data_chunk_*.pt, reaproveitando
    os build() existentes (só os diretórios de entrada/saída são trocados --
    os módulos leem esses caminhos de variáveis globais no momento da chamada)."""
    import scripts.build_data_chunks as b_grid
    import scripts.build_data_chunks_femm_mesh as b_v1
    import scripts.build_data_chunks_femm_mesh_v2 as b_v2

    builders = {
        'FNO2d':            (b_grid, '_NPZ_DIR',     '_DATA_DIR'),
        'FNO_GNN':          (b_v1,   '_STAGING_DIR', '_OUT_DIR'),
        'GNN_PostBase':     (b_v1,   '_STAGING_DIR', '_OUT_DIR'),
        'FNO_BipartiteGNN': (b_v2,   '_STAGING_DIR', '_OUT_DIR'),
    }
    for arch in UNIFIED_ARCHS:
        mod, in_attr, out_attr = builders[arch]
        orig = getattr(mod, in_attr), getattr(mod, out_attr)
        setattr(mod, in_attr, STAGING_ROOT / arch)
        setattr(mod, out_attr, CHUNKS_ROOT / arch)
        try:
            print(f"\n=== Etapa 2: chunks -- {arch} -> {CHUNKS_ROOT / arch} ===")
            mod.build(max_samples=max_samples, chunk_size=CHUNK_SIZE)
        finally:
            setattr(mod, in_attr, orig[0])
            setattr(mod, out_attr, orig[1])


def main():
    run_npz()
    run_chunks()


if __name__ == "__main__":
    main()
