"""
build_smooth_ans_chunks_direct.py
---------------------------------
Chunks dos 4 datasets com gabarito B SUAVIZADO do FEMM (node_y por prioridade
de material, y_hw = point_b smooth, polares) -- raw -> chunks direto, sem
staging .npz: pool novo por chunk, escrita atômica (src/data_gen/chunks.py),
retomável. Parser: parse_ans_gzip_sample_smooth (src/data_gen/parsers/
femm_mesh_smooth.py; CLAUDE.md "Suavização de B do FEMM a partir do `.ans`").

Raw de origem: data/raw/<DatagenConfig.dataset>/ -- chunking fixo (CHUNK_SIZE
amostras, mesma ordem), então o split_seed=12 dá o mesmo split treino/teste.
Saída: data/torch/data_chunks/<DatagenConfig.dataset>/<arch>/data_chunk_XXXX.pt

Execução (a partir da raiz do projeto):
    python -m scripts.build_smooth_ans_chunks_direct
"""
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.configs.datagen import DatagenConfig
from src.data_gen.parsers.femm_mesh_smooth import parse_ans_gzip_sample_smooth, SMOOTH_ARCHS
from src.data_gen.chunks import LAYOUT_BY_ARCH, sample_idx as _sample_idx, write_chunk

_dg = DatagenConfig()

# raw e chunks com o mesmo nome: data/raw/<dataset>/ -> data/torch/data_chunks/<dataset>/<arch>/
RAW_DATASET  = _dg.dataset
OUT_DATASET  = _dg.dataset
N_R, N_A     = _dg.n_r, _dg.n_a
ANG_1, ANG_2 = _dg.ang_1, _dg.ang_2
CHUNK_SIZE   = _dg.chunk_size
MAX_WORKERS  = _dg.chunk_max_workers

# Todas as amostras do raw (BLDC_138x276: 6000 -> 188 chunks, 187 × 32 + 1 × 16). Usado também
# por scripts/build_elem_smooth_chunks.py, scripts/build_smooth_datasets.py e
# scripts/postprocess.py. None = todas.
MAX_SAMPLES = None

RAW_DIR     = Path("data/raw") / RAW_DATASET
CHUNKS_ROOT = Path("data/torch/data_chunks") / OUT_DATASET


def _parse_one(path: Path, r_in: float, r_ext: float) -> dict:
    """Worker: 1 .ans.gz -> {arch: dict de arrays}."""
    return parse_ans_gzip_sample_smooth(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
                                         n_r=N_R, n_a=N_A)


def run(max_samples=MAX_SAMPLES, chunk_size=CHUNK_SIZE, chunks_root=CHUNKS_ROOT,
        max_workers=MAX_WORKERS):
    ans_paths = sorted(RAW_DIR.glob("sample_*.ans.gz"), key=_sample_idx)
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]

    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        rows = list(csv.DictReader(f))

    groups = [ans_paths[i:i + chunk_size] for i in range(0, len(ans_paths), chunk_size)]

    print("\n=== Chunks com B suavizado do FEMM, direto do raw ===")
    print(f"  origem  : {RAW_DIR}  ({len(ans_paths)} amostras)")
    print(f"  destino : {chunks_root}/{{{','.join(SMOOTH_ARCHS)}}}")
    print(f"  chunks  : {len(groups)} x {chunk_size}  |  workers: {max_workers}")

    t0 = time.time()
    feitos = 0
    for chunk_idx, group in enumerate(groups):
        name = f"data_chunk_{chunk_idx:04d}.pt"
        if all((chunks_root / a / name).exists() for a in SMOOTH_ARCHS):
            continue

        t = time.time()
        with ProcessPoolExecutor(max_workers=min(max_workers, len(group))) as ex:
            futs = []
            for p in group:
                row = rows[_sample_idx(p)]
                futs.append(ex.submit(_parse_one, p,
                                      float(row['inner_diameter [mm]']) / 2,
                                      float(row['outer_diameter [mm]']) / 2))
            # ordem preservada (mesma ordem de amostras em todas as archs)
            layouts = [f.result() for f in futs]

        for arch in SMOOTH_ARCHS:
            if not (chunks_root / arch / name).exists():
                write_chunk(chunks_root / arch, chunk_idx, [l[arch] for l in layouts],
                            LAYOUT_BY_ARCH[arch])
        del layouts

        feitos += 1
        print(f"  chunk {chunk_idx + 1}/{len(groups)} ok  ({time.time() - t:.0f}s, "
              f"total {(time.time() - t0) / 60:.1f} min)", flush=True)

    print(f"\n=== Resumo: {feitos} chunk(s) gerado(s) nesta execução ===")


def main():
    run()


if __name__ == "__main__":
    main()
