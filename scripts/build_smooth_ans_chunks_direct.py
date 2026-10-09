"""
build_smooth_ans_chunks_direct.py
---------------------------------
Chunks dos 4 datasets com gabarito B SUAVIZADO do FEMM (node_y por prioridade
de material, y_hw = point_b smooth) -- raw -> chunks direto, sem staging .npz.
Mesma mecânica de scripts/build_unified_ans_chunks_direct.py (pool novo por
chunk, escrita atômica em <arch>/.tmp/, retomável), só trocando o parser por
parse_ans_gzip_sample_smooth (ver src/data_gen/parsers/femm_mesh_smooth.py e
CLAUDE.md "Suavização de B do FEMM a partir do `.ans`"). 2026-10-09.

Raw de origem: o MESMO raw da bateria oficial sem smooth
(data/raw/mesh_ans_138x276/) -- mesmo chunking (CHUNK_SIZE amostras, mesma
ordem), então o split_seed=12 dá o mesmo split treino/teste.
Saída: data/torch/data_chunks/mesh_ans_138x276_smooth/<arch>/data_chunk_XXXX.pt

Execução (a partir da raiz do projeto):
    python -m scripts.build_smooth_ans_chunks_direct
"""
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.configs.datagen import DatagenConfig
from src.data_gen.parsers.femm_mesh_smooth import parse_ans_gzip_sample_smooth, SMOOTH_ARCHS
from scripts.build_unified_ans_chunks_direct import _sample_idx, _dim, _builders

_dg = DatagenConfig()

# raw da bateria oficial sem smooth (mesh_ans_138x276_unified) -- fonte única
RAW_DATASET  = 'mesh_ans_138x276'
OUT_DATASET  = 'mesh_ans_138x276_smooth'
N_R, N_A     = _dg.n_r, _dg.n_a
ANG_1, ANG_2 = _dg.ang_1, _dg.ang_2
CHUNK_SIZE   = _dg.chunk_size
MAX_WORKERS  = _dg.npz_max_workers

# None = todas as amostras do raw
MAX_SAMPLES = None

RAW_DIR     = Path("data/raw") / RAW_DATASET
CHUNKS_ROOT = Path("data/torch/data_chunks") / OUT_DATASET
TMP_PARSE   = Path("data/temp") / f"{OUT_DATASET}_parse"   # .ans descomprimido temporário


def _parse_one(path: Path, r_in: float, r_ext: float, tmp_dir: Path) -> dict:
    """Worker: 1 .ans.gz -> {arch: dict de arrays}."""
    return parse_ans_gzip_sample_smooth(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
                                         n_r=N_R, n_a=N_A, tmp_dir=tmp_dir)


def _write_chunk(chunk_idx: int, arch: str, samples: list, builders: dict, chunks_root: Path):
    make_bufs, mod, out_attr = builders[arch]
    final_dir = chunks_root / arch
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


def run(max_samples=MAX_SAMPLES, chunk_size=CHUNK_SIZE, chunks_root=CHUNKS_ROOT,
        max_workers=MAX_WORKERS):
    TMP_PARSE.mkdir(parents=True, exist_ok=True)
    ans_paths = sorted(RAW_DIR.glob("sample_*.ans.gz"), key=_sample_idx)
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]

    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        rows = list(csv.DictReader(f))

    groups = [ans_paths[i:i + chunk_size] for i in range(0, len(ans_paths), chunk_size)]
    builders = _builders()

    print(f"\n=== Chunks com B suavizado do FEMM, direto do raw ===")
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
                                      float(row['outer_diameter [mm]']) / 2,
                                      TMP_PARSE))
            # ordem preservada (mesma ordem de amostras em todas as archs)
            layouts = [f.result() for f in futs]

        for arch in SMOOTH_ARCHS:
            if not (chunks_root / arch / name).exists():
                _write_chunk(chunk_idx, arch, [l[arch] for l in layouts], builders, chunks_root)
        del layouts

        feitos += 1
        print(f"  chunk {chunk_idx + 1}/{len(groups)} ok  ({time.time() - t:.0f}s, "
              f"total {(time.time() - t0) / 60:.1f} min)", flush=True)

    print(f"\n=== Resumo: {feitos} chunk(s) gerado(s) nesta execução ===")


def main():
    run()


if __name__ == "__main__":
    main()
