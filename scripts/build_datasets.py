"""
build_datasets.py
-----------------
Gera os 5 datasets da bateria bldc (src/configs/batteries/bldc.py) com um comando, direto
do raw (data/raw/<DatagenConfig.dataset>/, BLDC_138x276 com correntes), nas MAX_SAMPLES
primeiras amostras (None = todas; 6000 -> 188 chunks):
  - FNO2d, FNO_GNN, GNN_PostBase, FNO_BipartiteGNN  (layouts de smooth_layouts)
  - FNO_BipartiteGNN_Elem                           (layout de elem_smooth)
Saída: data/torch/data_chunks/<DatagenConfig.dataset>/<arch>/data_chunk_XXXX.pt
Retomável: chunks já presentes são pulados (pode rodar de novo após interrupção).

Ordem: precheck do parser Elem (amostra 0) + disco -> 5 datasets numa passada -> contagem final.
Passada única (2026-10-22): cada amostra é lida e parseada UMA vez
(elem_smooth.parse_all_layouts) e o chunk de cada arch que faltar é gravado
(escrita atômica, src/data_gen/chunks.py) -- antes, os 4 datasets e a Elem eram duas passadas
pelo raw (scripts.build_chunks_main + scripts.build_chunks_elem, que
continuam executáveis sozinhos). Chunks idênticos aos das duas passadas (torch.equal).
Medido em 2026-10-13 (Windows local, 6 workers): ~15 s/chunk nos 4 datasets e ~15 s/chunk
na Elem; ~0,86 GB + ~0,36 GB por chunk -> 2000 amostras ≈ 32 min, ≈ 77 GB.

Execução (raiz do projeto):  python -m scripts.build_datasets
"""
import csv
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

from scripts import build_chunks_elem as elem
from scripts import build_chunks_main as smooth
from src.configs.datagen import DatagenConfig
from src.data_gen.chunks import LAYOUT_BY_ARCH, sample_idx as _sample_idx, write_chunk
from src.data_gen.parsers.elem_smooth import parse_all_layouts
from src.data_gen.parsers.smooth_layouts import SMOOTH_ARCHS

GB_PER_CHUNK = 0.86 + 0.36   # 4 datasets + Elem, medido 2026-10-13
ARCHS = (*SMOOTH_ARCHS, elem.OUT_NAME)


def _parse_one(path, r_in: float, r_ext: float) -> dict:
    """Worker: 1 .ans.gz -> {arch: dict de arrays} das 5 archs."""
    return parse_all_layouts(path, r_in, r_ext, ang_1=smooth.ANG_1, ang_2=smooth.ANG_2,
                                     n_r=smooth.N_R, n_a=smooth.N_A)


def run(max_samples=smooth.MAX_SAMPLES, chunk_size=smooth.CHUNK_SIZE,
        chunks_root=smooth.CHUNKS_ROOT, max_workers=smooth.MAX_WORKERS):
    """Gera os chunks que faltam das 5 archs numa passada pelo raw (retomável por chunk)."""
    ans_paths = sorted(smooth.RAW_DIR.glob("sample_*.ans.gz"), key=_sample_idx)
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]
    with open(smooth.RAW_DIR / "valid_designs.csv", newline='') as f:
        rows = list(csv.DictReader(f))

    groups = [ans_paths[i:i + chunk_size] for i in range(0, len(ans_paths), chunk_size)]
    print("\n=== Chunks dos 5 datasets smooth, direto do raw (passada única) ===")
    print(f"  origem  : {smooth.RAW_DIR}  ({len(ans_paths)} amostras)")
    print(f"  destino : {chunks_root}/{{{','.join(ARCHS)}}}")
    print(f"  chunks  : {len(groups)} x {chunk_size}  |  workers: {max_workers}")

    t0 = time.time()
    feitos = 0
    for chunk_idx, group in enumerate(groups):
        name = f"data_chunk_{chunk_idx:04d}.pt"
        missing = [a for a in ARCHS if not (chunks_root / a / name).exists()]
        if not missing:
            continue

        t = time.time()
        with ProcessPoolExecutor(max_workers=min(max_workers, len(group))) as ex:
            futs = []
            for p in group:
                row = rows[_sample_idx(p)]
                futs.append(ex.submit(_parse_one, p,
                                      float(row['inner_diameter [mm]']) / 2,
                                      float(row['outer_diameter [mm]']) / 2))
            layouts = [f.result() for f in futs]   # ordem preservada

        for arch in missing:
            write_chunk(chunks_root / arch, chunk_idx, [l[arch] for l in layouts],
                        LAYOUT_BY_ARCH[arch])
        del layouts

        feitos += 1
        print(f"  chunk {chunk_idx + 1}/{len(groups)} ok  ({time.time() - t:.0f}s, "
              f"total {(time.time() - t0) / 60:.1f} min)", flush=True)

    print(f"\n=== Resumo: {feitos} chunk(s) gerado(s) nesta execução ===")


def main():
    n_samples = smooth.MAX_SAMPLES
    if n_samples is None:   # todas (2026-10-14): o raw precisa estar completo (DatagenConfig.n_samples)
        n_raw = len(list(smooth.RAW_DIR.glob("sample_*.ans.gz")))
        if n_raw != DatagenConfig().n_samples:
            print(f"  ERRO: {smooth.RAW_DIR} tem {n_raw} .ans.gz, esperado "
                  f"{DatagenConfig().n_samples} (geração incompleta?) - nada foi gerado.")
            raise SystemExit(2)
        n_samples = n_raw
    n_chunks = math.ceil(n_samples / smooth.CHUNK_SIZE)
    need_gb = n_chunks * GB_PER_CHUNK
    free_gb = shutil.disk_usage('.').free / 2**30
    print(f"\n=== Datasets smooth: {n_samples} amostras -> {n_chunks} chunks x 5 datasets ===")
    print(f"  disco: {free_gb:.0f} GB livres, necessario ~ {need_gb:.0f} GB (se vazio)")
    if free_gb < need_gb * 1.1:
        print("  ERRO: disco insuficiente - nada foi gerado.")
        raise SystemExit(2)
    if not elem.precheck(max_samples=n_samples):
        print("\nPrecheck falhou — nada foi gerado.")
        raise SystemExit(2)

    t0 = time.time()
    run(max_samples=n_samples)

    print(f"\n=== Contagem final ({(time.time() - t0) / 60:.1f} min) ===")
    ok = True
    for arch in ARCHS:
        n = len(list((smooth.CHUNKS_ROOT / arch).glob("data_chunk_*.pt")))
        ok &= n == n_chunks
        print(f"  {arch:24s} {n}/{n_chunks} {'ok' if n == n_chunks else 'INCOMPLETO'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
