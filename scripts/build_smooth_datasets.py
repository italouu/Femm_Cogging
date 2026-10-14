"""
build_smooth_datasets.py
------------------------
Gera os 5 datasets da bateria smooth com um comando, direto do raw
(data/raw/mesh_ans_138x276/), nas MAX_SAMPLES primeiras amostras (2000 -> 63 chunks):
  - FNO2d, FNO_GNN, GNN_PostBase, FNO_BipartiteGNN  (scripts.build_smooth_ans_chunks_direct)
  - FNO_BipartiteGNN_Elem                           (scripts.build_elem_smooth_chunks)
Saída: data/torch/data_chunks/mesh_ans_138x276/<arch>/data_chunk_XXXX.pt
Retomável: chunks já presentes são pulados (pode rodar de novo após interrupção).

Ordem: precheck do parser Elem (amostra 0) + disco -> 4 datasets -> Elem -> contagem final.
Medido em 2026-10-13 (Windows local, 6 workers): ~15 s/chunk nos 4 datasets e ~15 s/chunk
na Elem; ~0,86 GB + ~0,36 GB por chunk -> 2000 amostras ≈ 32 min, ≈ 77 GB.

Execução (raiz do projeto):  python -m scripts.build_smooth_datasets
"""
import math
import shutil
import time

from scripts import build_elem_smooth_chunks as elem
from scripts import build_smooth_ans_chunks_direct as smooth
from src.data_gen.parsers.femm_mesh_smooth import SMOOTH_ARCHS

GB_PER_CHUNK = 0.86 + 0.36   # 4 datasets + Elem, medido 2026-10-13
ARCHS = (*SMOOTH_ARCHS, elem.OUT_NAME)


def main():
    n_samples = smooth.MAX_SAMPLES
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
    smooth.run(max_samples=n_samples, chunks_root=smooth.CHUNKS_ROOT)   # explícito: o default de run() é fixado na definição
    elem.run(max_samples=n_samples)

    print(f"\n=== Contagem final ({(time.time() - t0) / 60:.1f} min) ===")
    ok = True
    for arch in ARCHS:
        n = len(list((smooth.CHUNKS_ROOT / arch).glob("data_chunk_*.pt")))
        ok &= n == n_chunks
        print(f"  {arch:24s} {n}/{n_chunks} {'ok' if n == n_chunks else 'INCOMPLETO'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
