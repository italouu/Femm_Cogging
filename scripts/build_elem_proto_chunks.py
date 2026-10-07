"""
build_elem_proto_chunks.py
--------------------------
PROTÓTIPO (2026-10-07) -- chunks do FNO_BipartiteGNN_Elem (bipartite com os
papéis trocados: elementos = grafo principal com saída Bx,By por elemento,
vértices = auxiliar), direto do raw da bateria, sem staging .npz.

Mesma referência da bateria (scripts/build_unified_ans_chunks_direct.py):
raw data/raw/mesh_ans_138x276/, amostras em ordem de índice, CHUNK_SIZE=32
(125 chunks) -> mesmos chunks na mesma ordem -> mesmo split (split_seed=12)
das 4 archs da bateria. x_hw/y_hw idênticos aos da bateria (ver
src/data_gen/parsers/femm_mesh_elem.py).

Saída: data/torch/data_chunks/mesh_ans_138x276_unified/FNO_BipartiteGNN_Elem/
Escrita atômica via .tmp/ + rename; retomável (chunk existente é pulado).
Reaproveita _bufs_v2 (offsets: edge_index por L, cross linha 0 por elem_L,
linha 1 por L -- coerentes com o layout trocado) e o _flush do
build_data_chunks_femm_mesh_v2 (o log do _flush chama de "M_tot" o que aqui
são os vértices).

Executável 1 de 2 do protótipo (o 2º é scripts/run_elem_proto.py). Antes de
gerar, faz um precheck (raw com 4000 .ans.gz + valid_designs.csv, disco livre,
parser da amostra 0 conferido contra o layout da bateria) -- pular com
--no-precheck.

Execução (raiz do projeto; Windows ou VM Linux):
    python -m scripts.build_elem_proto_chunks                       # 4000 amostras
    python -m scripts.build_elem_proto_chunks --max-samples 64 --out-name _smoke_FNO_BipartiteGNN_Elem
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.build_unified_ans_chunks_direct import (
    RAW_DIR, CHUNKS_ROOT, N_R, N_A, ANG_1, ANG_2, CHUNK_SIZE, MAX_WORKERS,
    _sample_idx, _dim, _bufs_v2,
)
from src.data_gen.parsers.femm_mesh_elem import parse_ans_gzip_sample_elem

OUT_NAME  = 'FNO_BipartiteGNN_Elem'
TMP_PARSE = Path("data/temp") / "mesh_ans_138x276_elem_parse"


def _parse_one(path: Path, r_in: float, r_ext: float, tmp_dir: Path) -> dict:
    return parse_ans_gzip_sample_elem(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
                                      n_r=N_R, n_a=N_A, tmp_dir=tmp_dir)


def _write_chunk(chunk_idx: int, samples: list, out_dir: Path):
    import scripts.build_data_chunks_femm_mesh_v2 as b_v2
    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    name = f"data_chunk_{chunk_idx:04d}.pt"
    orig = b_v2._OUT_DIR
    b_v2._OUT_DIR = tmp_dir
    try:
        b_v2._flush(chunk_idx, _bufs_v2(samples), _dim(samples[0]))
    finally:
        b_v2._OUT_DIR = orig
    (tmp_dir / name).replace(out_dir / name)


def run(max_samples=None, chunk_size=CHUNK_SIZE, out_name=OUT_NAME):
    out_dir = CHUNKS_ROOT / out_name
    TMP_PARSE.mkdir(parents=True, exist_ok=True)
    ans_paths = sorted(RAW_DIR.glob("sample_*.ans.gz"), key=_sample_idx)
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]
    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        rows = list(csv.DictReader(f))

    groups = [ans_paths[i:i + chunk_size] for i in range(0, len(ans_paths), chunk_size)]
    print(f"\n=== Chunks FNO_BipartiteGNN_Elem (protótipo) direto do raw ===")
    print(f"  origem  : {RAW_DIR}  ({len(ans_paths)} amostras)")
    print(f"  destino : {out_dir}")
    print(f"  chunks  : {len(groups)} x {chunk_size}  |  workers: {MAX_WORKERS}")

    t0 = time.time()
    feitos = 0
    for chunk_idx, group in enumerate(groups):
        if (out_dir / f"data_chunk_{chunk_idx:04d}.pt").exists():
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
            samples = [f.result() for f in futs]   # ordem preservada
        _write_chunk(chunk_idx, samples, out_dir)
        del samples
        feitos += 1
        print(f"  chunk {chunk_idx + 1}/{len(groups)} ok  ({time.time() - t:.0f}s, "
              f"total {(time.time() - t0) / 60:.1f} min)", flush=True)

    n_final = len(list(out_dir.glob("data_chunk_*.pt")))
    print(f"\n=== Resumo: {feitos} chunk(s) gerado(s) nesta execução; "
          f"{n_final}/{len(groups)} presentes em {out_dir} ===")
    return n_final == len(groups)


N_RAW_EXPECTED = 4000
MIN_FREE_GB    = 55       # chunks completos ~47 GB (medido 2026-10-07: ~11,7 MB/amostra)


def precheck(out_name=OUT_NAME, max_samples=None) -> bool:
    """Raw, disco e parser (amostra 0: x_hw/y_hw idênticos à bateria, média
    nodal do B por elemento == node_y da bateria). True = pode gerar."""
    import shutil
    import numpy as np
    from src.data_gen.parsers.femm_mesh_v2 import parse_ans_gzip_sample
    from src.data_gen.parsers.ans_parsing import _node_mean_of_elements

    ok = True
    print("\n=== Precheck (chunks) ===", flush=True)
    n_raw = len(list(RAW_DIR.glob("sample_*.ans.gz")))
    has_csv = (RAW_DIR / "valid_designs.csv").exists()
    print(f"  raw    : {RAW_DIR}  {n_raw} .ans.gz  valid_designs.csv={'sim' if has_csv else 'NÃO'}")
    if not has_csv or (max_samples is None and n_raw != N_RAW_EXPECTED):
        print(f"  ERRO: esperado {N_RAW_EXPECTED} sample_*.ans.gz + valid_designs.csv")
        return False

    out_dir = CHUNKS_ROOT / out_name
    n_have = len(list(out_dir.glob("data_chunk_*.pt")))
    free_gb = shutil.disk_usage('.').free / 2**30
    print(f"  disco  : {free_gb:.0f} GB livres  |  chunks já presentes: {n_have}")
    if max_samples is None and n_have == 0 and free_gb < MIN_FREE_GB:
        print(f"  ERRO: < {MIN_FREE_GB} GB livres (chunks completos ~47 GB)")
        ok = False

    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        row = next(csv.DictReader(f))
    p = RAW_DIR / "sample_000000.ans.gz"
    r_in, r_ext = float(row['inner_diameter [mm]']) / 2, float(row['outer_diameter [mm]']) / 2
    TMP_PARSE.mkdir(parents=True, exist_ok=True)
    d = _parse_one(p, r_in, r_ext, TMP_PARSE)
    b = parse_ans_gzip_sample(p, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2, n_r=N_R, n_a=N_A,
                              tmp_dir=TMP_PARSE, target_field='B')
    cei = d['cross_edge_index']
    tri = cei[0][np.argsort(cei[1], kind='stable')].reshape(-1, 3)
    deg = np.bincount(d['edge_index'][1], minlength=int(d['L'][0]))
    checks = {
        'x_hw/y_hw idênticos à bateria': (np.array_equal(d['x_hw'], b['x_hw'])
                                          and np.array_equal(d['y_hw'], b['y_hw'])),
        'média nodal do B por elemento == node_y da bateria':
            np.array_equal(_node_mean_of_elements(tri, d['node_y'], int(d['elem_L'][0])), b['node_y']),
        'grau do grafo dual em [2,3]': bool(deg.min() >= 2 and deg.max() <= 3),
        'tudo finito': all(np.isfinite(v).all() for v in d.values()),
    }
    for msg, c in checks.items():
        print(f"  [{'ok' if c else 'FALHA'}] parser amostra 0: {msg}")
        ok &= bool(c)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--out-name', default=OUT_NAME)
    ap.add_argument('--no-precheck', action='store_true')
    a = ap.parse_args()
    if not a.no_precheck and not precheck(a.out_name, a.max_samples):
        print("\nPrecheck falhou — nada foi gerado.")
        raise SystemExit(2)
    ok = run(a.max_samples, a.chunk_size, a.out_name)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
