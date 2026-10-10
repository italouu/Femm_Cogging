"""
build_elem_smooth_chunks.py
---------------------------
Chunks do FNO_BipartiteGNN_Elem com gabarito B SUAVIZADO do FEMM (2026-10-10)
-- versão smooth de scripts/build_elem_proto_chunks.py (mesma mecânica: raw
-> chunks direto, pool novo por chunk, escrita atômica via .tmp/, retomável),
só trocando o parser por parse_ans_gzip_sample_elem_smooth (ver
src/data_gen/parsers/femm_mesh_elem_smooth.py).

Raw: o MESMO da bateria (data/raw/mesh_ans_138x276/), amostras em ordem de
índice, CHUNK_SIZE=32 (125 chunks) -> mesmo split (split_seed=12) da bateria
smooth (scripts/run_best_configs_smooth.py).
Saída: data/torch/data_chunks/mesh_ans_138x276_smooth/FNO_BipartiteGNN_Elem/

Precheck (amostra 0, pular com --no-precheck): x_hw/y_hw idênticos ao layout
smooth da bipartite; node_y == média dos 3 valores nodais suavizados do
elemento; estrutura (grafo dual, vértices, arestas cruzadas) idêntica à do
protótipo sem smooth.

Execução (raiz do projeto; Windows ou VM Linux):
    python -m scripts.build_elem_smooth_chunks
    python -m scripts.build_elem_smooth_chunks --max-samples 64 --out-name _smoke_FNO_BipartiteGNN_Elem
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.build_smooth_ans_chunks_direct import (
    RAW_DIR, CHUNKS_ROOT, N_R, N_A, ANG_1, ANG_2, CHUNK_SIZE, MAX_WORKERS,
)
from scripts.build_unified_ans_chunks_direct import _sample_idx
from scripts.build_elem_proto_chunks import _write_chunk, N_RAW_EXPECTED, MIN_FREE_GB
from src.data_gen.parsers.femm_mesh_elem_smooth import parse_ans_gzip_sample_elem_smooth

OUT_NAME  = 'FNO_BipartiteGNN_Elem'
TMP_PARSE = Path("data/temp") / "mesh_ans_138x276_elem_smooth_parse"


def _parse_one(path: Path, r_in: float, r_ext: float, tmp_dir: Path) -> dict:
    return parse_ans_gzip_sample_elem_smooth(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
                                             n_r=N_R, n_a=N_A, tmp_dir=tmp_dir)


def run(max_samples=None, chunk_size=CHUNK_SIZE, out_name=OUT_NAME):
    out_dir = CHUNKS_ROOT / out_name
    TMP_PARSE.mkdir(parents=True, exist_ok=True)
    ans_paths = sorted(RAW_DIR.glob("sample_*.ans.gz"), key=_sample_idx)
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]
    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        rows = list(csv.DictReader(f))

    groups = [ans_paths[i:i + chunk_size] for i in range(0, len(ans_paths), chunk_size)]
    print(f"\n=== Chunks FNO_BipartiteGNN_Elem (B suavizado) direto do raw ===")
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


def precheck(out_name=OUT_NAME, max_samples=None) -> bool:
    """Raw, disco e parser da amostra 0. True = pode gerar."""
    import shutil
    import numpy as np
    from src.data_gen.parsers.ans_b_smoothing import load_ans, element_b, nodal_b
    from src.data_gen.parsers.femm_mesh_elem import parse_ans_gzip_sample_elem
    from src.data_gen.parsers.femm_mesh_smooth import parse_ans_gzip_sample_smooth

    ok = True
    print("\n=== Precheck (chunks elem smooth) ===", flush=True)
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
    kw = dict(ang_1=ANG_1, ang_2=ANG_2, n_r=N_R, n_a=N_A, tmp_dir=TMP_PARSE)
    d = _parse_one(p, r_in, r_ext, TMP_PARSE)
    e = parse_ans_gzip_sample_elem(p, r_in, r_ext, **kw)
    sb = parse_ans_gzip_sample_smooth(p, r_in, r_ext, **kw)['FNO_BipartiteGNN']

    mesh = load_ans(p)
    B1, B2 = element_b(mesh)
    b1, b2 = nodal_b(mesh, B1, B2)
    ref = np.stack([b1.mean(axis=1), b2.mean(axis=1)], axis=1)
    err = np.abs(d['node_y'] - ref).max()
    same_struct = all(np.array_equal(d[k], e[k]) for k in
                      ('node_x', 'edge_index', 'edge_attr', 'elem_x', 'cross_edge_index',
                       'cross_edge_attr', 'x_hw', 'L', 'elem_L', 'E_L', 'C_L'))
    checks = {
        'x_hw/y_hw idênticos ao layout smooth da bipartite':
            np.array_equal(d['x_hw'], sb['x_hw']) and np.array_equal(d['y_hw'], sb['y_hw']),
        f'node_y == média dos 3 valores nodais suavizados (max |dif| {err:.1e} T)': err < 1e-5,
        'estrutura idêntica ao protótipo sem smooth': same_struct,
        'node_y difere do curl(A) P0 do protótipo sem smooth':
            not np.array_equal(d['node_y'], e['node_y']),
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
