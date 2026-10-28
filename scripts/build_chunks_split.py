"""
build_chunks_split.py
---------------------
Chunks do FNO_SplitGNN (grafo dual de elementos + vértices divididos por material),
gabarito B suavizado do FEMM: raw -> chunks direto, pool novo por chunk, escrita atômica
(src/data_gen/chunks.py), retomável. Parser: parse_split_smooth
(src/data_gen/parsers/split_layout.py). Não faz parte do scripts.build_datasets.

Raw: data/raw/<DatagenConfig.dataset>/ (o mesmo das outras archs), amostras em ordem de
índice, CHUNK_SIZE=32 (6000 amostras -> 188 chunks) -> mesmo split (split_seed=12) da
bateria bldc. Saída: data/torch/data_chunks/<DatagenConfig.dataset>/FNO_SplitGNN/

Precheck (amostra 0, pular com --no-precheck): node_y, x_hw, y_hw e Edge_dual idênticos aos
da Elem; elem_y == node_y do layout de vértices (FNO_GNN) na cópia do material escolhido
pela prioridade, fora do corte; Edge_aux ligando cada elemento a 3 cópias distintas do seu
material; vértices de 120° fundidos; sen/cos contínuos nas arestas.

Execução (raiz do projeto; Windows ou VM Linux):
    python -m scripts.build_chunks_split
    python -m scripts.build_chunks_split --max-samples 64 --chunk-size 8 --out-name _smoke_FNO_SplitGNN
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.build_chunks_main import (
    RAW_DIR, CHUNKS_ROOT, N_R, N_A, ANG_1, ANG_2, CHUNK_SIZE, MAX_WORKERS, MAX_SAMPLES,
)
from src.configs.datagen import DatagenConfig
from src.data_gen.chunks import LAYOUT_BY_ARCH, sample_idx as _sample_idx, write_chunk
from src.data_gen.parsers.split_layout import SPLIT_ARCH, parse_split_smooth

GB_PER_CHUNK = 0.66   # medido (32 amostras; 188 chunks ≈ 124 GB)
N_RAW_EXPECTED = DatagenConfig().n_samples   # precheck com max_samples=None: raw completo
OUT_NAME = SPLIT_ARCH


def _parse_one(path: Path, r_in: float, r_ext: float) -> dict:
    return parse_split_smooth(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2, n_r=N_R, n_a=N_A)


def run(max_samples=MAX_SAMPLES, chunk_size=CHUNK_SIZE, out_name=OUT_NAME):
    out_dir = CHUNKS_ROOT / out_name
    ans_paths = sorted(RAW_DIR.glob("sample_*.ans.gz"), key=_sample_idx)
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]
    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        rows = list(csv.DictReader(f))

    groups = [ans_paths[i:i + chunk_size] for i in range(0, len(ans_paths), chunk_size)]
    print(f"\n=== Chunks {SPLIT_ARCH} (B suavizado) direto do raw ===")
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
                                      float(row['outer_diameter [mm]']) / 2))
            samples = [f.result() for f in futs]   # ordem preservada
        write_chunk(out_dir, chunk_idx, samples, LAYOUT_BY_ARCH[SPLIT_ARCH])
        del samples
        feitos += 1
        print(f"  chunk {chunk_idx + 1}/{len(groups)} ok  ({time.time() - t:.0f}s, "
              f"total {(time.time() - t0) / 60:.1f} min)", flush=True)

    n_final = len(list(out_dir.glob("data_chunk_*.pt")))
    print(f"\n=== Resumo: {feitos} chunk(s) gerado(s) nesta execução; "
          f"{n_final}/{len(groups)} presentes em {out_dir} ===")
    return n_final == len(groups)


def precheck(out_name=OUT_NAME, max_samples=MAX_SAMPLES, chunk_size=CHUNK_SIZE) -> bool:
    """Raw, disco e parser da amostra 0. True = pode gerar."""
    import math
    import shutil
    import numpy as np
    from src.data_gen.parsers.ans_parsing import (read_ans, _parse_label_materials,
                                                  _wrap_edge_pairs)
    from src.data_gen.parsers.elem_smooth import parse_all_layouts
    from src.data_gen.parsers.split_layout import _split_vertices

    ok = True
    print(f"\n=== Precheck (chunks {SPLIT_ARCH}) ===", flush=True)
    n_raw = len(list(RAW_DIR.glob("sample_*.ans.gz")))
    has_csv = (RAW_DIR / "valid_designs.csv").exists()
    print(f"  raw    : {RAW_DIR}  {n_raw} .ans.gz  valid_designs.csv={'sim' if has_csv else 'NÃO'}")
    if not has_csv or (max_samples is None and n_raw != N_RAW_EXPECTED):
        print(f"  ERRO: esperado {N_RAW_EXPECTED} sample_*.ans.gz + valid_designs.csv")
        return False

    out_dir = CHUNKS_ROOT / out_name
    n_have = len(list(out_dir.glob("data_chunk_*.pt")))
    n_chunks = math.ceil((max_samples or n_raw) / chunk_size)
    need_gb = (n_chunks - n_have) * GB_PER_CHUNK * chunk_size / 32
    free_gb = shutil.disk_usage('.').free / 2**30
    print(f"  disco  : {free_gb:.0f} GB livres, necessario ~ {need_gb:.0f} GB  |  "
          f"chunks já presentes: {n_have}")
    if free_gb < need_gb * 1.1:
        print("  ERRO: disco insuficiente")
        ok = False

    with open(RAW_DIR / "valid_designs.csv", newline='') as f:
        row = next(csv.DictReader(f))
    p = RAW_DIR / "sample_000000.ans.gz"
    r_in, r_ext = float(row['inner_diameter [mm]']) / 2, float(row['outer_diameter [mm]']) / 2
    d = _parse_one(p, r_in, r_ext)
    lay = parse_all_layouts(p, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2, n_r=N_R, n_a=N_A)
    el, vx = lay['FNO_BipartiteGNN_Elem'], lay['FNO_GNN']

    # cópias recalculadas (mesma função do parser) para cruzar com o layout de vértices
    ans = read_ans(p)
    nodes, elems = ans.nodes, ans.elems
    mat_id, _, label_M = _parse_label_materials(ans.lines)
    elem_cls = mat_id[elems[:, 3]] * 4 + (np.rint(label_M[elems[:, 3]]).astype(np.int64) + 1)
    copy_of_corner, copy_vertex, _ = _split_vertices(nodes, elems, elem_cls, ANG_1, ANG_2)
    idx_1, idx_2 = _wrap_edge_pairs(nodes[:, :2], ANG_1, ANG_2)
    on_cut = np.zeros(nodes.shape[0], dtype=bool)
    on_cut[idx_1] = True
    on_cut[idx_2] = True
    sel = (~on_cut[copy_vertex]) & (d['elem_x'][:, 4] == vx['node_x'][copy_vertex, 0])
    err_y = np.abs(d['elem_y'][sel] - vx['node_y'][copy_vertex[sel]]).max()
    n_sel = np.unique(copy_vertex[sel]).size
    n_free = int((~on_cut).sum())

    cei = d['cross_edge_index']
    per_elem = cei[0].reshape(-1, 3)
    distinct = (per_elem[:, 0] != per_elem[:, 1]) & (per_elem[:, 1] != per_elem[:, 2]) \
        & (per_elem[:, 0] != per_elem[:, 2])
    copy_cls = np.full(d['elem_x'].shape[0], -1, dtype=np.int64)
    copy_cls[cei[0]] = elem_cls[cei[1]]   # com repetição, fica um dos valores
    same_cls = np.array_equal(copy_cls[cei[0]], elem_cls[cei[1]])
    n_vert = nodes.shape[0] - idx_2.size
    print(f"  amostra 0: N_aux {d['node_x'].shape[0]}  N_base {d['elem_x'].shape[0]} "
          f"(vértices fundidos {n_vert}, +{d['elem_x'].shape[0] - n_vert} cópias)  "
          f"Edge_dual {d['edge_index'].shape[1]}  Edge_aux {cei.shape[1]}")
    checks = {
        'node_y, x_hw, y_hw e Edge_dual idênticos aos da Elem':
            all(np.array_equal(d[k], el[k]) for k in ('node_y', 'x_hw', 'y_hw', 'edge_index')),
        'comprimentos (Edge_dual, Edge_aux) idênticos às distâncias da Elem':
            np.array_equal(d['edge_attr'][:, 3], el['edge_attr'][:, 2])
            and np.array_equal(d['cross_edge_attr'][:, 3], el['cross_edge_attr'][:, 0]),
        f'elem_y == node_y do layout de vértices na cópia do material do nó, fora do corte '
        f'({n_sel}/{n_free} vértices, max |dif| {err_y:.1e} T)':
            err_y < 1e-5 and n_sel == n_free,
        'cada elemento ligado a 3 cópias distintas, todas do seu material': bool(distinct.all())
            and same_cls,
        'vértices de 120° fundidos (nenhuma cópia aponta para eles)':
            not np.isin(copy_vertex, idx_2).any(),
        f"sen/cos contínuos (max |delta sen|, |delta cos| nas Edge_dual "
        f"{np.abs(d['edge_attr'][:, 1:3]).max():.3f})": np.abs(d['edge_attr'][:, 1:3]).max() < 0.1,
        'tudo finito': all(np.isfinite(v).all() for v in d.values() if v is not None),
    }
    for msg, c in checks.items():
        print(f"  [{'ok' if c else 'FALHA'}] {msg}")
        ok &= bool(c)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-samples', type=int, default=MAX_SAMPLES)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--out-name', default=OUT_NAME)
    ap.add_argument('--no-precheck', action='store_true')
    a = ap.parse_args()
    if not a.no_precheck and not precheck(a.out_name, a.max_samples, a.chunk_size):
        print("\nPrecheck falhou — nada foi gerado.")
        raise SystemExit(2)
    ok = run(a.max_samples, a.chunk_size, a.out_name)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
