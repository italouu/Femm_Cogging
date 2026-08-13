"""
gen_npz_structures.py
---------------------
Etapa intermediária: gera um arquivo .npz por amostra em data/temp/samples_npz/.
Cada .npz contém os arrays prontos (stream, grafo, grade) sem agrupamento em chunks.

mode='femm_mesh' (2026-07-27): mesmo papel (aplicar PARSER_REGISTRY[npz_parser],
gerar o .npz intermediário), mas fonte e lógica diferentes — lê o staging bruto
já pronto de data/raw/<dataset>/ (gerado por generate_data_femm_mesh.py, ver
scripts/generate_data.py) e só filtra colunas (sem simulação FEM nenhuma aqui,
por isso não usa multi_process/ProcessPoolExecutor como o branch grid/qtree
abaixo) — grava em data/temp/samples_npz/<dataset>_<npz_parser>/ (sufixo do
parser mantido também aqui e no chunk final, decisão 2026-07-27: aceitar a
mesma "redundância"/risco que o pipeline grid/qtree já tem sem sufixo, mas
com o sufixo pra não misturar parsers enquanto os chunks ainda são apontados
manualmente no treino).

Execução (a partir da raiz do projeto, mesmo comando pros três modos):
    python -m scripts.gen_npz_structures
"""

import csv
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed, process
from pathlib import Path

import numpy as np

from src.data_gen.data_utils import QtreeSampleUnifier, GridSampleUnifier
from src.data_gen.sample_processor import process_and_save_sample
from src.data_gen.parsers.femm_mesh_v2 import parse_ans_gzip_sample
from src.data_gen.parsers import PARSER_REGISTRY
from src.configs.datagen import DatagenConfig
_dg      = DatagenConfig()
N_R      = _dg.n_r
N_A      = _dg.n_a
ANG_1    = _dg.ang_1
ANG_2    = _dg.ang_2
N_PHASES = 1  # fase do rotor fixada em 0 (n_phases removido de DatagenConfig)
DATASET  = _dg.dataset

MODE               = _dg.mode
PARSER             = PARSER_REGISTRY[_dg.npz_parser]
SAMPLES_PER_WORKER = _dg.npz_samples_per_worker
MAX_WORKERS        = _dg.npz_max_workers
MAX_SAMPLES        = _dg.npz_max_samples

def multi_process(indices: list, unifier: QtreeSampleUnifier, out_dir: Path):
    num_workers = min(os.cpu_count() or 1, MAX_WORKERS)

    # divide índices em chunks fixos de SAMPLES_PER_WORKER
    chunks       = [indices[i:i + SAMPLES_PER_WORKER] for i in range(0, len(indices), SAMPLES_PER_WORKER)]
    n_chunks     = len(chunks)
    pending      = deque((c, 0) for c in chunks)   # (chunk, tentativas)
    total_ok     = 0
    total_falhas = []
    chunks_done  = 0

    def submit_one(ex, chunk):
        return ex.submit(process_and_save_sample, chunk, unifier, out_dir)

    while pending:
        ex = ProcessPoolExecutor(max_workers=num_workers) # remover max_tasks_per_child=1
        fut2meta = {}
        try:
            for _ in range(min(len(pending), num_workers)):
                chunk, tries = pending.popleft()
                fut2meta[submit_one(ex, chunk)] = (chunk, tries)

            while fut2meta:
                for fut in as_completed(list(fut2meta.keys())):
                    chunk, tries = fut2meta.pop(fut)
                    falhas = []
                    try:
                        paths, falhas = fut.result()
                        total_ok += len(paths)
                        total_falhas.extend(falhas)
                    except process.BrokenProcessPool:
                        pending.appendleft((chunk, tries))
                        for f2, (b2, t2) in list(fut2meta.items()):
                            pending.appendleft((b2, t2))
                        raise
                    except Exception as e:
                        falhas = [(idx, repr(e)) for idx in chunk]
                        total_falhas.extend(falhas)

                    chunks_done += 1
                    ex_msg = f"  ex: {falhas[0][1][:120]}" if falhas else ""
                    print(f"[{chunks_done}/{n_chunks}] falhas={len(total_falhas)}{ex_msg}")

                    while pending and len(fut2meta) < num_workers:
                        b3, t3 = pending.popleft()
                        fut2meta[submit_one(ex, b3)] = (b3, t3)

        except process.BrokenProcessPool:
            pass
        finally:
            ex.shutdown(wait=True)

    return total_ok, total_falhas


# ── mode='femm_mesh': aplica o parser sobre o staging bruto (data/raw/) ────────
# Metadados de geometria por amostra do staging bruto -- só rastreabilidade/
# depuração (ver src/data_gen/femm_mesh.generate_mesh_sample), não é dado
# final de treino.
_FEMM_MESH_DROP_KEYS = frozenset({'r_in_mm', 'r_ext_mm', 'ang_1_deg', 'ang_2_deg'})


def _load_npz_femm_mesh(path: Path) -> dict:
    return {key: v for key, v in np.load(path).items() if key not in _FEMM_MESH_DROP_KEYS}


def _apply_parser_femm_mesh(d: dict, cfg) -> dict:
    """Filtra node_x/node_y/x_hw/y_hw/edge_attr pelas colunas do parser.
    Demais chaves (node_A, a_hw, edge_index, L, dim_H/W) passam inalteradas
    — não têm variante "cheia" a filtrar.

    cfg.target_field:
        'B' (padrão) -> node_y/y_hw fatiados dos brutos node_y/y_hw (Bx,By).
        'A'          -> node_y/y_hw reconstruídos a partir de node_A/a_hw
                        (potencial vetor, 1 canal); node_y_cols/y_hw_cols
                        ignorados. node_A/a_hw brutos NÃO são removidos do
                        output — eval.py::fno_gnn_eval_fn usa 'node_A' in d
                        para detectar chunks femm_mesh e escolher o plot certo.
    """
    out = dict(d)
    out['node_x'] = d['node_x'][:, cfg.node_x_cols]
    out['x_hw']   = d['x_hw'][cfg.x_hw_cols]
    if cfg.target_field == 'A':
        out['node_y'] = d['node_A'][:, None]   # [S] -> [S, 1]
        out['y_hw']   = d['a_hw'][None]         # [H, W] -> [1, H, W]
    else:
        out['node_y'] = d['node_y'][:, cfg.node_y_cols]
        out['y_hw']   = d['y_hw'][cfg.y_hw_cols]
    if cfg.build_graph:
        out['edge_attr'] = d['edge_attr'][:, cfg.edge_attr_cols]
    else:
        out.pop('edge_index', None)
        out.pop('edge_attr', None)
    return out


def _run_femm_mesh(max_samples):
    raw_dir = Path("data/raw") / DATASET
    out_dir = Path("data/temp/samples_npz") / _dg.parsed_dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(raw_dir.glob("sample_*.npz"), key=lambda p: int(p.stem.split('_')[1]))
    if max_samples is not None:
        npz_paths = npz_paths[:max_samples]

    total = len(npz_paths)
    if total == 0:
        print("Nenhum .npz encontrado em", raw_dir)
        return

    pending = [p for p in npz_paths if not (out_dir / p.name).exists()]
    ja_prontos = total - len(pending)

    print(f"\n=== Parser '{_dg.npz_parser}' sobre staging femm_mesh ===")
    print(f"  origem  : {raw_dir}")
    print(f"  destino : {out_dir}")
    print(f"  Já processados: {ja_prontos}  |  A processar: {len(pending)}")

    if not pending:
        print("Nada a fazer.")
        return

    ok = 0
    for path in pending:
        d   = _load_npz_femm_mesh(path)
        out = _apply_parser_femm_mesh(d, PARSER)

        # escrita atômica: salva em .tmp.npz e renomeia
        stem     = path.stem
        out_path = out_dir / path.name
        tmp_path = out_dir / f"{stem}.tmp"        # np.savez adiciona .npz -> .tmp.npz
        tmp_npz  = out_dir / f"{stem}.tmp.npz"
        np.savez(tmp_path, **out)
        tmp_npz.replace(out_path)
        ok += 1

    print(f"\n=== Resumo gen_npz_structures (femm_mesh) ===")
    print(f"  Salvos     : {ok}")
    print(f"  Já prontos : {ja_prontos}")


# ── mode='femm_mesh_v2': deriva os 2 grafos + grade a partir do .ans.gz bruto ──
# Sem PARSER_REGISTRY aqui -- parse_ans_gzip_sample já produz o formato final
# (grafo de vértices + grafo de elementos + arestas cruzadas + grade), não há
# seleção de colunas a fazer (ao contrário de mode='femm_mesh', que tem várias
# variantes de parser sobre o mesmo staging bruto). Por isso não usa
# parsed_dataset_name (sufixo de parser) -- grava direto em
# data/temp/samples_npz/<dataset>/, mesma convenção do grid/qtree.

def _parse_and_save_one_v2(path: Path, r_in: float, r_ext: float, ang_1: float, ang_2: float,
                            n_r: int, n_a: int, out_dir: Path) -> int:
    """Worker (processo novo, spawn) de UMA amostra: parse_ans_gzip_sample +
    escrita atômica do .npz. parse_ans_gzip_sample é numpy/matplotlib.tri puro
    (sem FEMM/COM), então o pool pode ser persistente (sem max_tasks_per_child=1
    -- diferente de generate_data_femm_mesh_v2.py, que precisa reciclar workers
    por causa do vazamento de memória do ciclo openfemm()/closefemm())."""
    idx = int(path.name.split('_')[1].split('.')[0])
    d = parse_ans_gzip_sample(path, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                               n_r=n_r, n_a=n_a, tmp_dir=out_dir)

    stem     = path.name.removesuffix('.ans.gz')
    out_path = out_dir / f"{stem}.npz"
    tmp_path = out_dir / f"{stem}.tmp"        # np.savez adiciona .npz -> .tmp.npz
    tmp_npz  = out_dir / f"{stem}.tmp.npz"
    np.savez(tmp_path, **d)
    tmp_npz.replace(out_path)
    return idx


def _run_femm_mesh_v2(max_samples):
    raw_dir = Path("data/raw") / DATASET
    out_dir = Path("data/temp/samples_npz") / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    ans_paths = sorted(raw_dir.glob("sample_*.ans.gz"),
                        key=lambda p: int(p.name.split('_')[1].split('.')[0]))
    if max_samples is not None:
        ans_paths = ans_paths[:max_samples]

    total = len(ans_paths)
    if total == 0:
        print("Nenhum .ans.gz encontrado em", raw_dir)
        return

    with open(raw_dir / "valid_designs.csv", newline='') as f:
        design_rows = list(csv.DictReader(f))

    pending = [p for p in ans_paths
               if not (out_dir / f"{p.name.removesuffix('.ans.gz')}.npz").exists()]
    ja_prontos = total - len(pending)

    print(f"\n=== Parsing femm_mesh_v2 (.ans.gz -> grafo de vértices + grafo de elementos + grade) ===")
    print(f"  origem  : {raw_dir}")
    print(f"  destino : {out_dir}")
    print(f"  Já processados: {ja_prontos}  |  A processar: {len(pending)}")

    if not pending:
        print("Nada a fazer.")
        return

    # [REMOVIDO 2026-08-13] loop sequencial -- ~1,2-5,7s/amostra medido em
    # produção (10-11/08, ~6h18min pras 4000 amostras). parse_ans_gzip_sample
    # não abre FEMM/COM (só numpy/matplotlib.tri sobre o .ans.gz já em disco),
    # então dá pra paralelizar com ProcessPoolExecutor persistente, sem o
    # cuidado de max_tasks_per_child=1 que generate_data_femm_mesh_v2.py
    # precisa (não há vazamento de memória de ciclo openfemm()/closefemm()
    # aqui). Substituído pela versão paralela abaixo.
    # ok, falhas = 0, []
    # for path in pending:
    #     idx = int(path.name.split('_')[1].split('.')[0])
    #     try:
    #         row = design_rows[idx]
    #         r_in  = float(row['inner_diameter [mm]']) / 2
    #         r_ext = float(row['outer_diameter [mm]']) / 2
    #         d = parse_ans_gzip_sample(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
    #                                    n_r=N_R, n_a=N_A, tmp_dir=out_dir)
    #
    #         # escrita atômica: salva em .tmp.npz e renomeia
    #         stem     = path.name.removesuffix('.ans.gz')
    #         out_path = out_dir / f"{stem}.npz"
    #         tmp_path = out_dir / f"{stem}.tmp"        # np.savez adiciona .npz -> .tmp.npz
    #         tmp_npz  = out_dir / f"{stem}.tmp.npz"
    #         np.savez(tmp_path, **d)
    #         tmp_npz.replace(out_path)
    #         ok += 1
    #     except Exception as e:
    #         falhas.append((idx, repr(e)))
    #
    #     if ok % 100 == 0 or (ok + len(falhas)) == len(pending):
    #         print(f"[{ok + len(falhas)}/{len(pending)}] ok={ok} falhas={len(falhas)}")

    num_workers = min(os.cpu_count() or 1, MAX_WORKERS)
    print(f"  workers : {num_workers}")

    ok, falhas = 0, []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        fut2idx = {}
        for path in pending:
            idx   = int(path.name.split('_')[1].split('.')[0])
            row   = design_rows[idx]
            r_in  = float(row['inner_diameter [mm]']) / 2
            r_ext = float(row['outer_diameter [mm]']) / 2
            fut = ex.submit(_parse_and_save_one_v2, path, r_in, r_ext, ANG_1, ANG_2,
                             N_R, N_A, out_dir)
            fut2idx[fut] = idx

        for fut in as_completed(fut2idx):
            idx = fut2idx[fut]
            try:
                fut.result()
                ok += 1
            except Exception as e:
                falhas.append((idx, repr(e)))

            done = ok + len(falhas)
            if done % 100 == 0 or done == len(pending):
                print(f"[{done}/{len(pending)}] ok={ok} falhas={len(falhas)}")

    print(f"\n=== Resumo gen_npz_structures (femm_mesh_v2) ===")
    print(f"  Salvos     : {ok}")
    print(f"  Já prontos : {ja_prontos}")
    print(f"  Falhas     : {len(falhas)}")
    if falhas:
        print("  Índices com falha:")
        for idx, err in falhas:
            print(f"    [{idx}] {err}")


def run(max_samples=MAX_SAMPLES):
    """
    Executa a etapa intermediária de geração de .npz.

    Parâmetros
    ----------
    max_samples : int | None
        Limite de amostras a processar (None = todas).
        Aplicado sobre os índices ainda pendentes (não considera já prontos no limite).
    """
    if MODE == 'femm_mesh':
        return _run_femm_mesh(max_samples)
    if MODE == 'femm_mesh_v2':
        return _run_femm_mesh_v2(max_samples)

    out_dir = Path("data/temp/samples_npz") / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    if MODE == 'grid':
        unifier = GridSampleUnifier(n_r=N_R, n_a=N_A, n_phases=N_PHASES,
                                    parser_cfg=PARSER)
    else:
        unifier = QtreeSampleUnifier(n_r=N_R, n_a=N_A, n_phases=N_PHASES,
                                     parser_cfg=PARSER)
    total = len(unifier)
    print(f"Amostras disponíveis: {total}")

    # filtra índices já processados
    pending = [i for i in range(total) if not (out_dir / f"sample_{i:06d}.npz").exists()]
    ja_prontos = total - len(pending)

    # aplica limite sobre os pendentes
    if max_samples is not None:
        pending = pending[:max_samples]

    print(f"Já processados: {ja_prontos}  |  A processar: {len(pending)}"
          + (f"  (limite={max_samples})" if max_samples is not None else ""))

    if not pending:
        print("Nada a fazer.")
        return

    ok, falhas = multi_process(indices=pending, unifier=unifier, out_dir=out_dir)

    print("\n=== Resumo gen_npz_structures ===")
    print(f"  Salvos     : {ok}")
    print(f"  Já prontos : {ja_prontos}")
    print(f"  Falhas     : {len(falhas)}")
    if falhas:
        print("  Índices com falha:")
        for idx, err in falhas:
            print(f"    [{idx}] {err}")


def main():
    run(MAX_SAMPLES)


if __name__ == "__main__":
    main()
