"""
gen_npz_structures.py
---------------------
Etapa intermediária: gera um arquivo .npz por amostra em data/temp/samples_npz/.
Cada .npz contém os arrays prontos (stream, grafo, grade) sem agrupamento em chunks.

Execução (a partir da raiz do projeto):
    python -m scripts.gen_npz_structures
"""

import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed, process
from pathlib import Path

from src.data_gen.data_utils import QtreeSampleUnifier, GridSampleUnifier
from src.data_gen.sample_processor import process_and_save_sample
from src.data_gen.parsers import PARSER_REGISTRY
from src.configs.datagen import DatagenConfig
_dg      = DatagenConfig()
N_R      = _dg.n_r
N_A      = _dg.n_a
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


def run(max_samples=MAX_SAMPLES):
    """
    Executa a etapa intermediária de geração de .npz.

    Parâmetros
    ----------
    max_samples : int | None
        Limite de amostras a processar (None = todas).
        Aplicado sobre os índices ainda pendentes (não considera já prontos no limite).
    """
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
