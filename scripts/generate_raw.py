"""
generate_raw.py
---------------
Gera o raw: sorteia as geometrias (BLDC_Process.generate_samples) e as correntes de fase
(add_phase_currents), grava valid_designs.csv e, para cada amostra, desenha + malha +
mi_analyze() e salva o `.ans` INTEIRO (texto puro do FEMM, sem extração/derivação)
comprimido em gzip em data/raw/<dataset>/sample_XXXXXX.ans.gz -- ver
src/data_gen/femm_solve.py::save_ans_gzip_sample. Sem pós-processamento COM (sem
mi_loadsolution/mo_*). Retomável: amostras já presentes são puladas.

Os chunks de treino saem desse raw por scripts/build_datasets.py.

Execução (a partir da raiz do projeto):
    python -m scripts.generate_raw
"""
import csv
import math
import os
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed, process
from pathlib import Path

from src.data_gen.motor_model import BLDC_Process
from src.configs.datagen import DatagenConfig

_dg = DatagenConfig()
DATASET     = _dg.dataset
N_SAMPLES   = _dg.n_samples
MAX_WORKERS = _dg.raw_max_workers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR     = PROJECT_ROOT / "data" / "temp"          # só workdirs tmp_raw_*/ (scratch por amostra)
OUT_DIR      = PROJECT_ROOT / "data" / "raw" / DATASET


def _generate_one_ans_sample(idx, motor_params_list, out_dir):
    # import local: cada worker é um processo novo (Windows spawn)
    from src.data_gen.femm_solve import save_ans_gzip_sample

    motor_params = BLDC_Process.extract_params_at_index(motor_params=motor_params_list, code=idx)
    tmp_dir = TEMP_DIR / f"tmp_raw_{idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_{idx:06d}.ans.gz"

    cwd0 = os.getcwd()
    os.chdir(tmp_dir)
    try:
        save_ans_gzip_sample(motor_params, Path('.'), out_path)
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return idx


def multi_process(indices, motor_params_list, out_dir, max_workers):
    num_workers = min(os.cpu_count() or 1, max_workers)
    window = num_workers * 2
    max_retries = 1

    pending = deque((idx, 0) for idx in indices)
    done = 0
    failed = []

    def submit_one(ex, idx):
        return ex.submit(_generate_one_ans_sample, idx, motor_params_list, out_dir)

    while pending:
        # max_tasks_per_child=1: recicla o worker a cada amostra --
        # mitigação do vazamento de memória do ciclo openfemm()/closefemm()
        # via COM (pywin32)
        ex = ProcessPoolExecutor(max_workers=num_workers, max_tasks_per_child=1)
        fut2meta = {}
        try:
            for _ in range(min(len(pending), window)):
                idx, tries = pending.popleft()
                fut2meta[submit_one(ex, idx)] = (idx, tries)

            while fut2meta:
                for fut in as_completed(list(fut2meta.keys())):
                    idx, tries = fut2meta.pop(fut)
                    try:
                        fut.result()
                        done += 1
                    except Exception as e:
                        if isinstance(e, process.BrokenProcessPool):
                            pending.appendleft((idx, tries))
                            for f2, (i2, t2) in list(fut2meta.items()):
                                pending.appendleft((i2, t2))
                            raise
                        if tries < max_retries:
                            pending.append((idx, tries + 1))
                        else:
                            failed.append((idx, repr(e)))

                    while pending and len(fut2meta) < window:
                        i3, t3 = pending.popleft()
                        fut2meta[submit_one(ex, i3)] = (i3, t3)
        except process.BrokenProcessPool:
            pass
        finally:
            ex.shutdown(wait=True)
        print(f"[prog] ok={done} pend={len(pending)} falhas={len(failed)}")

    print(f"Finalizado: ok={done}, falhas ignoradas={len(failed)}")
    if failed:
        print("  Índices com falha:")
        for idx, err in failed:
            print(f"    [{idx}] {err}")
    return failed


def _check_existing_raw(motor_params_list):
    """Recusa completar um raw já existente com parâmetros de outras colunas (2026-10-14):
    ex. raw sem correntes + current_peak > 0 -> as amostras novas teriam corrente e as
    antigas não, e o valid_designs.csv (reescrito abaixo) não descreveria as antigas."""
    csv_path = OUT_DIR / "valid_designs.csv"
    if not csv_path.exists() or not any(OUT_DIR.glob("sample_*.ans.gz")):
        return
    with open(csv_path, newline='') as f:
        old_cols = next(csv.reader(f))
    new_cols = [f"{k} [{v['unit']}]" if v['unit'] else k for k, v in motor_params_list.items()]
    if old_cols != new_cols:
        raise RuntimeError(
            f"{OUT_DIR} já tem amostras geradas com outras colunas no valid_designs.csv "
            f"(faltando: {sorted(set(new_cols) - set(old_cols))}, sobrando: "
            f"{sorted(set(old_cols) - set(new_cols))}) — use outro DatagenConfig.dataset "
            f"ou confira DatagenConfig.current_peak")
    # 2026-10-21: mesmas colunas não bastam (ex.: stack_length fixo muda a sequência do gerador
    # e todas as geometrias, sem mudar nenhuma coluna) -- os VALORES sorteados também precisam
    # bater com o valid_designs.csv existente, nas linhas em comum.
    with open(csv_path, newline='') as f:
        old_rows = list(csv.reader(f))[1:]
    values = list(motor_params_list.values())
    for i, row in enumerate(old_rows[:len(values[0]['value'])]):
        bad = [c for c, v, p in zip(old_cols, row, values)
               if not math.isclose(float(v), float(p['value'][i]), rel_tol=1e-12, abs_tol=0.0)]
        if bad:
            raise RuntimeError(
                f"{OUT_DIR} já tem amostras geradas com outro sorteio: linha {i} do "
                f"valid_designs.csv difere em {bad} (ex.: {bad[0]} = {row[old_cols.index(bad[0])]} "
                f"no CSV x {values[old_cols.index(bad[0])]['value'][i]!r} agora) — o raw foi "
                f"gerado por outro sampler/semente/versão do código; use outro DatagenConfig.dataset")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    motor_params_list = BLDC_Process.generate_samples(num_samples=N_SAMPLES, seed=_dg.datagen_seed)
    # correntes de fase (2026-10-14) — sorteio próprio, não altera a geometria
    if _dg.current_peak > 0:
        BLDC_Process.add_phase_currents(motor_params_list, i_peak=_dg.current_peak,
                                        seed=_dg.datagen_seed)
    _check_existing_raw(motor_params_list)
    BLDC_Process.export_params(params=motor_params_list)

    pending = [i for i in range(N_SAMPLES) if not (OUT_DIR / f"sample_{i:06d}.ans.gz").exists()]
    ja_prontos = N_SAMPLES - len(pending)
    print(f"Já processados: {ja_prontos}  |  A processar: {len(pending)}")

    if not pending:
        print("Nada a fazer.")
        return

    multi_process(indices=pending, motor_params_list=motor_params_list, out_dir=OUT_DIR,
                  max_workers=MAX_WORKERS)


if __name__ == "__main__":
    run()
