"""
generate_data_femm_mesh.py
---------------------------
Gera dados via a malha real do FEMM (src/data_gen/femm_mesh.py) — método
alternativo ao pipeline CSV/qtree de generate_data.py. Grava um .npz de
staging por amostra diretamente, sem CSV intermediário (a extração do
grafo já acontece na mesma sessão FEMM — ver
src/data_gen/femm_mesh.generate_mesh_sample) — mesma convenção .npz de
sample_processor.py/gen_npz_structures.py no pipeline qtree.

Saída: data/temp/samples_mesh/<dataset>/sample_XXXXXX.npz
Agrupamento final em data_chunk_*.pt: scripts/build_data_chunks_femm_mesh.py

Execução (a partir da raiz do projeto):
    python -m scripts.generate_data_femm_mesh
"""
import os
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed, process
from pathlib import Path

from src.data_gen.motor_model import BLDC_Process
from src.configs.datagen import DatagenConfig

_dg = DatagenConfig()
N_R         = _dg.n_r
N_A         = _dg.n_a
ANG_1       = _dg.ang_1
ANG_2       = _dg.ang_2
DATASET     = _dg.dataset
N_SAMPLES   = _dg.n_samples
MAX_WORKERS = _dg.femm_mesh_max_workers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR     = PROJECT_ROOT / "data" / "temp"
OUT_DIR      = TEMP_DIR / "samples_mesh" / DATASET

_SAMPLE_METHODS = {
    'constrained':     BLDC_Process.generate_samples_constrained,
    'legacy':          BLDC_Process.generate_samples,
    'fixed_geometry':  BLDC_Process.generate_samples_fixed_geometry,
    'constrained_lhs': BLDC_Process.generate_samples_constrained_lhs_filtered,
}


def _generate_one_mesh_sample(idx, motor_params_list, out_dir, n_r, n_a, ang_1, ang_2):
    # import local: cada worker é um processo novo (Windows spawn)
    from src.data_gen.femm_mesh import generate_mesh_sample

    motor_params = BLDC_Process.extract_params_at_index(motor_params=motor_params_list, code=idx)
    tmp_dir = TEMP_DIR / f"tmp_femm_mesh_{idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_{idx:06d}.npz"

    cwd0 = os.getcwd()
    os.chdir(tmp_dir)
    try:
        generate_mesh_sample(motor_params, Path('.'), out_path,
                              n_r=n_r, n_a=n_a, ang_1=ang_1, ang_2=ang_2)
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return idx


def multi_process(indices, motor_params_list, out_dir, n_r, n_a, ang_1, ang_2, max_workers):
    num_workers = min(os.cpu_count() or 1, max_workers)
    window = num_workers * 2
    max_retries = 1

    pending = deque((idx, 0) for idx in indices)
    done = 0
    failed = []

    def submit_one(ex, idx):
        return ex.submit(_generate_one_mesh_sample, idx, motor_params_list, out_dir,
                          n_r, n_a, ang_1, ang_2)

    while pending:
        # max_tasks_per_child=1: recicla o worker a cada amostra -- a mesma
        # mitigacao usada em generate_data.py (linha ~45) para o vazamento de
        # memoria do ciclo openfemm()/closefemm() via COM (pywin32), que nao
        # libera 100% por chamada dentro do mesmo processo hospedeiro.
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


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gen = _SAMPLE_METHODS[_dg.sample_method]
    motor_params_list = gen(num_samples=N_SAMPLES, seed=_dg.datagen_seed)
    BLDC_Process.export_params(params=motor_params_list)

    pending = [i for i in range(N_SAMPLES) if not (OUT_DIR / f"sample_{i:06d}.npz").exists()]
    ja_prontos = N_SAMPLES - len(pending)
    print(f"Já processados: {ja_prontos}  |  A processar: {len(pending)}")

    if not pending:
        print("Nada a fazer.")
        return

    multi_process(indices=pending, motor_params_list=motor_params_list, out_dir=OUT_DIR,
                  n_r=N_R, n_a=N_A, ang_1=ANG_1, ang_2=ANG_2, max_workers=MAX_WORKERS)


if __name__ == "__main__":
    run()
