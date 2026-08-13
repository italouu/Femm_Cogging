"""
generate_data_femm_mesh_v2.py
-------------------------------
Gera dados via a malha real do FEMM (src/data_gen/femm_mesh.py) -- variante
"v2" de generate_data_femm_mesh.py que só faz a etapa de geração BRUTA:
desenha + malha + mi_analyze() e salva o `.ans` INTEIRO (texto puro do
FEMM, sem nenhuma extração/derivação) comprimido em gzip direto em
data/raw/<dataset>/sample_XXXXXX.ans.gz -- ver
src/data_gen/femm_mesh.py::save_ans_gzip_sample.

Corrige a lacuna encontrada em 2026-08-07 (ver CLAUDE.md, "Extração de
dados direto do arquivo .ans"): mode='femm_mesh' (generate_data_femm_mesh.py)
deriva node_x/edge_index/etc. na hora e descarta a malha original (elems)
-- sem ela, não dá pra recalcular nada (ex: B exato por elemento, validado
nesta sessão -- bate com mo_getb(smooth='off') a 1e-14 T) sem resimular.
Guardando o .ans inteiro, qualquer derivação futura parte só do arquivo,
sem FEMM.

Sem chamada de pós-processamento COM nenhuma (sem mi_loadsolution/mo_*) --
só o solve (mi_analyze) e cópia do arquivo, mais rápido que
generate_data_femm_mesh.py (que ainda faz o loop de mo_getb por nó).

Etapas de parser/chunk (gen_npz_structures.py/build_data_chunks.py) que leem
esse .ans.gz ainda NÃO existem -- fora do escopo desta mudança, ver
"Pendências conhecidas" no CLAUDE.md.

Execução (a partir da raiz do projeto):
    python -m scripts.generate_data_femm_mesh_v2
"""
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
MAX_WORKERS = _dg.femm_mesh_max_workers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR     = PROJECT_ROOT / "data" / "temp"          # só workdirs tmp_femm_mesh_v2_*/ (scratch por amostra)
OUT_DIR      = PROJECT_ROOT / "data" / "raw" / DATASET  # mesma pasta-base de mode='femm_mesh' (raw = fonte única)

_SAMPLE_METHODS = {
    'constrained':     BLDC_Process.generate_samples_constrained,
    'legacy':          BLDC_Process.generate_samples,
    'fixed_geometry':  BLDC_Process.generate_samples_fixed_geometry,
    'constrained_lhs': BLDC_Process.generate_samples_constrained_lhs_filtered,
}


def _generate_one_ans_sample(idx, motor_params_list, out_dir):
    # import local: cada worker é um processo novo (Windows spawn)
    from src.data_gen.femm_mesh import save_ans_gzip_sample

    motor_params = BLDC_Process.extract_params_at_index(motor_params=motor_params_list, code=idx)
    tmp_dir = TEMP_DIR / f"tmp_femm_mesh_v2_{idx}"
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
        # max_tasks_per_child=1: recicla o worker a cada amostra -- mesma
        # mitigação do vazamento de memória do ciclo openfemm()/closefemm()
        # via COM (pywin32) usada em generate_data.py/generate_data_femm_mesh.py
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
