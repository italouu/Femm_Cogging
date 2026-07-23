from src.data_gen.motor_model import BLDC_Process
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, process
from src.data_gen.data_utils import generate_one_batch, check_data
from src.configs.datagen import DatagenConfig
_dg      = DatagenConfig()
MODE      = _dg.mode
N_R       = _dg.n_r
N_A       = _dg.n_a
ANG_1     = _dg.ang_1
ANG_2     = _dg.ang_2
# [REMOVIDO] N_PHASES = _dg.n_phases  — phase do rotor fixado em 0
MAX_DEPTH = _dg.max_depth
N_SAMPLES = _dg.n_samples
from collections import deque

def multi_process(codes, mode, motor_params_list, n_r, n_a, ang_1, ang_2, max_depth):

    ######### Multi Process ##########

    all_codes = codes
    num_workers = min(os.cpu_count() or 1, 8)
    window = num_workers * 4      # tamanho da janela de submissão
    max_retries = 1               # ignora após 1 retry (ajuste se quiser)

    pending = deque((c, 0) for c in all_codes)  # (code, tentativas)
    done = 0
    failed = []

    def submit_one(ex, code):
        return ex.submit(
            generate_one_batch,
            mode=mode,
            motor_params_list=motor_params_list,
            code_list=[code],
            n_r=n_r,
            n_a=n_a,
            ang_1=ang_1,
            ang_2=ang_2,
            max_depth=max_depth,
        )

    while pending:
        # abre uma pool nova a cada janela; se quebrar, recriamos no próximo loop
        ex = ProcessPoolExecutor(max_workers=num_workers, max_tasks_per_child=1) # remover max_tasks_per_child=1
        fut2meta = {}
        try:
            # pré-enche a janela
            for _ in range(min(len(pending), window)):
                code, tries = pending.popleft()
                fut2meta[submit_one(ex, code)] = (code, tries)

            # processa conforme concluem e mantém a janela cheia
            while fut2meta:
                for fut in as_completed(list(fut2meta.keys())):
                    code, tries = fut2meta.pop(fut)
                    try:
                        fut.result()
                        done += 1
                    except Exception as e:
                        if isinstance(e, process.BrokenProcessPool):
                            # pool morreu: refile a atual + as pendentes dessa janela
                            pending.appendleft((code, tries))
                            for f2, (c2, t2) in list(fut2meta.items()):
                                pending.appendleft((c2, t2))
                            raise  # sai para recriar a pool
                        # falha do sample: tenta de novo ou marca como ignorado
                        if tries < max_retries:
                            pending.append((code, tries + 1))
                        else:
                            failed.append((code, repr(e)))

                    # completa a janela
                    while pending and len(fut2meta) < window:
                        c3, t3 = pending.popleft()
                        fut2meta[submit_one(ex, c3)] = (c3, t3)
        except process.BrokenProcessPool:
            pass
        finally:
            ex.shutdown(wait=True)
        print(f"[prog] ok={done} pend={len(pending)} falhas={len(failed)}")

    print(f"Finalizado: ok={done}, falhas ignoradas={len(failed)}")

_SAMPLE_METHODS = {
    'constrained':     BLDC_Process.generate_samples_constrained,
    'legacy':          BLDC_Process.generate_samples,
    'fixed_geometry':  BLDC_Process.generate_samples_fixed_geometry,
    'constrained_lhs': BLDC_Process.generate_samples_constrained_lhs_filtered,
}

if __name__ == "__main__":
    if MODE == 'femm_mesh':
        # pipeline via malha real do FEMM (ver src/data_gen/femm_mesh.py) —
        # generate_one_batch/check_data só aceitam 'grid'/'qtree', então esse
        # modo tem seu próprio fluxo, reaproveitado aqui em vez de duplicado.
        from scripts.generate_data_femm_mesh import run as run_femm_mesh
        run_femm_mesh()
    else:
        _gen = _SAMPLE_METHODS[_dg.sample_method]
        motor_params_list = _gen(num_samples=N_SAMPLES, seed=_dg.datagen_seed)
        BLDC_Process.export_params(params=motor_params_list)

        missing_list = check_data(mode=MODE)   # n_phases removido; default=1 em check_data
        print(f"missing [{len(missing_list)}] models")

        multi_process(codes=missing_list, mode=MODE, motor_params_list=motor_params_list,
                      n_r=N_R, n_a=N_A, ang_1=ANG_1, ang_2=ANG_2,
                      max_depth=MAX_DEPTH)
