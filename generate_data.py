from data_gen.motor_model import BLDC_Process
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, process
from data_gen.data_utils import generate_one_batch, check_data
from collections import deque

def multi_process(codes):

    ######### Multi Process ##########

    all_codes = codes
    num_workers = min(os.cpu_count() or 1, 20)
    window = num_workers * 4      # tamanho da janela de submissão
    max_retries = 1               # ignora após 1 retry (ajuste se quiser)

    pending = deque((c, 0) for c in all_codes)  # (code, tentativas)
    done = 0
    failed = []

    def submit_one(ex, code):
        return ex.submit(
            generate_one_batch,
            motor_params_list=motor_params_list,
            code_list=[code],      # <- um único sample por processo
            n_r=n_r,
            n_a=n_a,
            ang_1=ang_1,
            ang_2=ang_2,
            n_phases=n_phases
        )

    while pending:
        # abre uma pool nova a cada janela; se quebrar, recriamos no próximo loop
        ex = ProcessPoolExecutor(max_workers=num_workers, max_tasks_per_child=1)
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
            pass  # recria a pool no próximo ciclo
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        print(f"[prog] ok={done} pend={len(pending)} falhas={len(failed)}")

    print(f"Finalizado: ok={done}, falhas ignoradas={len(failed)}")

if __name__ == "__main__":
    ######### params ############
    ang_1 = 0
    ang_2 = 120
    r_in = 55/2
    r_ext = 90/2
    n_r = 80
    n_a = 240
    n_samples = 2000
    n_phases = 3

    ######### samples ###########
    # motor_params_list = BLDC_Process.generate_samples(num_samples=n_samples,seed=12)
    # BLDC_Process.export_params(params=motor_params_list)

    missing_models = check_data(n_phases=n_phases)
    print(f"missing [{len(missing_models)}] models")
    
    ###### process ######
    multi_process(codes=missing_models)