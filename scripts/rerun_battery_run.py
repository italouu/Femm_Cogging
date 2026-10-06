"""
rerun_battery_run.py — refaz UMA run da bateria (scripts/run_best_configs.py), com
exatamente a mesma configuração (2026-10-06).

Motivo: na bateria mesh_ans_138x276_unified_best_mse_mae (commit 3118c3f),
FNO_BipartiteGNN mse (run_0001) colapsou entre as épocas 21 e 25 (train_loss
0,04 -> 0,79, sem recuperação) e foi parada pelo GL na época 99 com best.pth na
época 19 — inutilizável. Esta é uma repetição idêntica (sem controle de semente,
como o resto da bateria): mesmo spec de BEST_CONFIGS, mesma loss, mesmo
make_nn_cfg (N_EPOCHS, MonitorCfg, INTERP_MODE, B2/B3/B4) — nada é redefinido aqui.

A run nova entra na mesma pasta, data/logs/<PROBLEM>/<ARCH>/, com o próximo número
(run_0003); a run que falhou NÃO é apagada nem movida.

Uso (da raiz do projeto, na VM — FNO_BipartiteGNN usa ~17,6 GiB de GPU):
    python -m scripts.rerun_battery_run
"""
import traceback

import scripts.run_best_configs as rbc
from scripts.train import run

ARCH = 'FNO_BipartiteGNN'
LOSS = 'mse'
REPEAT = 0


def main():
    specs = [s for s in rbc.build_run_list()
             if s['arch'] == ARCH and s['loss'] == LOSS and s['repeat'] == REPEAT]
    if len(specs) != 1:
        raise SystemExit(f"esperava 1 spec para {ARCH}/{LOSS}/rep={REPEAT}, achei {len(specs)}")
    spec = specs[0]
    label = f"{spec['arch']} / {spec['dataset']} / loss={spec['loss']} / rep={spec['repeat']}"
    print(f"{'=' * 80}\n{label}  (refazendo — problem={rbc.PROBLEM}, n_epochs={rbc.N_EPOCHS}, "
          f"interp={rbc.INTERP_MODE})\n{'=' * 80}", flush=True)
    try:
        status = run(rbc.make_nn_cfg(spec))
    except Exception:
        traceback.print_exc()
        status = 'failed'
    print(f"\n[{status}] {label}")


if __name__ == '__main__':
    main()
