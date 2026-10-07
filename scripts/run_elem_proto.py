"""
run_elem_proto.py
-----------------
PROTÓTIPO (2026-10-07) -- treina FNO_BipartiteGNN_Elem (bipartite com os
papéis trocados, saída Bx,By por elemento) com os MESMOS hiperparâmetros e
chaves de correção (B1-B4, N_EPOCHS) do FNO_BipartiteGNN da bateria
definitiva -- importados de scripts/run_best_configs.py, não copiados.

Dataset: data/torch/data_chunks/mesh_ans_138x276_unified/FNO_BipartiteGNN_Elem/
(python -m scripts.build_elem_proto_chunks). Mesmo split da bateria.
Logs em data/logs/mesh_ans_138x276_unified_elem_proto/FNO_BipartiteGNN_Elem/
(pasta separada da bateria -- é só um teste).

Atenção: mae_graph aqui é MAE por ELEMENTO, não comparável diretamente com o
mae_graph por nó da bateria.

Execução (raiz do projeto):
    python -m scripts.run_elem_proto
"""
import traceback

from src.configs.training import NnCfg, FNO_BipartiteGNN_ElemConfig
from src.configs.monitor import MonitorCfg
from scripts.run_best_configs import (
    UNIFIED_ROOT, INTERP_MODE, FNO_NODE_RESCALE, MODES1, MODES2, DATA_RES_BIPARTITE,
    N_EPOCHS, METRICS_EVERY_EPOCH,
)
from scripts.train import run

ARCH     = 'FNO_BipartiteGNN_Elem'
DATASET  = f'{UNIFIED_ROOT}/{ARCH}'
PROBLEM  = f'{UNIFIED_ROOT}_elem_proto'
LOSSES   = ['mse', 'mae']
# [REMOVIDO 2026-10-07] auxiliar com FNO@vértices — decisão do usuário: só posição
# AUX_FNO  = True          # False = troca "pura" (vértices só com posição)
AUX_FNO  = False         # auxiliar (vértices) só com posição [r_base, c_base]


def make_arch_cfg() -> FNO_BipartiteGNN_ElemConfig:
    # mesmos valores do FNO_BipartiteGNN em run_best_configs.BEST_CONFIGS
    return FNO_BipartiteGNN_ElemConfig(
        fno_modes1=MODES1, fno_modes2=MODES2, fno_conv_width=6, fno_conv_layers=4,
        fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
        data_res=DATA_RES_BIPARTITE, gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
        interp_mode=INTERP_MODE, fno_node_rescale=FNO_NODE_RESCALE, aux_fno=AUX_FNO,
    )


def make_nn_cfg(loss: str, dataset: str = DATASET, problem: str = PROBLEM, **overrides) -> NnCfg:
    kw = dict(
        dataset=dataset, problem=problem, arch=ARCH, loss=loss,
        lr=0.01, scheduler_gamma=0.6,          # = FNO_BipartiteGNN da bateria
        arch_cfg=make_arch_cfg(),
        n_epochs=N_EPOCHS,
        monitor_cfg=MonitorCfg(metrics_every_epoch=METRICS_EVERY_EPOCH),
    )
    kw.update(overrides)
    return NnCfg(**kw)


if __name__ == '__main__':
    summary = []
    for loss in LOSSES:
        label = f"{ARCH} / {DATASET} / loss={loss}"
        print(f"\n{'=' * 80}\n{label}\n{'=' * 80}", flush=True)
        try:
            summary.append((label, run(make_nn_cfg(loss)), None))
        except Exception as e:
            traceback.print_exc()
            summary.append((label, 'failed', str(e)))

    print(f"\n{'=' * 80}\nResumo\n{'=' * 80}")
    for label, status, err in summary:
        print(f"  [{status:8s}] {label}" + (f"  -- {err}" if err else ""))
    raise SystemExit(0 if all(s in ('done', 'stopped') for _, s, _ in summary) else 1)
