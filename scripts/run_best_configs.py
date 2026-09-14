"""
Roda uma sequência de treinos: para cada arquitetura (FNO2d, FNO_GNN,
GNN_PostBase, FNO_BipartiteGNN — v3 e graph_div_b_loss ficam de fora a pedido
do usuário), reconstrói os hiperparâmetros do melhor run já encontrado (ver
levantamento em CLAUDE.md / conversa) e roda duas vezes: loss='mse' e loss='mae'.
Sem baseline mse/mae pra FNO_BipartiteGNN até agora — graph_div_b_loss nunca foi
comparado contra as losses padrão nesse dataset, essa sequência cobre essa lacuna
(mesmo motivo pelo qual v3 fica de fora: ela já tinha mse/mae, faltava era
FNO_BipartiteGNN "puro").

Total: 4 arqs × 2 losses = 8 treinos, sequenciais (um processo, uma GPU).
Uma falha em um treino não aborta os demais — status/erro de cada um é
reportado no resumo final.
"""
import traceback

from src.configs.training import (
    NnCfg, FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig, FNO_BipartiteGNNConfig,
)
from scripts.train import run


# Cada entrada: (arch, dataset, lr, scheduler_gamma, arch_cfg) — valores
# reconstruídos do config.json do melhor run de cada arquitetura (mae_hw/
# mae_graph mais baixo entre os runs com treino substancial em datasets mesh_*).
BEST_CONFIGS = [
    dict(
        arch='FNO2d',
        dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0002 (mae_hw=0.0207 T)
        lr=0.005,
        scheduler_gamma=0.7,
        arch_cfg=FNOConfig(
            modes1=270, modes2=270, conv_width=8, conv_layers=4,
            lift_width=64, lift_layers=3, proj_width=64, proj_layers=3,
            data_res=(135, 270),
        ),
    ),
    dict(
        arch='FNO_GNN',
        dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0002 (mae_graph=0.1010 T)
        lr=0.003,
        scheduler_gamma=0.8,
        arch_cfg=FNO_GNNConfig(
            fno_modes1=270, fno_modes2=270, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(135, 270), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
        ),
    ),
    dict(
        arch='GNN_PostBase',
        dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0004 (mae_graph=0.1323 T)
        lr=0.001,
        scheduler_gamma=0.6,
        arch_cfg=GNN_PostBaseConfig(
            base_run_dir='data/logs/mesh_138x276_FEMM_MESH/FNO2d/run_0001',
            base_checkpoint='best',
            gnn_node_width=64, gnn_n_layers=6,
        ),
    ),
    dict(
        arch='FNO_BipartiteGNN',
        dataset='mesh_ans_138x276_B',           # melhor run: run_0015 (mae_graph=0.0503 T,
        lr=0.01,                                # lá com graph_div_b_loss — aqui mse/mae puros)
        scheduler_gamma=0.6,
        arch_cfg=FNO_BipartiteGNNConfig(
            fno_modes1=270, fno_modes2=270, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
        ),
    ),
]

LOSSES = ['mse', 'mae']


def build_run_list():
    runs = []
    for base in BEST_CONFIGS:
        for loss in LOSSES:
            runs.append(dict(base, loss=loss))
    return runs


if __name__ == '__main__':
    summary = []
    for spec in build_run_list():
        label = f"{spec['arch']} / {spec['dataset']} / loss={spec['loss']}"
        print(f"\n{'='*80}\n{label}\n{'='*80}", flush=True)
        try:
            _nn = NnCfg(
                dataset=spec['dataset'],
                # sufixo só no problem (diretório de logs, data/logs/{problem}/{arch}/) —
                # dataset continua apontando pros chunks reais; separa estas runs
                # (best hparams × mse/mae) das demais já registradas em data/logs/{dataset}/.
                problem=f"{spec['dataset']}_best_mse_mae",
                arch=spec['arch'],
                loss=spec['loss'],
                lr=spec['lr'],
                scheduler_gamma=spec['scheduler_gamma'],
                arch_cfg=spec['arch_cfg'],
            )
            status = run(_nn)
            summary.append((label, status, None))
        except Exception as e:
            traceback.print_exc()
            summary.append((label, 'failed', str(e)))

    print(f"\n{'='*80}\nResumo\n{'='*80}")
    for label, status, err in summary:
        line = f"  [{status:8s}] {label}"
        if err:
            line += f"  -- {err}"
        print(line)
