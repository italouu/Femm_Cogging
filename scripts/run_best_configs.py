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

2026-10-01 — adaptado para os datasets unificados (ver CLAUDE.md "Datasets
unificados a partir do raw mesh_ans_138x276"): as 4 archs passam a treinar em
data/torch/data_chunks/mesh_ans_138x276_unified/<arch>/ (mesmo raw, mesmo
gabarito B, mesmo split), gerados por scripts/build_unified_ans_chunks_direct.py.
Logs em data/logs/mesh_ans_138x276_unified_best_mse_mae/<arch>/run_XXXX/.
GNN_PostBase usa como base o FNO2d treinado NESTA bateria com a MESMA loss do
próprio GNN_PostBase (mse -> base FNO2d mse, mae -> base FNO2d mae) — a base
antiga (mesh_138x276_FEMM_MESH/FNO2d/run_0001) foi treinada contra o B
suavizado do FEMM (gabarito diferente). Por
isso FNO2d precisa vir antes de GNN_PostBase em BEST_CONFIGS, e o
GNN_PostBaseConfig é construído só na hora do treino (__post_init__ lê o
config.json da base, que ainda não existe na importação do módulo).
"""
import json
import traceback
from pathlib import Path

from src.configs.training import (
    NnCfg, FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig, FNO_BipartiteGNNConfig,
)
from scripts.train import run

UNIFIED_ROOT = 'mesh_ans_138x276_unified'       # data/torch/data_chunks/<UNIFIED_ROOT>/<arch>/
PROBLEM      = f'{UNIFIED_ROOT}_best_mse_mae'   # data/logs/<PROBLEM>/<arch>/
# [REMOVIDO 2026-10-01] base única com loss fixa — substituído por pareamento por loss
# (GNN_PostBase mse -> FNO2d mse, mae -> FNO2d mae), decisão do usuário.
# BASE_LOSS    = 'mae'   # loss do FNO2d desta bateria usado como base do GNN_PostBase
#                        # (mesma loss da base antiga, mesh_138x276_FEMM_MESH/FNO2d/run_0001)


# Cada entrada: (arch, dataset, lr, scheduler_gamma, arch_cfg) — valores
# reconstruídos do config.json do melhor run de cada arquitetura (mae_hw/
# mae_graph mais baixo entre os runs com treino substancial em datasets mesh_*).
BEST_CONFIGS = [
    dict(
        arch='FNO2d',
        # [REMOVIDO 2026-10-01] dataset antigo (raw v1, B suavizado do FEMM) — ver docstring
        # dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0002 (mae_hw=0.0207 T)
        dataset=f'{UNIFIED_ROOT}/FNO2d',
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
        # [REMOVIDO 2026-10-01] dataset antigo (raw v1, B suavizado do FEMM) — ver docstring
        # dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0002 (mae_graph=0.1010 T)
        dataset=f'{UNIFIED_ROOT}/FNO_GNN',
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
        # [REMOVIDO 2026-10-01] dataset antigo (raw v1, B suavizado do FEMM) — ver docstring
        # dataset='mesh_138x276_FEMM_MESH',       # melhor run: run_0004 (mae_graph=0.1323 T)
        dataset=f'{UNIFIED_ROOT}/GNN_PostBase',
        lr=0.001,
        scheduler_gamma=0.6,
        # [REMOVIDO 2026-10-01] base fixa antiga (treinada contra o B suavizado do FEMM) e
        # construção imediata — substituído por fábrica chamada na hora do treino
        # (_make_postbase_cfg), com base = FNO2d desta bateria (ver docstring).
        # arch_cfg=GNN_PostBaseConfig(
        #     base_run_dir='data/logs/mesh_138x276_FEMM_MESH/FNO2d/run_0001',
        #     base_checkpoint='best',
        #     gnn_node_width=64, gnn_n_layers=6,
        # ),
        # [REMOVIDO 2026-10-01] fábrica sem loss (base com BASE_LOSS fixa)
        # arch_cfg=lambda: _make_postbase_cfg(gnn_node_width=64, gnn_n_layers=6),
        # fábrica recebe a loss do treino -> base = FNO2d desta bateria com a mesma loss
        arch_cfg=lambda loss: _make_postbase_cfg(loss, gnn_node_width=64, gnn_n_layers=6),
    ),
    dict(
        arch='FNO_BipartiteGNN',
        # [REMOVIDO 2026-10-01] dataset antigo (mesmo raw, mas fora da pasta unificada)
        # dataset='mesh_ans_138x276_B',           # melhor run: run_0015 (mae_graph=0.0503 T,
        dataset=f'{UNIFIED_ROOT}/FNO_BipartiteGNN',  # melhor run antigo: mesh_ans_138x276_B/run_0015
        lr=0.01,                                # (mae_graph=0.0503 T, lá com graph_div_b_loss — aqui mse/mae puros)
        scheduler_gamma=0.6,
        arch_cfg=FNO_BipartiteGNNConfig(
            fno_modes1=270, fno_modes2=270, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
        ),
    ),
]

LOSSES = ['mse', 'mae']


# [REMOVIDO 2026-10-01] default loss=BASE_LOSS — loss agora sempre explícita
# def _find_base_run_dir(arch: str = 'FNO2d', loss: str = BASE_LOSS) -> str:
def _find_base_run_dir(loss: str, arch: str = 'FNO2d') -> str:
    """Run mais recente de data/logs/<PROBLEM>/<arch>/ treinado com `loss` e com
    checkpoints/best.pth — a base do GNN_PostBase desta bateria."""
    candidates = []
    for run_dir in sorted((Path('data/logs') / PROBLEM / arch).glob('run_*')):
        cfg_path = run_dir / 'config.json'
        if not cfg_path.exists() or not (run_dir / 'checkpoints' / 'best.pth').exists():
            continue
        with open(cfg_path, encoding='utf-8') as f:
            if json.load(f).get('loss') == loss:
                candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            f"nenhum run {arch} com loss={loss!r} e best.pth em data/logs/{PROBLEM}/{arch}/ "
            f"— base do GNN_PostBase precisa ser treinada antes (ordem de BEST_CONFIGS)")
    return candidates[-1].as_posix()


# [REMOVIDO 2026-10-01] versão sem loss (base com BASE_LOSS fixa)
# def _make_postbase_cfg(**kw) -> GNN_PostBaseConfig:
#     base_run_dir = _find_base_run_dir()
def _make_postbase_cfg(loss: str, **kw) -> GNN_PostBaseConfig:
    base_run_dir = _find_base_run_dir(loss)
    print(f"  GNN_PostBase base_run_dir = {base_run_dir}", flush=True)
    return GNN_PostBaseConfig(base_run_dir=base_run_dir, base_checkpoint='best', **kw)


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
                # [REMOVIDO 2026-10-01] problem derivado do dataset — com dataset em
                # subpasta (<UNIFIED_ROOT>/<arch>) viraria um problem por arch; um único
                # PROBLEM agrupa a bateria em data/logs/<PROBLEM>/<arch>/.
                # problem=f"{spec['dataset']}_best_mse_mae",
                problem=PROBLEM,
                arch=spec['arch'],
                loss=spec['loss'],
                lr=spec['lr'],
                scheduler_gamma=spec['scheduler_gamma'],
                # arch_cfg pode ser fábrica (GNN_PostBase — base só existe após o FNO2d;
                # recebe a loss do treino pra parear com o FNO2d de mesma loss)
                # [REMOVIDO 2026-10-01] fábrica sem argumento
                # arch_cfg=spec['arch_cfg']() if callable(spec['arch_cfg']) else spec['arch_cfg'],
                arch_cfg=(spec['arch_cfg'](spec['loss']) if callable(spec['arch_cfg'])
                          else spec['arch_cfg']),
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
