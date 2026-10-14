"""
Bateria smooth (gabarito B suavizado do FEMM) — a única bateria do projeto: 5 archs × mse/mae
(FNO2d, FNO_GNN, GNN_PostBase, FNO_BipartiteGNN e FNO_BipartiteGNN_Elem), na ordem em que
foram treinadas (2026-10-09/10). Cada run escrita por inteiro.
Logs: data/logs/mesh_ans_138x276/<arch>/run_XXXX/ (as 5 archs, inclusive a Elem).
Chunks: data/torch/data_chunks/mesh_ans_138x276/<arch>/ — python -m scripts.build_smooth_datasets.
(Até 2026-10-13: chunks em mesh_ans_138x276_smooth/, logs em mesh_ans_138x276_smooth_battery/
e, para a Elem, mesh_ans_138x276_smooth_elem_proto/ — runs de 4000 amostras.)

    python -m scripts.run_battery smooth [--dry-run]

2026-10-13: dataset smooth reduzido às 2000 primeiras amostras do raw (63 chunks: 62×32 +
1×16, scripts.build_smooth_datasets). Split 70/30: train_split=0.70 → 44 chunks de treino;
test_split=None → teste = complemento (19 chunks). test_split=0.30 daria int(18,9)=18 e
deixaria 1 chunk fora de tudo.
"""
# [REMOVIDO 2026-10-13] valores de todas as runs na bateria de 4000 amostras (125 chunks:
# 37 treino / 37 teste / 51 fora). Trocados para o dataset de 2000 amostras com 70/30 e
# 500 épocas (pedido do usuário):
#     dataset='mesh_ans_138x276_smooth/<arch>',
#     problem='mesh_ans_138x276_smooth_battery',   (Elem: 'mesh_ans_138x276_smooth_elem_proto')
#     n_epochs=350, ...
#     train_split=0.30, test_split=0.30, split_seed=12,
from src.configs.battery import BatteryCfg, BatteryRun
from src.configs.training import (NnCfg, FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig,
                                  FNO_BipartiteGNNConfig, FNO_BipartiteGNN_ElemConfig)
from src.configs.monitor import MonitorCfg
from src.configs.loss import MseLossCfg, MaeLossCfg


# ── FNO2d ─────────────────────────────────────────────────────────────────────
def fno2d_mse(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/FNO2d',
        problem='mesh_ans_138x276',
        lr=0.005, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.7,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNOConfig(
            modes1=69, modes2=139, conv_width=8, conv_layers=4,
            lift_width=64, lift_layers=3, proj_width=64, proj_layers=3,
            data_res=(138, 276), interp_mode='cell_centered_border'),
        loss_cfg=MseLossCfg(tail_alpha=0.0, tail_k_frac=0.0, subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def fno2d_mae(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/FNO2d',
        problem='mesh_ans_138x276',
        lr=0.005, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.7,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNOConfig(
            modes1=69, modes2=139, conv_width=8, conv_layers=4,
            lift_width=64, lift_layers=3, proj_width=64, proj_layers=3,
            data_res=(138, 276), interp_mode='cell_centered_border'),
        loss_cfg=MaeLossCfg(subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


# ── FNO_GNN ───────────────────────────────────────────────────────────────────
def fno_gnn_mse(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/FNO_GNN',
        problem='mesh_ans_138x276',
        lr=0.003, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.8,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_GNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
            interp_mode='cell_centered_border', fno_node_rescale=True),
        loss_cfg=MseLossCfg(tail_alpha=0.0, tail_k_frac=0.0, subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def fno_gnn_mae(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/FNO_GNN',
        problem='mesh_ans_138x276',
        lr=0.003, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.8,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_GNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
            interp_mode='cell_centered_border', fno_node_rescale=True),
        loss_cfg=MaeLossCfg(subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


# ── GNN_PostBase (base = FNO2d desta bateria com a mesma loss) ───────────────
def postbase_mse(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/GNN_PostBase',
        problem='mesh_ans_138x276',
        lr=0.001, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=GNN_PostBaseConfig(
            base_run_dir=runs['fno2d_mse'], base_checkpoint='best',
            gnn_node_width=64, gnn_n_layers=6, interp_mode='cell_centered_border'),
        loss_cfg=MseLossCfg(tail_alpha=0.0, tail_k_frac=0.0, subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def postbase_mae(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/GNN_PostBase',
        problem='mesh_ans_138x276',
        lr=0.001, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=GNN_PostBaseConfig(
            base_run_dir=runs['fno2d_mae'], base_checkpoint='best',
            gnn_node_width=64, gnn_n_layers=6, interp_mode='cell_centered_border'),
        loss_cfg=MaeLossCfg(subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


# ── FNO_BipartiteGNN ──────────────────────────────────────────────────────────
def bipartite_mse(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/FNO_BipartiteGNN',
        problem='mesh_ans_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_BipartiteGNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
            interp_mode='cell_centered_border', fno_node_rescale=True),
        loss_cfg=MseLossCfg(tail_alpha=0.0, tail_k_frac=0.0, subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def bipartite_mae(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/FNO_BipartiteGNN',
        problem='mesh_ans_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_BipartiteGNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
            interp_mode='cell_centered_border', fno_node_rescale=True),
        loss_cfg=MaeLossCfg(subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )

# ── FNO_BipartiteGNN_Elem ─────────────────────────────────────────────────────
def elem_mse(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/FNO_BipartiteGNN_Elem',
        problem='mesh_ans_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_BipartiteGNN_ElemConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
            interp_mode='cell_centered_border', fno_node_rescale=True, aux_fno=False),
        loss_cfg=MseLossCfg(tail_alpha=0.0, tail_k_frac=0.0, subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def elem_mae(runs):
    return NnCfg(
        dataset='mesh_ans_138x276/FNO_BipartiteGNN_Elem',
        problem='mesh_ans_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_BipartiteGNN_ElemConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
            interp_mode='cell_centered_border', fno_node_rescale=True, aux_fno=False),
        loss_cfg=MaeLossCfg(subtract_fno=False),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


BATTERY = BatteryCfg(
    name='smooth',
    n_repeats=1,
    runs=[
        BatteryRun('fno2d_mse',     fno2d_mse),
        BatteryRun('fno2d_mae',     fno2d_mae),
        BatteryRun('fno_gnn_mse',   fno_gnn_mse),
        BatteryRun('fno_gnn_mae',   fno_gnn_mae),
        BatteryRun('postbase_mse',  postbase_mse,  needs=('fno2d_mse',)),
        BatteryRun('postbase_mae',  postbase_mae,  needs=('fno2d_mae',)),
        BatteryRun('bipartite_mse', bipartite_mse),
        BatteryRun('bipartite_mae', bipartite_mae),
        BatteryRun('elem_mse',      elem_mse),
        BatteryRun('elem_mae',      elem_mae),
    ],
)
