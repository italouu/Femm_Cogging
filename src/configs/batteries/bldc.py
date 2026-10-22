"""
Bateria BLDC_138x276 (2026-10-14) — gabarito B suavizado do FEMM COM correntes de fase
(1 canal de entrada a mais, J [A/mm²], último canal de x_hw/elem_x/node_x; ver CLAUDE.md
"Correntes nas bobinas"). Mesmas 10 runs e hiperparâmetros da antiga bateria smooth
(5 archs × mse/mae, 500 épocas, 70/30; removida em 2026-10-21), só trocando
dataset/problem. Cada run escrita por inteiro.

Raw: data/raw/BLDC_138x276/ (6000 amostras, I_pico = 10 A) — scripts.generate_raw.
Chunks: data/torch/data_chunks/BLDC_138x276/<arch>/ (188 chunks: 187×32 + 1×16) —
python -m scripts.build_datasets. Split 70/30: train_split=0.70 → int(131,6) = 131
chunks de treino; test_split=None → teste = complemento (57 chunks).
Logs: data/logs/BLDC_138x276/<arch>/run_XXXX/.

Alvo POLAR (2026-10-17): os chunks foram regerados com node_y/y_hw = (Br, Bθ) (parsers smooth,
ans_parsing._to_polar); hiperparâmetros inalterados, ordem nova: Elem → Bipartite → FNO_GNN →
FNO2d → GNN_PostBase (por último; depende do FNO2d). Substitui a bateria xy: o manifesto
data/logs/_batteries/bldc/ e as runs xy de data/logs/BLDC_138x276/ precisavam sair do caminho antes
(configs idênticas -> mesmo hash -> o runner pularia as runs como já concluídas).
[2026-10-21] As runs atuais de data/logs/BLDC_138x276/ (run_0001/0002 de cada arch, 2026-10-19) já
são desta bateria polar (alvo Br, Bθ; conferido no torque: frame='polar' bate com o FEMM).

    python -m scripts.run_battery bldc [--dry-run]
"""
from src.configs.battery import BatteryCfg, BatteryRun
from src.configs.training import (NnCfg, FNOConfig, FNO_GNNConfig, GNN_PostBaseConfig,
                                  FNO_BipartiteGNNConfig, FNO_BipartiteGNN_ElemConfig)
from src.configs.monitor import MonitorCfg
from src.configs.loss import MseLossCfg, MaeLossCfg


# ── FNO2d ─────────────────────────────────────────────────────────────────────
def fno2d_mse(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO2d',
        problem='BLDC_138x276',
        lr=0.005, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.7,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNOConfig(
            modes1=69, modes2=139, conv_width=8, conv_layers=4,
            lift_width=64, lift_layers=3, proj_width=64, proj_layers=3,
            data_res=(138, 276)),
        loss_cfg=MseLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def fno2d_mae(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO2d',
        problem='BLDC_138x276',
        lr=0.005, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.7,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNOConfig(
            modes1=69, modes2=139, conv_width=8, conv_layers=4,
            lift_width=64, lift_layers=3, proj_width=64, proj_layers=3,
            data_res=(138, 276)),
        loss_cfg=MaeLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


# ── FNO_GNN ───────────────────────────────────────────────────────────────────
def fno_gnn_mse(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO_GNN',
        problem='BLDC_138x276',
        lr=0.003, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.8,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_GNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3,),
        loss_cfg=MseLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def fno_gnn_mae(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO_GNN',
        problem='BLDC_138x276',
        lr=0.003, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.8,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_GNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3,),
        loss_cfg=MaeLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


# ── GNN_PostBase (base = FNO2d desta bateria com a mesma loss) ───────────────
def postbase_mse(runs):
    return NnCfg(
        dataset='BLDC_138x276/GNN_PostBase',
        problem='BLDC_138x276',
        lr=0.001, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=GNN_PostBaseConfig(
            base_run_dir=runs['fno2d_mse'], base_checkpoint='best',
            gnn_node_width=64, gnn_n_layers=6),
        loss_cfg=MseLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def postbase_mae(runs):
    return NnCfg(
        dataset='BLDC_138x276/GNN_PostBase',
        problem='BLDC_138x276',
        lr=0.001, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=GNN_PostBaseConfig(
            base_run_dir=runs['fno2d_mae'], base_checkpoint='best',
            gnn_node_width=64, gnn_n_layers=6),
        loss_cfg=MaeLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


# ── FNO_BipartiteGNN ──────────────────────────────────────────────────────────
def bipartite_mse(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO_BipartiteGNN',
        problem='BLDC_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_BipartiteGNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3,),
        loss_cfg=MseLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def bipartite_mae(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO_BipartiteGNN',
        problem='BLDC_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_BipartiteGNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3,),
        loss_cfg=MaeLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )

# ── FNO_BipartiteGNN_Elem ─────────────────────────────────────────────────────
def elem_mse(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO_BipartiteGNN_Elem',
        problem='BLDC_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_BipartiteGNN_ElemConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3),
        loss_cfg=MseLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


def elem_mae(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO_BipartiteGNN_Elem',
        problem='BLDC_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.70, test_split=None, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_BipartiteGNN_ElemConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3),
        loss_cfg=MaeLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            log_grad_norm=False, save_best=True, metrics_every_epoch=True),
    )


BATTERY = BatteryCfg(
    name='bldc',
    n_repeats=1,
    runs=[
        BatteryRun('elem_mse',      elem_mse),
        BatteryRun('elem_mae',      elem_mae),
        BatteryRun('bipartite_mse', bipartite_mse),
        BatteryRun('bipartite_mae', bipartite_mae),
        BatteryRun('fno_gnn_mse',   fno_gnn_mse),
        BatteryRun('fno_gnn_mae',   fno_gnn_mae),
        BatteryRun('fno2d_mse',     fno2d_mse),
        BatteryRun('fno2d_mae',     fno2d_mae),
        BatteryRun('postbase_mse',  postbase_mse,  needs=('fno2d_mse',)),
        BatteryRun('postbase_mae',  postbase_mae,  needs=('fno2d_mae',)),
    ],
)
