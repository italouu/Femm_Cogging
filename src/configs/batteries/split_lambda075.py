"""
Bateria split_lambda075 — 2 runs do FNO_SplitGNN (mae/mse, 500 épocas) com a loss
(1 − λ)·L(N_aux) + λ·L(N_base), λ = 0,75 (mais peso em N_base que na bateria split).
Demais hiperparâmetros iguais aos da bateria split (src/configs/batteries/split.py); roda
mae → mse (a mae foi a melhor na split).

Chunks: data/torch/data_chunks/BLDC_138x276/FNO_SplitGNN/ (188 chunks). Split 60/30/10 por
chunk (split_seed=12), igual ao bldc/split.
Logs: data/logs/BLDC_138x276/FNO_SplitGNN/run_XXXX/.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.run_battery split_lambda075 [--dry-run]
"""
from src.configs.battery import BatteryCfg, BatteryRun
from src.configs.training import NnCfg, FNO_SplitGNNConfig
from src.configs.monitor import MonitorCfg
from src.configs.loss import MseLossCfg, MaeLossCfg

LOSS_LAMBDA = 0.75


def _cfg(loss_cfg):
    return NnCfg(
        dataset='BLDC_138x276/FNO_SplitGNN',
        problem='BLDC_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.60, val_split=0.30, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_SplitGNNConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3,
            loss_lambda=LOSS_LAMBDA),
        loss_cfg=loss_cfg,
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            save_best=True, metrics_every_epoch=False),
    )


def split_l075_mae(runs):
    return _cfg(MaeLossCfg())


def split_l075_mse(runs):
    return _cfg(MseLossCfg())


BATTERY = BatteryCfg(
    name='split_lambda075',
    n_repeats=1,
    runs=[
        # mae primeiro (a mse vem depois)
        BatteryRun('split_l075_mae', split_l075_mae),
        BatteryRun('split_l075_mse', split_l075_mse),
    ],
)
