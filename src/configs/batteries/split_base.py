"""
Bateria split_base — 2 runs do FNO_SplitGNN_Base (mse/mae, 500 épocas), alvo polar (Br, Bθ) do
gabarito B suavizado do FEMM só em N_base (vértices por material); N_aux sem gabarito.
Mesmos hiperparâmetros e dados da bateria split (runs elem_* da bldc, metrics_every_epoch=False
-- MAE de validação só no heartbeat). Cada run escrita por inteiro.

Raw: data/raw/BLDC_138x276/ (6000 amostras) — scripts.generate_raw.
Chunks: data/torch/data_chunks/BLDC_138x276/FNO_SplitGNN/ (188 chunks) —
python -m scripts.build_chunks_split. Split 60/30/10 por chunk (split_seed=12), igual ao bldc.
Logs: data/logs/BLDC_138x276/FNO_SplitGNN_Base/run_XXXX/.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.run_battery split_base [--dry-run]
"""
from src.configs.battery import BatteryCfg, BatteryRun
from src.configs.training import NnCfg, FNO_SplitGNN_BaseConfig
from src.configs.monitor import MonitorCfg
from src.configs.loss import MseLossCfg, MaeLossCfg


def split_base_mse(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO_SplitGNN',
        problem='BLDC_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.60, val_split=0.30, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_SplitGNN_BaseConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3),
        loss_cfg=MseLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            # [REMOVIDO] metrics_every_epoch=True -- motivo: a 2ª passada pela validação a cada
            # época custa ~18 s (~11%) com leitura de disco; o MAE fica só no heartbeat.
            save_best=True, metrics_every_epoch=False),
    )


def split_base_mae(runs):
    return NnCfg(
        dataset='BLDC_138x276/FNO_SplitGNN',
        problem='BLDC_138x276',
        lr=0.01, n_epochs=500,
        scheduler='step', scheduler_step=100, scheduler_gamma=0.6,
        batch_size=32, train_split=0.60, val_split=0.30, split_seed=12,
        buffer_size=64, num_workers=2, prefetch_factor=2,
        resume_run=None, resume_checkpoint='latest', resume_modified=False,
        normalize=True,
        arch_cfg=FNO_SplitGNN_BaseConfig(
            fno_modes1=69, fno_modes2=139, fno_conv_width=6, fno_conv_layers=4,
            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
            data_res=(138, 276), gnn_node_width=32, gnn_n_layers=3),
        loss_cfg=MaeLossCfg(),
        monitor_cfg=MonitorCfg(
            checkpoint_every=10, early_stop_patience=10, early_stop_min_delta=1e-6,
            gl_threshold=5.0, gl_patience=3, min_epochs=100,
            # [REMOVIDO] metrics_every_epoch=True -- motivo: a 2ª passada pela validação a cada
            # época custa ~18 s (~11%) com leitura de disco; o MAE fica só no heartbeat.
            save_best=True, metrics_every_epoch=False),
    )


BATTERY = BatteryCfg(
    name='split_base',
    n_repeats=1,
    runs=[
        BatteryRun('split_base_mse', split_base_mse),
        BatteryRun('split_base_mae', split_base_mae),
    ],
)
