from dataclasses import dataclass, field, fields
from typing import Any, Optional

from src.configs.monitor import MonitorCfg   # noqa: F401  re-exportado daqui por retrocompatibilidade
from src.configs.loss    import (            # noqa: F401  re-exportado daqui por retrocompatibilidade
    LossCfg, MseLossCfg, MaeLossCfg, RelativeL2LossCfg,
    MaskedFNOLossCfg, SingleMaterialFNOLossCfg, MaskedFNOGNNLossCfg,
    LOSS_CFG_REGISTRY,
)


# ── Arquiteturas ──────────────────────────────────────────────────────────────

@dataclass
class FNOConfig:
    in_channels: int = 2
    out_channels: int = 2
    modes1: int = 270
    modes2: int = 270
    conv_width: int = 6

    conv_layers: int = 4
    lift_width: int = 64
    lift_layers: int = 3
    proj_width: int = 64
    proj_layers: int = 3
    data_res: tuple = (135, 270)


@dataclass
class FNORefConfig:
    in_channels             : int   = 1
    out_channels            : int   = 1
    n_modes                 : tuple = (12, 12)
    hidden_channels         : int   = 64
    n_layers                : int   = 4
    lifting_channel_ratio   : float = 2.0
    projection_channel_ratio: float = 2.0
    data_res                : tuple = (128, 128)  # não passado ao construtor; documenta resolução esperada


@dataclass
class SingleMatFNOConfig:
    material_id : int   = 0        # 0=ferro, 1=ar, 2=ima, 3=cobre
    in_channels : int   = 2
    out_channels: int   = 2        # sempre 2: Bx, By — não alterar
    modes1      : int   = 240
    modes2      : int   = 240
    conv_width  : int   = 6
    conv_layers : int   = 4
    lift_width  : int   = 64
    lift_layers : int   = 3
    proj_width  : int   = 64
    proj_layers : int   = 3
    data_res    : tuple = (80, 240)


@dataclass
class MaskedFNO2dConfig:
    in_channels : int   = 2
    out_channels: int   = 8      # 4 materiais × 2 campos (Bx, By) — ferro=0, ar=1, ima=2, cobre=3
    modes1      : int   = 240
    modes2      : int   = 240
    conv_width  : int   = 6
    conv_layers : int   = 4
    lift_width  : int   = 64
    lift_layers : int   = 3
    proj_width  : int   = 64
    proj_layers : int   = 3
    data_res    : tuple = (80, 240)


@dataclass
class FNO_GNNConfig:
    fno_modes1: int = 270
    fno_modes2: int = 270
    fno_conv_width: int = 6
    fno_conv_layers: int = 4
    fno_lift_width: int = 64
    fno_lift_layers: int = 3
    fno_proj_width: int = 64
    fno_proj_layers: int = 3
    data_res: tuple = (135, 270)
    gnn_node_width: int = 32
    gnn_n_layers: int = 3
    lambda_loss: float = 0   # peso da loss de grade; loss_nós = 1 - lambda_loss


@dataclass
class FNO_GNN_v2Config(FNO_GNNConfig):
    """Mesmos campos de FNO_GNNConfig — FNO_GNN_v2 (src/neural_op/archs/fno_gnn_v2.py)
    só muda _EDGE_DIM (4→5) internamente, sem novo hiperparâmetro. Classe
    própria (em vez de reaproveitar FNO_GNNConfig) para que NnCfg.__post_init__
    valide arch↔arch_cfg corretamente e para runs registrarem no config.json
    qual variante (v1/v2) foi usada. Requer dataset gerado com
    npz_parser='FNO_GNN_v2' (edge_attr [E,5], inclui delta_mu direcional —
    ver src/data_gen/parsers/fno_gnn_v2.py)."""
    pass


@dataclass
class GNN_PostBaseConfig:
    # Treino em duas etapas (não end-to-end): base_run_dir aponta para um run já treinado
    # (FNO2d ou FNO_GNN), congelado; só o GNN novo é treinado.
    base_run_dir   : str = 'data/logs/motor_default_v2_135x270/FNO2d/run_0001'
    base_checkpoint: str = 'best'   # 'best', 'latest' ou 'final'
    gnn_node_width : int = 32
    gnn_n_layers   : int = 3

    # Snapshot de arch/arch_cfg/epoch de base_run_dir, capturado em __post_init__ e
    # gravado em config.json desta run (via NnCfg → ModelManager.open). Garante que os
    # parâmetros do modelo base não se percam mesmo que base_run_dir seja movido/apagado
    # depois — usado como fallback em GNN_PostBase._load_frozen_base.
    base_arch     : str           = field(default='', init=False)
    base_arch_cfg : dict          = field(default_factory=dict, init=False)
    base_epoch    : Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        import json
        from pathlib import Path
        import torch

        run_dir  = Path(self.base_run_dir)
        cfg_dict = json.loads((run_dir / 'config.json').read_text())
        self.base_arch     = cfg_dict['arch']
        self.base_arch_cfg = cfg_dict['arch_cfg']

        ckpt_path = (run_dir / 'model_final.pth') if self.base_checkpoint == 'final' \
            else (run_dir / 'checkpoints' / f'{self.base_checkpoint}.pth')
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            self.base_epoch = ckpt.get('epoch')

    @classmethod
    def from_dict(cls, d: dict):
        # Reconstrói a partir de um config.json já salvo (eval/resume) sem rechamar
        # __post_init__ — evita depender de base_run_dir ainda existir; usa o snapshot
        # (base_arch/base_arch_cfg/base_epoch) já gravado na própria run.
        obj = cls.__new__(cls)
        for f in fields(cls):
            setattr(obj, f.name, d[f.name])
        return obj


@dataclass
class MaskedFNO_GNNConfig:
    # FNO com 8 canais de saída (4 materiais × Bx, By); GNN corrige nos 8 canais
    fno_modes1     : int   = 240
    fno_modes2     : int   = 240
    fno_conv_width : int   = 8
    fno_conv_layers: int   = 4
    fno_lift_width : int   = 64
    fno_lift_layers: int   = 3
    fno_proj_width : int   = 64
    fno_proj_layers: int   = 3
    data_res       : tuple = (80, 240)
    gnn_node_width : int   = 32
    gnn_n_layers   : int   = 3
    lambda_loss    : float = 0.5   # peso da loss de grade; loss_nós = 1 - lambda_loss


# ── Config principal ──────────────────────────────────────────────────────────
#
# Referência rápida — arch e loss disponíveis
# ┌─────────────────┬──────────────────────┬──────────────────┬──────────────────────────┐
# │ arch            │ loss recomendada      │ loader_mode      │ arch_cfg                 │
# ├─────────────────┼──────────────────────┼──────────────────┼──────────────────────────┤
# │ FNO2d           │ mse / mae /          │ grid             │ FNOConfig                │
# │                 │   relative_l2        │                  │                          │
# │ FNO_ref         │ mse / mae /          │ grid             │ FNORefConfig             │
# │                 │   relative_l2        │                  │                          │
# │ MaskedFNO2d     │ masked_fno_loss      │ grid             │ MaskedFNO2dConfig        │
# │ FNO2d_SingleMat │ single_material_     │ grid             │ SingleMatFNOConfig       │
# │                 │   fno_loss           │                  │   (material_id=0..3)     │
# │ FNO_GNN         │ mse / mae /          │ qtree            │ FNO_GNNConfig            │
# │                 │   relative_l2        │                  │                          │
# │ FNO_GNN_v2      │ mse / mae /          │ qtree            │ FNO_GNN_v2Config         │
# │                 │   relative_l2        │                  │   (edge_attr c/ delta_mu │
# │                 │                      │                  │    direcional; parser    │
# │                 │                      │                  │    FNO_GNN_v2 no npz)    │
# │ MaskedFNO_GNN   │ masked_fno_gnn_loss  │ qtree            │ MaskedFNO_GNNConfig      │
# │ FNO_GNN_Field   │ mse / mae /          │ qtree            │ FNO_GNNConfig            │
# │                 │   relative_l2        │                  │   (campo direto nos nós) │
# │ GNN_PostBase    │ mse / mae /          │ qtree            │ GNN_PostBaseConfig       │
# │                 │   relative_l2        │                  │   (base FNO2d/FNO_GNN    │
# │                 │                      │                  │    congelada)            │
# └─────────────────┴──────────────────────┴──────────────────┴──────────────────────────┘
#
# Chaves de loss — LOSS_REGISTRY (src/neural_op/losses.py)
# Config correspondente — LOSS_CFG_REGISTRY (src/configs/loss.py)
# ┌──────────────────────┬─────────────────────────┬────────────────────────────────────┐
# │ chave                │ cfg_cls                 │ campos configuráveis               │
# ├──────────────────────┼─────────────────────────┼────────────────────────────────────┤
# │ 'mse'                │ MseLossCfg              │ tail_alpha, tail_k_frac,           │
# │                      │                         │ subtract_fno (FNO_GNN/PostBase)    │
# │ 'mae'                │ MaeLossCfg              │ subtract_fno (FNO_GNN/PostBase)    │
# │ 'relative_l2'        │ RelativeL2LossCfg       │ subtract_fno (FNO_GNN/PostBase)    │
# │ 'masked_fno_loss'    │ MaskedFNOLossCfg        │ —                                  │
# │ 'single_material_    │ SingleMaterialFNOLossCfg│ —                                  │
# │   fno_loss'          │                         │                                    │
# │ 'masked_fno_gnn_loss'│ MaskedFNOGNNLossCfg     │ — (lambda_loss vem de arch_cfg)    │
# └──────────────────────┴─────────────────────────┴────────────────────────────────────┘
#
# subtract_fno (MseLossCfg / MaeLossCfg / RelativeL2LossCfg):
#   False (padrão) → loss_nós = loss_fn(fno_at_nodes + delta, y_node)
#   True           → loss_nós = loss_fn(delta, y_node)
#                    delta = output bruto do GNN (sem soma da baseline FNO)
#                    inference permanece inalterada: retorna fno + delta

@dataclass
class NnCfg:
    dataset: str = 'motor_default_v3_135x270'
    arch: str = 'FNO2d'
    loss: str = 'mse'

    problem: str = 'motor_default_v3_135x270'

    # Treino
    lr: float = 1e-3
    n_epochs: int = 500
    scheduler: str = 'step'
    scheduler_step: int = 100
    scheduler_gamma: float = 0.8

    # Dataloader
    batch_size: int = 32
    train_split: float = 0.30
    test_split: Optional[float] = None  # None → complemento de train_split (1 - train_split);
                                         # valor explícito → fração fixa de chunks para teste,
                                         # útil para agilizar épocas com train_split pequeno
    buffer_size: int = 64
    num_workers: int = 2
    prefetch_factor: int = 2
    split_seed: int = 12

    # Resume
    resume_run:        Optional[str] = None     # caminho para run_XXXX/ existente
    resume_checkpoint: str           = 'latest' # 'latest' ou 'best'
    resume_modified:   bool          = False    # True = carrega só pesos; optimizer/scheduler do zero

    arch_cfg:     Any        = None   # None → auto-instanciado em __post_init__ via ARCH_REGISTRY
    monitor_cfg:  MonitorCfg = field(default_factory=MonitorCfg)
    loss_cfg:     Any        = None   # None → auto-instanciado em __post_init__ via LOSS_CFG_REGISTRY

    def __post_init__(self):
        from pathlib import Path
        from src.neural_op.archs import ARCH_REGISTRY
        if self.arch not in ARCH_REGISTRY:
            raise ValueError(
                f"arch='{self.arch}' não existe no ARCH_REGISTRY. "
                f"Disponíveis: {list(ARCH_REGISTRY)}"
            )
        expected = ARCH_REGISTRY[self.arch].cfg_cls
        if self.arch_cfg is None:
            self.arch_cfg = expected()
        elif not isinstance(self.arch_cfg, expected):
            raise TypeError(
                f"arch='{self.arch}' espera arch_cfg do tipo {expected.__name__}, "
                f"recebido {type(self.arch_cfg).__name__}"
            )
        if self.loss_cfg is None:
            cls = LOSS_CFG_REGISTRY.get(self.loss, MseLossCfg)
            self.loss_cfg = cls()
        if self.resume_run is not None and not Path(self.resume_run).exists():
            resolved = Path('data/logs') / self.problem / self.arch / self.resume_run
            self.resume_run = str(resolved)