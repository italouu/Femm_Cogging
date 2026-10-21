from dataclasses import dataclass


@dataclass
class MseLossCfg:
    """MSE: média de (pred − y)² sobre todos os escalares (espaço normalizado)."""


@dataclass
class MaeLossCfg:
    """MAE: média de |pred − y| sobre todos os escalares (espaço normalizado)."""


@dataclass
class MaskedFNOLossCfg:
    """Config para masked_fno_loss (MaskedFNO2d). Sem parâmetros configuráveis."""


@dataclass
class SingleMaterialFNOLossCfg:
    """Config para single_material_fno_loss (FNO2d_SingleMat). Sem parâmetros configuráveis."""


@dataclass
class MaskedFNOGNNLossCfg:
    """Config para masked_fno_gnn_loss (MaskedFNO_GNN). lambda_loss vem de arch_cfg."""


# Retrocompatibilidade: LossCfg era o único tipo antes da separação por loss.
LossCfg = MseLossCfg

LOSS_CFG_REGISTRY: dict = {
    'mse':                      MseLossCfg,
    'mae':                      MaeLossCfg,
    'masked_fno_loss':          MaskedFNOLossCfg,
    'single_material_fno_loss': SingleMaterialFNOLossCfg,
    'masked_fno_gnn_loss':      MaskedFNOGNNLossCfg,
}
