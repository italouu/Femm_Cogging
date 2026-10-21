from dataclasses import dataclass


@dataclass
class MseLossCfg:
    """MSE: média de (pred − y)² sobre todos os escalares (espaço normalizado)."""


@dataclass
class MaeLossCfg:
    """MAE: média de |pred − y| sobre todos os escalares (espaço normalizado)."""


LOSS_CFG_REGISTRY: dict = {
    'mse': MseLossCfg,
    'mae': MaeLossCfg,
}
