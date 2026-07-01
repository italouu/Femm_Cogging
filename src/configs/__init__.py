from src.configs.datagen  import DatagenConfig
from src.configs.training import NnCfg, FNOConfig, MaskedFNO2dConfig, FNO_GNNConfig, FNORefConfig, MaskedFNO_GNNConfig, SingleMatFNOConfig, GNN_PostBaseConfig
# [REMOVIDO] PhiDeepONetConfig — removida (2026-05-27)
from src.configs.monitor  import MonitorCfg
from src.configs.eval     import EvalCfg
from src.configs.bench    import BenchCfg
from src.configs.loss     import (LossCfg, MseLossCfg, MaeLossCfg, RelativeL2LossCfg,
                                  MaskedFNOLossCfg, SingleMaterialFNOLossCfg,
                                  MaskedFNOGNNLossCfg, LOSS_CFG_REGISTRY)
