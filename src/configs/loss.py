from dataclasses import dataclass

@dataclass
class MseLossCfg:
    """
    Hiperparâmetros da MSE com termo de cauda top-k opcional e subtract_fno.

    tail_alpha=0  → tail desligado; loss_final == MSE pura.
    subtract_fno  → FNO_GNN / GNN_PostBase: loss de nós em delta (output bruto do GNN)
                    vs y_node em vez de (fno_at_nodes + delta) vs y_node.
    """
    tail_alpha:   float = 0.0
    tail_k_frac:  float = 0.00
    subtract_fno: bool  = False


@dataclass
class MaeLossCfg:
    """
    Config para MAE com opção subtract_fno (FNO_GNN / GNN_PostBase).

    subtract_fno  → loss de nós em delta (output bruto do GNN) vs y_node.
    """
    subtract_fno: bool = False


@dataclass
class RelativeL2LossCfg:
    """
    Config para L2 relativo com opção subtract_fno (FNO_GNN / GNN_PostBase).

    subtract_fno  → loss de nós em delta (output bruto do GNN) vs y_node.
    """
    subtract_fno: bool = False


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
    'relative_l2':              RelativeL2LossCfg,
    'masked_fno_loss':          MaskedFNOLossCfg,
    'single_material_fno_loss': SingleMaterialFNOLossCfg,
    'masked_fno_gnn_loss':      MaskedFNOGNNLossCfg,
}
