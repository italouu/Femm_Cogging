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


@dataclass
class DivBLossCfg:
    """
    Config para graph_div_b_loss (src/neural_op/losses.py) -- só faz sentido
    pra FNO_BipartiteGNN (mode='femm_mesh_v2', target_field='B', ver CLAUDE.md
    "Grafo duplo vértices+elementos"): soma à loss de ajuste normal (`base_loss`)
    uma penalidade pelo divergente discreto (nodal, fraco/Galerkin -- mesmo
    operador de tests/proto_div_b_check.py::weak_nodal_divergence) da predição
    B nos NÓS do grafo.

    Funciona EXCLUSIVAMENTE na parcela de grafo (nós) da loss combinada --
    make_fno_bipartite_gnn_step detecta esta config (via `lambda_div`) e usa
    `base_loss` puro (sem termo de div) na parcela de grade (FNO/H×W), que não
    tem malha real nem conectividade de elementos pra calcular divergente.

    Geometria em mm reais, não aproximada: o chunk de treino só carrega
    node_x=[r_base,c_base] (coordenadas polares normalizadas, ver
    src/data_gen/parsers/femm_mesh_v2.py) -- mas a janela de amostragem
    (r_in/r_ext/ang_1/ang_2) é CONSTANTE pra todo o dataset mesh_ans_138x276
    (confirmado: outer_diameter/inner_diameter fixos em 93/57mm nas 4000
    amostras -- hardcoded em todos os métodos de BLDC_Process.generate_*;
    ang_1/ang_2 fixos em 0°/120° em DatagenConfig), então dá pra reconstruir
    (x,y) reais em mm de forma EXATA, sem dado extra por amostra:
        r = r_in_mm + r_base·(r_ext_mm - r_in_mm)
        θ = ang_1_deg + c_base·(ang_2_deg - ang_1_deg)
        x, y = r·cos(θ), r·sin(θ)
    (graph_weak_divergence, src/neural_op/losses.py, faz essa conversão antes
    de calcular os coeficientes de gradiente P1 -- usar r_base/c_base direto
    como se fossem x,y ignoraria a métrica polar, ds²=dr²+r²dθ², distorcendo
    direção/magnitude do gradiente calculado). Os 4 valores default abaixo
    batem com esse dataset; se um dataset futuro usar outra janela, ajustar
    aqui (não há como auto-detectar -- não fica salvo no chunk).

    Resta uma aproximação menor, essa sim inerente ao treino: y_nodes/node_y
    chegam z-score normalizados (Bx,By com std ligeiramente diferentes) --
    o divergente é calculado nessa escala normalizada, não em Tesla/metro
    calibrado. Ver tests/proto_div_b_model_check.py para a análise física
    completa (em mm reais E unidade física, fora do loop de treino) que
    motivou esta loss.
    """
    base_loss:  str   = 'mae'   # loss de ajuste (chave em LOSS_REGISTRY, sem assinatura estendida)
    lambda_div: float = 0.3     # peso do termo de divergente somado à loss de ajuste
    r_in_mm:    float = 28.5    # janela de amostragem (mesh_ans_138x276) -- inner_diameter/2
    r_ext_mm:   float = 46.5    # outer_diameter/2
    ang_1_deg:  float = 0.0     # DatagenConfig.ang_1
    ang_2_deg:  float = 120.0   # DatagenConfig.ang_2


# Retrocompatibilidade: LossCfg era o único tipo antes da separação por loss.
LossCfg = MseLossCfg

LOSS_CFG_REGISTRY: dict = {
    'mse':                      MseLossCfg,
    'mae':                      MaeLossCfg,
    'relative_l2':              RelativeL2LossCfg,
    'masked_fno_loss':          MaskedFNOLossCfg,
    'single_material_fno_loss': SingleMaterialFNOLossCfg,
    'masked_fno_gnn_loss':      MaskedFNOGNNLossCfg,
    'graph_div_b_loss':         DivBLossCfg,
}
