"""
Parser base para amostras quadtree do motor BLDC.

Define a dataclass de configuração (MotorQtreeParserConfig) e a função
intermediária (apply_parser_config) que seleciona o subconjunto de features
a exportar nos chunks .pt.

Para criar um novo parser basta instanciar MotorQtreeParserConfig com os
índices desejados e importar apply_parser_config — nenhum outro código muda.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MotorQtreeParserConfig:
    """Configuração de seleção de features para um parser do motor quadtree.

    Define quais colunas de node_x e node_y são exportadas,
    e quais canais de x_hw/y_hw são mantidos no chunk final.

    Referência de colunas — node_x [S, 9]:
        0  material_id  (derivado de mu_r: ferro=0, ar=1, ima=2, bobina=3)
        1  mu_r         (permeabilidade — point query no centro)
        2  M            (magnetização   — point query no centro)
        3  cell_area    (1/2^depth)²  área relativa à célula base
        4  r_base       coord radial normalizada do centro da célula base [0,1]
        5  c_base       coord angular normalizada do centro da célula base [0,1]
        6  frac_dom     fração de área do material dominante no bounding box
        7  normal_x     placeholder 0 (pós-processamento futuro)
        8  normal_y     placeholder 0 (pós-processamento futuro)

    Referência de colunas — node_y [S, 2]:
        0  Bx  campo radial (T)
        1  By  campo angular (T)

    Referência de canais — x_hw / y_hw:
        x_hw: canal 0 = mu_r,  canal 1 = M
        y_hw: canal 0 = Bx,    canal 1 = By

    Referência de colunas — edge_attr [E, 5]:
        0  delta_r       diferença radial  j−i (unidades de célula base)
        1  delta_c       diferença angular j−i (com wrap; unidades de célula base)
        2  shared_length comprimento da fronteira compartilhada
        3  center_dist   distância euclidiana entre centros
        4  delta_mu      mu_r[i] − mu_r[j] (origem − destino)  ← única feature que requer mu_r

    NOTA (2026-07-17): delta_mu usa convenção "origem − destino" (i−j), o
    INVERSO de delta_r/delta_c ("destino − origem", j−i). Positivo quando a
    aresta sai de alta permeabilidade para baixa. Ver build_graph_edges_motor
    em data_utils.py. Usada pelo parser FNO_GNN_V2_PARSER (fno_gnn_v2.py);
    FNO_GNN_PARSER e MASKED_FNO_GNN_PARSER continuam sem essa coluna.
    """
    name           : str
    node_x_cols    : list   # índices das colunas de node_x [S, 9] a exportar
    node_y_cols    : list   # índices das colunas de node_y [S, 2] a exportar
    x_hw_cols      : list = field(default_factory=lambda: [0, 1])        # canais de x_hw
    y_hw_cols      : list = field(default_factory=lambda: [0, 1])        # canais de y_hw
    edge_attr_cols : list = field(default_factory=lambda: [0, 1, 2, 3, 4])  # canais de edge_attr
    build_graph    : bool = True   # incluir edge_index e edge_attr no output


def apply_parser_config(sample: dict, cfg: MotorQtreeParserConfig) -> dict:
    """Seleciona features de um dict *completo* (node_x com todas as 9 colunas).

    Uso externo — para quem constrói o dict sem passar pelo QtreeSampleUnifier,
    ou em pipelines de pós-processamento que têm acesso ao dict bruto.

    NÃO chame esta função sobre samples já emitidos por QtreeSampleUnifier
    quando parser_cfg foi fornecido: nesse caso node_x já chega pré-filtrado
    (lazy build via build_node_x_motor(cols=...)) e seria re-sliciado incorretamente.

    Aplica cfg.node_x_cols / cfg.node_y_cols / cfg.x_hw_cols / cfg.y_hw_cols
    como slices de coluna/canal nos arrays correspondentes.
    Campos não afetados (x, y, depth, cells, L, dim) são passados inalterados.
    Se cfg.build_graph=False, remove edge_index e edge_attr do output.

    Parameters
    ----------
    sample : dict com node_x [S, 9] completo (todas as colunas disponíveis)
    cfg    : MotorQtreeParserConfig com as seleções

    Returns
    -------
    dict com os mesmos campos, mas node_x/node_y/x_hw/y_hw filtrados.
    """
    out = dict(sample)                                    # shallow copy

    out['node_x'] = sample['node_x'][:, cfg.node_x_cols] # [S, len(node_x_cols)]
    out['node_y'] = sample['node_y'][:, cfg.node_y_cols] # [S, len(node_y_cols)]
    out['x_hw']   = sample['x_hw'][cfg.x_hw_cols]        # [len(x_hw_cols), H, W]
    out['y_hw']   = sample['y_hw'][cfg.y_hw_cols]        # [len(y_hw_cols), H, W]

    if not cfg.build_graph:
        out.pop('edge_index', None)
        out.pop('edge_attr',  None)

    return out
