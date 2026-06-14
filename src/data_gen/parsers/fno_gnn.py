"""
Parser do motor quadtree para FNO_GNN.

FNO_GNN usa:
  - Grade: x_hw [B, 2, H, W]  (mu_r, M)   →  FNO2d  →  y_hw_fno [B, 2, H, W]
  - Grafo: node_x [S, 5]  →  GNN correção residual  →  y_nodes [S_tot, 2]

node_x selecionado [S, 5]:
    0  mu_r         — permeabilidade contínua
    1  M            — magnetização contínua
    2  cell_area    — tamanho relativo da célula (sinaliza refinamento)
    3  r_base       — coord radial normalizada  ← usado pelo interpolador bilinear
    4  c_base       — coord angular normalizada ← usado pelo interpolador bilinear

Omitidos:
    0  material_id  — não usado pelo modelo atual
    6  frac_dom     — não usado pelo modelo atual
    7  normal_x  ─┐ placeholders zero — sem informação até pós-processamento
    8  normal_y  ─┘

node_y selecionado [S, 2]:  Bx, By
edge_attr [E, 4]:  Δr, Δc, shared_length, center_dist
                   (delta_mu — col 4 — omitida; não usada pelo modelo atual)
"""
from ._base import MotorQtreeParserConfig

FNO_GNN_PARSER = MotorQtreeParserConfig(
    name           = 'FNO_GNN',
    node_x_cols    = [1, 2, 3, 4, 5],      # mu_r, M, cell_area, r_base, c_base
    node_y_cols    = [0, 1],               # Bx, By
    x_hw_cols      = [0, 1],              # mu_r, M  →  x_hw [2, H, W]
    y_hw_cols      = [0, 1],              # Bx, By   →  y_hw [2, H, W]
    edge_attr_cols = [0, 1, 2, 3],        # Δr, Δc, shared_length, center_dist
    build_graph    = True,
)
