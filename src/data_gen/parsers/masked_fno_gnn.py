"""
Parser do motor quadtree para MaskedFNO_GNN.

MaskedFNO_GNN combina FNO mascarado por material (8 canais de saída na grade)
com correção residual via GNN nos nós quadtree (também 8 canais):

  x_hw [B, 2, H, W]  (Mu_r, M)  →  FNO(out=8)  →  y_hw_8 [B, 8, H, W]
                                                       ↓ interpolação
  [node_x | y_fno@nodes]  →  GNN(in=13, out=8)  →  Δ_8
  y_nodes_8 = y_fno@nodes + Δ_8
              ↓ assemble_nodes(material_ids)
              y_nodes_assembled [S_tot, 2]

Formato idêntico ao FNO_GNN_PARSER — chunks são intercambiáveis.
Material ID derivado de node_x[:,0] (mu_r) em runtime com os mesmos
thresholds de _make_material_masks.

node_x selecionado [S, 5]:
    0  mu_r         — permeabilidade (derivação de material_id em runtime)
    1  M            — magnetização
    2  cell_area    — tamanho relativo da célula
    3  r_base       — coord radial normalizada ← interpolador bilinear
    4  c_base       — coord angular normalizada ← interpolador bilinear

node_y [S, 2]:  Bx, By
edge_attr [E, 4]:  Δr, Δc, shared_length, center_dist
"""
from ._base import MotorQtreeParserConfig

MASKED_FNO_GNN_PARSER = MotorQtreeParserConfig(
    name           = 'MaskedFNO_GNN',
    node_x_cols    = [1, 2, 3, 4, 5],   # mu_r, M, cell_area, r_base, c_base
    node_y_cols    = [0, 1],            # Bx, By
    x_hw_cols      = [0, 1],           # mu_r, M  →  x_hw [2, H, W]
    y_hw_cols      = [0, 1],           # Bx, By   →  y_hw [2, H, W]
    edge_attr_cols = [0, 1, 2, 3],     # Δr, Δc, shared_length, center_dist
    build_graph    = True,
)
