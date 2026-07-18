"""
Parser do motor quadtree para FNO_GNN_v2.

Variante de FNO_GNN_PARSER (fno_gnn.py) que adiciona delta_mu (permeabilidade
direcional) como feature de aresta. Criado em 2026-07-17 a pedido do usuário,
em vez de alterar FNO_GNN_PARSER/FNO_GNN — mantém os chunks e runs já
treinados com o parser original intactos.

FNO_GNN_v2 usa:
  - Grade: x_hw [B, 2, H, W]  (mu_r, M)   →  FNO2d  →  y_hw_fno [B, 2, H, W]
  - Grafo: node_x [S, 5]  →  GNN correção residual  →  y_nodes [S_tot, 2]

node_x selecionado [S, 5] — idêntico a FNO_GNN_PARSER:
    0  mu_r         — permeabilidade (point query no centro; já é o valor
                      real de PERMEABILITY[material], não material_id)
    1  M            — magnetização contínua
    2  cell_area    — tamanho relativo da célula (sinaliza refinamento)
    3  r_base       — coord radial normalizada  ← usado pelo interpolador bilinear
    4  c_base       — coord angular normalizada ← usado pelo interpolador bilinear

Omitidos (mesmo motivo de FNO_GNN_PARSER):
    0  material_id  — não usado pelo modelo atual
    6  frac_dom     — não usado pelo modelo atual
    7  normal_x  ─┐ placeholders zero — sem informação até pós-processamento
    8  normal_y  ─┘

node_y selecionado [S, 2]:  Bx, By

edge_attr [E, 5]:  Δr, Δc, shared_length, center_dist, delta_mu
    delta_mu = mu_r[origem] − mu_r[destino] — DIRECIONAL, convenção invertida
    em relação a Δr/Δc (que são destino−origem). Positivo quando a aresta sai
    de material de alta permeabilidade para baixa (ex: ferro→ar), negativo no
    sentido inverso. Ver nota completa em build_graph_edges_motor
    (src/data_gen/data_utils.py) e em MotorQtreeParserConfig (_base.py).

Requer modelo FNO_GNN_v2 (src/neural_op/archs/fno_gnn_v2.py), que declara
_EDGE_DIM=5 — chunks gerados por este parser NÃO são compatíveis com
FNO_GNN/FNO_GNN_Field (esperam edge_dim=4).
"""
from ._base import MotorQtreeParserConfig

FNO_GNN_V2_PARSER = MotorQtreeParserConfig(
    name           = 'FNO_GNN_v2',
    node_x_cols    = [1, 2, 3, 4, 5],      # mu_r, M, cell_area, r_base, c_base
    node_y_cols    = [0, 1],               # Bx, By
    x_hw_cols      = [0, 1],              # mu_r, M  →  x_hw [2, H, W]
    y_hw_cols      = [0, 1],              # Bx, By   →  y_hw [2, H, W]
    edge_attr_cols = [0, 1, 2, 3, 4],     # Δr, Δc, shared_length, center_dist, delta_mu
    build_graph    = True,
)
