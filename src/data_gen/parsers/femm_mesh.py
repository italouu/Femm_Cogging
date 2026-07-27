"""
Parser da malha real do FEMM (mode='femm_mesh') para arquiteturas tipo GNN.

Layout de origem diferente dos parsers qtree (fno_gnn.py etc.) — ver contrato
completo em src/data_gen/femm_mesh.py e CLAUDE.md ("Chunks — malha real do
FEMM"). node_x aqui tem 9 colunas (mesma contagem do qtree, mas col 3 é
node_dual_area em vez de cell_area) e edge_attr tem só 4 colunas fixas
(sem shared_length — não existe análogo real numa malha triangular).

node_x [S, 9] — colunas disponíveis:
    0  material_id      — categórico (derivado de mu_r)
    1  mu_r             — permeabilidade (valor da malha, não point query)
    2  M                — magnetização
    3  node_dual_area   — área lumped do nó (mm²), soma de 1/3 dos triângulos incidentes
    4  r_base           — coord radial normalizada [0,1]
    5  c_base           — coord angular normalizada [0,1]
    6  frac_dom         — fração de área do material dominante (votação por área)
    7  normal_x         — placeholder 0
    8  normal_y         — placeholder 0

node_x selecionado [S, 5] — mesmo critério do FNO_GNN_PARSER (features
contínuas + posição, sem categórico/placeholder):
    mu_r, M, node_dual_area, r_base, c_base

edge_attr [E, 4] — TODAS as colunas disponíveis (mesh já não tem
shared_length pra descartar; delta_mu aqui é a 4ª coluna fixa, não uma
extra opcional como no FNO_GNN_v2 do qtree):
    0  delta_r       diferença radial destino−origem
    1  delta_c       diferença angular destino−origem (com wrap)
    2  center_dist   distância euclidiana entre centros
    3  delta_mu      mu_r[origem] − mu_r[destino] — direcional, ver nota em
                      _base.py (convenção invertida em relação a delta_r/delta_c)

node_y selecionado [S, 2]: Bx, By

Se no futuro algum modelo precisar de material_id/frac_dom/normal_x/normal_y,
criar um parser novo (ex: FEMM_MESH_v2) em vez de alterar este.
"""
from ._base import MotorQtreeParserConfig

FEMM_MESH_PARSER = MotorQtreeParserConfig(
    name           = 'FEMM_MESH',
    node_x_cols    = [1, 2, 3, 4, 5],      # mu_r, M, node_dual_area, r_base, c_base
    node_y_cols    = [0, 1],               # Bx, By
    x_hw_cols      = [0, 1],               # mu_r, M  →  x_hw [2, H, W]
    y_hw_cols      = [0, 1],               # Bx, By   →  y_hw [2, H, W]
    edge_attr_cols = [0, 1, 2, 3],         # Δr, Δc, center_dist, delta_mu
    build_graph    = True,
)
