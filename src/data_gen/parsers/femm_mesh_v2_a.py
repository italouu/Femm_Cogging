"""
Parser da malha real do FEMM v2 (mode='femm_mesh_v2') — alvo A (potencial
vetor escalar). Mesmo racional de femm_mesh_v2_b.py — só target_field muda;
ver aquele módulo para o porquê de node_x_cols/node_y_cols/etc. ficarem
vazios ("ignorado").

node_y [S,1]: A — valor nodal exato do .ans (sem curl, sem média).
y_hw [1,H,W]: A — interpolação baricêntrica.
"""
from ._base import MotorQtreeParserConfig

FEMM_MESH_V2_A_PARSER = MotorQtreeParserConfig(
    name           = 'FEMM_MESH_V2_A',
    node_x_cols    = [],   # ignorado — ver femm_mesh_v2_b.py
    node_y_cols    = [],   # ignorado
    x_hw_cols      = [],   # ignorado
    y_hw_cols      = [],   # ignorado
    edge_attr_cols = [],   # ignorado
    build_graph    = True,
    target_field   = 'A',
)
