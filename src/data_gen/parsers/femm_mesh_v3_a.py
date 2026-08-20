"""
Parser da malha real do FEMM v3 (mode='femm_mesh_v2', npz_parser='FEMM_MESH_V3_A')
-- alvo A (potencial vetor escalar). Mesmo racional de femm_mesh_v3_b.py --
só target_field muda; ver aquele módulo e femm_mesh_v3.py para o porquê de
node_x_cols/node_y_cols/etc. ficarem vazios ("ignorado") e de node_x ter 3
colunas (não 2, como em FEMM_MESH_V2_A_PARSER).

node_y [S,1]: A -- valor nodal exato do .ans (sem curl, sem média).
node_x [S,3]: r_base,c_base,node_cell_count.
"""
from ._base import MotorQtreeParserConfig

FEMM_MESH_V3_A_PARSER = MotorQtreeParserConfig(
    name           = 'FEMM_MESH_V3_A',
    node_x_cols    = [],   # ignorado — ver femm_mesh_v3_b.py
    node_y_cols    = [],   # ignorado
    x_hw_cols      = [],   # ignorado
    y_hw_cols      = [],   # ignorado
    edge_attr_cols = [],   # ignorado
    build_graph    = True,
    target_field   = 'A',
)
