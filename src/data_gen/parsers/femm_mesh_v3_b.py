"""
Parser da malha real do FEMM v3 (mode='femm_mesh_v2', npz_parser='FEMM_MESH_V3')
-- alvo B (Bx,By). Mesmo mecanismo de femm_mesh_v2_b.py: só cfg.target_field
é consumido (por scripts/gen_npz_structures.py::_run_femm_mesh_v2), repassado
pra src/data_gen/parsers/femm_mesh_v3.py::parse_ans_gzip_sample_v3 -- que já
produz o formato final (node_x com node_cell_count, ver docstring daquele
módulo), sem seleção de coluna.

node_y [S,2]: Bx,By -- idêntico a FEMM_MESH_V2_PARSER.
node_x [S,3]: r_base,c_base,node_cell_count (diferente do v2 -- ver
femm_mesh_v3.py).
"""
from ._base import MotorQtreeParserConfig

FEMM_MESH_V3_PARSER = MotorQtreeParserConfig(
    name           = 'FEMM_MESH_V3',
    node_x_cols    = [],   # ignorado — node_x já sai fixo [r_base,c_base,node_cell_count]
    node_y_cols    = [],   # ignorado — node_y decidido por target_field dentro do parser
    x_hw_cols      = [],   # ignorado — x_hw já sai fixo [Mu_r,M]
    y_hw_cols      = [],   # ignorado — idem node_y_cols
    edge_attr_cols = [],   # ignorado — edge_attr já sai fixo [delta_r,delta_c,center_dist]
    build_graph    = True,
    target_field   = 'B',
)
