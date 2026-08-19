"""
Parser da malha real do FEMM v2 (mode='femm_mesh_v2') — alvo B (Bx,By).

Diferente dos parsers qtree/femm_mesh v1 (MotorQtreeParserConfig sobre um
dict JÁ com todas as colunas calculadas, filtrando por índice), o pipeline
v2 (src/data_gen/parsers/femm_mesh_v2.py::parse_ans_gzip_sample) decide o
que CALCULAR internamente a partir de `target_field` — não há colunas extras
pra descartar depois (node_x/elem_x/edge_attr já saem no formato final,
fixo, nos dois casos). Por isso node_x_cols/node_y_cols/x_hw_cols/y_hw_cols/
edge_attr_cols abaixo ficam vazios ("ignorado") — só cfg.target_field é
consumido, por scripts/gen_npz_structures.py::_run_femm_mesh_v2, que repassa
pra parse_ans_gzip_sample(..., target_field=cfg.target_field). Reaproveitar
MotorQtreeParserConfig aqui (em vez de uma classe própria) é deliberado: dá
pra selecionar o alvo do femm_mesh_v2 pelo mesmo mecanismo já usado em todo
o resto do projeto (DatagenConfig.npz_parser -> PARSER_REGISTRY), em vez de
um campo dedicado (femm_mesh_v2_target_field, descontinuado — ver
src/configs/datagen.py).

node_y [S,2]: Bx,By — curl(A) fechado por elemento + média nodal (ver
docstring de parse_ans_gzip_sample).
y_hw [2,H,W]: Bx,By — interpolação baricêntrica (mesmo padrão de A).
"""
from ._base import MotorQtreeParserConfig

FEMM_MESH_V2_PARSER = MotorQtreeParserConfig(
    name           = 'FEMM_MESH_V2',
    node_x_cols    = [],   # ignorado — node_x já sai fixo [r_base,c_base] de parse_ans_gzip_sample
    node_y_cols    = [],   # ignorado — node_y decidido por target_field dentro do parser
    x_hw_cols      = [],   # ignorado — x_hw já sai fixo [Mu_r,M]
    y_hw_cols      = [],   # ignorado — idem node_y_cols
    edge_attr_cols = [],   # ignorado — edge_attr já sai fixo [delta_r,delta_c,center_dist]
    build_graph    = True,
    target_field   = 'B',
)
