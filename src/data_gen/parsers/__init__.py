"""
Parsers de features por arquitetura — motor quadtree.

Cada parser é uma instância de MotorQtreeParserConfig que declara
quais colunas de node_x / node_y / canais de x_hw / y_hw exportar.

Uso típico em build_data_chunks.py:
    from src.data_gen.parsers import FNO_GNN_PARSER
    save_qtree_chunks(unifier, parser_cfg=FNO_GNN_PARSER)

    from src.data_gen.parsers import FNO2D_PARSER
    save_qtree_chunks(unifier, parser_cfg=FNO2D_PARSER)

Para adicionar um novo parser, crie um arquivo <arch>.py neste diretório
instanciando MotorQtreeParserConfig e exporte-o aqui.
"""
from ._base            import MotorQtreeParserConfig, apply_parser_config
from .fno_gnn          import FNO_GNN_PARSER
from .fno_gnn_v2       import FNO_GNN_V2_PARSER
from .fno2d            import FNO2D_PARSER
from .masked_fno2d     import MASKED_FNO2D_PARSER
from .masked_fno_gnn   import MASKED_FNO_GNN_PARSER
from .femm_mesh        import FEMM_MESH_PARSER
from .femm_mesh_a       import FEMM_MESH_A_PARSER
from .femm_mesh_v2_b    import FEMM_MESH_V2_PARSER
from .femm_mesh_v2_a    import FEMM_MESH_V2_A_PARSER

PARSER_REGISTRY = {
    'FNO_GNN':        FNO_GNN_PARSER,
    'FNO_GNN_v2':     FNO_GNN_V2_PARSER,   # FNO_GNN_PARSER + delta_mu direcional em edge_attr
    'FNO2D':          FNO2D_PARSER,
    'MaskedFNO2d':    MASKED_FNO2D_PARSER,
    'MaskedFNO_GNN':  MASKED_FNO_GNN_PARSER,
    'FEMM_MESH':      FEMM_MESH_PARSER,    # mode='femm_mesh' — layout de 9/4 colunas, sem shared_length
    'FEMM_MESH_A':    FEMM_MESH_A_PARSER,  # mode='femm_mesh' — alvo A (1 canal) em vez de B
    # mode='femm_mesh_v2' — só carregam target_field (ver docstring de
    # femm_mesh_v2_b.py/femm_mesh_v2_a.py); demais campos de
    # MotorQtreeParserConfig não se aplicam (parse_ans_gzip_sample já produz
    # o formato final, sem seleção de coluna). Arch: FNO_BipartiteGNN.
    'FEMM_MESH_V2':   FEMM_MESH_V2_PARSER,    # alvo B (Bx,By via curl(A))
    'FEMM_MESH_V2_A': FEMM_MESH_V2_A_PARSER,  # alvo A (potencial vetor)
}

__all__ = [
    'MotorQtreeParserConfig',
    'apply_parser_config',
    'FNO_GNN_PARSER',
    'FNO_GNN_V2_PARSER',
    'FNO2D_PARSER',
    'MASKED_FNO2D_PARSER',
    'MASKED_FNO_GNN_PARSER',
    'FEMM_MESH_PARSER',
    'FEMM_MESH_A_PARSER',
    'FEMM_MESH_V2_PARSER',
    'FEMM_MESH_V2_A_PARSER',
    'PARSER_REGISTRY',
]
