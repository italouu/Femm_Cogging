"""
Parser do motor quadtree para MaskedFNO2d.

MaskedFNO2d opera na grade regular com 8 canais de saída (4 materiais × Bx, By).
As máscaras de material são derivadas on-the-fly de x_hw[:,0] (Mu_r) durante o
treino — nenhum campo extra precisa ser salvo no chunk.

  x_hw [B, 2, H, W]  (Mu_r, M)  →  MaskedFNO2d  →  pred [B, 8, H, W]
                                                       ↓ assemble(masks)
                                                       [B, 2, H, W]  (Bx, By)

build_graph = False: sem grafo, mesmo comportamento do FNO2D_PARSER.
Os chunks gerados com este parser são compatíveis com FNO2d e MaskedFNO2d.
"""
from ._base import MotorQtreeParserConfig

MASKED_FNO2D_PARSER = MotorQtreeParserConfig(
    name           = 'MaskedFNO2d',
    node_x_cols    = [1, 2],    # mu_r, M — mínimo para chunk válido; descartado no loader grid
    node_y_cols    = [0, 1],    # Bx, By  — idem
    x_hw_cols      = [0, 1],    # Mu_r, M  →  x_hw [2, H, W]
    y_hw_cols      = [0, 1],    # Bx, By   →  y_hw [2, H, W]
    edge_attr_cols = [],        # sem arestas
    build_graph    = False,     # MaskedFNO2d não usa grafo
)
