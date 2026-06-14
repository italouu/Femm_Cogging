"""
Parser do motor quadtree para FNO2d pura.

FNO2d opera exclusivamente na grade regular:
  - x_hw [B, 2, H, W]  (mu_r, M)   →  FNO2d  →  y_hw [B, 2, H, W]  (Bx, By)

Sem uso de grafo — toda a informação do quadtree é projetada na grade base
por média ponderada por área (qtree_to_lowres_hw) antes de ser salva no chunk.

build_graph = False poupa a construção do grafo de adjacência (passo mais custoso
da geração de chunks) sem impacto algum no treino/eval do FNO2d.

node_x_cols / node_y_cols mantêm um subconjunto mínimo (mu_r, M | Bx, By)
apenas para que o chunk seja válido — ChunkStreamDataset (modo grid) descarta
node_x/y imediatamente ao carregar o chunk; o overhead de disco é desprezível.
"""
from ._base import MotorQtreeParserConfig

FNO2D_PARSER = MotorQtreeParserConfig(
    name           = 'FNO2d',
    node_x_cols    = [1, 2],    # mu_r, M — mínimo para chunk válido; descartado no loader grid
    node_y_cols    = [0, 1],    # Bx, By  — idem
    x_hw_cols      = [0, 1],    # mu_r, M  →  x_hw [2, H, W]
    y_hw_cols      = [0, 1],    # Bx, By   →  y_hw [2, H, W]
    edge_attr_cols = [],        # sem arestas — build_graph=False pula o loop de adjacência
    build_graph    = False,     # FNO2d não usa grafo — poupa o passo mais custoso
)
