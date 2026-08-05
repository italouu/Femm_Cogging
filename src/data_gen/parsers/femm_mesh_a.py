"""
Parser da malha real do FEMM (mode='femm_mesh') com alvo A em vez de B.

Mesma entrada (node_x, edge_attr) do FEMM_MESH_PARSER — só o alvo muda:
node_y/y_hw passam a vir de node_A/a_hw (potencial vetor, 1 canal escalar)
em vez de node_y/y_hw brutos (Bx,By, 2 canais). Ver target_field em
_base.py e _apply_parser_femm_mesh em scripts/gen_npz_structures.py.

node_x selecionado [S, 5] — igual ao FEMM_MESH_PARSER:
    mu_r, M, node_dual_area, r_base, c_base

node_y [S, 1]: A (potencial vetor, valor nodal exato do .ans)
y_hw   [1, H, W]: A (point-query ao vivo)

node_y_cols/y_hw_cols abaixo são ignorados (target_field='A' desvia a
construção de node_y/y_hw para node_A/a_hw em _apply_parser_femm_mesh) —
mantidos vazios só para deixar explícito que não se aplicam aqui.

B (Bx, By) não é exportado por este parser.
"""
from ._base import MotorQtreeParserConfig

FEMM_MESH_A_PARSER = MotorQtreeParserConfig(
    name           = 'FEMM_MESH_A',
    node_x_cols    = [1, 2, 3, 4, 5],      # mu_r, M, node_dual_area, r_base, c_base
    node_y_cols    = [],                    # ignorado — node_y vem de node_A
    x_hw_cols      = [0, 1],                # mu_r, M  →  x_hw [2, H, W]
    y_hw_cols      = [],                    # ignorado — y_hw vem de a_hw
    edge_attr_cols = [0, 1, 2, 3],          # Δr, Δc, center_dist, delta_mu
    build_graph    = True,
    target_field   = 'A',
)
