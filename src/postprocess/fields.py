"""
Operações sobre o campo: |B| e raio do entreferro (posição das figuras).

Origem: tests/resultados/common.py (2026-10-13), sem mudança de lógica
(_mag -> mag, agora pública).
Não migrados (sem uso desde a remoção do cogging): arc_theta_deg, _barycentric,
arc_elements, arc_weights, arc_values_from_nodes, to_polar.
"""
import numpy as np
import torch

from src.data_gen.parsers.femm_mesh_smooth import _node_b_by_material


def mag(t):
    """|B|: nós [S,2] (2 componentes nas colunas) ou grade [2,H,W] (canais)."""
    if t.dim() == 2:
        return torch.hypot(t[:, 0], t[:, 1]).numpy()
    return torch.hypot(t[0], t[1]).numpy()


def gap_radius_mm(row):
    """r_m = stator_outer_d/2 + gap/2, gap = (rotor_inner_d - stator_outer_d)/2."""
    so = float(row['stator_outer_diameter [mm]'])
    ri = float(row['rotor_inner_diameter [mm]'])
    gap = (ri - so) / 2.0
    return so / 2.0 + gap / 2.0, gap


# --------------------------------------------------------------------------- #
# Projeções elemento <-> nó (origem: tests/resultados/elem_smooth_eval.py, 2026-10-20).
# Média componente a componente: com alvo polar (Br, Bθ) cada ponto tem o seu
# referencial, mas vizinhos diferem de ~1° (tamanho da malha / raio) e |B| não muda
# de forma apreciável -- mantida a lógica original.
# --------------------------------------------------------------------------- #
def elem_to_nodes(lay, B_elem):
    """[M,2] por elemento -> [S,2] por nó (média dos incidentes do material do nó).
    lay: saída de samples.parse_smooth(..., with_elem=True). B_elem: saída da Elem, na
    ordem dos elementos (contrato do layout Elem: docstring de src/data_gen/parsers/femm_mesh_elem.py)."""
    b1 = np.repeat(B_elem[:, :1], 3, axis=1)
    b2 = np.repeat(B_elem[:, 1:], 3, axis=1)
    return _node_b_by_material(lay['tri'], lay['elem_mat'], lay['node_mat'], b1, b2)


def nodes_to_elem(lay, B_node):
    """[S,2] por nó -> [M,2] por elemento (média dos 3 vértices = P1 no centróide)."""
    return B_node[lay['tri']].mean(axis=1)
