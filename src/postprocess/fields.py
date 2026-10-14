"""
Operações sobre o campo: |B| e raio do entreferro (posição das figuras).

Origem: tests/resultados/common.py (2026-10-13), sem mudança de lógica
(_mag -> mag, agora pública).
Não migrados (sem uso desde a remoção do cogging): arc_theta_deg, _barycentric,
arc_elements, arc_weights, arc_values_from_nodes, to_polar.
Os helpers de elemento de src/neural_op/archs/eval.py só entram aqui se algum
eval_fn em uso precisar deles.
"""
import torch


def mag(t):
    """|B|: nós [S,2] (Bx,By nas colunas) ou grade [2,H,W] (canais)."""
    if t.dim() == 2:
        return torch.hypot(t[:, 0], t[:, 1]).numpy()
    return torch.hypot(t[0], t[1]).numpy()


def gap_radius_mm(row):
    """r_m = stator_outer_d/2 + gap/2, gap = (rotor_inner_d - stator_outer_d)/2."""
    so = float(row['stator_outer_diameter [mm]'])
    ri = float(row['rotor_inner_diameter [mm]'])
    gap = (ri - so) / 2.0
    return so / 2.0 + gap / 2.0, gap
