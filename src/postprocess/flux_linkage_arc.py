"""
Fluxo concatenado e torque de EXCITAÇÃO do projeto (adotado em 2026-11-19): lei de Faraday num ARCO
ÚNICO por bobina (T0 = 0, o cogging não entra).

Bobina do dente k (lado de ida coil_k_0 na ranhura k, volta coil_k_1 na ranhura k+1): a superfície é
o arco no raio do topo da bobina, r = r_bs0, da linha central da ranhura k à da ranhura k+1; cobre os
dois lados da bobina e atravessa o dente. Em 2D, ∫ B·n dl no arco = ∫ Br r dθ = A(θ1) − A(θ0) para B
exato; regra do ponto médio com K_SAMP pontos. ψ_k = −N·L·∫ Br r dθ (mesmo sinal de
⟨A⟩_ida − ⟨A⟩_volta do c1). Fases, Clarke e torque: flux_linkage_direct.torque_from_psi.
Geometria da ranhura pela mesma construção de motor_model._coil_coords (linha do valid_designs.csv).

Física: o arco supõe todas as espiras no topo da bobina e deixa de fora o fluxo de dispersão da
ranhura abaixo dele. Com B exato, o ψ fica a 0,74% (p90 1,6%) do ψ do A do FEMM, contra 0,03% do c1
(flux_linkage_direct, média de área, o método exato). Subdividir em M arcos radiais (N_j = N·S_j/S)
converge para o c1 (M = 8: 0,22%), mas piora o torque contra o Maxwell: o arco único fica mais
perto do Maxwell porque a dispersão omitida compensa parte da diferença Maxwell × fluxo concatenado.

Validação (80 amostras × 16 θe, Split mae run_0002, T0 = 0; avaliacao.md §3.1): erro de torque
mediano (p90) contra o GT Maxwell 4,23% (8,2%), c1 4,66% (9,1%), superfície única 4,59% (8,7%);
com o B do GT 2,95% (5,3%), c1 3,65% (5,8%). Fluxo: um arco por bobina (< 2 ms por solução); o
PointField (~0,5 s) domina.
"""
import numpy as np

from src.data_gen.motor_constants import SLOT_PITCH_DEG, WINDING_SECTOR
from src.postprocess.flux_linkage_direct import K_SAMP, PointField, coil_geometry, torque_from_psi

EPS = 1e-6                                   # mm, deslocamento para dentro da bobina (o topo é aresta da malha)


def slot_geometry(row):
    """Ranhura de um design (linha do valid_designs.csv) [mm, rad]: r_in (canto, raio interno da
    sapata), op (abertura angular), r_bs0 (topo da bobina), x_b (fundo, coroa), bs0 (largura do topo)."""
    f = lambda k: float(row[k])
    r_in = f('stator_outer_diameter [mm]') / 2 - f('slot_Hs0 [mm]')
    bs0 = f('slot_Bs0 [mm]')
    op = np.arccos((2 * r_in ** 2 - bs0 ** 2) / (2 * r_in ** 2))
    r_bs0 = r_in * np.cos(op / 2)
    return dict(r_in=r_in, op=op, r_bs0=r_bs0, x_b=r_bs0 - f('slot_Hs1 [mm]') - f('slot_Hs2 [mm]'), bs0=bs0)


def arc_flux(F, B, r, k):
    """∫ Br r dθ [Wb/m] no arco de raio r [mm], da linha central da ranhura k à da k+1; B [M,2] por elemento."""
    a0, a1 = np.deg2rad(SLOT_PITCH_DEG * k), np.deg2rad(SLOT_PITCH_DEG * (k + 1))
    th = a0 + (a1 - a0) * (np.arange(K_SAMP) + .5) / K_SAMP
    Br, _ = F.polar(np.full_like(th, r), th, B)
    return Br.mean() * (a1 - a0) * r * 1e-3


def psi_arc(F, B, slot, geo, depth_m):
    """ψ [Wb] por bobina no formato de torque_from_psi: coil_k_0 = ψ da bobina, coil_k_1 = 0."""
    r = slot['r_bs0'] - EPS
    psi = {}
    for k, g in enumerate(geo):
        psi[f'coil_{k}_0'] = -depth_m * g['N'] * arc_flux(F, B, r, k)
        psi[f'coil_{k}_1'] = 0.0
    return psi


def flux_torque_arc(lines, xy, elems, B, depth_m, i_ph, slot, field=None):
    """Torque de excitação [N·m] (T0 = 0) e ψ por fase pelo arco único.
    lines = linhas do .ans; xy [N,2] mm; elems [M,4] = (n1, n2, n3, rótulo); B [M,2] = (Bx, By) em T
    por elemento; depth_m = profundidade [m]; i_ph = phase_currents(I, θe); slot = slot_geometry(row).
    `field` reaproveita um PointField já montado na mesma malha."""
    B = np.asarray(B)
    if B.shape != (len(elems), 2):
        raise ValueError(f'B com forma {B.shape}, esperado ({len(elems)}, 2) = (Bx, By) por elemento')
    F = field or PointField(xy, elems[:, :3])
    geo = coil_geometry(lines, xy, elems, len(WINDING_SECTOR))
    return torque_from_psi(psi_arc(F, B, slot, geo, depth_m), i_ph)
