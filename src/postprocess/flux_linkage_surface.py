"""
Fluxo concatenado e torque de EXCITAÇÃO pela lei de Faraday numa SUPERFÍCIE ÚNICA por bobina
(alternativa ao c1 de flux_linkage_direct; T0 = 0, o cogging não entra).

ψ da bobina = N·Φ, Φ = L·∫ B·n dl ao longo de uma linha aberta (em 2D, a superfície é a linha ×
profundidade L). Bobina do dente k: lado de ida coil_k_0 na ranhura k, volta coil_k_1 na ranhura k+1.
    L do lado de ida   : do fundo da ranhura (coroa) pela linha central da ranhura k (contorno externo
                         do lado) até o topo da bobina, depois pelo topo até o canto que encosta no dente;
    arco               : no raio do canto (r_in = R_so − Hs0), atravessando o dente até o outro canto;
    L do lado de volta : topo e linha central da ranhura k+1, descendo até a coroa.
Geometria da ranhura pela mesma construção de motor_model._coil_coords (lida da linha do
valid_designs.csv). ∫ B·n dl = ∫ (Bx dy − By dx) (= ΔA entre as pontas para B exato), regra do ponto
médio com K_SAMP pontos por trecho; os trechos sobre arestas da malha (linha central, topo) são
avaliados EPS para dentro do lado da bobina. Sinal: ψ = N·L·(A(início) − A(fim)), o mesmo de
⟨A⟩_ida − ⟨A⟩_volta no c1. Fases, Clarke e torque: flux_linkage_direct.torque_from_psi.

Diferença física para o c1: a superfície única supõe todas as espiras no contorno externo de cada
lado; o c1 (e o FEMM) usam a média de A na área. O fluxo de dispersão da ranhura que isso acrescenta
é ∝ i da própria fase e não gera torque: o ψ por fase erra 8–13% (#3748, B do GT), o torque não.

Validação (80 amostras × 16 θe, Split mae run_0002, T0 = 0; avaliacao.md §3.1): erro de torque
mediano (p90) contra o GT Maxwell 4,59% (8,7%), c1 4,66% (9,1%); com o B do GT, torque a 0,34% do
ψ do A do FEMM (c1: 0,03%). Fluxo ~7,5 ms por solução contra ~320 ms do c1 (sem o PointField, ~0,5 s,
comum aos dois).
"""
import numpy as np

from src.data_gen.motor_constants import SLOT_PITCH_DEG, WINDING_SECTOR
from src.postprocess.flux_linkage_direct import K_SAMP, PointField, coil_geometry, torque_from_psi

EPS = 1e-6                                   # mm, deslocamento para dentro do lado da bobina


def slot_geometry(row):
    """Ranhura de um design (linha do valid_designs.csv) [mm, rad]: r_in (canto, raio interno da
    sapata), op (abertura angular), r_bs0 (topo da bobina), x_b (fundo, coroa), bs0 (largura do topo)."""
    f = lambda k: float(row[k])
    r_in = f('stator_outer_diameter [mm]') / 2 - f('slot_Hs0 [mm]')
    bs0 = f('slot_Bs0 [mm]')
    op = np.arccos((2 * r_in ** 2 - bs0 ** 2) / (2 * r_in ** 2))
    r_bs0 = r_in * np.cos(op / 2)
    return dict(r_in=r_in, op=op, r_bs0=r_bs0, x_b=r_bs0 - f('slot_Hs1 [mm]') - f('slot_Hs2 [mm]'), bs0=bs0)


def _rot(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])


def coil_surface(G, k):
    """Trechos da superfície da bobina k, em ordem (início na coroa da ranhura k):
    ('reta', p, q, p_aval, q_aval) com as pontas geométricas e as deslocadas EPS; ('arco', r, θ0, θ1)."""
    a0, a1 = np.deg2rad(SLOT_PITCH_DEG * k), np.deg2rad(SLOT_PITCH_DEG * (k + 1))
    xb, rb, h = G['x_b'], G['r_bs0'], G['bs0'] / 2
    return [('reta', _rot((xb, 0), a0), _rot((rb, 0), a0), _rot((xb, EPS), a0), _rot((rb - EPS, EPS), a0)),
            ('reta', _rot((rb, 0), a0), _rot((rb, h), a0), _rot((rb - EPS, EPS), a0), _rot((rb - EPS, h), a0)),
            ('arco', G['r_in'], a0 + G['op'] / 2, a1 - G['op'] / 2),
            ('reta', _rot((rb, -h), a1), _rot((rb, 0), a1), _rot((rb - EPS, -h), a1), _rot((rb - EPS, -EPS), a1)),
            ('reta', _rot((rb, 0), a1), _rot((xb, 0), a1), _rot((rb - EPS, -EPS), a1), _rot((xb, -EPS), a1))]


def surface_flux(F, B, seg):
    """∫ B·n dl ao longo dos trechos [Wb/m] (= A(fim) − A(início) para B exato); B [M,2] por elemento."""
    t = (np.arange(K_SAMP) + .5) / K_SAMP
    tot = 0.0
    for s in seg:
        if s[0] == 'arco':
            _, r, th0, th1 = s
            th = th0 + (th1 - th0) * t
            Br, _ = F.polar(np.full_like(th, r), th, B)
            tot += Br.mean() * (th1 - th0) * r * 1e-3
        else:
            _, p, q, pe, qe = s
            pts = pe[None] + (qe - pe)[None] * t[:, None]
            Bxy = F.B(pts[:, 0], pts[:, 1], B)
            d = q - p                                                 # comprimento geométrico do trecho
            tot += (Bxy[:, 0] * d[1] - Bxy[:, 1] * d[0]).mean() * 1e-3
    return tot


def psi_surface(F, B, G, geo, depth_m):
    """ψ [Wb] por bobina no formato de torque_from_psi: coil_k_0 = ψ da bobina, coil_k_1 = 0."""
    psi = {}
    for k, g in enumerate(geo):
        psi[f'coil_{k}_0'] = -depth_m * g['N'] * surface_flux(F, B, coil_surface(G, k))
        psi[f'coil_{k}_1'] = 0.0
    return psi


def flux_torque_surface(lines, xy, elems, B, depth_m, i_ph, slot, field=None):
    """Torque de excitação [N·m] (T0 = 0) e ψ por fase pela superfície única.
    lines = linhas do .ans; xy [N,2] mm; elems [M,4] = (n1, n2, n3, rótulo); B [M,2] = (Bx, By) em T
    por elemento; depth_m = profundidade [m]; i_ph = phase_currents(I, θe); slot = slot_geometry(row).
    `field` reaproveita um PointField já montado na mesma malha."""
    B = np.asarray(B)
    if B.shape != (len(elems), 2):
        raise ValueError(f'B com forma {B.shape}, esperado ({len(elems)}, 2) = (Bx, By) por elemento')
    F = field or PointField(xy, elems[:, :3])
    geo = coil_geometry(lines, xy, elems, len(WINDING_SECTOR))
    return torque_from_psi(psi_surface(F, B, slot, geo, depth_m), i_ph)
