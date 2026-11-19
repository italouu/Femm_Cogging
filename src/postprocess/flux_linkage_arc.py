"""
Estimativa do torque de EXCITAÇÃO pelo fluxo concatenado: lei de Faraday num ARCO ÚNICO por bobina
(T0 = 0, o cogging não entra). Alternativa ao Arkkio (torque.band_torque) para as predições, em que
o tensor de Maxwell amplifica o erro do campo (avaliacao.md §3.1).

Método. Bobinas concentradas: a bobina do dente k tem o lado de ida (circuito coil_k_0) na ranhura k
e o de volta (coil_k_1) na ranhura k+1. A superfície da bobina é o arco no raio do topo da bobina,
r = r_bs0, da linha central da ranhura k à da ranhura k+1: cobre os dois lados e atravessa o dente.
Em 2D, ∫ B·n dl no arco = ∫ Br r dθ (= A(θ1) − A(θ0) para B exato), regra do ponto médio com K_SAMP
pontos; ψ_k = −N·L·∫ Br r dθ. Por fase, ψ = Σ_k s_k ψ_k (WINDING_SECTOR); Clarke amplitude-invariante
e T = 3 setores · (3/2)·p·(ψα iβ − ψβ iα), p = 21. Geometria da ranhura pela mesma construção de
motor_model._coil_coords (linha do valid_designs.csv); espiras N lidas do .ans.

Hipótese física: todas as espiras estão no topo da bobina; o fluxo de dispersão da ranhura abaixo do
arco fica de fora do ψ.

Validação (80 amostras × 16 θe do teste, Split mae run_0002):
  - ψ: com o B exato curl(A) do GT, o torque pelo arco fica a 0,74% (p90 1,6%) do calculado pelo ψ
    do A do FEMM (média de A na área de cada circuito), diferença que vem da hipótese acima.
  - torque contra o Arkkio do GT: erro relativo mediano 4,23% (p90 8,2%) com o B do Split e
    2,95% (p90 5,3%) com o B do GT. A proximidade com o Arkkio é uma observação nesse conjunto, não
    uma prova de exatidão: o ψ pelo arco não é o ψ exato, e parte da diferença Arkkio × fluxo
    concatenado pode ser compensada pela dispersão omitida.
"""
import numpy as np
from matplotlib.tri import Triangulation

from src.data_gen.motor_constants import N_POLES, SLOT_PITCH_DEG, WINDING_SECTOR

P_PAIRS = N_POLES // 2                       # 21 pares de polos da máquina inteira
K_SAMP = 120                                 # pontos (regra do ponto médio) por arco
EPS = 1e-6                                   # mm, deslocamento para dentro da bobina (o topo é aresta da malha)


# --------------------------------------------------------------------------- circuitos e espiras
def circuits(lines):
    """Nomes dos circuitos ([CircuitProps]) e, POR RÓTULO de bloco ([NumBlockLabels]), índice do
    circuito (−1 = nenhum) e espiras. lines = linhas do .ans. Para levar a cada elemento,
    indexe com o rótulo do elemento (elems[:, 3])."""
    L = [l.rstrip('\r\n') for l in lines]
    i = next(k for k, l in enumerate(L) if l.startswith('[CircuitProps]'))
    names = []
    for l in L[i + 1:]:
        if l.strip().startswith('<CircuitName>'):
            names.append(l.split('=')[1].strip().strip('"'))
        if l.startswith('['):
            break
    j = next(k for k, l in enumerate(L) if l.startswith('[NumBlockLabels]')); n = int(L[j].split('=')[1])
    lab = [l.split() for l in L[j + 1:j + 1 + n]]
    return names, np.array([int(f[4]) - 1 for f in lab]), np.array([float(f[7]) for f in lab])


def coil_turns(lines, elems, n_coils):
    """Espiras N [n_coils] de cada bobina. Erro se faltar um circuito em [CircuitProps], se um lado
    não tiver elementos ou se as espiras diferirem entre lados/blocos da bobina."""
    names, lab_c, lab_t = circuits(lines)
    ec, et = lab_c[elems[:, 3]], lab_t[elems[:, 3]]
    idx = {nm: ci for ci, nm in enumerate(names)}
    N = np.empty(n_coils)
    for k in range(n_coils):
        sides = [f'coil_{k}_0', f'coil_{k}_1']
        miss = [s for s in sides if s not in idx]
        if miss:
            raise ValueError(f'bobina {k}: circuito(s) {miss} ausente(s) em [CircuitProps]')
        m0, m1 = ec == idx[sides[0]], ec == idx[sides[1]]
        for s, m in zip(sides, (m0, m1)):
            if not m.any():
                raise ValueError(f'bobina {k}: o lado {s} não tem elementos')
        turns = np.unique(et[m0 | m1])
        if len(turns) != 1:
            raise ValueError(f'bobina {k}: espiras diferentes entre lados/blocos: {turns.tolist()}')
        N[k] = turns[0]
    return N


# --------------------------------------------------------------------------- B em pontos
class PointField:
    """B (Bx, By) constante por elemento, avaliado em pontos arbitrários da malha (xy em mm, E [M,3])."""

    def __init__(self, xy, E):
        self.xy, self.E = np.asarray(xy, np.float64), np.asarray(E)
        self._tf = Triangulation(self.xy[:, 0], self.xy[:, 1], self.E).get_trifinder()

    def locate(self, x, y):
        e = self._tf(x, y)
        if (e < 0).any():
            raise RuntimeError(f'{int((e < 0).sum())} ponto(s) fora da malha')
        return e

    def B(self, x, y, B):
        """B [M,2] por elemento -> (Bx, By) nos pontos (x, y)."""
        return np.asarray(B)[self.locate(x, y)]

    def polar(self, r, th, B):
        """(Br, Bθ) em (r [mm], θ [rad]), arrays de mesma forma."""
        x, y = r * np.cos(th), r * np.sin(th)
        Bxy = self.B(x.ravel(), y.ravel(), B)
        c, s = np.cos(th.ravel()), np.sin(th.ravel())
        return ((Bxy[:, 0] * c + Bxy[:, 1] * s).reshape(r.shape),
                (-Bxy[:, 0] * s + Bxy[:, 1] * c).reshape(r.shape))


# --------------------------------------------------------------------------- geometria, ψ e torque
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


def psi_arc(F, B, slot, N, depth_m):
    """ψ [Wb] por bobina [n_coils]; N = coil_turns(...)."""
    r = slot['r_bs0'] - EPS
    return np.array([-depth_m * N[k] * arc_flux(F, B, r, k) for k in range(len(N))])


def torque_from_psi(psi, i_ph):
    """ψ por bobina -> fases (WINDING_SECTOR) + Clarke: T = 3 setores · (3/2)·p·(ψα iβ − ψβ iα) [N·m];
    devolve (T, ψ por fase)."""
    ph = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    for k, (p_, sg) in enumerate(WINDING_SECTOR):
        ph[p_] += sg * psi[k]
    clarke = lambda a, b, c: ((2 / 3) * (a - b / 2 - c / 2), (b - c) / np.sqrt(3))
    pa, pb = clarke(ph['A'], ph['B'], ph['C']); ia, ib = clarke(i_ph['A'], i_ph['B'], i_ph['C'])
    return 3 * 1.5 * P_PAIRS * (pa * ib - pb * ia), ph


def flux_torque_arc(lines, xy, elems, B, depth_m, i_ph, slot, field=None):
    """Torque de excitação [N·m] (T0 = 0) e ψ por fase pelo arco único.
    lines = linhas do .ans; xy [N,2] mm; elems [M,4] = (n1, n2, n3, rótulo); B [M,2] = (Bx, By) em T
    por elemento; depth_m = profundidade [m]; i_ph = motor_constants.phase_currents(I, θe);
    slot = slot_geometry(row). `field` reaproveita um PointField já montado na mesma malha."""
    B = np.asarray(B)
    if B.shape != (len(elems), 2):
        raise ValueError(f'B com forma {B.shape}, esperado ({len(elems)}, 2) = (Bx, By) por elemento')
    F = field or PointField(xy, elems[:, :3])
    N = coil_turns(lines, elems, len(WINDING_SECTOR))
    return torque_from_psi(psi_arc(F, B, slot, N, depth_m), i_ph)
