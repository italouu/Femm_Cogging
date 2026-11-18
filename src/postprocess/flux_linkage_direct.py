"""
Fluxo concatenado e torque de EXCITAÇÃO do projeto: ψ direto do B por elemento (integral de
linha), sem reconstruir A; T0 = 0 (o cogging não entra).

Em 2D, ∇A = (−By, Bx): a diferença de A entre dois pontos é o fluxo de B que atravessa a linha
que os liga, e só a componente de B normal ao caminho entra:
    arco   (r fixo):  A(r, θ2) − A(r, θ1) = ∫ Br · r dθ
    radial (θ fixo):  A(r2, θ) − A(r1, θ) = −∫ Bθ dr
Bobinas concentradas no dente k (eixo θ_k = (k + ½)·passo de ranhura; lados coil_k_0 = ida e
coil_k_1 = volta nas ranhuras vizinhas). Por bobina, A em cada nó dos dois lados vem da integral
de B a partir da referência (r0, θ_k) no eixo do dente (r0 = meio radial da bobina), pelo caminho:
radial no eixo do dente (r0 -> r_nó), depois arco em r_nó (θ_k -> θ_nó). ψ só depende de
⟨A⟩_ida − ⟨A⟩_volta, então a referência cancela. Depois: ψ_c = L·N·⟨A⟩_c (média de área, como o
FEMM), fases (WINDING_SECTOR), Clarke e T = 3·(3/2)·p·(ψα iβ − ψβ iα).
B entra constante por elemento, (Bx, By) [M,2] em T; xy em mm.

Validação (80 amostras × 16 θe, Split mae run_0002, T0 = 0): com o B exato do GT reproduz o ψ
do A do FEMM a 0,03%; erro de torque mediano do Split 4,66% (p90 9,1%). Registro da comparação
(avaliacao.md §3.1): caminho alternativo (arco em r0, depois radial) 4,78%; B interpolado P1 nas
cópias da N_base 4,66%; A reconstruído por mínimos quadrados P1 (removido em 2026-11-18) 4,56%.
"""
import numpy as np
from matplotlib.tri import Triangulation

from src.data_gen.motor_constants import N_POLES, SECTOR_DEG, SLOT_PITCH_DEG, WINDING_SECTOR
from src.data_gen.motor_constants import phase_currents  # noqa: F401  (reexportado: i_ph de flux_torque_direct)
from src.postprocess.torque import grad_ops

P_PAIRS = N_POLES // 2                       # 21 pares de polos da máquina inteira
K_SAMP = 120                                 # pontos (regra do ponto médio) por trecho de integral de linha
TOOTH_PITCH_DEG = SLOT_PITCH_DEG             # 10°
TOOTH_0_DEG = SLOT_PITCH_DEG / 2             # 5°: eixo do dente 0, entre as ranhuras 0 (corte 0°) e 1


# --------------------------------------------------------------------------- circuitos
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


# --------------------------------------------------------------------------- integrais de linha
def _t():
    return (np.arange(K_SAMP) + .5) / K_SAMP


def line_arc(F, r, th0, th1, B):
    """∫_{th0}^{th1} Br r dθ por nó [Wb/m]; r, th1 arrays [n]; th0 escalar ou [n]."""
    th0 = np.broadcast_to(th0, np.shape(th1))
    th = th0[:, None] + (th1 - th0)[:, None] * _t()[None]
    rr = np.repeat(np.asarray(r, np.float64)[:, None], K_SAMP, 1)
    Br, _ = F.polar(rr, th, B)
    return Br.mean(1) * (th1 - th0) * rr[:, 0] * 1e-3


def line_rad(F, th, r0, r1, B):
    """−∫_{r0}^{r1} Bθ dr por nó [Wb/m]; th escalar ou [n]; r0 escalar; r1 [n]."""
    rr = r0 + (r1 - r0)[:, None] * _t()[None]
    tt = np.repeat(np.broadcast_to(th, np.shape(r1)).astype(np.float64)[:, None], K_SAMP, 1)
    _, Bt = F.polar(rr, tt, B)
    return -Bt.mean(1) * (r1 - r0) * 1e-3


# --------------------------------------------------------------------------- bobinas e ψ
def coil_geometry(lines, xy, elems, n_coils):
    """Por bobina k: nós dos dois lados, máscaras dos elementos de ida/volta, espiras, θ_k, r0.
    Erro se um lado não tiver elementos ou se as espiras diferirem entre lados/blocos da bobina."""
    names, lab_c, lab_t = circuits(lines)
    xy = np.asarray(xy, np.float64); E = elems[:, :3]
    ec, et = lab_c[elems[:, 3]], lab_t[elems[:, 3]]
    idx = {nm: ci for ci, nm in enumerate(names)}
    lo, hi = np.deg2rad([1e-7, SECTOR_DEG - 1e-7])                     # nós no corte: dentro da malha
    out = []
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
        nodes = np.unique(E[m0 | m1]); rn = np.hypot(*xy[nodes].T)
        thn = np.clip(np.arctan2(xy[nodes, 1], xy[nodes, 0]), lo, hi)
        out.append(dict(nodes=nodes, rn=rn, thn=thn, m0=m0, m1=m1, N=float(turns[0]),
                        thk=np.deg2rad(TOOTH_0_DEG + TOOTH_PITCH_DEG * k), r0=0.5 * (rn.min() + rn.max())))
    return out


def coil_A(F, g, B):
    """A [Wb/m] nos nós g['nodes'] da bobina, com A(r0, θ_k) = 0: radial no eixo do dente
    (r0 -> r_nó), depois arco em r_nó (θ_k -> θ_nó)."""
    rn, thn, thk, r0 = g['rn'], g['thn'], g['thk'], g['r0']
    return line_rad(F, thk, r0, rn, B) + line_arc(F, rn, thk, thn, B)


def psi_direct(F, geo, elems, xy, depth_m, B):
    """ψ [Wb] por lado de bobina ('coil_k_0', 'coil_k_1'), ψ_c = L·N·⟨A⟩_c (média de área)."""
    E = elems[:, :3]; _, _, S = grad_ops(np.asarray(xy, np.float64), E)
    psi = {}
    for k, g in enumerate(geo):
        An = np.full(len(xy), np.nan); An[g['nodes']] = coil_A(F, g, B)
        for side, m in (('0', g['m0']), ('1', g['m1'])):
            psi[f'coil_{k}_{side}'] = depth_m * g['N'] * (An[E[m]].mean(1) * S[m]).sum() / S[m].sum()
    return psi


def torque_from_psi(psi, i_ph):
    """Fases (WINDING_SECTOR) + Clarke: T = 3 setores · (3/2)·p·(ψα iβ − ψβ iα) [N·m]; devolve (T, ψ por fase)."""
    ph = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    for k, (p_, sg) in enumerate(WINDING_SECTOR):
        ph[p_] += sg * (psi[f'coil_{k}_0'] - psi[f'coil_{k}_1'])
    clarke = lambda a, b, c: ((2 / 3) * (a - b / 2 - c / 2), (b - c) / np.sqrt(3))
    pa, pb = clarke(ph['A'], ph['B'], ph['C']); ia, ib = clarke(i_ph['A'], i_ph['B'], i_ph['C'])
    return 3 * 1.5 * P_PAIRS * (pa * ib - pb * ia), ph


def flux_torque_direct(lines, xy, elems, B, depth_m, i_ph, field=None):
    """Torque de excitação [N·m] (T0 = 0) e ψ por fase a partir de B por elemento.
    lines = linhas do .ans; xy [N,2] mm; elems [M,4] = (n1, n2, n3, rótulo); B [M,2] = (Bx, By)
    em T por elemento; depth_m = profundidade [m]; i_ph = phase_currents(I, θe).
    `field` reaproveita um PointField já montado na mesma malha."""
    B = np.asarray(B)
    if B.shape != (len(elems), 2):
        raise ValueError(f'B com forma {B.shape}, esperado ({len(elems)}, 2) = (Bx, By) por elemento')
    F = field or PointField(xy, elems[:, :3])
    geo = coil_geometry(lines, xy, elems, len(WINDING_SECTOR))
    return torque_from_psi(psi_direct(F, geo, elems, xy, depth_m, B), i_ph)
