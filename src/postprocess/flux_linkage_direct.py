"""
Fluxo concatenado e torque de excitação do projeto: ψ DIRETO do B (integral de linha), sem
reconstruir A. Padrão: caminho 'c1'.

Em 2D, ∇A = (−By, Bx): a diferença de A entre dois pontos é o fluxo de B que atravessa a linha
que os liga, e só a componente de B normal ao caminho entra:
    arco   (r fixo):  A(r, θ2) − A(r, θ1) = ∫ Br · r dθ
    radial (θ fixo):  A(r2, θ) − A(r1, θ) = −∫ Bθ dr
Bobinas concentradas no dente k (eixo θ_k = 5° + 10°·k; lados coil_k_0 = ida, coil_k_1 = volta
nas ranhuras vizinhas). Por bobina, A em cada nó dos dois lados vem da integral de B a partir da
referência (r0, θ_k) no eixo do dente (r0 = meio radial da bobina); ψ só depende de
⟨A⟩_ida − ⟨A⟩_volta, então a referência cancela. Depois: ψ_c = L·N·⟨A⟩_c (média de área, como o
FEMM), fases (WINDING_SECTOR), Clarke e T = 3·(3/2)·p·(ψα iβ − ψβ iα).

Caminhos (com B exato = rot A são equivalentes; com B previsto, a diferença mede a parte não
conservativa — ~1% no torque do Split):
    'c1': radial no eixo do dente (r0 -> r_nó), depois arco em r_nó (θ_k -> θ_nó)
    'c2': arco em r0 (θ_k -> θ_nó), depois radial em θ_nó (r0 -> r_nó)
Campo B nos pontos:
    Bel     [M,2]   (Bx, By) constante por elemento (P0)
    Bcorner [M,3,2] (Bx, By) nos 3 cantos de cada elemento, interpolado linearmente (P1)

Validação (80 amostras × 16 θe, Split mae run_0002, T0 = 0): com o B exato do GT reproduz o ψ
do A do FEMM a 0,03% (c1) / 0,05% (c2); erro de torque mediano do Split 4,66% (c1) / 4,78% (c2).
Registro: a reconstrução de A por mínimos quadrados P1 (removida em 2026-11-18) dava 4,56%,
a 0,5% (c1) do método direto — ver docs/claude/avaliacao.md §3.1.
"""
import numpy as np
from matplotlib.tri import Triangulation

from src.postprocess.torque import grad_ops

P_PAIRS = 21                       # pares de polos da máquina inteira
K_SAMP = 120                       # pontos (regra do ponto médio) por trecho de integral de linha
TOOTH_0_DEG, TOOTH_PITCH_DEG = 5.0, 10.0
SECTOR_DEG = 120.0


# --------------------------------------------------------------------------- circuitos e correntes
def circuits(lines):
    """Nomes dos circuitos ([CircuitProps]) e, por elemento, índice do circuito (−1 = nenhum)
    e espiras. lines = linhas do .ans; elem_label = rótulo de cada elemento."""
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


def phase_currents(I, th_deg):
    t = np.deg2rad(th_deg)       # = BLDC_Process.phase_currents
    return {'A': I * np.cos(t), 'B': I * np.cos(t - 2 * np.pi / 3), 'C': I * np.cos(t + 2 * np.pi / 3)}


# --------------------------------------------------------------------------- B em pontos
class PointField:
    """B (Bx, By) em pontos arbitrários da malha (xy em mm, E [M,3])."""

    def __init__(self, xy, E):
        self.xy, self.E = np.asarray(xy, np.float64), np.asarray(E)
        self._tf = Triangulation(self.xy[:, 0], self.xy[:, 1], self.E).get_trifinder()

    def locate(self, x, y):
        e = self._tf(x, y)
        if (e < 0).any():
            raise RuntimeError(f'{int((e < 0).sum())} ponto(s) fora da malha')
        return e

    def B(self, x, y, Bel=None, Bcorner=None):
        e = self.locate(x, y)
        if Bcorner is None:
            return np.asarray(Bel)[e]
        P = self.xy[self.E[e]]                                           # [n,3,2]
        x0, y0 = P[:, 0, 0], P[:, 0, 1]
        d = (P[:, 1, 0] - x0) * (P[:, 2, 1] - y0) - (P[:, 2, 0] - x0) * (P[:, 1, 1] - y0)
        l1 = ((x - x0) * (P[:, 2, 1] - y0) - (P[:, 2, 0] - x0) * (y - y0)) / d
        l2 = ((P[:, 1, 0] - x0) * (y - y0) - (x - x0) * (P[:, 1, 1] - y0)) / d
        w = np.stack([1 - l1 - l2, l1, l2], 1)
        return (w[:, :, None] * np.asarray(Bcorner)[e]).sum(1)

    def polar(self, r, th, **B):
        """(Br, Bθ) em (r [mm], θ [rad]), arrays de mesma forma."""
        x, y = r * np.cos(th), r * np.sin(th)
        Bxy = self.B(x.ravel(), y.ravel(), **B)
        c, s = np.cos(th.ravel()), np.sin(th.ravel())
        return ((Bxy[:, 0] * c + Bxy[:, 1] * s).reshape(r.shape),
                (-Bxy[:, 0] * s + Bxy[:, 1] * c).reshape(r.shape))


# --------------------------------------------------------------------------- integrais de linha
def _t():
    return (np.arange(K_SAMP) + .5) / K_SAMP


def line_arc(F, r, th0, th1, **B):
    """∫_{th0}^{th1} Br r dθ por nó [Wb/m]; r, th1 arrays [n]; th0 escalar ou [n]."""
    th0 = np.broadcast_to(th0, np.shape(th1))
    th = th0[:, None] + (th1 - th0)[:, None] * _t()[None]
    rr = np.repeat(np.asarray(r, np.float64)[:, None], K_SAMP, 1)
    Br, _ = F.polar(rr, th, **B)
    return Br.mean(1) * (th1 - th0) * rr[:, 0] * 1e-3


def line_rad(F, th, r0, r1, **B):
    """−∫_{r0}^{r1} Bθ dr por nó [Wb/m]; th escalar ou [n]; r0 escalar; r1 [n]."""
    rr = r0 + (r1 - r0)[:, None] * _t()[None]
    tt = np.repeat(np.broadcast_to(th, np.shape(r1)).astype(np.float64)[:, None], K_SAMP, 1)
    _, Bt = F.polar(rr, tt, **B)
    return -Bt.mean(1) * (r1 - r0) * 1e-3


# --------------------------------------------------------------------------- bobinas e ψ
def coil_geometry(lines, xy, elems, n_coils):
    """Por bobina k: nós dos dois lados, máscaras dos elementos de ida/volta, espiras, θ_k, r0."""
    names, lab_c, lab_t = circuits(lines)
    xy = np.asarray(xy, np.float64); E = elems[:, :3]
    ec, et = lab_c[elems[:, 3]], lab_t[elems[:, 3]]
    idx = {nm: ci for ci, nm in enumerate(names)}
    lo, hi = np.deg2rad([1e-7, SECTOR_DEG - 1e-7])                     # nós no corte: dentro da malha
    out = []
    for k in range(n_coils):
        m0, m1 = ec == idx[f'coil_{k}_0'], ec == idx[f'coil_{k}_1']
        nodes = np.unique(E[m0 | m1]); rn = np.hypot(*xy[nodes].T)
        thn = np.clip(np.arctan2(xy[nodes, 1], xy[nodes, 0]), lo, hi)
        out.append(dict(nodes=nodes, rn=rn, thn=thn, m0=m0, m1=m1, N=float(et[m0][0]),
                        thk=np.deg2rad(TOOTH_0_DEG + TOOTH_PITCH_DEG * k), r0=0.5 * (rn.min() + rn.max())))
    return out


def coil_A(F, g, path='c1', **B):
    """A [Wb/m] nos nós g['nodes'] da bobina, com A(r0, θ_k) = 0."""
    rn, thn, thk, r0 = g['rn'], g['thn'], g['thk'], g['r0']
    if path == 'c1':
        return line_rad(F, thk, r0, rn, **B) + line_arc(F, rn, thk, thn, **B)
    if path == 'c2':
        return line_arc(F, np.full_like(rn, r0), thk, thn, **B) + line_rad(F, thn, r0, rn, **B)
    raise ValueError(f"path deve ser 'c1' ou 'c2', não {path!r}")


def psi_direct(F, geo, elems, xy, depth_m, path='c1', **B):
    """ψ [Wb] por lado de bobina ('coil_k_0', 'coil_k_1'), ψ_c = L·N·⟨A⟩_c (média de área)."""
    E = elems[:, :3]; _, _, S = grad_ops(np.asarray(xy, np.float64), E)
    psi = {}
    for k, g in enumerate(geo):
        An = np.full(len(xy), np.nan); An[g['nodes']] = coil_A(F, g, path, **B)
        for side, m in (('0', g['m0']), ('1', g['m1'])):
            psi[f'coil_{k}_{side}'] = depth_m * g['N'] * (An[E[m]].mean(1) * S[m]).sum() / S[m].sum()
    return psi


def torque_from_psi(psi, i_ph):
    """Fases (WINDING_SECTOR) + Clarke: T = 3 setores · (3/2)·p·(ψα iβ − ψβ iα) [N·m]; devolve (T, ψ por fase)."""
    from src.data_gen.motor_constants import WINDING_SECTOR
    ph = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    for k, (p_, sg) in enumerate(WINDING_SECTOR):
        ph[p_] += sg * (psi[f'coil_{k}_0'] - psi[f'coil_{k}_1'])
    clarke = lambda a, b, c: ((2 / 3) * (a - b / 2 - c / 2), (b - c) / np.sqrt(3))
    pa, pb = clarke(ph['A'], ph['B'], ph['C']); ia, ib = clarke(i_ph['A'], i_ph['B'], i_ph['C'])
    return 3 * 1.5 * P_PAIRS * (pa * ib - pb * ia), ph


def flux_torque_direct(lines, xy, elems, depth_m, i_ph, path='c1', Bel=None, Bcorner=None, field=None):
    """Torque de excitação [N·m] e ψ por fase a partir de B direto (Bel P0 ou Bcorner P1).
    lines = linhas do .ans, elems [M,4] = (n1, n2, n3, rótulo), depth_m = profundidade [m],
    i_ph = correntes de fase (phase_currents). `field` reaproveita um PointField já montado
    na mesma malha."""
    from src.data_gen.motor_constants import WINDING_SECTOR
    if (Bel is None) == (Bcorner is None):
        raise ValueError('passe exatamente um de Bel (P0) ou Bcorner (P1)')
    F = field or PointField(xy, elems[:, :3])
    geo = coil_geometry(lines, xy, elems, len(WINDING_SECTOR))
    B = dict(Bel=Bel) if Bel is not None else dict(Bcorner=Bcorner)
    return torque_from_psi(psi_direct(F, geo, elems, xy, depth_m, path, **B), i_ph)
