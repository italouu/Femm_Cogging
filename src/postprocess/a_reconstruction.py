"""
A nodal (P1) a partir de B por elemento, por mínimos quadrados, e torque pelo fluxo
concatenado. Porte de tests/check_split_torque_fluxlinkage.py (grad_ops, reconstruct_A,
circuits, phase_currents, flux_torque) + a variante COM restrição de contorno.

  reconstruct_A(xy, E, Bxy, constraints=None)
      min_A Σ_e S_e |∇A_e − (−By_e, Bx_e)|²     (B = rot(A ẑ) => ∇A = (−By, Bx))
      constraints=None : (a) sem restrição, gauge A[0] = 0 — idêntico ao de tests/
      constraints=dict : (b) A = P·u com A = 0 nas classes Dirichlet e nós periódicos
                         amarrados (ans_rewrite.ans_constraints) — o mesmo espaço do FEMM.
  shift_to_dirichlet(A, cons) : (a) deslocada pela constante que zera a média de A nos nós
                         Dirichlet (B e torque inalterados; só para gravar no .ans).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

P_PAIRS = 21                     # pares de polos da máquina inteira


def grad_ops(xy, E):
    """Gradiente P1 por elemento: ∂A/∂x = Σ gx·A, ∂A/∂y = Σ gy·A; área S."""
    X, Y = xy[E, 0], xy[E, 1]
    b = np.stack([Y[:, 1] - Y[:, 2], Y[:, 2] - Y[:, 0], Y[:, 0] - Y[:, 1]], 1)
    c = np.stack([X[:, 2] - X[:, 1], X[:, 0] - X[:, 2], X[:, 1] - X[:, 0]], 1)
    a2 = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    return b / a2[:, None], c / a2[:, None], np.abs(a2) / 2


def _G_t(xy, E, Bxy):
    gx, gy, S = grad_ops(xy, E); M, N = len(E), len(xy); w = np.sqrt(S)
    r = np.repeat(np.arange(M), 3)
    Gx = sp.csr_matrix(((gx * w[:, None]).ravel(), (r, E.ravel())), shape=(M, N))
    Gy = sp.csr_matrix(((gy * w[:, None]).ravel(), (r, E.ravel())), shape=(M, N))
    t = np.r_[-Bxy[:, 1] * 1e-3 * w, Bxy[:, 0] * 1e-3 * w]          # mm -> A em Wb/m
    return sp.vstack([Gx, Gy]).tocsc(), t


def constraint_map(cons, N):
    """P [N, K]: A = P·u. Classes com Dirichlet -> linha nula; demais -> sign·u_classe."""
    root, sign, zero = cons['root'], cons['sign'], cons['zero']
    free_roots = np.unique(root[~zero])
    col = -np.ones(N, int); col[free_roots] = np.arange(len(free_roots))
    keep = ~zero
    rows = np.flatnonzero(keep)
    return sp.csr_matrix((sign[keep], (rows, col[root[keep]])), shape=(N, len(free_roots)))


def reconstruct_A(xy, E, Bxy, constraints=None):
    """A nodal [Wb/m] cujo gradiente melhor reproduz (−By, Bx) por elemento (xy mm, B T)."""
    G, t = _G_t(np.asarray(xy, np.float64), np.asarray(E), np.asarray(Bxy, np.float64))
    if constraints is None:                                         # (a): A[0] = 0
        G1 = G[:, 1:]
        return np.r_[0.0, spsolve((G1.T @ G1).tocsc(), G1.T @ t)]
    P = constraint_map(constraints, len(xy)).tocsc()                 # (b)
    GP = (G @ P).tocsc()
    return P @ spsolve((GP.T @ GP).tocsc(), GP.T @ t)


def shift_to_dirichlet(A, cons):
    return A - A[cons['dirichlet']].mean()


def curl_P0(xy, E, A):
    """B = rot(A ẑ) por elemento (P1 -> P0), (Bx, By) em T (xy em mm)."""
    gx, gy, _ = grad_ops(np.asarray(xy, np.float64), np.asarray(E))
    Ae = np.asarray(A)[E]
    return np.stack([(gy * Ae).sum(1), -(gx * Ae).sum(1)], 1) * 1e3


# --------------------------------------------------------------------------- fluxo concatenado
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


def flux_torque(lines, xy, elems, A, depth_m, i_ph):
    """Torque de excitação do motor inteiro [N·m] a partir de A nodal; devolve também ψ por fase.
    elems [M,4] = (n1, n2, n3, rótulo). Mesma conta de tests/check_split_torque_fluxlinkage."""
    from src.data_gen.motor_constants import WINDING_SECTOR
    names, lab_c, lab_t = circuits(lines)
    E = elems[:, :3]; ec, et = lab_c[elems[:, 3]], lab_t[elems[:, 3]]
    _, _, S = grad_ops(xy, E); Ae = np.asarray(A)[E].mean(1)
    psi = {nm: depth_m * et[ec == ci][0] * (Ae[ec == ci] * S[ec == ci]).sum() / S[ec == ci].sum()
           for ci, nm in enumerate(names) if (ec == ci).any()}
    ph = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    for k, (p_, sg) in enumerate(WINDING_SECTOR):
        ph[p_] += sg * (psi[f'coil_{k}_0'] - psi[f'coil_{k}_1'])
    clarke = lambda a, b, c: ((2 / 3) * (a - b / 2 - c / 2), (b - c) / np.sqrt(3))
    pa, pb = clarke(ph['A'], ph['B'], ph['C']); ia, ib = clarke(i_ph['A'], i_ph['B'], i_ph['C'])
    return 3 * 1.5 * P_PAIRS * (pa * ib - pb * ia), ph
