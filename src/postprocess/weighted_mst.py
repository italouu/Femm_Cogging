"""
VERIFICAÇÃO, fora dos resultados: o T22 do FEMM (mo_blockintegral(22), tensor de Maxwell
PONDERADO) reproduzido em Python, com B por elemento dado diretamente. A referência de torque
do projeto é o Arkkio (torque.band_torque).

Wrap obrigatório: os cortes θ = 0° e 120° do setor são periódicos; sem amarrar g nos pares
periódicos, os termos de contorno de 𝕋·∇g nos dois cortes não se cancelam (7–11% de erro na #1177).

  weight_function(xy, E, lab_elem, sel_label, air_label, cons)
      g nodal: Laplace (P1) nos elementos de AR não selecionados, g = 1 nos nós dos elementos
      selecionados (rotor), g = 0 nos nós dos demais elementos que não são ar (estator);
      Neumann natural no resto. cons (ans_rewrite.ans_constraints, obrigatório): nós periódicos
      amarrados (g_i = g_raiz).
  weighted_torque(B, xy, E, g, depth_m, n_sectors=3)
      T = −n·L·Σ_e (r_e × (𝕋_e·∇g_e))·S_e,  𝕋 = (B Bᵀ − ½|B|² I)/μ0.
      Sinal: no ar ∇·𝕋 = 0, g = 1 no corpo e 0 no estator => F_corpo = −∫ 𝕋·∇g dV.
      B e ∇g constantes por elemento e r linear => r no centróide é exato.
  rotor_labels(lines, r_m) : rótulos com r > r_m (a mesma regra estrita de femm_solve).
  t22_python(ans, B, depth_m, r_m, air='vacuum') : atalho do .ans lido (restrições do próprio .ans).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from src.postprocess.torque import grad_ops

MU0 = 4e-7 * np.pi


def rotor_labels(lines, r_m):
    L = [l.rstrip('\r\n') for l in lines]
    j = next(i for i, l in enumerate(L) if l.startswith('[NumBlockLabels]'))
    n = int(L[j].split('=')[1])
    xy = np.array([[float(v) for v in l.split()[:2]] for l in L[j + 1:j + 1 + n]])
    return np.hypot(xy[:, 0], xy[:, 1]) > r_m


def weight_function(xy, E, lab_elem, sel_label, air_label, cons):
    """g nodal [N] (ver o topo do módulo); cons = ans_rewrite.ans_constraints (wrap obrigatório)."""
    xy = np.asarray(xy, np.float64); N = len(xy)
    sel_e = sel_label[lab_elem]; air_e = air_label[lab_elem] & ~sel_e
    fixed = np.zeros(N, bool); g = np.zeros(N)
    fixed[E[~air_e].ravel()] = True                     # nós de elementos não-ar (rotor ou estator)
    g[E[~air_e & ~sel_e].ravel()] = 0.0
    g[E[sel_e].ravel()] = 1.0                            # rotor prevalece num nó comum (não ocorre aqui)
    gx, gy, S = grad_ops(xy, E[air_e])
    Ke = S[:, None, None] * (gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :])
    Ea = E[air_e]
    K = sp.csr_matrix((Ke.ravel(), (np.repeat(Ea, 3, 1).ravel(), np.tile(Ea, (1, 3)).ravel())), shape=(N, N))
    # mapa nó -> incógnita, com os nós periódicos amarrados à raiz da classe
    root = cons['root']
    fix_r = np.zeros(N, bool); np.logical_or.at(fix_r, root, fixed)
    gr = np.zeros(N); np.maximum.at(gr, root, np.where(fixed, g, -np.inf))
    g = np.where(fix_r[root], gr[root], g); fixed = fix_r[root]
    free_roots = np.unique(root[~fixed])
    col = -np.ones(N, int); col[free_roots] = np.arange(len(free_roots))
    rows = np.flatnonzero(~fixed)
    P = sp.csr_matrix((np.ones(len(rows)), (rows, col[root[rows]])), shape=(N, len(free_roots)))
    rhs = -(P.T @ (K @ np.where(fixed, g, 0.0)))
    u = spsolve((P.T @ K @ P).tocsc(), rhs)
    return np.where(fixed, g, P @ u)


def weighted_torque(B, xy, E, g, depth_m, n_sectors=3):
    """B [M,2] (Bx, By) T por elemento; xy mm; torque no corpo com g = 1 [N·m]."""
    xy = np.asarray(xy, np.float64)
    gx, gy, S = grad_ops(xy, E)
    ge = np.asarray(g)[E]
    dgx, dgy = (gx * ge).sum(1) * 1e3, (gy * ge).sum(1) * 1e3          # 1/m
    bx, by = B[:, 0], B[:, 1]; b2 = bx ** 2 + by ** 2
    fx = ((bx ** 2 - b2 / 2) * dgx + bx * by * dgy) / MU0                # (𝕋·∇g), N/m³
    fy = (bx * by * dgx + (by ** 2 - b2 / 2) * dgy) / MU0
    c = xy[E].mean(1) * 1e-3
    return -n_sectors * depth_m * float(((c[:, 0] * fy - c[:, 1] * fx) * S * 1e-6).sum())


def t22_python(ans, B, depth_m, r_m, air='vacuum', return_g=False):
    """T22 em Python (verificação) [N·m]. ans = AnsData; B [M,2] (Bx, By) por elemento;
    air: 'vacuum' (só vácuo) ou 'mu1' (vácuo + cobre, μr = 1). O wrap periódico vem sempre
    de ans_constraints do próprio .ans."""
    from src.data_gen.motor_constants import MATERIAL_ID
    from src.data_gen.parsers.ans_parsing import _parse_label_materials
    from src.postprocess.ans_rewrite import ans_constraints
    mat = np.asarray(_parse_label_materials(ans.lines)[0])
    air_ids = {'vacuum': (MATERIAL_ID['vacuum'],), 'mu1': (MATERIAL_ID['vacuum'], MATERIAL_ID['copper'])}[air]
    air_label = np.isin(mat, air_ids)
    sel_label = rotor_labels(ans.lines, r_m)
    cons = ans_constraints(ans.lines, ans.nodes)
    E = ans.elems[:, :3]; xy = ans.nodes[:, :2]
    g = weight_function(xy, E, ans.elems[:, 3], sel_label, air_label, cons)
    T = weighted_torque(B, xy, E, g, depth_m)
    return (T, g) if return_g else T
