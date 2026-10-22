"""
Métricas de erro de |B| (convenção do CLAUDE.md, "Convenção de erro dos resultados"):
  e = | |B|_pred - |B|_true | por ponto; L1_A = Σe·a/Σa, L2_A = √(Σe²·a/Σa);
  MAE_pt = média simples de e; % relativo a B_ref = √(Σ|B|_true²·a/Σa);
  a = área dual do nó (malha) / área do elemento / r·dr·dθ (grade).

Origem: tests/resultados/common.py (2026-10-13), sem mudança de lógica.
L2 ponto a ponto na Accum: 2026-10-15. Erro grade × malha por interseção célula ∩
triângulo (célula = quadrilátero de lados retos, FEMM = B suavizado P1): grid_mesh_overlay
+ Accum.add_surface, 2026-10-20.
"""
import math

import numpy as np

# Domínio de amostragem fixo (setor r ∈ [28,5; 46,5] mm, θ ∈ [0°, 120°]) -- grade × malha
R_IN_MM, R_EXT_MM  = 28.5, 46.5
ANG1_DEG, ANG2_DEG = 0.0, 120.0


class Accum:
    """Acumula e = | |B|_pred - |B|_true | ponderado por área sobre amostras."""
    FIELDS = ('num_l1', 'num_l2', 'den_area', 'num_ref', 'sum_abs', 'sum_sq', 'n_pts')

    def __init__(self):
        self.num_l1 = self.num_l2 = self.den_area = self.num_ref = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0          # Σe² sem peso -> L2 ponto a ponto (2026-10-15)
        self.n_pts = 0

    def add(self, mag_pred, mag_true, area):
        e = np.abs(mag_pred - mag_true).astype(np.float64)
        a = np.broadcast_to(area, e.shape).astype(np.float64)
        t = mag_true.astype(np.float64)
        self.num_l1   += float((e * a).sum())
        self.num_l2   += float((e ** 2 * a).sum())
        self.num_ref  += float((t ** 2 * a).sum())
        self.den_area += float(a.sum())
        self.sum_abs  += float(e.sum())
        self.sum_sq   += float((e ** 2).sum())
        self.n_pts    += e.size

    def report(self):
        b_ref = math.sqrt(self.num_ref / self.den_area)
        l1 = self.num_l1 / self.den_area
        l2 = math.sqrt(self.num_l2 / self.den_area)
        out = dict(B_ref=b_ref, L1_area=l1, L1_area_pct=100 * l1 / b_ref,
                   L2_area=l2, L2_area_pct=100 * l2 / b_ref, n_pts=self.n_pts)
        if self.n_pts:
            mae = self.sum_abs / self.n_pts
            l2pt = math.sqrt(self.sum_sq / self.n_pts)
            out.update(MAE_pt=mae, MAE_pt_pct=100 * mae / b_ref,
                       L2_pt=l2pt, L2_pt_pct=100 * l2pt / b_ref)
        return out


    def add_surface(self, mag_pred, mag_true, w):
        """Só as integrais de superfície (L1_A, L2_A, B_ref) -- pontos de quadratura
        com peso w (grid_mesh_overlay); o ponto a ponto não se aplica (2026-10-20)."""
        e = np.abs(mag_pred - mag_true).astype(np.float64)
        w = np.asarray(w, np.float64)
        t = mag_true.astype(np.float64)
        self.num_l1   += float((e * w).sum())
        self.num_l2   += float((e ** 2 * w).sum())
        self.num_ref  += float((t ** 2 * w).sum())
        self.den_area += float(w.sum())


def merge(accs):
    m = Accum()
    for a in accs:
        for f in Accum.FIELDS:
            setattr(m, f, getattr(m, f) + getattr(a, f))
    return m


def pstats(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)))


def _cell_corners(r_in, r_ext, a1, a2, n_r, n_a):
    """Cantos (x,y) [n_r*n_a, 4, 2] das células da grade (row-major r, depois θ),
    ligados por segmentos retos -- célula = quadrilátero, não setor (decisão 2026-10-10)."""
    r = np.linspace(r_in, r_ext, n_r + 1)
    a = np.linspace(a1, a2, n_a + 1)
    R0, A0 = np.meshgrid(r[:-1], a[:-1], indexing='ij')
    R1, A1 = np.meshgrid(r[1:], a[1:], indexing='ij')
    rr = np.stack([R0, R1, R1, R0], axis=-1).reshape(-1, 4)
    aa = np.stack([A0, A0, A1, A1], axis=-1).reshape(-1, 4)
    return np.stack([rr * np.cos(aa), rr * np.sin(aa)], axis=-1)


def grid_mesh_overlay(xy, tri, b1, b2, r_in, r_ext, n_r, n_a,
                      ang_1=ANG1_DEG, ang_2=ANG2_DEG, n_sub=1):
    """Sobreposição grade H×W ∩ malha do FEMM para o erro de superfície grade × malha
    (CLAUDE.md, "Convenção de erro": interseção célula ∩ triângulo, célula de lados
    retos; valor do FEMM = B suavizado P1 no triângulo -- decisão do usuário 2026-10-20).

    xy [N,2] nós (mm), tri [M,3], b1/b2 [M,3] B suavizado por vértice de cada elemento
    (ans_b_smoothing.nodal_b, P1 descontínuo). Cada pedaço célula ∩ triângulo é
    triangulado em leque, cada subtriângulo dividido em 4^n_sub, e integrado com os
    3 pontos médios das arestas (exata até grau 2). n_sub=1: o |·| do erro tem dobra
    onde |G| cruza |B|_FEMM -- amostra 224, piso y_hw: L1 3,633% (n_sub=0), 3,672% (1),
    3,670% (2), 3,669% (3); Monte Carlo 20M pontos 3,670 ± 0,004%; L2 17,584% em todos.
    Retorna dict de arrays por ponto de quadratura:
      cell [Q] índice da célula (row-major), tri [Q] triângulo do FEMM, w [Q] peso (mm²),
      mag [Q] |B|_FEMM (T);
    e 'area_cells' (Σ área dos quadriláteros) para conferir a cobertura.
    O erro de uma saída em grade G [H,W] é então e_q = | |G|[cell_q] - mag_q |."""
    import shapely
    a1, a2 = np.deg2rad(ang_1), np.deg2rad(ang_2)
    dr, da = (r_ext - r_in) / n_r, (a2 - a1) / n_a
    corners = _cell_corners(r_in, r_ext, a1, a2, n_r, n_a)

    # --- pares candidatos: caixa (r, θ) de cada triângulo, com folga --------------
    P = xy[tri]                                                   # [M,3,2]
    r = np.hypot(P[..., 0], P[..., 1])
    th = np.arctan2(P[..., 1], P[..., 0])
    edge = np.linalg.norm(P - np.roll(P, 1, axis=1), axis=-1).max(axis=1)
    # uma aresta reta chega mais perto da origem que os vértices: flecha ≤ L²/(8r)
    tol_r = edge ** 2 / (8 * r.min(axis=1)) + 1e-3
    tol_a = 1e-3 / r.min(axis=1)
    i0 = np.clip(np.floor((r.min(1) - tol_r - r_in) / dr), 0, n_r - 1).astype(np.int64)
    i1 = np.clip(np.floor((r.max(1) + tol_r - r_in) / dr), 0, n_r - 1).astype(np.int64)
    j0 = np.clip(np.floor((th.min(1) - tol_a - a1) / da), 0, n_a - 1).astype(np.int64)
    j1 = np.clip(np.floor((th.max(1) + tol_a - a1) / da), 0, n_a - 1).astype(np.int64)
    ni, nj = i1 - i0 + 1, j1 - j0 + 1
    cnt = ni * nj
    t_idx = np.repeat(np.arange(len(tri)), cnt)
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)  # posição local
    c_idx = (np.repeat(i0, cnt) + k // np.repeat(nj, cnt)) * n_a \
        + np.repeat(j0, cnt) + k % np.repeat(nj, cnt)

    # --- recorte (convexo ∩ convexo) ---------------------------------------------
    tri_poly = shapely.polygons(P)
    cell_poly = shapely.polygons(corners)
    pieces = shapely.intersection(tri_poly[t_idx], cell_poly[c_idx])
    keep = shapely.area(pieces) > 1e-14
    pieces, t_idx, c_idx = pieces[keep], t_idx[keep], c_idx[keep]
    assert (shapely.get_type_id(pieces) == 3).all(), 'interseção não poligonal'

    # --- leque + pontos médios ---------------------------------------------------
    coords, pid = shapely.get_coordinates(pieces, return_index=True)
    nv = np.bincount(pid, minlength=len(pieces))                 # inclui o ponto de fecho
    start = np.cumsum(nv) - nv
    lp = np.arange(len(pid)) - start[pid]                         # posição no anel
    sel = (lp >= 1) & (lp <= nv[pid] - 3)                         # v_i, i = 1..k-2 (k = nv-1)
    s_piece = pid[sel]
    v0, vi, vj = coords[start[s_piece]], coords[sel], coords[np.flatnonzero(sel) + 1]
    for _ in range(n_sub):     # cada subtriângulo -> 4 (pontos médios das arestas)
        m01, m12, m20 = (v0 + vi) / 2, (vi + vj) / 2, (vj + v0) / 2
        v0, vi, vj = (np.concatenate(z) for z in ((v0, m01, m20, m01), (m01, vi, m12, m12),
                                                   (m20, m12, vj, m20)))
        s_piece = np.tile(s_piece, 4)
    area = 0.5 * np.abs((vi[:, 0] - v0[:, 0]) * (vj[:, 1] - v0[:, 1])
                        - (vj[:, 0] - v0[:, 0]) * (vi[:, 1] - v0[:, 1]))
    q = np.concatenate([(v0 + vi) / 2, (vi + vj) / 2, (vj + v0) / 2])   # [3T,2]
    q_piece = np.tile(s_piece, 3)
    q_w = np.tile(area / 3, 3)

    # --- B suavizado P1 do triângulo no ponto (baricêntricas) -------------------
    t = t_idx[q_piece]
    A, B, C = P[t, 0], P[t, 1], P[t, 2]
    det = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])
    l1 = ((q[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (q[:, 1] - A[:, 1])) / det
    l2 = ((B[:, 0] - A[:, 0]) * (q[:, 1] - A[:, 1]) - (q[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])) / det
    lam = np.stack([1 - l1 - l2, l1, l2], axis=1)
    bx = (lam * b1[t]).sum(1)
    by = (lam * b2[t]).sum(1)
    sh = corners
    area_cells = 0.5 * np.abs((sh[:, 2, 0] - sh[:, 0, 0]) * (sh[:, 3, 1] - sh[:, 1, 1])
                              - (sh[:, 3, 0] - sh[:, 1, 0]) * (sh[:, 2, 1] - sh[:, 0, 1])).sum()
    return dict(cell=c_idx[q_piece].astype(np.int32), tri=t.astype(np.int32),
                w=q_w.astype(np.float32), mag=np.hypot(bx, by).astype(np.float32),
                area_cells=float(area_cells))


def _ref_quadrature(n_sub):
    """Pontos (baricêntricas [K,3]) do triângulo de referência dividido em 4^n_sub
    subtriângulos, 3 pontos médios de aresta por subtriângulo; peso igual 1/K."""
    tris = [np.eye(3)]                                   # vértices em baricêntricas
    for _ in range(n_sub):
        new = []
        for v in tris:
            m01, m12, m20 = (v[0] + v[1]) / 2, (v[1] + v[2]) / 2, (v[2] + v[0]) / 2
            new += [np.stack(z) for z in ((v[0], m01, m20), (m01, v[1], m12),
                                          (m20, m12, v[2]), (m01, m12, m20))]
        tris = new
    pts = [(v[a] + v[b]) / 2 for v in tris for a, b in ((0, 1), (1, 2), (2, 0))]
    return np.array(pts)


def mesh_surface_quadrature(xy, tri, b1, b2, n_sub=2):
    """Quadratura da superfície do FEMM na PRÓPRIA malha (erro de superfície malha × malha,
    sem interseção -- 2026-10-20). FEMM = B suavizado P1 descontínuo (b1/b2 [M,3] de
    nodal_b); cada triângulo dividido em 4^n_sub, 3 pontos médios por subtriângulo.
    Retorna bary [K,3], area [M] (mm²), mag [M,K] |B|_FEMM nos pontos.
    Superfície do modelo nos mesmos pontos: surface_values_p1 (saída por nó) ou o valor
    do elemento repetido (saída por elemento, P0)."""
    bary = _ref_quadrature(n_sub)
    P = xy[tri]
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    bx, by = b1 @ bary.T, b2 @ bary.T                    # [M,K]
    return dict(bary=bary.astype(np.float32), area=area.astype(np.float32),
                mag=np.hypot(bx, by).astype(np.float32))


def surface_values_p1(tri, B_node, bary):
    """|B| [M,K] da superfície P1 contínua definida pelos valores nodais B_node [S,2]
    (interpolação linear das componentes no triângulo).

    B_node vem em componentes POLARES (Br, Bθ), cada nó na base do seu próprio ângulo
    (modelos nodais e o piso P1_nos), enquanto a superfície do FEMM
    (mesh_surface_quadrature) interpola as componentes CARTESIANAS b1, b2. A base polar
    gira dentro do triângulo, então as duas interpolações não coincidem exatamente
    (|·| idem). Efeito no L1_A/L2_A não medido."""
    V = B_node[tri]                                      # [M,3,2]
    return np.hypot(V[..., 0] @ bary.T, V[..., 1] @ bary.T)


def surface_error_per_triangle(n_tri, tri_idx, mag_model, mag_femm, w):
    """Densidade do erro de superfície por triângulo do FEMM: Σ e·w / Σ w dos pontos de
    quadratura de cada triângulo (mapa do erro; a média ponderada pela área de todos os
    triângulos é o L1_A)."""
    e = np.abs(np.asarray(mag_model, np.float64) - mag_femm)
    w = np.asarray(w, np.float64)
    num = np.bincount(tri_idx, e * w, minlength=n_tri)
    den = np.bincount(tri_idx, w, minlength=n_tri)
    return np.divide(num, den, out=np.zeros(n_tri), where=den > 0)
