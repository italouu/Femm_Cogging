"""
Geometria das fronteiras entre materiais na malha do FEMM.

  interface_edges — arestas compartilhadas por elementos de materiais diferentes
                    (contornos nos mapas de erro; estratificação espacial).
  corner_nodes / classify_triangles / RegionAccum — estratificação espacial do erro de
                    superfície em interior / interface / canto.

Material por elemento = material do rótulo (samples.parse_smooth(..., with_elem=True)
-> lay['elem_mat'], via ans_parsing._parse_label_materials).
"""
import numpy as np


def _edge_owners(tri):
    """Arestas não orientadas da malha: (edges [E,2] ordenadas, owners [E,2] elementos;
    -1 na 2ª coluna = aresta de contorno do domínio)."""
    tri = np.asarray(tri, np.int64)
    m = len(tri)
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(axis=1)
    own = np.tile(np.arange(m, dtype=np.int64), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, own = e[order], own[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    first = np.ones(len(e), bool)
    first[1:] = ~same
    idx = np.flatnonzero(first)
    owners = np.full((len(idx), 2), -1, np.int64)
    owners[:, 0] = own[idx]
    # segundo dono: a entrada seguinte, quando é a mesma aresta
    nxt = idx + 1
    ok = nxt < len(e)
    sec = np.zeros(len(idx), bool)
    sec[ok] = np.all(e[nxt[ok]] == e[idx[ok]], axis=1)
    owners[sec, 1] = own[nxt[sec]]
    return e[idx], owners


def interface_edges(tri, elem_mat):
    """Arestas [K,2] (índices de nó) entre 2 elementos de materiais diferentes."""
    edges, own = _edge_owners(tri)
    elem_mat = np.asarray(elem_mat)
    inner = own[:, 1] >= 0
    diff = np.zeros(len(edges), bool)
    diff[inner] = elem_mat[own[inner, 0]] != elem_mat[own[inner, 1]]
    return edges[diff]


# --------------------------------------------------------------------------- #
# Estratificação espacial do erro de superfície
# --------------------------------------------------------------------------- #
REGIONS = ('interior', 'interface', 'canto')     # código = índice; canto > interface


def corner_nodes(xy, tri, elem_mat, iface, ang_deg):
    """Vértices de fronteira entre materiais que são 'cantos': a fronteira muda de direção
    acima de ang_deg (vértice com 2 arestas de interface), ou ramifica (≥ 3 arestas de
    interface), ou o vértice toca ≥ 3 materiais (elementos incidentes)."""
    n = len(xy)
    deg = np.bincount(iface.ravel(), minlength=n)
    corner = deg >= 3
    # mudança de direção nos vértices de grau 2: ângulo entre (p - a) e (b - p)
    v2 = np.flatnonzero(deg == 2)
    if len(v2):
        ends = np.concatenate([iface, iface[:, ::-1]])            # (vértice, vizinho)
        order = np.argsort(ends[:, 0], kind='stable')
        ends = ends[order]
        start = np.searchsorted(ends[:, 0], v2)
        a, b = ends[start, 1], ends[start + 1, 1]
        u, w = xy[v2] - xy[a], xy[b] - xy[v2]
        cos = (u * w).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
        turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        corner[v2[turn > ang_deg]] = True
    # vértices que tocam ≥ 3 materiais
    tri = np.asarray(tri, np.int64)
    pairs = np.unique(np.stack([tri.ravel(), np.repeat(np.asarray(elem_mat), 3)], 1), axis=0)
    corner |= np.bincount(pairs[:, 0], minlength=n) >= 3
    return np.flatnonzero(corner)


def classify_triangles(xy, tri, elem_mat, d_int, d_canto, ang_canto):
    """Região de cada triângulo (código em REGIONS): canto = a até d_canto (mm) de um vértice
    de canto; interface = a até d_int (mm) de uma aresta entre materiais; interior = resto.
    Distâncias exatas triângulo↔segmento / triângulo↔ponto (shapely dwithin).
    Retorna (region [M] int8, iface [K,2], corners [C])."""
    import shapely
    xy = np.asarray(xy, np.float64)
    iface = interface_edges(tri, elem_mat)
    corners = corner_nodes(xy, tri, elem_mat, iface, ang_canto)
    polys = shapely.polygons(xy[np.asarray(tri, np.int64)])
    region = np.zeros(len(polys), np.int8)
    segs = shapely.linestrings(xy[iface])
    hit = shapely.STRtree(segs).query(polys, predicate='dwithin', distance=d_int)[0]
    region[np.unique(hit)] = 1
    if len(corners):
        pts = shapely.points(xy[corners])
        hit = shapely.STRtree(polys).query(pts, predicate='dwithin', distance=d_canto)[1]
        region[np.unique(hit)] = 2
    return region, iface, corners


class RegionAccum:
    """Erro de superfície acumulado por região (pontos de quadratura -> região do triângulo)
    sobre amostras; % sempre relativo ao B_ref GLOBAL (todas as regiões)."""

    def __init__(self):
        n = len(REGIONS)
        self.num_l1, self.num_l2, self.area = np.zeros(n), np.zeros(n), np.zeros(n)
        self.num_ref = 0.0

    def add(self, mag_model, mag_femm, w, point_region):
        e = np.abs(np.asarray(mag_model, np.float64) - mag_femm)
        w = np.asarray(w, np.float64)
        n = len(REGIONS)
        self.num_l1 += np.bincount(point_region, e * w, minlength=n)
        self.num_l2 += np.bincount(point_region, e ** 2 * w, minlength=n)
        self.area += np.bincount(point_region, w, minlength=n)
        self.num_ref += float((np.asarray(mag_femm, np.float64) ** 2 * w).sum())

    def report(self):
        A = self.area.sum()
        b_ref = float(np.sqrt(self.num_ref / A))
        out = dict(B_ref_global=b_ref,
                   total=dict(L1_area_pct=100 * self.num_l1.sum() / A / b_ref,
                              L2_area_pct=100 * np.sqrt(self.num_l2.sum() / A) / b_ref))
        for i, name in enumerate(REGIONS):
            a = self.area[i]
            l1 = self.num_l1[i] / a if a else float('nan')
            l2 = np.sqrt(self.num_l2[i] / a) if a else float('nan')
            out[name] = dict(L1_area=float(l1), L1_area_pct=float(100 * l1 / b_ref),
                             L2_area=float(l2), L2_area_pct=float(100 * l2 / b_ref),
                             area_frac=float(a / A),
                             L1_contrib_frac=float(self.num_l1[i] / self.num_l1.sum()))
        return out
