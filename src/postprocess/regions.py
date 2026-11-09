"""
Geometria das fronteiras entre materiais na malha do FEMM.

  interface_edges — arestas compartilhadas por elementos de materiais diferentes
                    (contornos nos mapas de erro; estratificação espacial).
  corner_nodes      — vértices de canto (𝒱_c) da fronteira entre materiais.
  band_distances / band_edges / band_points / BandAccum — erro de superfície por faixas de
                    distância (docs/claude/avaliacao.md §2): Ω_f(d) = {dist(x, Γ) ≤ d},
                    Ω_q(d) = {dist(x, 𝒱_c) ≤ d}, Ω_int(d) = Ω ∖ Ω_f(d); cada ponto de
                    quadratura pela SUA distância (elementos cortados).
  classify_triangles — classificação antiga por triângulo (só scripts/artigo_v1.py).

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


# [REMOVIDO] RegionAccum: região fixa por TRIÂNGULO (classify_triangles); substituído por
# BandAccum (faixas por distância de cada ponto de quadratura, elementos cortados).
# class RegionAccum:
#     """Erro de superfície acumulado por região (pontos de quadratura -> região do triângulo)
#     sobre amostras; % sempre relativo ao B_ref GLOBAL (todas as regiões)."""
#
#     def __init__(self):
#         n = len(REGIONS)
#         self.num_l1, self.num_l2, self.area = np.zeros(n), np.zeros(n), np.zeros(n)
#         self.num_ref = 0.0
#
#     def add(self, mag_model, mag_femm, w, point_region):
#         e = np.abs(np.asarray(mag_model, np.float64) - mag_femm)
#         w = np.asarray(w, np.float64)
#         n = len(REGIONS)
#         self.num_l1 += np.bincount(point_region, e * w, minlength=n)
#         self.num_l2 += np.bincount(point_region, e ** 2 * w, minlength=n)
#         self.area += np.bincount(point_region, w, minlength=n)
#         self.num_ref += float((np.asarray(mag_femm, np.float64) ** 2 * w).sum())
#
#     def report(self):
#         A = self.area.sum()
#         b_ref = float(np.sqrt(self.num_ref / A))
#         out = dict(B_ref_global=b_ref,
#                    total=dict(L1_area_pct=100 * self.num_l1.sum() / A / b_ref,
#                               L2_area_pct=100 * np.sqrt(self.num_l2.sum() / A) / b_ref))
#         for i, name in enumerate(REGIONS):
#             a = self.area[i]
#             l1 = self.num_l1[i] / a if a else float('nan')
#             l2 = np.sqrt(self.num_l2[i] / a) if a else float('nan')
#             out[name] = dict(L1_area=float(l1), L1_area_pct=float(100 * l1 / b_ref),
#                              L2_area=float(l2), L2_area_pct=float(100 * l2 / b_ref),
#                              area_frac=float(a / A),
#                              L1_contrib_frac=float(self.num_l1[i] / self.num_l1.sum()))
#         return out


# --------------------------------------------------------------------------- #
# Faixas por distância (elementos cortados: cada ponto de quadratura pela SUA distância)
#   Γ   = união das arestas de interface (interface_edges)
#   𝒱_c = vértices de canto (corner_nodes)
#   Ω_f(d) = {dist(x, Γ) ≤ d}, Ω_q(d) = {dist(x, 𝒱_c) ≤ d}, Ω_int(d) = Ω ∖ Ω_f(d)
# --------------------------------------------------------------------------- #
BANDS = ('quinas', 'fronteira_sem_quinas', 'interior')   # disjuntas; código = índice
BAND_UNIONS = {'fronteira': ('quinas', 'fronteira_sem_quinas')}   # Ω_f = Ω_q ∪ (Ω_f ∖ Ω_q)


def region_geometry(xy, tri, elem_mat, ang_deg):
    """(iface [K,2], corners [C]) = Γ e 𝒱_c da malha (critérios de interface_edges e
    corner_nodes)."""
    iface = interface_edges(tri, elem_mat)
    corners = corner_nodes(np.asarray(xy, np.float64), tri, elem_mat, iface, ang_deg)
    return iface, corners


def _dist_point_segments(p, a, b):
    """Distância exata ponto p [n,2] ↔ segmento [a, b] ([n,...,2], broadcast)."""
    ab = b - a
    ap = p - a
    den = (ab * ab).sum(-1)
    t = np.clip(np.divide((ap * ab).sum(-1), den, out=np.zeros_like(den), where=den > 0), 0, 1)
    return np.linalg.norm(ap - t[..., None] * ab, axis=-1)


def _dist_to_segments(pts, A, B, d_cap=np.inf, h=0.02, ks=(16, 64), pairs=4_000_000):
    """Distância euclidiana exata de pts [Q,2] à união dos segmentos [A_i, B_i], vetorizada.
    Cada segmento é dividido em sub-segmentos de comprimento ≤ h (a união é o MESMO
    conjunto: a distância não muda). Para cada ponto: k pontos médios mais próximos
    (cKDTree), distância exata a esses k sub-segmentos, mínimo d_k. Qualquer outro
    sub-segmento fica a ≥ lb = (distância ao k-ésimo ponto médio) - h/2, então:
      lb ≥ d_k     -> d_k é exato;
      lb > d_cap   -> dist > d_cap garantido (d_k é um limite superior; só o fato de estar
                      além de d_cap importa -- bin de excesso das faixas);
      senão        -> tenta o próximo k de ks; no fim, shapely (query_nearest).
    Retorna (dist [Q], n_fallback)."""
    import shapely
    from scipy.spatial import cKDTree
    L = np.linalg.norm(B - A, axis=1)
    n = np.maximum(1, np.ceil(L / h).astype(np.int64))
    sid = np.repeat(np.arange(len(A)), n)
    j = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t0, t1 = j / n[sid], (j + 1) / n[sid]
    a = A[sid] + t0[:, None] * (B - A)[sid]
    b = A[sid] + t1[:, None] * (B - A)[sid]
    hs = float(np.linalg.norm(b - a, axis=1).max())
    tree = cKDTree((a + b) / 2)
    dist = np.empty(len(pts))
    todo = np.arange(len(pts))
    for k in ks:
        k = min(k, len(a))
        chunk = max(1, pairs // k)        # blocos de ~pairs pares ponto-segmento (memória)
        bad = []
        for s in range(0, len(todo), chunk):
            ii = todo[s:s + chunk]
            p = pts[ii]
            dm, im = tree.query(p, k=k)
            if k == 1:
                dm, im = dm[:, None], im[:, None]
            d = _dist_point_segments(p[:, None, :], a[im], b[im]).min(axis=1)
            dist[ii] = d
            lb = dm[:, -1] - hs / 2
            bad.append(ii[(lb < d) & (lb <= d_cap)])
        todo = np.concatenate(bad)
        if not len(todo) or k == len(a):
            break
    bad = todo if len(a) > max(ks) else np.zeros(0, np.int64)
    if len(bad):
        segs = shapely.linestrings(np.stack([A, B], axis=1))
        _, dist[bad] = shapely.STRtree(segs).query_nearest(
            shapely.points(pts[bad]), return_distance=True, all_matches=False)
    return dist, int(len(bad))


def band_distances(xy, iface, corners, pts, method='kdtree', d_cap=np.inf):
    """Distância euclidiana exata de cada ponto pts [Q,2] a Γ (segmentos xy[iface]) e a
    𝒱_c (pontos xy[corners]).
      method='kdtree' (padrão): Γ por _dist_to_segments (vetorizado, exato com conferência
        e fallback no shapely); 𝒱_c por cKDTree. d_cap: pontos com dist_f > d_cap garantido
        ficam com um limite superior (> d_cap) em vez do valor exato -- band_edges põe
        tudo acima de d_max num só bin, então d_cap = d_max não muda nenhuma saída.
      method='shapely': shapely STRtree.query_nearest com distância (referência; ~8 µs por
        ponto -> ~60 s por amostra com 7 M pontos).
    Retorna (dist_f [Q], dist_q [Q], viol, n_fallback): como 𝒱_c são vértices de Γ,
    dist_f ≤ dist_q ponto a ponto; dist_f é levado a min(dist_f, dist_q) para que Ω_q ⊆ Ω_f
    valha também no arredondamento, e viol = maior violação antes disso (mm)."""
    import shapely
    xy = np.asarray(xy, np.float64)
    pts = np.asarray(pts, np.float64)
    nfb = 0
    if method == 'kdtree':
        from scipy.spatial import cKDTree
        d_f, nfb = _dist_to_segments(pts, xy[iface[:, 0]], xy[iface[:, 1]], d_cap=d_cap)
        d_q = cKDTree(xy[corners]).query(pts)[0] if len(corners) else np.full(len(pts), np.inf)
    elif method == 'shapely':
        p = shapely.points(pts)
        _, d_f = shapely.STRtree(shapely.linestrings(xy[iface])).query_nearest(
            p, return_distance=True, all_matches=False)
        if len(corners):
            _, d_q = shapely.STRtree(shapely.points(xy[corners])).query_nearest(
                p, return_distance=True, all_matches=False)
        else:
            d_q = np.full(len(p), np.inf)
    else:
        raise ValueError(f'método desconhecido: {method}')
    ok = d_f <= d_cap                      # valores exatos (os demais: limite superior)
    viol = float(np.max(d_f[ok] - d_q[ok], initial=0.0))
    return np.minimum(d_f, d_q), d_q, viol, nfb


def mesh_domain(xy, tri):
    """Domínio Ω (shapely) = união dos triângulos da malha."""
    import shapely
    return shapely.union_all(shapely.polygons(np.asarray(xy, np.float64)[np.asarray(tri, np.int64)]))


def band_polygons(xy, iface, corners, d, domain):
    """Geometria exata das faixas num d (figura e conferência da quadratura):
    {'fronteira': Ω_f, 'quinas': Ω_q, 'fronteira_sem_quinas': Ω_f ∖ Ω_q}, cada uma =
    buffer de Γ / 𝒱_c recortado ao `domain` (aproximação poligonal do arco do buffer:
    quad_segs=64)."""
    import shapely
    xy = np.asarray(xy, np.float64)
    f = shapely.intersection(shapely.buffer(shapely.multilinestrings(xy[iface]), d, quad_segs=64),
                             domain)
    q = shapely.intersection(shapely.buffer(shapely.multipoints(xy[corners]), d, quad_segs=64),
                             domain) if len(corners) else shapely.Polygon()
    return dict(fronteira=f, quinas=q, fronteira_sem_quinas=shapely.difference(f, q))


def band_edges(d_list, d_max, n_bins, scale='lin'):
    """Bordas dos bins de distância (mm): 0, n_bins bins uniformes ('lin') ou log ('log',
    de d_max/1e3 a d_max) até d_max, com os d de d_list inseridos como bordas exatas.
    Bin k (band_points) = {e[k-1] < dist ≤ e[k]}, bin 0 = {dist = 0}, último = {dist > d_max}."""
    d_list = np.asarray(d_list, np.float64)
    if (d_list <= 0).any() or (d_list > d_max).any():
        raise ValueError(f'--d fora de (0, d_max={d_max}]: {d_list.tolist()}')
    if scale == 'lin':
        e = np.linspace(0.0, d_max, n_bins + 1)
    elif scale == 'log':
        e = np.concatenate([[0.0], np.geomspace(d_max / 1e3, d_max, n_bins)])
    else:
        raise ValueError(f'escala de bins desconhecida: {scale}')
    return np.unique(np.concatenate([e, d_list]))


def band_points(dist_f, dist_q, edges, d_list):
    """Índices de bin (bin_f, bin_q [Q], convenção de band_edges) e código da região
    disjunta (BANDS) de cada ponto para cada d da lista (reg [n_d, Q] int8):
    quinas = dist_q ≤ d; fronteira_sem_quinas = dist_f ≤ d e dist_q > d; interior = resto."""
    it = np.int16 if len(edges) + 1 <= np.iinfo(np.int16).max else np.int32
    bf = np.searchsorted(edges, dist_f, side='left').astype(it)
    bq = np.searchsorted(edges, dist_q, side='left').astype(it)
    reg = np.full((len(d_list), len(dist_f)), 2, np.int8)
    for i, d in enumerate(d_list):
        reg[i, dist_f <= d] = 1
        reg[i, dist_q <= d] = 0
    return dict(bin_f=bf, bin_q=bq, reg=reg)


class BandAccum:
    """Erro de superfície acumulado por distância sobre amostras (pontos de quadratura com
    peso w; e = | |B|_modelo - |B|_FEMM |):
      - histogramas em dist_f e dist_q: Σw, Σe·w, Σe²·w por bin (bordas `edges`) -> curvas
        acumuladas de Ω_f(d) e Ω_q(d) em qualquer borda, sem reprocessar;
      - para cada d da lista, as 3 regiões disjuntas (BANDS) acumuladas diretamente pelo
        código do ponto (a fronteira sem quinas usa os dois limiares no mesmo ponto);
      - global: Σw, Σe·w, Σe²·w e Σ|B|_FEMM²·w (B_ref).
    % sempre relativo ao B_ref GLOBAL."""

    def __init__(self, edges, d_list):
        self.edges = np.asarray(edges, np.float64)
        self.d_list = np.asarray(d_list, np.float64)
        nb = len(self.edges) + 1
        self.hist = {k: np.zeros((3, nb)) for k in ('f', 'q')}      # (Σw, Σe·w, Σe²·w)
        self.tab = np.zeros((len(self.d_list), 3, len(BANDS)))       # [d, (Σw,Σew,Σe²w), região]
        self.tot = np.zeros(3)
        self.num_ref = 0.0

    def add(self, mag_model, mag_femm, w, bp):
        """bp = band_points(...) do conjunto de pontos (mesma ordem de mag_*/w)."""
        e = np.abs(np.asarray(mag_model, np.float64) - mag_femm)
        w = np.asarray(w, np.float64)
        vals = (w, e * w, e ** 2 * w)
        nb = len(self.edges) + 1
        for k, b in (('f', bp['bin_f']), ('q', bp['bin_q'])):
            for j, v in enumerate(vals):
                self.hist[k][j] += np.bincount(b, v, minlength=nb)
        for i in range(len(self.d_list)):
            for j, v in enumerate(vals):
                self.tab[i, j] += np.bincount(bp['reg'][i], v, minlength=len(BANDS))
        self.tot += [v.sum() for v in vals]
        self.num_ref += float((np.asarray(mag_femm, np.float64) ** 2 * w).sum())

    def _stats(self, a, l1, l2, b_ref):
        la = l1 / a if a else float('nan')
        lb = np.sqrt(l2 / a) if a else float('nan')
        return dict(L1_area=float(la), L1_area_pct=float(100 * la / b_ref),
                    L2_area=float(lb), L2_area_pct=float(100 * lb / b_ref),
                    area_frac=float(a / self.tot[0]), L1_contrib_frac=float(l1 / self.tot[1]))

    def report(self):
        """Tabela por d (regiões disjuntas + 'fronteira' = Ω_f inteira) e total."""
        A = self.tot[0]
        b_ref = float(np.sqrt(self.num_ref / A))
        out = dict(B_ref_global=b_ref,
                   total=dict(L1_area_pct=float(100 * self.tot[1] / A / b_ref),
                              L2_area_pct=float(100 * np.sqrt(self.tot[2] / A) / b_ref)),
                   by_d={})
        for i, d in enumerate(self.d_list):
            r = {name: self._stats(*self.tab[i, :, k], b_ref) for k, name in enumerate(BANDS)}
            for u, parts in BAND_UNIONS.items():
                r[u] = self._stats(*sum(self.tab[i, :, BANDS.index(p)] for p in parts), b_ref)
            out['by_d'][f'{d:g}'] = dict(d_mm=float(d), **r)
        return out

    def curves(self):
        """Curvas acumuladas nas bordas (d = edges), para Ω_f ('f') e Ω_q ('q'): area_frac,
        eta (fração de ∫e dA total), L1_pct e L2_pct da região (% do B_ref global); e as
        somas cruas cum_w / cum_e / cum_e2 (mm², T·mm², T²·mm²) para recálculo."""
        A, L1 = self.tot[0], self.tot[1]
        b_ref = float(np.sqrt(self.num_ref / A))
        n = len(self.edges)
        out = dict(d=self.edges.copy(), B_ref_global=b_ref, area_total=A, L1_total=L1,
                   L2_total=self.tot[2])
        with np.errstate(invalid='ignore', divide='ignore'):
            for k in ('f', 'q'):
                c = np.cumsum(self.hist[k], axis=1)[:, :n]
                out[f'{k}_cum_w'], out[f'{k}_cum_e'], out[f'{k}_cum_e2'] = c
                out[f'{k}_area_frac'] = c[0] / A
                out[f'{k}_eta'] = c[1] / L1
                out[f'{k}_L1_pct'] = 100 * c[1] / c[0] / b_ref
                out[f'{k}_L2_pct'] = 100 * np.sqrt(c[2] / c[0]) / b_ref
        return out
