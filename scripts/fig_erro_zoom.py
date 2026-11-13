"""Figura do artigo, seção "Distribuição espacial do erro": erro de |B| em 3 janelas de zoom
(Quina, Interface, Entreferro) × 4 archs (FNO2d, FNO_GNN, FNO_Bipartite, FNO_SplitGNN; loss
mae), numa amostra do teste (padrão #2184, a do erro_dominio).

Só visualização: processa uma amostra e não recalcula nem grava métricas.

Grandeza: Δ|B| = | |B|_mod − |B|_FEMM | [T], nos pontos de avaliação das métricas de
superfície, com as mesmas funções do regions/surface:
  - FNO2d (grade): pedaços da interseção célula × triângulo (metrics.grid_mesh_overlay), com a
    célula constante;
  - FNO_GNN (P1_nos) e FNO_SplitGNN (P1_mat): 48 pontos por triângulo
    (mesh_surface_quadrature, 16 subtriângulos × 3 pontos médios);
  - FNO_Bipartite (P0_elem): constante no elemento.
FEMM = B suavizado P1 descontínuo. Layout, Ω e Γ vêm de postprocess._regions_sample.

Modos (o que é desenhado):
  - fine=False (padrão): 1 valor por triângulo, a média ponderada Σ e·w / Σ w
    (surface_error_per_triangle). A média por área desses valores é o L1_A.
  - fine=True: resolução dos pontos de avaliação, sem interpolação adicional:
      FNO2d    -> cada pedaço célula ∩ triângulo, com Σ e·w / Σ w dos pontos do pedaço;
      P1 e Bipartite -> cada um dos 16 subtriângulos da quadratura, com a média dos seus 3
                  pontos (pesos iguais; a Bipartite é constante no elemento, o FEMM não).

Região por linha (ROW_REGION): só os pontos da região da linha são coloridos -- Quina =
quinas (dist_q <= d), Interface = fronteira sem quinas (dist_f <= d < dist_q), com a
classificação do regions por ponto de avaliação; Entreferro = ar entre o raio externo do
estator e o interno do rotor. Cada polígono desenhado é recortado pela forma exata da região
(region_zone: discos de raio d nas quinas, faixa de largura d em Γ menos os discos, ou os
triângulos do entreferro; tudo ∩ Ω) e só a parte de dentro é pintada, com a média dos pontos
da região no polígono; o resto da janela fica cinza. A borda da região é a mesma nas 4 archs.

Escala por linha: vmin = 0; vmax = percentil P (99) dos valores efetivamente desenhados na
janela (polígonos coloridos), juntando as 4 archs; colorbar só retângulo (sem
seta), valores acima de vmax com a cor do topo. Fora de Ω em cinza. Coluna da esquerda: o
domínio completo com o retângulo da janela da linha.

Saída: docs/figures/erro_zoom_regioes{,_fino}.{pdf,png} (7,16 in, 8 pt) e
docs/figures/erro_zoom_regioes_localizacao.{pdf,png} (geometria + as 3 janelas).
Janelas, vmax e contagens são impressos no terminal e salvos em
resultados/BLDC_138x276/dados/fig_erro_zoom/sample_XXXXXX_<loss>.json.

    python -m scripts.fig_erro_zoom [--sample 2184] [--loss mae] [--mode both|coarse|fine]
                                    [--band] [--d MM] [--percentile 99]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.postprocess import (ARTICLE_ARCH_NAME, ARTICLE_FIG_DIR, OUT_ROOT, REGIONS_ARCHS,
                                 REGIONS_D_MM, _regions_runs, _regions_sample, _split_points,
                                 _surface_points, design_rows, predict, predict_split,
                                 regions_setup, sample_path, to_torch_sample)
from src.postprocess.metrics import _cell_corners, surface_error_per_triangle
from src.postprocess.plots.errors import OUT_OMEGA_COLOR
from src.postprocess.plots.primitives import apply_article_style, apply_style
from src.postprocess.samples import ANG_1, ANG_2, N_A, N_R

FIG_NAME = 'erro_zoom_regioes'
FIG_WIDTH_IN = 7.16               # duas colunas IEEE
FONT_PT = 8
CMAP = 'viridis'
CB_LABEL = r'$\Delta|\mathbf{B}|$ (T)'
ROWS = ('Quina', 'Interface', 'Entreferro')
# região pintada em cada linha (o resto da janela fica cinza): código de regions.BANDS
# (0 quinas, 1 fronteira sem quinas, 2 interior) ou 'entreferro' (ar entre estator e rotor)
ROW_REGION = {'Quina': 0, 'Interface': 1, 'Entreferro': 'entreferro'}
# Janelas (centro x, y [mm], lado [mm]), escolhidas na geometria da #2184 (em polar):
#   Quina      -> canto da sapata do dente na abertura da ranhura: r = 38,614 mm (raio externo
#                 do estator), θ = 59,16°;
#   Interface  -> abertura da ranhura entre dois dentes (pontas em 49,16° e 50,84°, de
#                 r = 37,514 a 38,614 mm) e um pouco das pontas dos dentes: r = 38,0 mm, θ = 50,0°
#                 [antes: face ímã–coroa do rotor, r = 42,175 mm, θ = 49,63° -- (27.33, 32.13)];
#   Entreferro -> meio do entreferro (38,614–39,925 mm), r = 39,27 mm, θ = 90°: faixa quase
#                 horizontal, janela maior (ENTREFERRO_SIDE_MM) com dentes embaixo e ímãs em cima
#                 [antes: θ = 66,0°, lado igual às outras -- (15.97, 35.88)].
# [REMOVIDO] janelas de 3,0 mm -- aumentadas para 4,5 mm a pedido (mais área no zoom)
# DEFAULT_WINDOWS = {
#     'Quina': (19.79, 33.16, 3.0),
#     'Interface': (27.33, 32.13, 3.0),
#     'Entreferro': (15.97, 35.88, 3.0),
# }
WINDOW_SIDE_MM = 4.5
ENTREFERRO_SIDE_MM = 8.0          # janela do entreferro maior (faixa horizontal em θ = 90°)
DEFAULT_WINDOWS = {
    'Quina': (19.79, 33.16, WINDOW_SIDE_MM),
    # [REMOVIDO] face ímã–coroa do rotor: 'Interface': (27.33, 32.13, WINDOW_SIDE_MM),
    'Interface': (24.43, 29.11, WINDOW_SIDE_MM),
    # [REMOVIDO] θ = 66°, mesmo lado das outras: 'Entreferro': (15.97, 35.88, WINDOW_SIDE_MM),
    'Entreferro': (0.0, 39.27, ENTREFERRO_SIDE_MM),
}


# --------------------------------------------------------------------------------------------
# dados (1 amostra)
# --------------------------------------------------------------------------------------------
def compute_errors(sample_idx, loss='mae', d=None):
    """Layout + Ω + Γ da amostra (_regions_sample) e, para cada arch do artigo, o erro
    e = | |B|_mod − |B|_FEMM | nos pontos de avaliação das métricas (mesmas funções do
    regions_add_run); código da região (regions.BANDS, no d dado) de cada ponto e os
    triângulos do entreferro."""
    d = REGIONS_D_MM[0] if d is None else float(d)
    d_list, edges, _ = regions_setup(d_list=[d])
    di = d_list.index(d)
    row = design_rows()[sample_idx]
    lay = _regions_sample(sample_path(sample_idx), row, 30.0, edges, d_list)
    M, K = lay['mesh_quad']['mag'].shape
    runs = {k: r for k, r in _regions_runs().items() if r['loss'] == loss}
    ts = to_torch_sample(lay)
    E = {}
    for r in runs.values():
        a = r['arch']
        if a == 'FNO_SplitGNN':
            _, y_base = predict_split(r['model'], r['normalizer'], lay['split'])
            mm, mf, w, ps = _split_points(lay, y_base)
            t_idx = np.repeat(np.arange(M), K)
        else:
            out_hw, yn = predict(a, r['model'], r['normalizer'], ts)
            mm, mf, w, t_idx, ps = _surface_points(a, lay, out_hw, yn, with_set=True)
        e = np.abs(np.asarray(mm, np.float64) - mf)
        E[a] = dict(e=e, w=np.asarray(w, np.float64), t=np.asarray(t_idx), pset=ps,
                    e_tri=surface_error_per_triangle(M, t_idx, mm, mf, w),
                    run=r['run_dir'], epoch=r['epoch'])
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    cells = _cell_corners(r_in, r_ext, np.deg2rad(ANG_1), np.deg2rad(ANG_2), N_R, N_A)
    code = {}
    for ps, bp in lay['bands'].items():                 # -1 = fora de Ω
        c = np.full(len(bp['keep']), -1, np.int8)
        c[bp['keep']] = bp['reg'][di]
        code[ps] = c
    from src.data_gen.motor_constants import MATERIAL_ID
    xy, tri = lay['xy'], lay['tri']
    rc = np.hypot(*xy[tri].mean(1).T)
    r_so = float(row['stator_outer_diameter [mm]']) / 2
    r_ri = float(row['rotor_inner_diameter [mm]']) / 2
    gap_tri = (lay['elem_mat'] == MATERIAL_ID['vacuum']) & (rc > r_so) & (rc < r_ri)
    return dict(sample_idx=sample_idx, loss=loss, row=row, xy=xy, tri=tri, d=d,
                in_omega=lay['in_omega'], iface=lay['iface'], corners=lay['corners'],
                elem_mat=lay['elem_mat'], bary=lay['mesh_quad']['bary'].astype(np.float64),
                K=K, cell=lay['overlay']['cell'], cells=cells, E=E, code=code, gap_tri=gap_tri,
                vac_tri=lay['elem_mat'] == MATERIAL_ID['vacuum'], r_so=r_so, r_ri=r_ri)


def _tris_in_window(xy, tri, box):
    x0, x1, y0, y1 = box
    P = xy[tri]
    return np.flatnonzero((P[..., 0].min(1) <= x1) & (P[..., 0].max(1) >= x0)
                          & (P[..., 1].min(1) <= y1) & (P[..., 1].max(1) >= y0))


# [REMOVIDO] panel_polys sem máscara de região (pintava tudo que caía na janela) -- substituída
# pela versão abaixo, que pinta só os pontos da região da linha e deixa o resto em cinza.
# def panel_polys(D, arch, box, fine):
#     """Polígonos [lista de [n,2]], valores [T] e flag 'em Ω' do que é desenhado na janela."""
#     import shapely
#     xy, tri, in_om = D['xy'], D['tri'], D['in_omega']
#     ex = D['E'][arch]
#     tw = _tris_in_window(xy, tri, box)
#     if not fine or arch == 'FNO_BipartiteGNN_Elem':
#         return list(xy[tri[tw]]), ex['e_tri'][tw], in_om[tw]
#     if ex['pset'] == 'overlay':                    # FNO2d: pedaços célula ∩ triângulo
#         sel = np.isin(ex['t'], tw)
#         t, c = ex['t'][sel], D['cell'][sel]
#         key, inv = np.unique(t.astype(np.int64) * len(D['cells']) + c, return_inverse=True)
#         val = np.bincount(inv, ex['e'][sel] * ex['w'][sel]) / np.bincount(inv, ex['w'][sel])
#         wsum = np.bincount(inv, ex['w'][sel])
#         tu, cu = key // len(D['cells']), key % len(D['cells'])
#         pieces = shapely.intersection(shapely.polygons(xy[tri[tu]]), shapely.polygons(D['cells'][cu]))
#         area = shapely.area(pieces)
#         D.setdefault('_piece_check', []).append(float(np.max(np.abs(area - wsum) / area)))
#         polys = [np.asarray(p.exterior.coords)[:-1] for p in pieces]
#         return polys, val, in_om[tu]
#     # P1 (FNO_GNN, Split): 16 subtriângulos da quadratura, média dos 3 pontos médios
#     K, bary = D['K'], D['bary']
#     e = ex['e'].reshape(-1, K)[tw].reshape(len(tw), K // 3, 3)
#     m = np.einsum('kj,tjd->tkd', bary, xy[tri[tw]]).reshape(len(tw), K // 3, 3, 2)
#     m01, m12, m20 = m[:, :, 0], m[:, :, 1], m[:, :, 2]
#     V = np.stack([m01 + m20 - m12, m01 + m12 - m20, m12 + m20 - m01], axis=2)   # vértices
#     return list(V.reshape(-1, 3, 2)), e.mean(-1).ravel(), np.repeat(in_om[tw], K // 3)


# [REMOVIDO] panel_polys com a regra dos 50% (polígono inteiro colorido ou cinza: borda da
# região em escada, diferente por arch) e a Bipartite sem subdivisão no fino -- substituída
# pela versão abaixo, que recorta cada polígono pela forma exata da região (region_zone).
# def panel_polys(D, arch, box, fine, m):
#     """O que é desenhado na janela: polígonos [lista de [n,2]], valor [G] e 'colorido' [G].
#     m [Q] bool: pontos de avaliação da arch que pertencem à região da linha (fora de Ω = False).
#     Polígono = triângulo (normal, e a Bipartite nos dois modos), pedaço célula ∩ triângulo
#     (FNO2d fino) ou subtriângulo da quadratura (P1 fino). Colorido se >= 50% do peso dos seus
#     pontos está na região; valor = Σ e·w / Σ w só dos pontos da região."""
#     import shapely
#     xy, tri = D['xy'], D['tri']
#     ex = D['E'][arch]
#     tw = _tris_in_window(xy, tri, box)                       # ordenado (flatnonzero)
#     sel = np.flatnonzero(np.isin(ex['t'], tw))
#     t, e, w, mr = ex['t'][sel], ex['e'][sel], ex['w'][sel], m[sel]
#     if not fine or arch == 'FNO_BipartiteGNN_Elem':
#         grp, polys, G = np.searchsorted(tw, t), list(xy[tri[tw]]), len(tw)
#     elif ex['pset'] == 'overlay':                            # FNO2d: pedaços célula ∩ triângulo
#         nc = len(D['cells'])
#         key, grp = np.unique(t.astype(np.int64) * nc + D['cell'][sel], return_inverse=True)
#         tu, cu = key // nc, key % nc
#         pieces = shapely.intersection(shapely.polygons(xy[tri[tu]]), shapely.polygons(D['cells'][cu]))
#         area = shapely.area(pieces)
#         D.setdefault('_piece_check', []).append(
#             float(np.max(np.abs(area - np.bincount(grp, w)) / area)))
#         polys, G = [np.asarray(p.exterior.coords)[:-1] for p in pieces], len(key)
#     else:                                                    # P1: 16 subtriângulos da quadratura
#         K, bary = D['K'], D['bary']
#         grp = np.searchsorted(tw, sel // K) * (K // 3) + (sel % K) // 3
#         mid = np.einsum('kj,tjd->tkd', bary, xy[tri[tw]]).reshape(len(tw), K // 3, 3, 2)
#         m01, m12, m20 = mid[:, :, 0], mid[:, :, 1], mid[:, :, 2]
#         V = np.stack([m01 + m20 - m12, m01 + m12 - m20, m12 + m20 - m01], axis=2)   # vértices
#         polys, G = list(V.reshape(-1, 3, 2)), len(tw) * (K // 3)
#     den = np.bincount(grp, w, minlength=G)
#     den_r = np.bincount(grp, w * mr, minlength=G)
#     num_r = np.bincount(grp, e * w * mr, minlength=G)
#     on = (den > 0) & (den_r >= 0.5 * den)
#     val = np.divide(num_r, den_r, out=np.zeros(G), where=den_r > 0)
#     return polys, val, on


def region_zone(D, region, box):
    """Forma exata da região na janela (shapely), recortada por Ω: os mesmos conjuntos que o
    regions usa para classificar cada ponto (distância euclidiana exata, d = D['d']):
      0 quinas               = {dist(x, 𝒱_c) <= d}           (discos nas quinas)
      1 fronteira sem quinas = {dist(x, Γ) <= d} − quinas      (faixa nas interfaces)
      2 interior             = resto
      'entreferro'           = anel r_so <= r <= r_ri (raio externo do estator, interno do
                               rotor) ∩ ar."""
    import shapely
    xy, tri, d = D['xy'], D['tri'], D['d']
    x0, x1, y0, y1 = box
    m = 2 * d
    win = shapely.box(x0 - m, y0 - m, x1 + m, y1 + m)
    tw = _tris_in_window(xy, tri, (x0 - m, x1 + m, y0 - m, y1 + m))
    omega = shapely.union_all(shapely.polygons(xy[tri[tw[D['in_omega'][tw]]]]))
    if region == 'entreferro':
        # [REMOVIDO] união dos triângulos de ar pelo centróide (degrau na abertura da ranhura):
        # g = tw[D['gap_tri'][tw]]
        # return shapely.intersection(shapely.union_all(shapely.polygons(xy[tri[g]])), omega)
        o = shapely.Point(0.0, 0.0)
        ring = shapely.difference(o.buffer(D['r_ri'], quad_segs=512), o.buffer(D['r_so'], quad_segs=512))
        air = shapely.union_all(shapely.polygons(xy[tri[tw[D['vac_tri'][tw]]]]))
        return shapely.intersection(shapely.intersection(ring, air), omega)
    C = xy[D['corners']]
    C = C[(C[:, 0] >= x0 - 2 * m) & (C[:, 0] <= x1 + 2 * m) & (C[:, 1] >= y0 - 2 * m) & (C[:, 1] <= y1 + 2 * m)]
    Q = shapely.union_all(shapely.buffer(shapely.points(C), d, quad_segs=32)) if len(C) else shapely.Polygon()
    S = xy[D['iface']]
    near = ((S[..., 0].min(1) <= x1 + 2 * m) & (S[..., 0].max(1) >= x0 - 2 * m)
            & (S[..., 1].min(1) <= y1 + 2 * m) & (S[..., 1].max(1) >= y0 - 2 * m))
    F = shapely.union_all(shapely.buffer(shapely.linestrings(S[near]), d, quad_segs=32))
    z = {0: Q, 1: shapely.difference(F, Q), 2: shapely.difference(win, F)}[region]
    return shapely.intersection(shapely.intersection(z, omega), win)


def _poly_path(g):
    """Path do matplotlib de um Polygon do shapely (com furos)."""
    from matplotlib.path import Path
    verts, codes = [], []
    for ring in (g.exterior, *g.interiors):
        c = np.asarray(ring.coords)
        verts.append(c)
        codes.append([Path.MOVETO] + [Path.LINETO] * (len(c) - 2) + [Path.CLOSEPOLY])
    return Path(np.concatenate(verts), np.concatenate(codes))


def panel_polys(D, arch, box, fine, m, zone):
    """O que é pintado na janela: (paths [lista de Path], valor [G], True [G], área recortada).
    Polígonos: triângulo (normal), pedaço célula ∩ triângulo (FNO2d fino) ou subtriângulo da
    quadratura (P1 e Bipartite no fino: 16 por triângulo, 3 pontos médios cada). Cada polígono
    é recortado pela forma exata da região (zone, region_zone) e só a parte de dentro é
    pintada. Valor = Σ e·w / Σ w dos pontos da região no polígono (m [Q] bool); se nenhum
    ponto do polígono cai na região (lasca na borda), a média de todos os seus pontos."""
    import shapely
    xy, tri = D['xy'], D['tri']
    ex = D['E'][arch]
    tw = _tris_in_window(xy, tri, box)                       # ordenado (flatnonzero)
    sel = np.flatnonzero(np.isin(ex['t'], tw))
    t, e, w, mr = ex['t'][sel], ex['e'][sel], ex['w'][sel], m[sel]
    if not fine:
        grp, geoms, G = np.searchsorted(tw, t), shapely.polygons(xy[tri[tw]]), len(tw)
    elif ex['pset'] == 'overlay':                            # FNO2d: pedaços célula ∩ triângulo
        nc = len(D['cells'])
        key, grp = np.unique(t.astype(np.int64) * nc + D['cell'][sel], return_inverse=True)
        tu, cu = key // nc, key % nc
        geoms = shapely.intersection(shapely.polygons(xy[tri[tu]]), shapely.polygons(D['cells'][cu]))
        area = shapely.area(geoms)
        D.setdefault('_piece_check', []).append(
            float(np.max(np.abs(area - np.bincount(grp, w)) / area)))
        G = len(key)
    else:                                                    # P1 e Bipartite: 16 subtriângulos
        K, bary = D['K'], D['bary']
        grp = np.searchsorted(tw, sel // K) * (K // 3) + (sel % K) // 3
        mid = np.einsum('kj,tjd->tkd', bary, xy[tri[tw]]).reshape(len(tw), K // 3, 3, 2)
        m01, m12, m20 = mid[:, :, 0], mid[:, :, 1], mid[:, :, 2]
        V = np.stack([m01 + m20 - m12, m01 + m12 - m20, m12 + m20 - m01], axis=2)   # vértices
        geoms, G = shapely.polygons(V.reshape(-1, 3, 2)), len(tw) * (K // 3)
    den = np.bincount(grp, w, minlength=G)
    num = np.bincount(grp, e * w, minlength=G)
    den_r = np.bincount(grp, w * mr, minlength=G)
    num_r = np.bincount(grp, e * w * mr, minlength=G)
    val = np.where(den_r > 0, num_r / np.where(den_r > 0, den_r, 1), num / np.where(den > 0, den, 1))
    shapely.prepare(zone)
    hit = np.flatnonzero(shapely.intersects(geoms, zone) & (den > 0))
    clip = shapely.intersection(geoms[hit], zone)
    paths, vals, a_tot = [], [], 0.0
    for g, v in zip(clip, val[hit]):
        for part in shapely.get_parts(g):
            if shapely.get_type_id(part) == 3 and part.area > 0:
                paths.append(_poly_path(part))
                vals.append(v)
                a_tot += part.area
    vals = np.asarray(vals)
    return paths, vals, np.ones(len(vals), bool), a_tot


def row_mask(D, arch, region):
    """Pontos de avaliação da arch na região: int = código de regions.BANDS para o d da
    figura (0 quinas, 1 fronteira sem quinas, 2 interior); 'entreferro' = pontos dos
    triângulos de ar com centróide entre o raio externo do estator e o interno do rotor."""
    ex = D['E'][arch]
    if region == 'entreferro':
        # [REMOVIDO] return D['gap_tri'][ex['t']] & D['in_omega'][ex['t']]
        # pontos dos triângulos de ar (o recorte pelo anel é feito em region_zone)
        return D['vac_tri'][ex['t']] & D['in_omega'][ex['t']]
    return D['code'][ex['pset']] == region


# --------------------------------------------------------------------------------------------
# figura
# --------------------------------------------------------------------------------------------
def _box(win):
    x, y, L = win
    return (x - L / 2, x + L / 2, y - L / 2, y + L / 2)


def _band_boundary(D, box, d):
    """Contorno de {dist(x, Γ) ≤ d} perto da janela (Γ = arestas de interface)."""
    import shapely
    xy, iface = D['xy'], D['iface']
    x0, x1, y0, y1 = box
    S = xy[iface]
    m = 2 * d
    near = ((S[..., 0].min(1) <= x1 + m) & (S[..., 0].max(1) >= x0 - m)
            & (S[..., 1].min(1) <= y1 + m) & (S[..., 1].max(1) >= y0 - m))
    zone = shapely.union_all(shapely.buffer(shapely.linestrings(S[near]), d))
    return shapely.boundary(zone)


def _comma(v, _=None):
    return f'{v:g}'.replace('.', ',')


def _draw_domain(ax, D, win=None, color='#c0392b'):
    """Setor completo (materiais em tons claros, interfaces finas) com o retângulo da
    janela win = (cx, cy, lado) -- coluna de localização da figura."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from matplotlib.tri import Triangulation
    from src.data_gen.motor_constants import MATERIAL_ID
    xy, tri = D['xy'], D['tri']
    col = {MATERIAL_ID['iron_1008']: '#c9ced6', MATERIAL_ID['vacuum']: '#ffffff',
           MATERIAL_ID['N35p']: '#f3c9a8', MATERIAL_ID['copper']: '#e8d9a0'}
    ids = sorted(col)
    code = np.searchsorted(ids, D['elem_mat'])
    ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], tri), facecolors=code.astype(float),
                 cmap=ListedColormap([col[k] for k in ids]), vmin=-0.5, vmax=len(ids) - 0.5,
                 rasterized=True)
    ax.add_collection(LineCollection(xy[D['iface']], colors='k', linewidths=0.15))
    if win is not None:
        x0, x1, y0, y1 = _box(win)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=0.9))
    ax.autoscale()
    ax.set_aspect('equal', adjustable='datalim')    # painel no tamanho cheio (título alinhado)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def render(D, windows=None, fine=False, percentile=99.0, d=None, show_band=False, cmap=CMAP,
           out_dir=ARTICLE_FIG_DIR, name=None):
    """Grade 3 × 5 (linhas = janelas; colunas = domínio com o retângulo da janela + 4 archs),
    uma colorbar por linha (retângulo, sem seta; acima de vmax = cor do topo)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    windows = windows or DEFAULT_WINDOWS
    d = D['d'] if d is None else d
    name = name or (FIG_NAME + ('_fino' if fine else ''))
    archs = [a for a in REGIONS_ARCHS if a in D['E']]
    xy, tri, iface = D['xy'], D['tri'], D['iface']
    report = dict(mode='fine' if fine else 'coarse', percentile=percentile, rows={})

    # layout absoluto (polegadas)
    w_lab, w_cb, w_cbl, gap, gap_cb, h_t, gap_r, h_b = 0.16, 0.07, 0.50, 0.05, 0.06, 0.17, 0.07, 0.05
    # [REMOVIDO] 4 colunas (só as archs) -- agora + coluna do domínio à esquerda
    # pw = (FIG_WIDTH_IN - w_lab - 3 * gap - gap_cb - w_cb - w_cbl) / 4
    pw = (FIG_WIDTH_IN - w_lab - 4 * gap - gap_cb - w_cb - w_cbl) / 5
    ph = pw
    H = h_t + 3 * ph + 2 * gap_r + h_b
    with plt.rc_context():
        apply_style()
        apply_article_style(FONT_PT)
        fig = plt.figure(figsize=(FIG_WIDTH_IN, H))
        for i, rname in enumerate(windows):
            box = _box(windows[rname])
            x0, x1, y0, y1 = box
            # [REMOVIDO] tudo colorido: P = {a: panel_polys(D, a, box, fine) for a in archs}
            reg = ROW_REGION.get(rname, 'entreferro')
            # [REMOVIDO] sem recorte: P = {a: panel_polys(D, a, box, fine, row_mask(D, a, reg)) for a in archs}
            import shapely
            zone = shapely.intersection(region_zone(D, reg, box), shapely.box(x0, y0, x1, y1))
            Pz = {a: panel_polys(D, a, box, fine, row_mask(D, a, reg), zone) for a in archs}
            P = {a: v[:3] for a, v in Pz.items()}
            zone_check = {ARTICLE_ARCH_NAME[a]: float(abs(v[3] / zone.area - 1)) for a, v in Pz.items()}
            vals = np.concatenate([v[om] for _, v, om in P.values()])
            vmax = float(np.percentile(vals, percentile))
            over = {a: int((v[om] > vmax).sum()) for a, (_, v, om) in P.items()}
            norm = Normalize(0.0, vmax)
            yb = H - h_t - (i + 1) * ph - i * gap_r
            tw = _tris_in_window(xy, tri, box)
            ed = np.sort(np.concatenate([tri[tw][:, [0, 1]], tri[tw][:, [1, 2]], tri[tw][:, [2, 0]]]), axis=1)
            mesh_seg = xy[np.unique(ed, axis=0)]
            band = _band_boundary(D, box, d) if show_band else None
            axd = fig.add_axes([w_lab / FIG_WIDTH_IN, yb / H, pw / FIG_WIDTH_IN, ph / H])
            _draw_domain(axd, D, windows[rname])
            if i == 0:
                axd.set_title('Domínio', pad=4)
            for j, a in enumerate(archs):
                polys, v, om = P[a]
                ax = fig.add_axes([(w_lab + (j + 1) * (pw + gap)) / FIG_WIDTH_IN, yb / H,
                                   pw / FIG_WIDTH_IN, ph / H])
                cols = plt.get_cmap(cmap)(norm(v))
                # [REMOVIDO] polígonos inteiros, cinza fora da região/Ω:
                # cols[~om] = plt.matplotlib.colors.to_rgba(OUT_OMEGA_COLOR)
                # ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols,
                #                                  linewidths=0.1))
                ax.set_facecolor(OUT_OMEGA_COLOR)            # fora da região (e de Ω): cinza
                ax.add_collection(PathCollection(polys, facecolors=cols, edgecolors=cols,
                                                 linewidths=0.1))
                ax.add_collection(LineCollection(mesh_seg, colors='0.5', linewidths=0.2, alpha=0.4))
                ax.add_collection(LineCollection(xy[iface], colors='k', linewidths=0.6))
                if band is not None:
                    import shapely
                    for g in getattr(band, 'geoms', [band]):
                        c = np.asarray(g.coords) if not g.is_empty else None
                        if c is not None:
                            ax.plot(c[:, 0], c[:, 1], ls=(0, (2, 1.5)), color='w', lw=0.5)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.5)
                if i == 0:
                    ax.set_title(ARTICLE_ARCH_NAME[a], pad=4)
            fig.text(0.5 * w_lab / FIG_WIDTH_IN, (yb + ph / 2) / H, rname, rotation=90,
                     ha='center', va='center')
            cax = fig.add_axes([(w_lab + 5 * pw + 4 * gap + gap_cb) / FIG_WIDTH_IN, yb / H,
                                w_cb / FIG_WIDTH_IN, ph / H])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            # [REMOVIDO] seta extend='max' quando havia valor acima de vmax -- escala só retângulo
            # cb = fig.colorbar(sm, cax=cax, extend='max' if any(over.values()) else 'neither')
            cb = fig.colorbar(sm, cax=cax)
            cb.ax.yaxis.set_major_locator(MaxNLocator(5))
            cb.ax.yaxis.set_major_formatter(FuncFormatter(_comma))
            cb.ax.grid(False)
            cb.outline.set_linewidth(0.5)
            cb.set_label(CB_LABEL)
            report['rows'][rname] = dict(
                window=dict(cx=windows[rname][0], cy=windows[rname][1], side=windows[rname][2]),
                vmax_T=vmax, n_values=int(len(vals)),
                above_vmax={ARTICLE_ARCH_NAME[a]: over[a] for a in archs},
                n_polys={ARTICLE_ARCH_NAME[a]: int(len(P[a][1])) for a in archs},
                max_T={ARTICLE_ARCH_NAME[a]: float(P[a][1][P[a][2]].max()) for a in archs},
                region=str(reg), zone_area_mm2=float(zone.area),
                zone_area_rel_diff=zone_check)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f'{name}.pdf')
        fig.savefig(out_dir / f'{name}.png', dpi=300)
        plt.close(fig)
    report['figure'] = str(out_dir / f'{name}.pdf')
    report['size_in'] = [FIG_WIDTH_IN, H]
    return report


def render_locator(D, windows=None, out_dir=ARTICLE_FIG_DIR, name=FIG_NAME + '_localizacao'):
    """Painel auxiliar: setor completo (materiais em tons claros, interfaces) com os 3
    retângulos de zoom rotulados."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from matplotlib.tri import Triangulation
    from src.data_gen.motor_constants import MATERIAL_ID
    windows = windows or DEFAULT_WINDOWS
    xy, tri = D['xy'], D['tri']
    col = {MATERIAL_ID['iron_1008']: '#c9ced6', MATERIAL_ID['vacuum']: '#ffffff',
           MATERIAL_ID['N35p']: '#f3c9a8', MATERIAL_ID['copper']: '#e8d9a0'}
    ids = sorted(col)
    code = np.searchsorted(ids, D['elem_mat'])
    with plt.rc_context():
        apply_style()
        apply_article_style(FONT_PT)
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        ax.tripcolor(Triangulation(xy[:, 0], xy[:, 1], tri), facecolors=code.astype(float),
                     cmap=ListedColormap([col[k] for k in ids]), vmin=-0.5, vmax=len(ids) - 0.5,
                     rasterized=True)
        ax.add_collection(LineCollection(xy[D['iface']], colors='k', linewidths=0.2))
        r_out = float(np.hypot(xy[:, 0], xy[:, 1]).max())
        for k, (rname, win) in enumerate(windows.items()):
            x0, x1, y0, y1 = _box(win)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='#c0392b', lw=0.9))
            # rótulo radial, para fora do rotor, com linha de chamada a partir da janela
            th = np.arctan2(win[1], win[0])
            u = np.array([np.cos(th), np.sin(th)])
            r_c = float(np.hypot(win[0], win[1]))
            p0, p1 = u * (r_c + win[2] / 2), u * (r_out + 2.0)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='#c0392b', lw=0.6)
            ax.text(*(u * (r_out + 2.6)), rname, color='#c0392b', rotation=np.degrees(th),
                    rotation_mode='anchor', ha='left', va='center')
        ax.set_aspect('equal')
        ax.autoscale()
        ax.axis('off')
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(out_dir / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    return str(out_dir / f'{name}.pdf')


def make_figure(sample_idx=2184, windows=None, d=None, percentile=99.0, fine=False,
                show_band=False, loss='mae', D=None):
    """Figura (1 modo) + painel de localização; D: saída de compute_errors (reaproveitada)."""
    D = D if D is not None else compute_errors(sample_idx, loss)
    rep = render(D, windows, fine=fine, percentile=percentile, d=d, show_band=show_band)
    rep['locator'] = render_locator(D, windows)
    return D, rep


def _print_windows(windows):
    print('janelas (centro x, y e lado em mm; polar do centro):')
    for k, (x, y, L) in windows.items():
        print(f'  {k:10s} cx = {x:7.3f}  cy = {y:7.3f}  lado = {L:.2f}   '
              f'(r = {np.hypot(x, y):.3f} mm, θ = {np.degrees(np.arctan2(y, x)):.2f}°)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sample', type=int, default=2184)
    ap.add_argument('--loss', default='mae')
    ap.add_argument('--mode', choices=('both', 'coarse', 'fine'), default='both')
    ap.add_argument('--band', action='store_true', help='contorno da faixa dist(Γ) <= d, tracejado')
    ap.add_argument('--d', type=float, default=REGIONS_D_MM[0])
    ap.add_argument('--percentile', type=float, default=99.0)
    args = ap.parse_args()

    print(f'amostra {args.sample} ({args.loss}); d = {args.d:.4f} mm; percentil {args.percentile:g}')
    _print_windows(DEFAULT_WINDOWS)
    D = compute_errors(args.sample, args.loss, args.d)
    out = dict(sample_idx=args.sample, loss=args.loss, d_mm=args.d, band=args.band,
               runs={ARTICLE_ARCH_NAME[a]: dict(run_dir=v['run'], epoch=v['epoch'])
                     for a, v in D['E'].items()}, modes={})
    modes = {'both': (False, True), 'coarse': (False,), 'fine': (True,)}[args.mode]
    for fine in modes:
        _, rep = make_figure(args.sample, DEFAULT_WINDOWS, args.d, args.percentile, fine,
                             args.band, args.loss, D=D)
        out['modes'][rep['mode']] = rep
        print(f"\n[{rep['mode']}] {rep['figure']} ({rep['size_in'][0]:.2f} × {rep['size_in'][1]:.2f} in)")
        for rname, x in rep['rows'].items():
            print(f"  {rname:10s} vmax = {x['vmax_T']:.4f} T (p{args.percentile:g} de {x['n_values']} valores); "
                  f"acima: {x['above_vmax']}; máx: " + ', '.join(f'{k} {v:.3f}' for k, v in x['max_T'].items()))
    if D.get('_piece_check'):
        print(f"\nconferência dos pedaços do FNO2d: max |área(pedaço) − Σw| / área = {max(D['_piece_check']):.1e}")
    print(f"localização: {out['modes'][next(iter(out['modes']))]['locator']}")
    jd = OUT_ROOT / 'dados' / 'fig_erro_zoom'
    jd.mkdir(parents=True, exist_ok=True)
    (jd / f'sample_{args.sample:06d}_{args.loss}.json').write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
