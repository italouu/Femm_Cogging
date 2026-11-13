"""Figura do artigo, seção "Representações de entrada": a mesma janela da mesma amostra (padrão
#2184) nas 4 representações de entrada, painéis (a)–(d):

  (a) FNO2d                  grade polar centrada em células (x_hw), célula pintada pelo
                             material do seu μ_r/M (elemento que contém o centro do pixel);
  (b) FNO_GNN                grafo de vértices: nós nos vértices da malha, material do nó
                             (node_x: μ_r, M = prioridade ferro > ímã > cobre > ar), arestas
                             da malha (edge_index);
  (c) FNO_Bipartite          layout Elem: nós principais nos centróides (node_x: μ_r, M do
                             elemento), arestas duais (edge_index), vértices auxiliares sem
                             material (elem_x só tem posição) e arestas cruzadas vértice →
                             elemento (cross_edge_index);
  (d) FNO_SplitGNN           N_aux nos centróides (node_x), Edge_dual (edge_index), cópias de
                             vértice N_base, 1 por classe (material, M) (elem_x: μ_r, M), e
                             Edge_aux de cada elemento até a cópia do seu material
                             (cross_edge_index).

O material de cada marcador vem das features de entrada do próprio layout (μ_r e M), não de
um recálculo. Conferência: as mesmas classes recalculadas do .ans (elem_mat, node_mat e o
elemento que contém o centro do pixel) -- resultado impresso e salvo no JSON.

No painel (d) as 2 ou 3 cópias de um vértice de interface ocupam a mesma posição: cada cópia
é deslocada de COPY_SHIFT (fração) da distância vértice → centróide médio dos elementos do
seu material, só nos vértices com mais de uma cópia (as cópias únicas ficam no vértice).

Malha do FEMM ao fundo (cinza claro) e interfaces entre classes de material (preto) em todos
os painéis. Cores: Okabe–Ito, distinguíveis em tons de cinza.

Saída: docs/figures/representacoes_entrada{,_sem_titulo}.{pdf,png} (18 cm, 8 pt) e
resultados/BLDC_138x276/dados/fig_representacoes/sample_XXXXXX.json (janela, materiais,
contagens, deslocamento, conferência).

    python -m scripts.fig_representacoes [--sample 2184] [--center X Y] [--side MM]
                                         [--layout 2x2|1x4]
"""
import argparse
import json

import numpy as np

from scripts.postprocess import ARTICLE_FIG_DIR, OUT_ROOT, design_rows, sample_path
from src.data_gen.motor_constants import MATERIAL_ID, PERMEABILITY
from src.data_gen.parsers.ans_parsing import read_ans
from src.data_gen.parsers.split_layout import parse_split_smooth
from src.postprocess.metrics import _cell_corners
from src.postprocess.plots.primitives import apply_article_style, apply_style
from src.postprocess.regions import interface_edges
from src.postprocess.samples import ANG_1, ANG_2, N_A, N_R, parse_smooth

FIG_NAME = 'representacoes_entrada'
FIG_WIDTH_IN = 18 / 2.54          # figure* (duas colunas)
FONT_PT = 8
# Janela padrão: abertura da ranhura em θ = 90° (direção radial = vertical na figura), na
# #2184. Abertura de 89,16° a 90,84° (Bs0 = 1,105 mm), de r = 37,514 a 38,614 mm; cobre
# abaixo de 37,514 mm; entreferro acima de 38,614 mm. Contém ferro (2 pontas de dente), ar
# (abertura + entreferro), cobre, as 2 quinas da sapata (r = 38,614) e os 2 pontos triplos
# ferro/ar/cobre (r = 37,514). O ímã (r >= 39,925) não cabe junto com o cobre num lado <= 2,5 mm.
# [REMOVIDO] janela inteira da abertura (2 dentes, 2,2 mm): nós demais (649 vértices, 1292
# elementos) e as quinas refinadas viram borrões -- trocada por meia abertura (1 dente).
# DEFAULT_CENTER = (0.0, 38.10)
# DEFAULT_SIDE_MM = 2.2
# Meia abertura: ponta do dente esquerdo (θ = 90,84°, x ≈ −0,56 mm) com a quina da sapata
# (r = 38,614), o ponto triplo ferro/ar/cobre (r = 37,514), a abertura, o cobre e o entreferro.
# [REMOVIDO] meia abertura com 1,6 mm -- zoom aumentado na quina a pedido:
# DEFAULT_CENTER = (-0.45, 37.95)
# DEFAULT_SIDE_MM = 1.6
# Zoom no ponto triplo ferro/ar/cobre da ponta do dente esquerdo (vértice 24668 da #2184 em
# (−0,552; 37,510) mm, r = 37,514, θ = 90,84°): quina da interface (canto da abertura com a
# diagonal da cunha) e o único tipo de vértice com 3 cópias no Split.
# [REMOVIDO] zoom no ponto triplo (0,5 mm): malha refinada demais (centenas de nós) para uma
# figura que é só representação -- trocado por um trecho reto de interface com malha grossa.
# DEFAULT_CENTER = (-0.52, 37.54)
# DEFAULT_SIDE_MM = 0.5
# Flanco do dente dentro da ranhura (interface ferro/cobre, longe das quinas), r = 34,1 mm,
# θ = 58,0°: malha grossa (14 vértices, 29 elementos na janela, menor aresta local ~0,24 mm)
# e ~180 células da grade. [antes: (18.42, 29.54), com um trecho refinado no canto]
DEFAULT_CENTER = (18.07, 28.94)
DEFAULT_SIDE_MM = 2.5
# [REMOVIDO] limites absolutos (bons só para ~1,6 mm):
# MARK_FRAC, MARK_MIN_MM, MARK_MAX_MM = 0.40, 0.006, 0.045
# diâmetro do marcador = MARK_FRAC × tamanho local da malha, limitado a
# [MARK_MIN_FRAC, MARK_MAX_FRAC] × lado da janela
MARK_FRAC, MARK_MIN_FRAC, MARK_MAX_FRAC = 0.40, 0.0038, 0.028
COPY_SHIFT = 0.20                 # (d): fração da distância vértice → centróide médio do material

# classes de material desenhadas (código, rótulo, cor) -- Okabe–Ito; cinza ≈ 87/218/162/106/188
CLASSES = {
    'ferro': ('Ferro', '#0072B2'),
    'ar': ('Ar', '#BFE3F6'),
    'cobre': ('Cobre', '#E69F00'),
    'ima+': (r'Ímã ($M>0$)', '#009E73'),
    'ima-': (r'Ímã ($M<0$)', '#8FD3BC'),
}
CLS_ORDER = list(CLASSES)
# [REMOVIDO] arestas em tons de cinza próximos (difíceis de distinguir):
# MESH_COLOR, MESH_LW = '0.80', 0.25
# IFACE_COLOR, IFACE_LW = 'k', 0.8
# GRAPH_EDGE_COLOR, GRAPH_EDGE_LW = '0.45', 0.45
# CROSS_EDGE_COLOR, CROSS_EDGE_LW = '0.55', 0.4
# NEUTRAL_FACE, NEUTRAL_EDGE = 'white', '0.45'
# [REMOVIDO] arestas mais claras (sem o fundo de material):
# MESH_COLOR, MESH_LW = '0.80', 0.5
# IFACE_COLOR, IFACE_LW = 'k', 2.2
# GRAPH_EDGE_COLOR, GRAPH_EDGE_LW = '0.30', 1.0
# CROSS_EDGE_COLOR, CROSS_EDGE_LW = '#CC79A7', 1.0
MESH_COLOR, MESH_LW = '0.62', 0.6                           # malha: cinza médio, fina
IFACE_COLOR, IFACE_LW = 'k', 2.2                            # interface: preta, grossa
GRAPH_EDGE_COLOR, GRAPH_EDGE_LW = '0.15', 1.2               # grafo: quase preto, sólida
CROSS_EDGE_COLOR, CROSS_EDGE_LW = '#B0407F', 1.1            # cruzada / Edge_aux: rosa escuro, tracejada
# fundo dos painéis de grafo (b–d): material de cada elemento (classe das features da Elem)
# em tons muito fracos das mesmas cores
CELL_INSET, CELL_LW = 0.10, 0.7      # (a): recuo do contorno da célula (fração) e espessura
FAINT = {'ferro': '#E9F2F9', 'ar': '#F5FAFD', 'cobre': '#FFF8E3', 'ima+': '#E8F6F1', 'ima-': '#F2FAF7'}
CROSS_EDGE_LS = (0, (3, 2))
NEUTRAL_FACE, NEUTRAL_EDGE = 'white', '0.55'
# todos os nós circulares: elemento = disco cheio; vértice / cópia = anel na cor do material;
# vértice sem material (c) = anel cinza fino. Diâmetros em fração do lado da janela.
D_ELEM_FRAC, D_VERT_FRAC = 0.040, 0.046
RING_LW, RING_LW_NEUTRAL, DISC_EDGE_LW = 1.8, 0.9, 0.4
TITLES = ('Grade', 'Grafo de vértices', 'Elementos + vértices', 'Elementos + vértices por material')
ARCH_OF_PANEL = ('FNO2d', 'FNO_GNN', 'FNO_Bipartite', 'FNO_SplitGNN')


# --------------------------------------------------------------------------------------------
# classes de material a partir das features (μ_r, M) e do .ans
# --------------------------------------------------------------------------------------------
def cls_from_features(mu, M):
    """Classe de cada entrada a partir de μ_r e M (features de entrada): índice em CLS_ORDER."""
    mu, M = np.asarray(mu, np.float64), np.asarray(M, np.float64)
    out = np.full(mu.shape, -1, np.int64)
    out[mu > 100] = CLS_ORDER.index('ferro')
    out[np.isclose(mu, PERMEABILITY['vacuum'], atol=1e-5)] = CLS_ORDER.index('ar')
    out[np.isclose(mu, PERMEABILITY['copper'], atol=1e-5)] = CLS_ORDER.index('cobre')
    mag = np.isclose(mu, PERMEABILITY['N35p'], atol=1e-4)
    out[mag & (M > 0)] = CLS_ORDER.index('ima+')
    out[mag & (M <= 0)] = CLS_ORDER.index('ima-')
    return out


def cls_from_mat(mat_id, M):
    """Mesma classe a partir do material_id do .ans (MATERIAL_ID) e do sinal de M."""
    inv = {MATERIAL_ID['iron_1008']: 'ferro', MATERIAL_ID['vacuum']: 'ar',
           MATERIAL_ID['copper']: 'cobre'}
    mat_id, M = np.asarray(mat_id), np.asarray(M, np.float64)
    out = np.array([CLS_ORDER.index(inv[m]) if m in inv else -1 for m in mat_id], np.int64)
    mag = mat_id == MATERIAL_ID['N35p']
    out[mag & (M > 0)] = CLS_ORDER.index('ima+')
    out[mag & (M <= 0)] = CLS_ORDER.index('ima-')
    return out


# --------------------------------------------------------------------------------------------
# dados (1 amostra)
# --------------------------------------------------------------------------------------------
def load_layouts(sample_idx):
    """Layouts de entrada da amostra (os mesmos dos datasets): grade, vértices, Elem e Split,
    mais malha (xy, tri) e os materiais do .ans para a conferência."""
    row = design_rows()[sample_idx]
    p = sample_path(sample_idx)
    ans = read_ans(p)
    lay = parse_smooth(p, row, with_elem=True, ans=ans)
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    split = parse_split_smooth(p, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2, n_r=N_R, n_a=N_A, ans=ans)
    xy = np.stack([ans.mesh.x, ans.mesh.y], axis=1).astype(np.float64)
    return dict(sample_idx=sample_idx, row=row, xy=xy, tri=lay['tri'], cent=xy[lay['tri']].mean(1),
                elem_mat=lay['elem_mat'], node_mat=lay['node_mat'], vertex=lay['vertex'],
                elem=lay['elem'], split=split, r_in=r_in, r_ext=r_ext)


def build(L, box):
    """Tudo o que é desenhado, com as classes vindas das features, e a conferência."""
    from matplotlib.tri import Triangulation
    xy, tri, cent = L['xy'], L['tri'], L['cent']
    x0, x1, y0, y1 = box
    M_el = L['elem']['node_x'][:, 3].astype(np.float64)         # M do elemento (Elem: col 3)

    def inside(P, m=0.0):
        return (P[:, 0] >= x0 - m) & (P[:, 0] <= x1 + m) & (P[:, 1] >= y0 - m) & (P[:, 1] <= y1 + m)

    side = max(x1 - x0, y1 - y0)
    mg = 0.25 * side                                           # margem: o que cruza a borda
    chk = {}

    # (a) grade: x_hw [3,H,W] = μ_r, M, J; cantos retos (_cell_corners), row-major (r, θ)
    xh = L['vertex']['x_hw']
    cells = _cell_corners(L['r_in'], L['r_ext'], np.deg2rad(ANG_1), np.deg2rad(ANG_2), N_R, N_A)
    cc = cells.mean(1)
    # [REMOVIDO] margem só da janela -- em janelas pequenas faltavam células na borda:
    # sel_c = np.flatnonzero(inside(cc, mg))
    sel_c = np.flatnonzero(inside(cc, mg + 0.4))               # + ~1 célula (Δθ·r ≈ 0,29 mm)
    c_cls = cls_from_features(xh[0].ravel()[sel_c], xh[1].ravel()[sel_c])
    # conferência: elemento que contém o centro do pixel (centro polar, como _grid_polar_xy)
    H, W = N_R, N_A
    ii, jj = np.unravel_index(sel_c, (H, W))
    rr = L['r_in'] + (ii + 0.5) * (L['r_ext'] - L['r_in']) / H
    aa = np.deg2rad(ANG_1 + (jj + 0.5) * (ANG_2 - ANG_1) / W)
    t = Triangulation(xy[:, 0], xy[:, 1], tri).get_trifinder()(rr * np.cos(aa), rr * np.sin(aa))
    ok_t = t >= 0
    ref = cls_from_mat(L['elem_mat'][t[ok_t]], M_el[t[ok_t]])
    chk['a_grade'] = dict(n=int(ok_t.sum()), n_diff=int((ref != c_cls[ok_t]).sum()),
                          n_fora_malha=int((~ok_t).sum()),
                          regra='classe(μ_r, M de x_hw) × classe(elem_mat, M) do elemento que '
                                'contém o centro polar do pixel')

    # (b) vértices: node_x = μ_r, M, área dual, r_base, c_base, J
    nx = L['vertex']['node_x']
    v_cls_all = cls_from_features(nx[:, 0], nx[:, 1])
    # M do nó: polaridade dos ímãs incidentes; para a referência, o M só importa no ímã
    ref_v = cls_from_mat(L['node_mat'], nx[:, 1])
    chk['b_vertices'] = dict(n=int(len(nx)), n_diff=int((ref_v != v_cls_all).sum()),
                             regra='classe(μ_r, M de node_x) × node_mat (prioridade do .ans)')
    ei = L['vertex']['edge_index']
    e_b = np.unique(np.sort(ei.T, axis=1), axis=0)
    e_b = e_b[np.linalg.norm(xy[e_b[:, 0]] - xy[e_b[:, 1]], axis=1) > 0]   # tira o wrap (d = 0)

    # (c) Elem: node_x = r_base, c_base, μ_r, M, área, J (elementos); elem_x = vértices (só posição)
    enx = L['elem']['node_x']
    e_cls_all = cls_from_features(enx[:, 2], enx[:, 3])
    ref_e = cls_from_mat(L['elem_mat'], enx[:, 3])
    chk['c_elem'] = dict(n=int(len(enx)), n_diff=int((ref_e != e_cls_all).sum()),
                         elem_x_cols=int(L['elem']['elem_x'].shape[1]),
                         regra='classe(μ_r, M de node_x da Elem) × elem_mat; elem_x (vértices) '
                               'só tem r_base, c_base: sem material')
    de = np.unique(np.sort(L['elem']['edge_index'].T, axis=1), axis=0)
    cr_c = L['elem']['cross_edge_index'].T                     # [3M, 2]: vértice, elemento

    # (d) Split: node_x = r, c, sin, cos, μ_r, M, J (N_aux); elem_x idem (N_base)
    sp = L['split']
    s_aux = cls_from_features(sp['node_x'][:, 4], sp['node_x'][:, 5])
    s_cp = cls_from_features(sp['elem_x'][:, 4], sp['elem_x'][:, 5])
    ce = sp['cross_edge_index']                                # linha 0 = cópia, 1 = elemento
    Mt = len(tri)
    assert np.array_equal(ce[1].reshape(Mt, 3), np.repeat(np.arange(Mt)[:, None], 3, 1))
    copy_of = ce[0].reshape(Mt, 3)                             # cópia do canto j do elemento e
    K = len(sp['elem_x'])
    copy_v = np.full(K, -1, np.int64)
    copy_v[copy_of.ravel()] = tri.ravel()                      # vértice original da cópia
    # média dos centróides dos elementos ligados a cada cópia (= elementos do seu material)
    nb = np.bincount(copy_of.ravel(), minlength=K)
    mc = np.stack([np.bincount(copy_of.ravel(), np.repeat(cent[:, k], 3), minlength=K)
                   for k in (0, 1)], 1) / np.maximum(nb, 1)[:, None]
    n_cp_v = np.bincount(copy_v, minlength=len(xy))
    multi = n_cp_v[copy_v] > 1
    pos_cp = xy[copy_v].copy()
    pos_cp[multi] += COPY_SHIFT * (mc[multi] - xy[copy_v][multi])
    ref_cp = cls_from_mat(L['elem_mat'][np.arange(Mt).repeat(3)], M_el[np.arange(Mt).repeat(3)])
    d_cp = int((s_cp[copy_of.ravel()] != ref_cp).sum())
    chk['d_split'] = dict(
        n_aux=int(len(s_aux)), n_aux_diff=int((s_aux != ref_e).sum()),
        n_edge_aux=int(copy_of.size), n_edge_aux_diff=d_cp,
        n_aux_vs_elem_layout_diff=int((s_aux != e_cls_all).sum()),
        regra='classe(μ_r, M de node_x do Split) × elem_mat; classe da cópia (elem_x) × '
              'classe do elemento de cada Edge_aux (a cópia é a do material do elemento)')

    # seleção na janela (com margem, para o que cruza a borda)
    vin = np.flatnonzero(inside(xy, mg))
    ein = np.flatnonzero(inside(cent, mg))
    seg_in = lambda A, B: inside(A, mg) | inside(B, mg)
    mesh_e = np.unique(np.sort(np.concatenate([tri[ein][:, [0, 1]], tri[ein][:, [1, 2]],
                                               tri[ein][:, [2, 0]]]), axis=1), axis=0)
    e_cls_mat = cls_from_mat(L['elem_mat'], M_el)
    iface = interface_edges(tri, e_cls_mat)                    # entre classes (inclui polaridade)
    iface = iface[seg_in(xy[iface[:, 0]], xy[iface[:, 1]])]
    eb = e_b[seg_in(xy[e_b[:, 0]], xy[e_b[:, 1]])]
    dd = de[seg_in(cent[de[:, 0]], cent[de[:, 1]])]
    crc = cr_c[seg_in(xy[cr_c[:, 0]], cent[cr_c[:, 1]])]
    cps = np.flatnonzero(inside(pos_cp, mg))
    ea = np.stack([copy_of.ravel(), np.arange(Mt).repeat(3)], 1)
    ea = ea[seg_in(pos_cp[ea[:, 0]], cent[ea[:, 1]])]

    # contagens dentro da janela exata
    vin0, ein0 = inside(xy), inside(cent)
    mid = lambda A, B: inside(0.5 * (A + B))
    v_iface = np.zeros(len(xy), bool)
    v_iface[interface_edges(tri, e_cls_mat).ravel()] = True
    nv = n_cp_v[vin0]
    counts = dict(
        a_celulas=int(inside(cc).sum()),
        a_celulas_por_classe={CLASSES[CLS_ORDER[k]][0]: int(v) for k, v in
                              zip(*np.unique(cls_from_features(xh[0].ravel()[inside(cc)],
                                                               xh[1].ravel()[inside(cc)]),
                                             return_counts=True))},
        b_nos=int(vin0.sum()), b_nos_interface=int((vin0 & v_iface).sum()),
        b_arestas=int(mid(xy[e_b[:, 0]], xy[e_b[:, 1]]).sum()),
        c_nos_elemento=int(ein0.sum()), c_vertices_aux=int(vin0.sum()),
        c_arestas_duais=int(mid(cent[de[:, 0]], cent[de[:, 1]]).sum()),
        c_arestas_cruzadas=int(mid(xy[cr_c[:, 0]], cent[cr_c[:, 1]]).sum()),
        d_n_aux=int(ein0.sum()), d_copias=int(inside(xy[copy_v]).sum()),
        d_vertices_1_copia=int((nv == 1).sum()), d_vertices_2_copias=int((nv == 2).sum()),
        d_vertices_3_copias=int((nv == 3).sum()), d_vertices_4plus=int((nv >= 4).sum()),
        d_edge_dual=int(mid(cent[de[:, 0]], cent[de[:, 1]]).sum()),
        d_edge_aux=int(mid(xy[copy_v[ea[:, 0]]], cent[ea[:, 1]]).sum() if False else
                       mid(xy[copy_v[copy_of.ravel()]], cent[np.arange(Mt).repeat(3)]).sum()),
        malha_triangulos=int(ein0.sum()),
    )
    present = sorted({CLASSES[CLS_ORDER[k]][0] for k in e_cls_mat[ein0] if k >= 0})
    # tamanho local da malha [mm]: vértice = média das arestas incidentes; elemento = √(2·área)
    ed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    ln = np.linalg.norm(xy[ed[:, 0]] - xy[ed[:, 1]], axis=1)
    h_v = (np.bincount(ed.ravel(), np.repeat(ln, 2), minlength=len(xy))
           / np.maximum(np.bincount(ed.ravel(), minlength=len(xy)), 1))
    P = xy[tri]
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    h_e = np.sqrt(2 * area)
    return dict(box=box, cells=cells[sel_c], tri_poly=xy[tri[ein]], tri_cls=e_cls_all[ein], v_h=h_v[vin], e_h=h_e[ein], cp_h=h_v[copy_v[cps]], c_cls=c_cls, mesh_seg=xy[mesh_e], iface_seg=xy[iface],
                v_xy=xy[vin], v_cls=v_cls_all[vin], b_seg=xy[eb],
                e_xy=cent[ein], e_cls=e_cls_all[ein], d_seg=cent[dd],
                c_seg=np.stack([xy[crc[:, 0]], cent[crc[:, 1]]], 1),
                cp_xy=pos_cp[cps], cp_cls=s_cp[cps],
                ea_seg=np.stack([pos_cp[ea[:, 0]], cent[ea[:, 1]]], 1),
                n_aux_xy=cent[ein], n_aux_cls=s_aux[ein],
                check=chk, counts=counts, materiais=present)


# --------------------------------------------------------------------------------------------
# figura
# --------------------------------------------------------------------------------------------
def _style_ax(ax, box):
    x0, x1, y0, y1 = box
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def render(F, layout='2x2', titles=True, out_dir=ARTICLE_FIG_DIR, name=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    name = name or (FIG_NAME + ('' if titles else '_sem_titulo'))
    col = np.array([plt.matplotlib.colors.to_rgba(CLASSES[k][1]) for k in CLS_ORDER])
    box = F['box']
    side = box[1] - box[0]
    nr, nc = (2, 2) if layout == '2x2' else (1, 4)
    # [REMOVIDO] gap, h_t, h_leg = 0.08, (0.16 if titles else 0.02), 0.62  (legenda com 3 linhas)
    gap, h_t, h_leg = 0.08, (0.16 if titles else 0.02), 0.80
    pw = (FIG_WIDTH_IN - (nc - 1) * gap - 0.04) / nc
    ph = pw * (box[3] - box[2]) / side
    h_lab = 0.16                                               # rótulo (a)–(d) abaixo
    H = nr * (h_t + ph + h_lab) + h_leg
    # tamanho dos marcadores proporcional à malha local (pt por mm)
    pt_mm = pw * 72 / side
    # [REMOVIDO] tamanho fixo -- embolava nas quinas refinadas:
    # s_v = (0.055 * pt_mm) ** 2
    # s_e = (0.050 * pt_mm) ** 2
    # [REMOVIDO] diâmetro ∝ malha local (para janelas com malha muito refinada):
    # size = lambda h: (np.clip(MARK_FRAC * h, MARK_MIN_FRAC * side, MARK_MAX_FRAC * side) * pt_mm) ** 2
    # s_v = (MARK_MAX_FRAC * side * 0.8 * pt_mm) ** 2
    # s_e = s_v
    # lw_m = 0.3
    s_v = (D_VERT_FRAC * side * pt_mm) ** 2                    # anel (vértice / cópia)
    s_e = (D_ELEM_FRAC * side * pt_mm) ** 2                    # disco (elemento)
    size = lambda h: None
    lw_m = DISC_EDGE_LW
    ring = dict(s=s_v, marker='o', facecolors='white', linewidths=RING_LW, zorder=5)
    disc = dict(s=s_e, marker='o', edgecolors='k', linewidths=DISC_EDGE_LW, zorder=4)
    with plt.rc_context():
        apply_style()
        apply_article_style(FONT_PT)
        fig = plt.figure(figsize=(FIG_WIDTH_IN, H))
        axs = []
        for k in range(4):
            i, j = divmod(k, nc)
            left = 0.02 + j * (pw + gap)
            bottom = H - (i + 1) * (h_t + ph + h_lab) + h_lab
            ax = fig.add_axes([left / FIG_WIDTH_IN, bottom / H, pw / FIG_WIDTH_IN, ph / H])
            _style_ax(ax, box)
            ax.add_collection(LineCollection(F['mesh_seg'], colors=MESH_COLOR, linewidths=MESH_LW,
                                             zorder=1))
            if titles:
                ax.set_title(TITLES[k], pad=3)
            ax.text(0.5, -0.02, f'({"abcd"[k]}) {ARCH_OF_PANEL[k]}',
                    transform=ax.transAxes, ha='center', va='top')
            axs.append(ax)
        a, b, c, d = axs
        # [REMOVIDO] células cheias na cor forte (confundia a discretização com a geometria):
        # a.add_collection(PolyCollection(F['cells'], facecolors=col[F['c_cls']], edgecolors='white',
        #                                 linewidths=0.3, zorder=0))
        # a.add_collection(LineCollection(F['mesh_seg'], colors='0.35', linewidths=MESH_LW,
        #                                 alpha=0.6, zorder=2))
        # (a) fundo = material real (tom fraco, como b–d); cada célula só com o contorno fino na cor
        # forte do SEU material (x_hw), recuado para dentro (CELL_INSET) para que células vizinhas
        # não dividam a borda: a célula que atravessa a interface mostra a cor errada sobre o fundo
        cc = F['cells'].mean(1, keepdims=True)
        inset = cc + (1 - CELL_INSET) * (F['cells'] - cc)
        a.add_collection(PolyCollection(inset, facecolors='none', edgecolors=col[F['c_cls']],
                                        linewidths=CELL_LW, zorder=2))
        # [REMOVIDO] marcadores antigos (vértice = círculo cheio, elemento = quadrado):
        # # (b) grafo de vértices
        # b.add_collection(LineCollection(F['b_seg'], colors=GRAPH_EDGE_COLOR,
        #                                 linewidths=GRAPH_EDGE_LW, zorder=2))
        # b.scatter(*F['v_xy'].T, s=size(F['v_h']), c=col[F['v_cls']], marker='o', edgecolors='k',
        #           linewidths=lw_m, zorder=4)
        # # (c) Elem: duais + cruzadas + vértices neutros + elementos
        # c.add_collection(LineCollection(F['c_seg'], colors=CROSS_EDGE_COLOR, linewidths=CROSS_EDGE_LW,
        #                                 linestyles=(0, (2, 1.5)), zorder=2))
        # c.add_collection(LineCollection(F['d_seg'], colors=GRAPH_EDGE_COLOR, linewidths=GRAPH_EDGE_LW,
        #                                 zorder=2))
        # c.scatter(*F['v_xy'].T, s=size(F['v_h']), c=NEUTRAL_FACE, marker='o', edgecolors=NEUTRAL_EDGE,
        #           linewidths=lw_m, zorder=4)
        # c.scatter(*F['e_xy'].T, s=size(F['e_h']), c=col[F['e_cls']], marker='s', edgecolors='k',
        #           linewidths=lw_m, zorder=4)
        # # (d) Split: Edge_dual + Edge_aux + cópias + N_aux
        # d.add_collection(LineCollection(F['ea_seg'], colors=CROSS_EDGE_COLOR, linewidths=CROSS_EDGE_LW,
        #                                 linestyles=(0, (2, 1.5)), zorder=2))
        # d.add_collection(LineCollection(F['d_seg'], colors=GRAPH_EDGE_COLOR, linewidths=GRAPH_EDGE_LW,
        #                                 zorder=2))
        # d.scatter(*F['cp_xy'].T, s=size(F['cp_h']), c=col[F['cp_cls']], marker='o', edgecolors='k',
        #           linewidths=lw_m, zorder=4)
        # d.scatter(*F['n_aux_xy'].T, s=size(F['e_h']), c=col[F['n_aux_cls']], marker='s', edgecolors='k',
        #           linewidths=lw_m, zorder=4)
        # fundo: região de cada material, em tom muito fraco (b, c, d)
        faint = np.array([plt.matplotlib.colors.to_rgba(FAINT[k]) for k in CLS_ORDER])
        for ax in (a, b, c, d):
            fc = faint[F['tri_cls']]
            ax.add_collection(PolyCollection(F['tri_poly'], facecolors=fc, edgecolors=fc,
                                             linewidths=0.3, zorder=0))
        dash = dict(colors=CROSS_EDGE_COLOR, linewidths=CROSS_EDGE_LW, linestyles=CROSS_EDGE_LS, zorder=2)
        solid = dict(colors=GRAPH_EDGE_COLOR, linewidths=GRAPH_EDGE_LW, zorder=2)
        # (b) grafo de vértices: arestas da malha = arestas do grafo; nós = anéis do material
        b.add_collection(LineCollection(F['b_seg'], **solid))
        b.scatter(*F['v_xy'].T, edgecolors=col[F['v_cls']], **ring)
        # (c) Elem: duais (sólidas) + cruzadas (tracejadas) + vértices sem material + elementos
        c.add_collection(LineCollection(F['c_seg'], **dash))
        c.add_collection(LineCollection(F['d_seg'], **solid))
        c.scatter(*F['v_xy'].T, s=s_v, marker='o', facecolors=NEUTRAL_FACE, edgecolors=NEUTRAL_EDGE,
                  linewidths=RING_LW_NEUTRAL, zorder=5)
        c.scatter(*F['e_xy'].T, c=col[F['e_cls']], **disc)
        # (d) Split: Edge_dual (sólidas) + Edge_aux (tracejadas) + cópias (anéis) + N_aux (discos)
        d.add_collection(LineCollection(F['ea_seg'], **dash))
        d.add_collection(LineCollection(F['d_seg'], **solid))
        d.scatter(*F['cp_xy'].T, edgecolors=col[F['cp_cls']], **ring)
        d.scatter(*F['n_aux_xy'].T, c=col[F['n_aux_cls']], **disc)
        for ax in axs:
            ax.add_collection(LineCollection(F['iface_seg'], colors=IFACE_COLOR, linewidths=IFACE_LW,
                                             zorder=3))
        # barra de escala na (a)
        # [REMOVIDO] L = 0.5 if side > 1.2 else 0.2
        L = max(v for v in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if v <= side / 3.5)
        x0, y0 = box[0] + 0.06 * side, box[2] + 0.06 * side
        a.plot([x0, x0 + L], [y0, y0], color='k', lw=1.4, solid_capstyle='butt', zorder=6)
        a.text(x0 + L / 2, y0 + 0.025 * side, f'{L:g} mm'.replace('.', ','), ha='center',
               va='bottom', zorder=6, bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
        # legenda única
        present = [k for k in CLS_ORDER if CLASSES[k][0] in F['materiais']]
        # [REMOVIDO] h1 = [Patch(fc=CLASSES[k][1], ec='k', lw=0.4, label=CLASSES[k][0]) for k in present]
        h1 = [Patch(fc=FAINT[k], ec=CLASSES[k][1], lw=1.2, label=CLASSES[k][0]) for k in present]
        h1.append(Patch(fc='none', ec='0.35', lw=CELL_LW, label='Célula da grade (contorno, cor do material)'))
        # [REMOVIDO] legenda com quadrado para elemento e cinzas próximos nas arestas
        ms_l = 7.0
        h2 = [Line2D([], [], ls='', marker='o', ms=ms_l, mfc='white', mec='0.35', mew=RING_LW,
                     label='Nó de vértice / cópia (anel, cor do material)'),
              Line2D([], [], ls='', marker='o', ms=ms_l * 0.9, mfc='0.6', mec='k', mew=DISC_EDGE_LW,
                     label='Nó de elemento (disco, cor do material)'),
              Line2D([], [], ls='', marker='o', ms=ms_l, mfc=NEUTRAL_FACE, mec=NEUTRAL_EDGE,
                     mew=RING_LW_NEUTRAL, label='Vértice sem material (c)')]
        h3 = [Line2D([], [], color=MESH_COLOR, lw=1.0, label='Aresta da malha'),
              Line2D([], [], color=GRAPH_EDGE_COLOR, lw=GRAPH_EDGE_LW,
                     label='Aresta do grafo (vértices, duais, Edge_dual)'),
              Line2D([], [], color=CROSS_EDGE_COLOR, lw=CROSS_EDGE_LW, ls=CROSS_EDGE_LS,
                     label='Aresta cruzada / Edge_aux'),
              Line2D([], [], color=IFACE_COLOR, lw=IFACE_LW, label='Interface')]
        fig.legend(handles=h1 + h2 + h3, loc='lower center', ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.2,
                   borderaxespad=0.2)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f'{name}.pdf')
        fig.savefig(out_dir / f'{name}.png', dpi=300)
        plt.close(fig)
    return dict(figure=str(out_dir / f'{name}.pdf'), size_in=[FIG_WIDTH_IN, H], layout=layout,
                titles=titles)


def make_figure(sample_idx=2184, center=DEFAULT_CENTER, side=DEFAULT_SIDE_MM, layout='2x2'):
    L = load_layouts(sample_idx)
    cx, cy = center
    box = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)
    F = build(L, box)
    reps = [render(F, layout, titles=t) for t in (True, False)]
    return L, F, reps


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sample', type=int, default=2184)
    ap.add_argument('--center', type=float, nargs=2, default=DEFAULT_CENTER, metavar=('X', 'Y'))
    ap.add_argument('--side', type=float, default=DEFAULT_SIDE_MM)
    ap.add_argument('--layout', choices=('2x2', '1x4'), default='2x2')
    args = ap.parse_args()
    cx, cy = args.center
    L, F, reps = make_figure(args.sample, (cx, cy), args.side, args.layout)
    r, th = float(np.hypot(cx, cy)), float(np.degrees(np.arctan2(cy, cx)))
    print(f'amostra {args.sample}: janela centro ({cx:.3f}, {cy:.3f}) mm = (r {r:.3f} mm, θ {th:.2f}°), '
          f'lado {args.side:.2f} mm')
    print(f"materiais na janela: {', '.join(F['materiais'])}")
    print('contagens:', json.dumps(F['counts'], ensure_ascii=False))
    print(f'(d) deslocamento das cópias: {COPY_SHIFT:.0%} da distância vértice → centróide médio '
          f'dos elementos do material da cópia, só em vértices com 2+ cópias')
    for k, v in F['check'].items():
        print(f'conferência {k}: ' + ', '.join(f'{a}={b}' for a, b in v.items() if a != 'regra'))
    for rep in reps:
        print(f"figura: {rep['figure']} ({rep['size_in'][0]:.2f} × {rep['size_in'][1]:.2f} in)")
    out = dict(sample_idx=args.sample,
               janela=dict(cx=cx, cy=cy, lado_mm=args.side, r_mm=r, theta_deg=th, box=list(F['box'])),
               materiais=F['materiais'], contagens=F['counts'],
               deslocamento_d=dict(fracao=COPY_SHIFT,
                                   regra='cópia = vértice + f·(centróide médio dos elementos do '
                                         'material da cópia − vértice), só em vértices com 2+ '
                                         'cópias; cópias únicas no vértice'),
               conferencia_material_features=F['check'], figuras=reps,
               cores={CLASSES[k][0]: CLASSES[k][1] for k in CLS_ORDER})
    jd = OUT_ROOT / 'dados' / 'fig_representacoes'
    jd.mkdir(parents=True, exist_ok=True)
    (jd / f'sample_{args.sample:06d}.json').write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
