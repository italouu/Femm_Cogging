"""
parsers/split_layout.py
-----------------------
Layout do FNO_SplitGNN: grafo dual de ELEMENTOS (N_aux) + VÉRTICES DIVIDIDOS POR
MATERIAL (N_base), ligados por arestas elemento <-> vértice (Edge_aux). Arch:
src/neural_op/archs/fno_split_gnn.py. Ver docs/claude/dados.md §3.5.

Construção (a partir do raw, sem FEMM):
  - N_aux  = 1 nó por elemento (centróide);
  - Edge_dual = elementos que compartilham uma aresta da malha + wrap θ=0/120°
    (elem_structure._build_dual_edges), bidirecional;
  - N_base = vértices com os pares θ=0°/θ=120° FUNDIDOS (_wrap_edge_pairs; o nó de
    120° vira o de 0°) e DIVIDIDOS por material: 1 cópia por (vértice, classe), com
    classe = (material_id, M). Vértice interno -> 1 cópia; fronteira de 2 materiais ->
    2 cópias; raros vértices triplos -> 3;
  - Edge_aux = cada elemento ligado aos seus 3 vértices, na cópia do SEU material
    (3M arestas). As arestas da malha (vértice-vértice) não existem: os materiais só
    se comunicam pelas Edge_dual.

Coordenadas: r_base e o ângulo do setor (0-120° levado a 0-360°, φ = 2π·c_base) como
(sen φ, cos φ) -- contínuos no corte. Os deltas das arestas usam essas coordenadas
finais (r_base, sen, cos), então atravessam o corte pelo caminho curto sem regra
especial; o comprimento é medido com as coordenadas reais (no wrap da Edge_dual,
centróide girado de −120°; na Edge_aux, o vértice ORIGINAL do elemento, antes da fusão).

Gabarito (B suavizado do FEMM, polar):
  node_y  = B no centróide (= média dos 3 valores nodais do elemento; mesmo da Elem);
  elem_y  = por cópia, média dos valores nodais b_e(k) dos elementos da cópia (mesmo
            material), cada um em polar no ângulo do vértice original. Nos cantos e no
            corte, onde o FEMM dá valores diferentes por elemento, vale a média.

=============================================================================
CONTRATO DO LAYOUT Split (FNO_SplitGNN) -- fonte única.
Chaves do bipartite (loader/collate/offsets de chunks.py reaproveitados), com
node_* = N_aux (elementos, grafo principal) e elem_* = N_base (cópias):

  chave             shape  significado
  ----------------  -----  ----------------------------------------------------
  node_x            [M,7]  N_aux: r_base_c, c_base_c (CRUS, interpolação), sin, cos,
                           mu_r, M, J [A/mm²]
  node_y            [M,2]  (Br, Bθ) suavizado no centróide [T]
  edge_index        [2,F]  Edge_dual, os dois sentidos (i->j, depois j->i)
  edge_attr         [F,7]  dst − src: delta_r, delta_sin, delta_cos, length [mm],
                           delta_mu, delta_J, delta_M
  elem_x            [K,7]  N_base: r_base, c_base (CRUS; fundido = lado θ=0°), sin,
                           cos, mu_r, M, J (média por área dos elementos da cópia)
  elem_y            [K,2]  (Br, Bθ) da cópia [T]
  cross_edge_index  [2,C]  Edge_aux: linha 0 = cópia (N_base), linha 1 = elemento
                           (N_aux); C = 3M, na ordem (elemento, vértice local 0..2)
  cross_edge_attr   [C,7]  vértice − centróide: delta_r, delta_sin, delta_cos,
                           length [mm]; e mu_r, J, M do elemento
  x_hw / y_hw    [3|2,H,W] grade, idêntica à das demais archs
  L / elem_L / E_L / C_L   M / K / F / C
=============================================================================
"""
from pathlib import Path

import numpy as np

from src.data_gen.parsers.ans_b_smoothing import element_b, nodal_b
from src.data_gen.parsers.ans_parsing import (
    AnsData, read_ans, _grid_polar_xy, _parse_label_materials, _to_polar, _wrap_edge_pairs,
)
from src.data_gen.parsers.bipartite_structure import parse_bipartite_structure, _build_trifinder
from src.data_gen.parsers.elem_smooth import _fill_elem_smooth
from src.data_gen.parsers.elem_structure import _build_dual_edges
from src.data_gen.parsers.smooth_layouts import _grid_smooth_b

SPLIT_ARCH = 'FNO_SplitGNN'
_N_CLS = 16   # classes (material_id, M) codificadas como material_id*4 + (M+1)


def check_split_layout(d: dict):
    """Falha alto se `d` (1 amostra, numpy ou tensores) não seguir o contrato Split."""
    m, k = d['node_x'].shape[0], d['elem_x'].shape[0]
    cei = d['cross_edge_index']
    if cei.shape[1] != 3 * m:
        raise ValueError(f"layout Split: {cei.shape[1]} Edge_aux, esperado 3M = {3 * m}")
    if d['elem_y'] is not None and d['elem_y'].shape[0] != k:
        raise ValueError(f"layout Split: elem_y com {d['elem_y'].shape[0]} linhas, elem_x com {k}")
    if int(cei[0].max()) >= k or int(cei[1].max()) >= m:
        raise ValueError("layout Split: Edge_aux fora do intervalo de N_base/N_aux")
    if int(d['edge_index'].max()) >= m:
        raise ValueError("layout Split: Edge_dual fora do intervalo de N_aux")


def _split_vertices(nodes, elems, elem_cls, ang_1, ang_2):
    """Funde 0°/120° e divide os vértices por classe. Devolve (copy_of_corner [3M],
    copy_vertex [K] = vértice canônico de cada cópia, first_corner [K])."""
    n = nodes.shape[0]
    idx_1, idx_2 = _wrap_edge_pairs(nodes[:, :2], ang_1, ang_2)
    canon = np.arange(n, dtype=np.int64)
    canon[idx_2] = idx_1
    k_flat = canon[elems[:, :3].reshape(-1)]
    e_flat = np.repeat(np.arange(elems.shape[0], dtype=np.int64), 3)
    key = k_flat * _N_CLS + elem_cls[e_flat]
    uniq, first_corner, copy_of_corner = np.unique(key, return_index=True, return_inverse=True)
    return copy_of_corner.astype(np.int64), uniq // _N_CLS, first_corner


def parse_split_smooth(ans_gz_path: Path, r_in: float, r_ext: float,
                       ang_1: float = 0.0, ang_2: float = 120.0,
                       n_r: int = 138, n_a: int = 276,
                       ans: AnsData = None) -> dict:
    """1 `.ans.gz` -> layout Split (ver docstring do módulo). `ans`: `.ans` já lido
    (read_ans); None = lê de `ans_gz_path`."""
    if ans is None:
        ans = read_ans(ans_gz_path)
    bip = parse_bipartite_structure(None, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                                    n_r=n_r, n_a=n_a, ans=ans)
    nodes, elems, mesh = ans.nodes, ans.elems, ans.mesh
    assert np.array_equal(mesh.p, elems[:, :3]), "conectividade divergente entre leitores do .ans"
    n_elems = elems.shape[0]

    # --- N_aux: elementos (elem_x do bipartite = mu_r, M, area, r_base_c, c_base_c, J) ---
    ex = bip['elem_x'].astype(np.float64)
    e_mu, e_M, e_area, e_r, e_c, e_J = (ex[:, i] for i in range(6))
    e_sin, e_cos = np.sin(2 * np.pi * e_c), np.cos(2 * np.pi * e_c)
    node_x = np.stack([e_r, e_c, e_sin, e_cos, e_mu, e_M, e_J], axis=1).astype(np.float32)

    # --- N_base: vértices fundidos no corte e divididos por (material, M) ---
    label_material_id, _, _ = _parse_label_materials(ans.lines)
    elem_cls = label_material_id[elems[:, 3]] * 4 + (np.rint(e_M).astype(np.int64) + 1)
    copy_of_corner, copy_vertex, first_corner = _split_vertices(nodes, elems, elem_cls,
                                                                ang_1, ang_2)
    n_copies = copy_vertex.shape[0]
    e_flat = np.repeat(np.arange(n_elems, dtype=np.int64), 3)
    e_first = e_flat[first_corner]
    v_r = bip['node_x'][:, 0].astype(np.float64)
    v_c = bip['node_x'][:, 1].astype(np.float64)
    c_r, c_c = v_r[copy_vertex], v_c[copy_vertex]
    c_sin, c_cos = np.sin(2 * np.pi * c_c), np.cos(2 * np.pi * c_c)
    w = np.bincount(copy_of_corner, e_area[e_flat], minlength=n_copies)
    c_J = np.bincount(copy_of_corner, (e_area * e_J)[e_flat], minlength=n_copies) / w
    elem_x = np.stack([c_r, c_c, c_sin, c_cos, e_mu[e_first], e_M[e_first], c_J],
                      axis=1).astype(np.float32)

    # --- Edge_dual: deltas dst − src nas coordenadas finais; comprimento real ---
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    i_int, j_int, i_wrap, j_wrap = _build_dual_edges(nodes, elems, ang_1, ang_2)
    len_int = np.linalg.norm(centroids[j_int] - centroids[i_int], axis=1)
    phi = -np.deg2rad(ang_2 - ang_1)
    cj = centroids[j_wrap]
    cj_rot = np.stack([cj[:, 0] * np.cos(phi) - cj[:, 1] * np.sin(phi),
                       cj[:, 0] * np.sin(phi) + cj[:, 1] * np.cos(phi)], axis=1)
    len_wrap = np.linalg.norm(cj_rot - centroids[i_wrap], axis=1)
    i = np.concatenate([i_int, i_wrap])
    j = np.concatenate([j_int, j_wrap])
    length = np.concatenate([len_int, len_wrap])
    src, dst = np.concatenate([i, j]), np.concatenate([j, i])
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr = np.stack([
        e_r[dst] - e_r[src], e_sin[dst] - e_sin[src], e_cos[dst] - e_cos[src],
        np.concatenate([length, length]),
        e_mu[dst] - e_mu[src], e_J[dst] - e_J[src], e_M[dst] - e_M[src],
    ], axis=1).astype(np.float32)

    # --- Edge_aux: cópia (linha 0) -> elemento (linha 1); vértice − centróide ---
    v_orig = elems[:, :3].reshape(-1)
    aux_len = np.linalg.norm(nodes[v_orig, :2] - centroids[e_flat], axis=1)
    cross_edge_index = np.stack([copy_of_corner, e_flat], axis=0).astype(np.int64)
    cross_edge_attr = np.stack([
        c_r[copy_of_corner] - e_r[e_flat], c_sin[copy_of_corner] - e_sin[e_flat],
        c_cos[copy_of_corner] - e_cos[e_flat], aux_len,
        e_mu[e_flat], e_J[e_flat], e_M[e_flat],
    ], axis=1).astype(np.float32)

    # --- gabarito: B suavizado do FEMM ---
    B1, B2 = element_b(mesh)
    b1, b2 = nodal_b(mesh, B1, B2)
    br, bt = _to_polar(b1.reshape(-1), b2.reshape(-1), nodes[v_orig, 0], nodes[v_orig, 1])
    cnt = np.bincount(copy_of_corner, minlength=n_copies)
    elem_y = np.stack([np.bincount(copy_of_corner, br, minlength=n_copies) / cnt,
                       np.bincount(copy_of_corner, bt, minlength=n_copies) / cnt],
                      axis=1).astype(np.float32)

    Xg, Yg = _grid_polar_xy(r_in, r_ext, np.deg2rad(ang_1), np.deg2rad(ang_2), n_r, n_a)
    _, trifinder = _build_trifinder(nodes, elems)
    Bx_hw, By_hw = _grid_smooth_b(mesh, B1, B2, b1, b2, trifinder, centroids, Xg, Yg)
    Br_hw, Bt_hw = _to_polar(Bx_hw, By_hw, Xg, Yg)
    y_hw = np.stack([Br_hw.reshape(n_r, n_a), Bt_hw.reshape(n_r, n_a)], axis=0)

    out = {
        'node_x': node_x, 'node_y': None,
        'edge_index': edge_index, 'edge_attr': edge_attr,
        'elem_x': elem_x, 'elem_y': elem_y,
        'cross_edge_index': cross_edge_index, 'cross_edge_attr': cross_edge_attr,
        'x_hw': bip['x_hw'], 'y_hw': None,
        'L': np.array([n_elems], dtype=np.int64),
        'elem_L': np.array([n_copies], dtype=np.int64),
        'E_L': np.array([edge_index.shape[1]], dtype=np.int64),
        'C_L': np.array([cross_edge_index.shape[1]], dtype=np.int64),
        'dim_H': bip['dim_H'], 'dim_W': bip['dim_W'],
    }
    _fill_elem_smooth(out, ans, B1, B2, b1, b2, y_hw)   # node_y no centróide + y_hw
    out['node_y'] = out['node_y'].astype(np.float32)
    check_split_layout(out)
    return out
