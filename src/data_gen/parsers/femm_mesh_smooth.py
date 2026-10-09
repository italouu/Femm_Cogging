"""
parsers/femm_mesh_smooth.py
----------------------------
Mesmos 4 layouts de `femm_mesh_unified.py` (FNO2d, FNO_GNN, GNN_PostBase,
FNO_BipartiteGNN), a partir do MESMO raw (`.ans.gz`, `mode='femm_mesh_v2'`),
mas com o gabarito B trocado pela suavização do pós-processador do FEMM
(`ans_b_smoothing.py`, porte de xfemm validado a ~1e-15 contra o FEMM) e o
material do nó por PRIORIDADE em vez de voto por área (2026-10-09, ver
CLAUDE.md "Suavização de B do FEMM a partir do `.ans`").

Diferenças em relação a parse_ans_gzip_sample_unified:

  material do nó  -- prioridade ferro > ímã > cobre > ar entre os elementos
                     incidentes (_node_material_priority), em vez do voto por
                     área (_node_material_stats).
  node_y [S,2]    -- (as 4 archs) média dos valores nodais suavizados
                     (b1,b2 do FEMM, P1 descontínuo) dos elementos incidentes
                     DO MATERIAL ESCOLHIDO para o nó. Antes: média simples do
                     curl(A) P0 de todos os incidentes.
  y_hw [2,H,W]    -- (as 4 archs) point_b(smooth=True) no centro do pixel,
                     elemento via trifinder (= mo_getb com mo_smooth('on')).
                     Antes: interpolação baricêntrica do node_y antigo.
  node_x [S,5]    -- (FNO_GNN/GNN_PostBase) mu_r pela prioridade; M segue o
                     material escolhido (nó ímã -> +-1 pela polaridade dos
                     elementos de ímã incidentes; qualquer outro -> 0 --
                     decisão do usuário 2026-10-09, opção (a)).
  edge_attr [E,4] -- delta_mu recalculado com o mu_r novo.

Idênticos aos unificados: x_hw, a_hw, node_A, node_dual_area, r_base/c_base,
edge_index, e todo o resto do layout bipartite (node_x, elem_x, arestas,
arestas cruzadas, contagens).
"""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.data_gen.parsers.ans_b_smoothing import load_ans, element_b, nodal_b, point_b
from src.data_gen.parsers.ans_parsing import (
    _parse_label_materials, _build_edges, _element_areas, _node_material_stats,
    _node_magnet_polarity, _node_material_priority, _wrap_edge_pairs,
    _build_bidirectional_edge_attrs, _grid_polar_xy, _MU_BY_ID, _MAGNET_ID,
)
from src.data_gen.parsers.femm_mesh_v2 import (
    parse_ans_gzip_sample, _build_trifinder, _grid_barycentric,
)
from src.data_gen.parsers.femm_mesh_unified import _read_ans_gz, UNIFIED_ARCHS

SMOOTH_ARCHS = UNIFIED_ARCHS


def _node_b_by_material(elems: np.ndarray, elem_material_id: np.ndarray,
                        node_material_id: np.ndarray, b1: np.ndarray, b2: np.ndarray):
    """Média dos valores nodais (b1,b2 [M,3]) dos elementos incidentes cujo
    material == material do nó. Retorna [S,2] float32."""
    n_nodes = node_material_id.shape[0]
    k = elems[:, :3].reshape(-1)                                    # [3M]
    sel = np.repeat(elem_material_id, 3) == node_material_id[k]
    s1 = np.zeros(n_nodes)
    s2 = np.zeros(n_nodes)
    cnt = np.zeros(n_nodes)
    np.add.at(s1, k[sel], b1.reshape(-1)[sel])
    np.add.at(s2, k[sel], b2.reshape(-1)[sel])
    np.add.at(cnt, k[sel], 1.0)
    if (cnt == 0).any():
        raise ValueError("nó sem elemento incidente do material escolhido")
    return np.stack([s1 / cnt, s2 / cnt], axis=1).astype(np.float32)


def _grid_smooth_b(mesh, B1, B2, b1, b2, trifinder, centroids, Xg, Yg):
    """B suavizado (point_b smooth=True) nos pixels; fallback pro elemento de
    centróide mais próximo nos poucos pixels fora da triangulação (mesmo
    critério de _grid_const_per_element)."""
    elem_idx = trifinder(Xg, Yg)
    invalid = elem_idx < 0
    if invalid.any():
        elem_idx = elem_idx.copy()
        elem_idx[invalid] = cKDTree(centroids).query(
            np.stack([Xg[invalid], Yg[invalid]], axis=1))[1]
    bx, by = point_b(mesh, B1, B2, b1, b2, elem_idx, Xg, Yg, smooth=True)
    return bx.astype(np.float32), by.astype(np.float32)


def parse_ans_gzip_sample_smooth(ans_gz_path: Path, r_in: float, r_ext: float,
                                  ang_1: float = 0.0, ang_2: float = 120.0,
                                  n_r: int = 138, n_a: int = 276,
                                  tmp_dir: Path = None) -> dict:
    """Retorna {arch: dict_de_arrays} para as 4 archs de SMOOTH_ARCHS -- ver
    docstring do módulo."""
    ans_gz_path = Path(ans_gz_path)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else ans_gz_path.parent

    # --- layout bipartite (estrutura; node_y/y_hw substituídos abaixo) ---
    bip = parse_ans_gzip_sample(ans_gz_path, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                                 n_r=n_r, n_a=n_a, tmp_dir=tmp_dir, target_field='B')

    lines, nodes, elems = _read_ans_gz(ans_gz_path, tmp_dir)
    label_material_id, _, label_M = _parse_label_materials(lines)
    elem_material_id = label_material_id[elems[:, 3]]
    elem_M = label_M[elems[:, 3]]
    n_nodes = nodes.shape[0]
    assert n_nodes == bip['node_y'].shape[0], "ordem/contagem de nós divergente entre layouts"

    # --- B suavizado do FEMM (só .ans) ---
    mesh = load_ans(ans_gz_path)
    assert np.array_equal(mesh.p, elems[:, :3]), "conectividade divergente entre leitores do .ans"
    B1, B2 = element_b(mesh)
    b1, b2 = nodal_b(mesh, B1, B2)

    # --- material do nó por prioridade + node_y ---
    node_material_id = _node_material_priority(elems, elem_material_id, n_nodes)
    node_y = _node_b_by_material(elems, elem_material_id, node_material_id, b1, b2)

    # --- grade H×W: B suavizado no centro do pixel ---
    ang_1_rad, ang_2_rad = np.deg2rad(ang_1), np.deg2rad(ang_2)
    Xg, Yg = _grid_polar_xy(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)
    tri, trifinder = _build_trifinder(nodes, elems)
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    Bx_hw, By_hw = _grid_smooth_b(mesh, B1, B2, b1, b2, trifinder, centroids, Xg, Yg)
    y_hw = np.stack([Bx_hw.reshape(n_r, n_a), By_hw.reshape(n_r, n_a)], axis=0)

    bip['node_y'] = node_y
    bip['y_hw'] = y_hw

    # --- layout v1 (grafo único de vértices, material por prioridade) ---
    r_node = np.hypot(nodes[:, 0], nodes[:, 1])
    th_node = np.arctan2(nodes[:, 1], nodes[:, 0])
    r_base = (r_node - r_in) / (r_ext - r_in)
    c_base = (th_node - ang_1_rad) / (ang_2_rad - ang_1_rad)

    area = _element_areas(nodes, elems)
    _, _frac_dom, node_dual_area = _node_material_stats(nodes, elems, area, elem_material_id)
    node_mu = _MU_BY_ID[node_material_id]
    node_M = np.where(node_material_id == _MAGNET_ID,
                      _node_magnet_polarity(nodes, elems, area, elem_material_id, elem_M),
                      0.0).astype(np.float32)

    edges_undirected = _build_edges(elems)
    wrap_idx_1, wrap_idx_2 = _wrap_edge_pairs(nodes, ang_1, ang_2)
    edge_index, edge_attr = _build_bidirectional_edge_attrs(
        nodes, edges_undirected, wrap_idx_1, wrap_idx_2, r_base, c_base, node_mu, n_r, n_a)

    node_A = nodes[:, 2].astype(np.float32)
    a_hw = _grid_barycentric(tri, node_A, Xg, Yg).reshape(n_r, n_a)

    node_x = np.stack([
        node_mu.astype(np.float32),          # mu_r (prioridade)
        node_M.astype(np.float32),           # M (segue o material do nó)
        node_dual_area.astype(np.float32),   # node_dual_area (mm^2)
        r_base.astype(np.float32),           # r_base
        c_base.astype(np.float32),           # c_base
    ], axis=1)

    mesh_v1 = {
        'node_x':     node_x,
        'node_y':     node_y,
        'node_A':     node_A,
        'edge_index': edge_index,
        'edge_attr':  edge_attr,
        'x_hw':       bip['x_hw'],
        'y_hw':       y_hw,
        'a_hw':       a_hw.astype(np.float32),
        'L':          np.array([n_nodes], dtype=np.int64),
        'dim_H':      bip['dim_H'],
        'dim_W':      bip['dim_W'],
    }
    grid = {
        'x_hw':  bip['x_hw'],
        'y_hw':  y_hw,
        'dim_H': bip['dim_H'],
        'dim_W': bip['dim_W'],
    }
    return {
        'FNO2d':            grid,
        'FNO_GNN':          mesh_v1,
        'GNN_PostBase':     mesh_v1,
        'FNO_BipartiteGNN': bip,
    }
