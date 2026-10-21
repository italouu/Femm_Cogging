"""
parsers/femm_mesh_smooth.py
----------------------------
Layouts das 4 archs de SMOOTH_ARCHS (FNO2d, FNO_GNN, GNN_PostBase,
FNO_BipartiteGNN) a partir do raw (`.ans.gz`), com o gabarito B da suavização
do pós-processador do FEMM
(`ans_b_smoothing.py`, porte de xfemm validado a ~1e-15 contra o FEMM) e o
material do nó por PRIORIDADE em vez de voto por área (2026-10-09, ver
CLAUDE.md "Suavização de B do FEMM a partir do `.ans`").

Diferenças em relação ao gabarito "unificado" (curl(A); parser removido em 2026-10-21):

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

  Referencial (2026-10-17): node_y e y_hw são POLARES (Br, Bθ), girados pelo
  ângulo atan2(y, x) do nó / centro do pixel (_to_polar). Antes eram (Bx, By).
  node_x [S,6]    -- (FNO_GNN/GNN_PostBase) mu_r pela prioridade; M segue o
                     material escolhido (nó ímã -> +-1 pela polaridade dos
                     elementos de ímã incidentes; qualquer outro -> 0 --
                     decisão do usuário 2026-10-09, opção (a)). J [A/mm²]
                     (2026-10-14, última coluna) segue o mesmo critério: nó de
                     cobre -> média por área do J dos elementos de cobre
                     incidentes; outro -> 0 (_node_current_density).
  edge_attr [E,4] -- delta_mu recalculado com o mu_r novo.

Idênticos aos unificados: x_hw, node_dual_area, r_base/c_base,
edge_index, e todo o resto do layout bipartite (node_x, elem_x, arestas,
arestas cruzadas, contagens).

Sem node_A/a_hw (2026-10-21): o layout v1 deixou de carregar o potencial A,
que nenhum código do pipeline atual lia (só os eval_fn antigos de
archs/eval.py); o gabarito do femm_mesh_v2 (curl(A)/interpolação) não é mais
calculado (target_field=None).
"""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.data_gen.parsers.ans_b_smoothing import load_ans, element_b, nodal_b, point_b
from src.data_gen.parsers.ans_parsing import (
    _read_ans_gz, _parse_label_materials, _build_edges, _element_areas, _node_material_stats,
    _node_magnet_polarity, _node_material_priority, _wrap_edge_pairs,
    _build_bidirectional_edge_attrs, _grid_polar_xy, _MU_BY_ID, _MAGNET_ID,
    _node_current_density, _to_polar,
)
from src.data_gen.parsers.femm_mesh_v2 import parse_ans_gzip_sample, _build_trifinder

SMOOTH_ARCHS = ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN')


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
                                 n_r=n_r, n_a=n_a, tmp_dir=tmp_dir, target_field=None)

    lines, nodes, elems = _read_ans_gz(ans_gz_path, tmp_dir)
    label_material_id, _, label_M = _parse_label_materials(lines)
    elem_material_id = label_material_id[elems[:, 3]]
    elem_M = label_M[elems[:, 3]]
    n_nodes = nodes.shape[0]
    assert n_nodes == bip['node_x'].shape[0], "ordem/contagem de nós divergente entre layouts"

    # --- B suavizado do FEMM (só .ans) ---
    mesh = load_ans(ans_gz_path)
    assert np.array_equal(mesh.p, elems[:, :3]), "conectividade divergente entre leitores do .ans"
    B1, B2 = element_b(mesh)
    b1, b2 = nodal_b(mesh, B1, B2)

    # --- material do nó por prioridade + node_y ---
    node_material_id = _node_material_priority(elems, elem_material_id, n_nodes)
    node_bxy = _node_b_by_material(elems, elem_material_id, node_material_id, b1, b2)
    node_y = np.stack(_to_polar(node_bxy[:, 0], node_bxy[:, 1], nodes[:, 0], nodes[:, 1]), axis=1)

    # --- grade H×W: B suavizado no centro do pixel ---
    ang_1_rad, ang_2_rad = np.deg2rad(ang_1), np.deg2rad(ang_2)
    Xg, Yg = _grid_polar_xy(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)
    _, trifinder = _build_trifinder(nodes, elems)
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    Bx_hw, By_hw = _grid_smooth_b(mesh, B1, B2, b1, b2, trifinder, centroids, Xg, Yg)
    Br_hw, Bt_hw = _to_polar(Bx_hw, By_hw, Xg, Yg)
    y_hw = np.stack([Br_hw.reshape(n_r, n_a), Bt_hw.reshape(n_r, n_a)], axis=0)

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
    # J (2026-10-14): elem_x do bipartite, coluna 5 (mesma ordem de elementos)
    node_J = _node_current_density(elems, area, bip['elem_x'][:, 5], elem_material_id,
                                   node_material_id)

    edges_undirected = _build_edges(elems)
    wrap_idx_1, wrap_idx_2 = _wrap_edge_pairs(nodes, ang_1, ang_2)
    edge_index, edge_attr = _build_bidirectional_edge_attrs(
        nodes, edges_undirected, wrap_idx_1, wrap_idx_2, r_base, c_base, node_mu, n_r, n_a)

    node_x = np.stack([
        node_mu.astype(np.float32),          # mu_r (prioridade)
        node_M.astype(np.float32),           # M (segue o material do nó)
        node_dual_area.astype(np.float32),   # node_dual_area (mm^2)
        r_base.astype(np.float32),           # r_base
        c_base.astype(np.float32),           # c_base
        node_J,                              # J [A/mm²] (2026-10-14; antes node_x [S,5])
    ], axis=1)

    mesh_v1 = {
        'node_x':     node_x,
        'node_y':     node_y,
        'edge_index': edge_index,
        'edge_attr':  edge_attr,
        'x_hw':       bip['x_hw'],
        'y_hw':       y_hw,
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
