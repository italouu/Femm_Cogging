"""
parsers/femm_mesh_unified.py
-----------------------------
Deriva, a partir de UM `.ans.gz` bruto (raw `mode='femm_mesh_v2'`), os
layouts de entrada das 4 arquiteturas comparadas (FNO2d, FNO_GNN,
GNN_PostBase, FNO_BipartiteGNN) com o MESMO gabarito B -- sem FEMM aberto,
sem chamada COM (2026-10-01, ver CLAUDE.md "Datasets unificados a partir do
raw mesh_ans_138x276").

Motivação: a tabela de resultados comparava FNO2d/FNO_GNN/GNN_PostBase
(treinados em `mesh_138x276_FEMM_MESH`, raw v1 -- B via `mo_getb` com a
suavização padrão do FEMM) contra FNO_BipartiteGNN (treinado em
`mesh_ans_138x276_B`, raw v2 -- B = curl(A) por elemento + média simples
por nó). Mesmas 4000 simulações (valid_designs.csv idênticos, A nodal
idêntico a ~1e-10), mas gabaritos B diferentes nas interfaces (mediana
|ΔB| ~1e-3 T, p95 ~0,4 T). Aqui TODOS os layouts saem do `.ans`, com o B
do v2 -- única definição de B reconstruível a partir do raw v2 (o B
suavizado do FEMM não é reproduzível fora dele).

Layouts devolvidos (dict arch -> dict de arrays, já no formato final do
staging .npz -- `dim_H`/`dim_W` como escalares):

  'FNO_BipartiteGNN' -- exatamente parse_ans_gzip_sample(target_field='B'),
                        sem mudança nenhuma.
  'FNO_GNN' / 'GNN_PostBase' -- layout v1 já filtrado pelo FEMM_MESH_PARSER
                        (mesmas colunas de mesh_138x276_FEMM_MESH):
        node_x     [S,5]   mu_r, M, node_dual_area, r_base, c_base
                           (votação de material por área -- mesmo código de
                           femm_mesh.py::generate_mesh_sample, via ans_parsing)
        node_y     [S,2]   Bx, By  (== node_y do layout bipartite)
        node_A     [S]     A nodal (mantido -- eval.py::fno_gnn_eval_fn usa
                           'node_A' in d pra escolher o plot de malha)
        edge_index [2,E]   bidirecional, malha + wrap periódico
        edge_attr  [E,4]   delta_r, delta_c, center_dist, delta_mu
        x_hw [2,H,W] / y_hw [2,H,W]  == do layout bipartite
        a_hw [H,W]         A, interpolação baricêntrica (mesma de y_hw)
        L [1], dim_H, dim_W
  'FNO2d' -- só grade: x_hw [2,H,W], y_hw [2,H,W], dim_H, dim_W
             (== do layout bipartite).

Os dois dicts de FNO_GNN/GNN_PostBase são o MESMO objeto (decisão do
usuário: uma pasta por arch, mesmo com conteúdo idêntico).
"""
import gzip
import shutil
from pathlib import Path

import numpy as np

from src.data_gen.parsers.ans_parsing import (
    _parse_solution, _parse_block_materials, _block_magnet_polarity,
    _build_edges, _element_areas, _node_material_stats, _node_magnet_polarity,
    _wrap_edge_pairs, _build_bidirectional_edge_attrs, _MU_BY_ID,
    _parse_label_materials,
)
from src.data_gen.parsers.femm_mesh_v2 import (
    parse_ans_gzip_sample, _build_trifinder, _grid_barycentric,
)
from src.data_gen.parsers.ans_parsing import _grid_polar_xy
from src.data_gen.motor_constants import N_POLES_SECTOR as _N_POLES_SECTOR

UNIFIED_ARCHS = ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN')


def _read_ans_gz(ans_gz_path: Path, tmp_dir: Path):
    stem = ans_gz_path.name.removesuffix('.ans.gz')
    tmp_ans = tmp_dir / f"{stem}.tmp_unified.ans"
    with gzip.open(ans_gz_path, 'rb') as f_in, open(tmp_ans, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    try:
        return _parse_solution(str(tmp_ans))
    finally:
        tmp_ans.unlink(missing_ok=True)


def parse_ans_gzip_sample_unified(ans_gz_path: Path, r_in: float, r_ext: float,
                                   ang_1: float = 0.0, ang_2: float = 120.0,
                                   n_r: int = 138, n_a: int = 276,
                                   tmp_dir: Path = None) -> dict:
    """Retorna {arch: dict_de_arrays} para as 4 archs de UNIFIED_ARCHS --
    ver docstring do módulo."""
    ans_gz_path = Path(ans_gz_path)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else ans_gz_path.parent

    # --- layout bipartite (v2) -- fonte única de x_hw / y_hw / node_y (B) ---
    bip = parse_ans_gzip_sample(ans_gz_path, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                                 n_r=n_r, n_a=n_a, tmp_dir=tmp_dir, target_field='B')

    # --- layout v1 (grafo único de vértices com material votado) ---
    lines, nodes, elems = _read_ans_gz(ans_gz_path, tmp_dir)
    # [REMOVIDO 2026-10-05] mesmo motivo de femm_mesh_v2.py (ver _parse_label_materials)
    # block_material_id, block_mu = _parse_block_materials(lines)
    # block_M = _block_magnet_polarity(block_material_id, _N_POLES_SECTOR)
    # elem_material_id = block_material_id[elems[:, 3]]
    # elem_M = block_M[elems[:, 3]]
    label_material_id, _, label_M = _parse_label_materials(lines)
    elem_material_id = label_material_id[elems[:, 3]]
    elem_M = label_M[elems[:, 3]]

    n_nodes = nodes.shape[0]
    assert n_nodes == bip['node_y'].shape[0], "ordem/contagem de nós divergente entre layouts"

    ang_1_rad, ang_2_rad = np.deg2rad(ang_1), np.deg2rad(ang_2)
    r_node = np.hypot(nodes[:, 0], nodes[:, 1])
    th_node = np.arctan2(nodes[:, 1], nodes[:, 0])
    r_base = (r_node - r_in) / (r_ext - r_in)
    c_base = (th_node - ang_1_rad) / (ang_2_rad - ang_1_rad)

    area = _element_areas(nodes, elems)
    node_material_id, _frac_dom, node_dual_area = _node_material_stats(
        nodes, elems, area, elem_material_id)
    node_mu = _MU_BY_ID[node_material_id]
    node_M = _node_magnet_polarity(nodes, elems, area, elem_material_id, elem_M)

    edges_undirected = _build_edges(elems)
    wrap_idx_1, wrap_idx_2 = _wrap_edge_pairs(nodes, ang_1, ang_2)
    edge_index, edge_attr = _build_bidirectional_edge_attrs(
        nodes, edges_undirected, wrap_idx_1, wrap_idx_2, r_base, c_base, node_mu, n_r, n_a)

    node_A = nodes[:, 2].astype(np.float32)
    Xg, Yg = _grid_polar_xy(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)
    tri, _ = _build_trifinder(nodes, elems)
    a_hw = _grid_barycentric(tri, node_A, Xg, Yg).reshape(n_r, n_a)

    node_x = np.stack([
        node_mu.astype(np.float32),          # mu_r
        node_M.astype(np.float32),           # M
        node_dual_area.astype(np.float32),   # node_dual_area (mm^2)
        r_base.astype(np.float32),           # r_base
        c_base.astype(np.float32),           # c_base
    ], axis=1)

    mesh_v1 = {
        'node_x':     node_x,
        'node_y':     bip['node_y'],
        'node_A':     node_A,
        'edge_index': edge_index,
        'edge_attr':  edge_attr,
        'x_hw':       bip['x_hw'],
        'y_hw':       bip['y_hw'],
        'a_hw':       a_hw.astype(np.float32),
        'L':          np.array([n_nodes], dtype=np.int64),
        'dim_H':      bip['dim_H'],
        'dim_W':      bip['dim_W'],
    }
    grid = {
        'x_hw':  bip['x_hw'],
        'y_hw':  bip['y_hw'],
        'dim_H': bip['dim_H'],
        'dim_W': bip['dim_W'],
    }
    return {
        'FNO2d':            grid,
        'FNO_GNN':          mesh_v1,
        'GNN_PostBase':     mesh_v1,
        'FNO_BipartiteGNN': bip,
    }
