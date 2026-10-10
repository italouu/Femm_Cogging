"""
parsers/femm_mesh_elem_smooth.py
--------------------------------
FNO_BipartiteGNN_Elem com gabarito B SUAVIZADO do FEMM (2026-10-10) -- versão
smooth de `femm_mesh_elem.py` (bipartite com papéis trocados: elementos =
grafo principal, vértices = auxiliar), coerente com os datasets
`mesh_ans_138x276_smooth/<arch>` (`femm_mesh_smooth.py`).

Reaproveita parse_ans_gzip_sample_elem SEM mudança (x_hw, grafo dual, wrap,
vértices, arestas cruzadas, contagens) e troca só o gabarito:

  y_hw [2,H,W]  -- point_b(smooth=True) no centro do pixel (_grid_smooth_b de
                   femm_mesh_smooth.py) -- idêntico ao y_hw das 4 archs de
                   mesh_ans_138x276_smooth.
  node_y [M,2]  -- B suavizado NO CENTRÓIDE de cada elemento:
                   point_b(smooth=True) no centróide = média dos 3 valores
                   nodais suavizados (b1,b2, P1 descontínuo) do elemento.
                   Mesmo campo do y_hw, só amostrado no centróide em vez do
                   pixel (decisão do usuário 2026-10-10; alternativas
                   descartadas: curl(A) P0 exato, ou P1 completo [M,6], que
                   exigiria mudar a arch).

Layout/chaves idênticos ao de femm_mesh_elem.py -- loader/collate/
Normalizer/step_fn/metric_fn/arch reaproveitados sem mudança.
"""
from pathlib import Path

import numpy as np

from src.data_gen.parsers.ans_b_smoothing import load_ans, element_b, nodal_b, point_b
from src.data_gen.parsers.ans_parsing import _grid_polar_xy
from src.data_gen.parsers.femm_mesh_elem import parse_ans_gzip_sample_elem
from src.data_gen.parsers.femm_mesh_smooth import _grid_smooth_b
from src.data_gen.parsers.femm_mesh_v2 import _build_trifinder
from src.data_gen.parsers.femm_mesh_unified import _read_ans_gz


def parse_ans_gzip_sample_elem_smooth(ans_gz_path: Path, r_in: float, r_ext: float,
                                      ang_1: float = 0.0, ang_2: float = 120.0,
                                      n_r: int = 138, n_a: int = 276,
                                      tmp_dir: Path = None) -> dict:
    """1 `.ans.gz` -> layout de femm_mesh_elem.py com node_y/y_hw suavizados
    (ver docstring do módulo)."""
    ans_gz_path = Path(ans_gz_path)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else ans_gz_path.parent

    d = parse_ans_gzip_sample_elem(ans_gz_path, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                                   n_r=n_r, n_a=n_a, tmp_dir=tmp_dir)

    _, nodes, elems = _read_ans_gz(ans_gz_path, tmp_dir)
    mesh = load_ans(ans_gz_path)
    assert np.array_equal(mesh.p, elems[:, :3]), "conectividade divergente entre leitores do .ans"
    assert elems.shape[0] == d['node_y'].shape[0], "contagem de elementos divergente"
    B1, B2 = element_b(mesh)
    b1, b2 = nodal_b(mesh, B1, B2)

    # --- elementos: B suavizado no centróide ---
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    m_idx = np.arange(elems.shape[0])
    bx_c, by_c = point_b(mesh, B1, B2, b1, b2, m_idx, centroids[:, 0], centroids[:, 1], smooth=True)
    node_y = np.stack([bx_c, by_c], axis=1).astype(np.float32)

    # --- grade H×W: B suavizado no centro do pixel (= femm_mesh_smooth) ---
    Xg, Yg = _grid_polar_xy(r_in, r_ext, np.deg2rad(ang_1), np.deg2rad(ang_2), n_r, n_a)
    _, trifinder = _build_trifinder(nodes, elems)
    Bx_hw, By_hw = _grid_smooth_b(mesh, B1, B2, b1, b2, trifinder, centroids, Xg, Yg)
    y_hw = np.stack([Bx_hw.reshape(n_r, n_a), By_hw.reshape(n_r, n_a)], axis=0)

    d['node_y'] = node_y
    d['y_hw'] = y_hw
    return d
