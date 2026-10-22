"""
parsers/elem_smooth.py
----------------------
FNO_BipartiteGNN_Elem com gabarito B SUAVIZADO do FEMM -- versão
smooth de `elem_structure.py` (bipartite com papéis trocados: elementos =
grafo principal, vértices = auxiliar), coerente com os datasets das outras 4 archs
`<DatagenConfig.dataset>/<arch>` (`smooth_layouts.py`).

Reaproveita parse_elem_structure SEM mudança (x_hw, grafo dual, wrap,
vértices, arestas cruzadas, contagens) e troca só o gabarito:

  y_hw [2,H,W]  -- point_b(smooth=True) no centro do pixel (_grid_smooth_b de
                   smooth_layouts.py) -- idêntico ao y_hw das outras 4 archs.
  node_y [M,2]  -- B suavizado NO CENTRÓIDE de cada elemento:
                   point_b(smooth=True) no centróide = média dos 3 valores
                   nodais suavizados (b1,b2, P1 descontínuo) do elemento.
                   Mesmo campo do y_hw, só amostrado no centróide em vez do
                   pixel (1 valor por elemento, a saída da arch).

Referencial: node_y e y_hw são POLARES (Br, Bθ), girados pelo ângulo atan2(y, x)
do centróide / centro do pixel (ans_parsing._to_polar).

Layout/chaves idênticos ao de elem_structure.py -- loader/collate/
Normalizer/step_fn/metric_fn/arch reaproveitados sem mudança (contrato do
layout Elem: docstring de elem_structure.py).
"""
from pathlib import Path

import numpy as np

from src.data_gen.parsers.ans_b_smoothing import element_b, nodal_b, point_b
from src.data_gen.parsers.ans_parsing import AnsData, read_ans, _grid_polar_xy, _to_polar
from src.data_gen.parsers.elem_structure import parse_elem_structure
from src.data_gen.parsers.smooth_layouts import _grid_smooth_b, _parse_smooth
from src.data_gen.parsers.bipartite_structure import _build_trifinder

ELEM_ARCH = 'FNO_BipartiteGNN_Elem'


def parse_elem_smooth(ans_gz_path: Path, r_in: float, r_ext: float,
                                      ang_1: float = 0.0, ang_2: float = 120.0,
                                      n_r: int = 138, n_a: int = 276,
                                      ans: AnsData = None) -> dict:
    """1 `.ans.gz` -> layout de elem_structure.py com node_y/y_hw suavizados
    (ver docstring do módulo). `ans`: `.ans` já lido (read_ans); None = lê de
    `ans_gz_path`."""
    if ans is None:
        ans = read_ans(ans_gz_path)
    d = parse_elem_structure(None, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                                   n_r=n_r, n_a=n_a, target_field=None, ans=ans)
    mesh = ans.mesh
    B1, B2 = element_b(mesh)
    b1, b2 = nodal_b(mesh, B1, B2)

    # --- grade H×W: B suavizado no centro do pixel (= smooth_layouts) ---
    nodes, elems = ans.nodes, ans.elems
    Xg, Yg = _grid_polar_xy(r_in, r_ext, np.deg2rad(ang_1), np.deg2rad(ang_2), n_r, n_a)
    _, trifinder = _build_trifinder(nodes, elems)
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    Bx_hw, By_hw = _grid_smooth_b(mesh, B1, B2, b1, b2, trifinder, centroids, Xg, Yg)
    Br_hw, Bt_hw = _to_polar(Bx_hw, By_hw, Xg, Yg)
    y_hw = np.stack([Br_hw.reshape(n_r, n_a), Bt_hw.reshape(n_r, n_a)], axis=0)
    return _fill_elem_smooth(d, ans, B1, B2, b1, b2, y_hw)


def _fill_elem_smooth(d: dict, ans: AnsData, B1, B2, b1, b2, y_hw) -> dict:
    """Preenche o gabarito smooth do layout Elem `d`: node_y = B suavizado no centróide
    (Br, Bθ) e y_hw (o mesmo das 4 archs de smooth_layouts)."""
    mesh, nodes, elems = ans.mesh, ans.nodes, ans.elems
    assert np.array_equal(mesh.p, elems[:, :3]), "conectividade divergente entre leitores do .ans"
    assert elems.shape[0] == d['node_x'].shape[0], "contagem de elementos divergente"

    # --- elementos: B suavizado no centróide ---
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    m_idx = np.arange(elems.shape[0])
    bx_c, by_c = point_b(mesh, B1, B2, b1, b2, m_idx, centroids[:, 0], centroids[:, 1], smooth=True)
    node_y = np.stack(_to_polar(bx_c, by_c, centroids[:, 0], centroids[:, 1]), axis=1)

    d['node_y'] = node_y
    d['y_hw'] = y_hw
    return d


def parse_all_layouts(ans_gz_path: Path, r_in: float, r_ext: float,
                              ang_1: float = 0.0, ang_2: float = 120.0,
                              n_r: int = 138, n_a: int = 276) -> dict:
    """1 `.ans.gz` -> {arch: dict_de_arrays} dos 5 datasets smooth (as 4 de SMOOTH_ARCHS +
    ELEM_ARCH) numa passada: `.ans` lido uma vez, layout bipartite e B
    suavizado calculados uma vez; a Elem é derivada do bipartite e reaproveita o y_hw
    (mesma conta de parse_elem_smooth, resultado idêntico)."""
    ans = read_ans(ans_gz_path)
    layouts, sm = _parse_smooth(ans, r_in, r_ext, ang_1, ang_2, n_r, n_a)
    bip = layouts['FNO_BipartiteGNN']
    d = parse_elem_structure(None, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                                   n_r=n_r, n_a=n_a, target_field=None, ans=ans, bip=bip)
    layouts[ELEM_ARCH] = _fill_elem_smooth(d, ans, sm['B1'], sm['B2'], sm['b1'], sm['b2'],
                                           bip['y_hw'])
    return layouts
