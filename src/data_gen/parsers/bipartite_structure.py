"""
parsers/bipartite_structure.py
------------------------------
Construtor da ESTRUTURA do layout bipartido (FNO_BipartiteGNN) a partir do `.ans.gz`
bruto (scripts/generate_raw.py / src/data_gen/femm_solve.py::
save_ans_gzip_sample) -- sem FEMM aberto, sem chamada COM: grafo de vértices, grafo de
elementos, arestas (malha + wrap periódico), arestas cruzadas, node_x/elem_x e x_hw.
O gabarito (node_y/y_hw) NÃO é calculado aqui: sai como None e é preenchido pelos
parsers smooth (smooth_layouts, elem_smooth); o curl(A) exato por elemento
do torque fica em elem_structure (target_field='B').

Desenho (em relação ao grafo único de vértices do layout de vértices, FNO_GNN):

  Material (mu_r, M) não é feature de vértice -- um nó na fronteira ferro/ar
     teria que receber UM material (ambíguo, perde informação). Aqui é um GRAFO
     SEPARADO, de elementos/centróides (1 material por elemento, exato),
     conectado ao grafo de vértices só por arestas de INJEÇÃO
     (elemento -> vértice, unidirecional): cada elemento manda info pros
     seus 3 vértices, mas nunca recebe nada de volta -- o grafo de
     elementos não é iterado (sem estado evoluindo por camada; a mensagem F2
     da BipartiteEdgeConvLayer tem pesos próprios por camada, mas sempre lê o
     mesmo elem_x estático).

Grafo 1 -- vértices/campo (nós da malha real):
    node_x [S,2]  r_base, c_base
    node_y        None (preenchido pelo chamador)
    edge_index [2,E]  bidirecional -- malha real + wrap periódico θ=0/120°
    edge_attr [E,3]  delta_r, delta_c, center_dist  (SEM delta_mu -- mu_r
                     não é mais feature de vértice, não tem o que subtrair)

Grafo 2 -- elementos/centróides (1 nó por triângulo, SEM arestas internas):
    elem_x [M,6]  mu_r, M, area, r_base_centróide, c_base_centróide, J
                  (J [A/mm²] = Turns·I/área do bloco, 0 fora das bobinas --
                  _parse_label_currents; última coluna, fora dos índices
                  estruturais da normalização)

Arestas cruzadas (elemento -> vértice, unidirecional, conexão imediata --
cada elemento liga só aos seus 3 vértices, sem busca por raio/k-NN):
    cross_edge_index [2,C]  linha 0 = índice do elemento, linha 1 = índice do vértice
    cross_edge_attr [C,1]  center_dist (distância centróide-vértice, mm)

Grade H×W (amostrada só do grafo, sem COM):
    x_hw [3,H,W]  Mu_r, M, J -- constante por elemento (trifinder + valor do
                  elemento; fallback pro elemento de centróide mais
                  próximo se o pixel cair fora da triangulação -- SEM
                  fallback por nó, nó não carrega material nesse design)
    y_hw          None (preenchido pelo chamador)

r_in/r_ext/ang_1/ang_2 (janela de amostragem) são passados pelo chamador --
essa função não abre FEMM nem instancia o modelo geométrico, só lê o
arquivo; quem chama obtém r_in/r_ext de valid_designs.csv (colunas
'inner_diameter [mm]'/'outer_diameter [mm]').
"""
from pathlib import Path

import numpy as np
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

from src.data_gen.parsers.ans_parsing import (
    AnsData, read_ans,
    _build_edges, _element_areas, _wrap_edge_pairs, _grid_polar_xy,
    _parse_label_materials,
    _parse_label_currents,
)


# ---------------------------------------------------------------------------
# grade H×W -- só a partir dos dados já extraídos da malha (sem COM)
# ---------------------------------------------------------------------------

def _build_trifinder(nodes: np.ndarray, elems: np.ndarray):
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elems[:, :3])
    return tri, tri.get_trifinder()


def _grid_const_per_element(trifinder, elem_field: np.ndarray, centroids: np.ndarray,
                             Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    """Amostra um campo constante-por-elemento (mu_r, M) na grade -- valor
    exato do triângulo que contém o pixel; fallback pro elemento de
    centróide mais próximo nos poucos pixels fora da triangulação."""
    elem_idx = trifinder(Xg, Yg)
    valid = elem_idx >= 0
    out = np.empty(Xg.shape[0], dtype=np.float32)
    out[valid] = elem_field[elem_idx[valid]]
    n_invalid = int((~valid).sum())
    if n_invalid:
        nearest = cKDTree(centroids).query(np.stack([Xg[~valid], Yg[~valid]], axis=1))[1]
        out[~valid] = elem_field[nearest]
    return out


# ---------------------------------------------------------------------------
# pipeline completo por amostra
# ---------------------------------------------------------------------------

def parse_bipartite_structure(ans_gz_path: Path, r_in: float, r_ext: float,
                           ang_1: float = 0.0, ang_2: float = 120.0,
                           n_r: int = 138, n_a: int = 276,
                           ans: AnsData = None) -> dict:
    """Estrutura do layout bipartido (os dois grafos + x_hw) de UMA amostra a partir do
    `sample_XXXXXX.ans.gz` bruto. Sem FEMM aberto, sem chamada COM -- só lê o arquivo e
    faz numpy/matplotlib.tri puro. `node_y`/`y_hw` saem como None, para o chamador
    preencher (parsers smooth: smooth_layouts, elem_smooth).

    `ans`: `.ans` já lido (ans_parsing.read_ans) -- evita descomprimir/ler o
    arquivo de novo; None = lê de `ans_gz_path` (read_ans, em memória).

    Retorna dict com todos os campos descritos na docstring do módulo, mais
    L/elem_L/E_L/C_L (contagens por amostra, pra agrupamento em chunks) e
    dim_H/dim_W.
    """
    if ans is None:
        ans = read_ans(ans_gz_path)
    lines, nodes, elems = ans.lines, ans.nodes, ans.elems

    # material/polaridade pelo rótulo de cada bloco (_parse_label_materials): a ordem e a
    # contagem dos blocos de ímã variam com a fase do rotor
    _, label_mu, label_M = _parse_label_materials(lines)
    elem_mu = label_mu[elems[:, 3]]
    elem_M = label_M[elems[:, 3]]
    elem_area = _element_areas(nodes, elems)
    # densidade de corrente por elemento -- 0 sem correntes
    elem_J = _parse_label_currents(lines, elems, elem_area)[elems[:, 3]]

    ang_1_rad, ang_2_rad = np.deg2rad(ang_1), np.deg2rad(ang_2)

    # --- grafo 1: vértices ---
    n_nodes = nodes.shape[0]
    r_node = np.hypot(nodes[:, 0], nodes[:, 1])
    th_node = np.arctan2(nodes[:, 1], nodes[:, 0])
    r_base = ((r_node - r_in) / (r_ext - r_in)).astype(np.float32)
    c_base = ((th_node - ang_1_rad) / (ang_2_rad - ang_1_rad)).astype(np.float32)

    edges_undirected = _build_edges(elems)
    wrap_idx_1, wrap_idx_2 = _wrap_edge_pairs(nodes, ang_1, ang_2)
    i = np.concatenate([edges_undirected[:, 0], wrap_idx_1])
    j = np.concatenate([edges_undirected[:, 1], wrap_idx_2])
    n_wrap = len(wrap_idx_1)

    dx = nodes[j, 0] - nodes[i, 0]
    dy = nodes[j, 1] - nodes[i, 1]
    center_dist = np.hypot(dx, dy).astype(np.float32)
    delta_r = ((r_base[j] - r_base[i]) * n_r).astype(np.float32)
    delta_c = ((c_base[j] - c_base[i]) * n_a).astype(np.float32)
    if n_wrap:
        center_dist[-n_wrap:] = 0.0
        delta_r[-n_wrap:] = 0.0
        delta_c[-n_wrap:] = 0.0

    edge_attr_fwd = np.stack([delta_r, delta_c, center_dist], axis=1)
    edge_attr_bwd = np.stack([-delta_r, -delta_c, center_dist], axis=1)
    edge_index = np.stack([np.concatenate([i, j]), np.concatenate([j, i])], axis=0).astype(np.int64)
    edge_attr = np.concatenate([edge_attr_fwd, edge_attr_bwd], axis=0).astype(np.float32)

    node_x = np.stack([r_base, c_base], axis=1).astype(np.float32)

    # --- grafo 2: elementos/centróides (sem arestas internas) ---
    n_elems = elems.shape[0]
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    r_elem = np.hypot(centroids[:, 0], centroids[:, 1])
    th_elem = np.arctan2(centroids[:, 1], centroids[:, 0])
    r_base_elem = ((r_elem - r_in) / (r_ext - r_in)).astype(np.float32)
    c_base_elem = ((th_elem - ang_1_rad) / (ang_2_rad - ang_1_rad)).astype(np.float32)

    elem_x = np.stack(
        [elem_mu, elem_M, elem_area.astype(np.float32), r_base_elem, c_base_elem, elem_J], axis=1
    ).astype(np.float32)

    # --- arestas cruzadas: cada elemento -> seus 3 vértices (unidirecional) ---
    tri_idx = np.repeat(np.arange(n_elems, dtype=np.int64), 3)
    vtx_idx = elems[:, :3].reshape(-1).astype(np.int64)
    cross_dist = np.linalg.norm(nodes[vtx_idx, :2] - centroids[tri_idx], axis=1).astype(np.float32)
    cross_edge_index = np.stack([tri_idx, vtx_idx], axis=0)
    cross_edge_attr = cross_dist[:, None]

    # --- grade H×W ---
    Xg, Yg = _grid_polar_xy(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)
    _, trifinder = _build_trifinder(nodes, elems)
    Mu_hw = _grid_const_per_element(trifinder, elem_mu, centroids, Xg, Yg).reshape(n_r, n_a)
    M_hw = _grid_const_per_element(trifinder, elem_M, centroids, Xg, Yg).reshape(n_r, n_a)
    J_hw = _grid_const_per_element(trifinder, elem_J, centroids, Xg, Yg).reshape(n_r, n_a)
    return {
        'node_x': node_x, 'node_y': None,     # gabarito: preenchido pelo chamador
        'edge_index': edge_index, 'edge_attr': edge_attr,
        'elem_x': elem_x,
        'cross_edge_index': cross_edge_index, 'cross_edge_attr': cross_edge_attr,
        'x_hw': np.stack([Mu_hw, M_hw, J_hw], axis=0).astype(np.float32),
        'y_hw': None,
        'L': np.array([n_nodes], dtype=np.int64),
        'elem_L': np.array([n_elems], dtype=np.int64),
        'E_L': np.array([edge_index.shape[1]], dtype=np.int64),
        'C_L': np.array([cross_edge_index.shape[1]], dtype=np.int64),
        'dim_H': np.array(n_r, dtype=np.int64),
        'dim_W': np.array(n_a, dtype=np.int64),
    }
