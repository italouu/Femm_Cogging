"""
parsers/femm_mesh_v2.py
------------------------
Deriva o grafo de vértices + grafo de elementos (dual, não iterado) + grade
H×W a partir do `.ans.gz` bruto salvo por mode='femm_mesh_v2'
(scripts/generate_data_femm_mesh_v2.py / src/data_gen/femm_mesh.py::
save_ans_gzip_sample) -- sem FEMM aberto, sem nenhuma chamada COM, só o
arquivo já salvo em disco.

Relocado de src/data_gen/femm_mesh_v2.py em 2026-08-13 -- essa etapa é,
junto com ans_parsing.py, todo o "serviço de parsing" do pipeline
femm_mesh_v2 (raw .ans.gz -> arrays de treino), então mora com os demais
parsers do projeto em src/data_gen/parsers/ (ver PARSER_REGISTRY em
src/data_gen/parsers/__init__.py -- esse módulo não entra nesse registry,
já que não é um MotorQtreeParserConfig de seleção de colunas, mas é o mesmo
tipo de responsabilidade: transformar dado bruto em formato de treino).
Motivo original da separação: src/data_gen/femm_mesh.py (v1) tem
`import femm` no topo (necessário só pras funções que abrem o FEMM de
verdade) -- importar qualquer coisa de lá arrastava esse import junto,
mesmo pro caminho 100% numpy daqui. Ver docstring de ans_parsing.py.

Duas mudanças de design em relação a mode='femm_mesh' (v1, femm_mesh.py):

  1. B deixa de ser alvo -- só A (potencial vetor), extraído nodal exato do
     .ans. Sem o loop de mo_getb por nó (não tem como -- não existe sessão
     FEMM aberta aqui, só o arquivo). Ver CLAUDE.md "Raw -- malha real do
     FEMM v2".

  2. Material (mu_r, M) deixa de ser feature de vértice -- no v1, um nó bem
     na fronteira ferro/ar tinha que "votar" por UM material (ambíguo, perde
     informação). Aqui vira um GRAFO SEPARADO, de elementos/centróides
     (1 material por elemento, sempre exato, sem ambiguidade nenhuma),
     conectado ao grafo de vértices só por arestas de INJEÇÃO
     (elemento -> vértice, unidirecional): cada elemento manda info pros
     seus 3 vértices, mas nunca recebe nada de volta -- o grafo de
     elementos não é iterado (sem estado evoluindo por camada; ver
     conversa de 2026-08-10 sobre a camada F2 na GNN, que roda com pesos
     próprios por camada mas sempre lendo o mesmo elem_x estático).

Grafo 1 -- vértices/campo (nós da malha real):
    node_x [S,2]  r_base, c_base
    node_y [S,1]  A (potencial vetor, valor nodal exato)
    edge_index [2,E]  bidirecional -- malha real + wrap periódico θ=0/120°
    edge_attr [E,3]  delta_r, delta_c, center_dist  (SEM delta_mu -- mu_r
                     não é mais feature de vértice, não tem o que subtrair)

Grafo 2 -- elementos/centróides (1 nó por triângulo, SEM arestas internas):
    elem_x [M,5]  mu_r, M, area, r_base_centróide, c_base_centróide

Arestas cruzadas (elemento -> vértice, unidirecional, conexão imediata --
cada elemento liga só aos seus 3 vértices, sem busca por raio/k-NN):
    cross_edge_index [2,C]  linha 0 = índice do elemento, linha 1 = índice do vértice
    cross_edge_attr [C,1]  center_dist (distância centróide-vértice, mm)

Grade H×W (amostrada só do grafo, sem COM):
    x_hw [2,H,W]  Mu_r, M -- constante por elemento (trifinder + valor do
                  elemento; fallback pro elemento de centróide mais
                  próximo se o pixel cair fora da triangulação -- SEM
                  fallback por nó, nó não carrega material nesse design)
    y_hw [1,H,W]  A -- interpolação baricêntrica nos 3 nós do elemento que
                  contém o pixel (mesma função de forma do FEM linear;
                  validado MAE≈0 contra point-query em 2026-07-23, ver
                  CLAUDE.md "Point-query vs malha para a grade H×W")

r_in/r_ext/ang_1/ang_2 (janela de amostragem) são passados pelo chamador --
essa função não abre FEMM nem instancia o modelo geométrico, só lê o
arquivo; quem chama pode obter r_in/r_ext de valid_designs.csv (colunas
'inner_diameter [mm]'/'outer_diameter [mm]', ver
tests/proto_femm_mesh_v2_element_b_plot.py::_read_r_bounds_mm).
"""
import gzip
import shutil
from pathlib import Path

import numpy as np
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

from src.data_gen.parsers.ans_parsing import (
    _parse_solution, _parse_block_materials, _block_magnet_polarity,
    _build_edges, _element_areas, _wrap_edge_pairs, _grid_polar_xy,
)
from src.data_gen.motor_model import BLDC_FEMM_Model_Sym120_Annular

_N_POLES_SECTOR = BLDC_FEMM_Model_Sym120_Annular.N_POLES_SECTOR  # 14 -- constante de classe, sem instanciar/abrir FEMM


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


def _grid_barycentric(tri, node_field: np.ndarray, Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    """Amostra um campo nodal contínuo (A) na grade via interpolação
    baricêntrica -- mesma função de forma do FEM linear dentro do elemento
    que contém o pixel; validado MAE≈0 contra point-query (ver docstring do
    módulo). Fallback pro nó mais próximo nos poucos pixels fora da malha."""
    interp = mtri.LinearTriInterpolator(tri, node_field)
    out = np.ma.filled(interp(Xg, Yg).astype(np.float32), np.nan)
    invalid = np.isnan(out)
    if invalid.any():
        node_xy = np.stack([tri.x, tri.y], axis=1)
        nearest = cKDTree(node_xy).query(np.stack([Xg[invalid], Yg[invalid]], axis=1))[1]
        out[invalid] = node_field[nearest]
    return out


# ---------------------------------------------------------------------------
# pipeline completo por amostra
# ---------------------------------------------------------------------------

def parse_ans_gzip_sample(ans_gz_path: Path, r_in: float, r_ext: float,
                           ang_1: float = 0.0, ang_2: float = 120.0,
                           n_r: int = 138, n_a: int = 276,
                           tmp_dir: Path = None) -> dict:
    """Deriva os dois grafos + grade H×W de UMA amostra a partir do
    `sample_XXXXXX.ans.gz` bruto. Sem FEMM aberto, sem chamada COM --
    só descompacta o arquivo e faz numpy/matplotlib.tri puro.

    Retorna dict com todos os campos descritos na docstring do módulo, mais
    L/elem_L/E_L/C_L (contagens por amostra, pra agrupamento em chunks) e
    dim_H/dim_W.
    """
    ans_gz_path = Path(ans_gz_path)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else ans_gz_path.parent
    stem = ans_gz_path.name.removesuffix('.ans.gz')
    tmp_ans = tmp_dir / f"{stem}.tmp_parse.ans"

    with gzip.open(ans_gz_path, 'rb') as f_in, open(tmp_ans, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    try:
        lines, nodes, elems = _parse_solution(str(tmp_ans))
    finally:
        tmp_ans.unlink(missing_ok=True)

    block_material_id, block_mu = _parse_block_materials(lines)
    block_M = _block_magnet_polarity(block_material_id, _N_POLES_SECTOR)

    elem_mu = block_mu[elems[:, 3]]
    elem_M = block_M[elems[:, 3]]
    elem_area = _element_areas(nodes, elems)

    ang_1_rad, ang_2_rad = np.deg2rad(ang_1), np.deg2rad(ang_2)

    # --- grafo 1: vértices ---
    n_nodes = nodes.shape[0]
    r_node = np.hypot(nodes[:, 0], nodes[:, 1])
    th_node = np.arctan2(nodes[:, 1], nodes[:, 0])
    r_base = ((r_node - r_in) / (r_ext - r_in)).astype(np.float32)
    c_base = ((th_node - ang_1_rad) / (ang_2_rad - ang_1_rad)).astype(np.float32)
    node_A = nodes[:, 2].astype(np.float32)

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
    node_y = node_A[:, None]

    # --- grafo 2: elementos/centróides (sem arestas internas) ---
    n_elems = elems.shape[0]
    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    r_elem = np.hypot(centroids[:, 0], centroids[:, 1])
    th_elem = np.arctan2(centroids[:, 1], centroids[:, 0])
    r_base_elem = ((r_elem - r_in) / (r_ext - r_in)).astype(np.float32)
    c_base_elem = ((th_elem - ang_1_rad) / (ang_2_rad - ang_1_rad)).astype(np.float32)

    elem_x = np.stack(
        [elem_mu, elem_M, elem_area.astype(np.float32), r_base_elem, c_base_elem], axis=1
    ).astype(np.float32)

    # --- arestas cruzadas: cada elemento -> seus 3 vértices (unidirecional) ---
    tri_idx = np.repeat(np.arange(n_elems, dtype=np.int64), 3)
    vtx_idx = elems[:, :3].reshape(-1).astype(np.int64)
    cross_dist = np.linalg.norm(nodes[vtx_idx, :2] - centroids[tri_idx], axis=1).astype(np.float32)
    cross_edge_index = np.stack([tri_idx, vtx_idx], axis=0)
    cross_edge_attr = cross_dist[:, None]

    # --- grade H×W ---
    Xg, Yg = _grid_polar_xy(r_in, r_ext, ang_1_rad, ang_2_rad, n_r, n_a)
    tri, trifinder = _build_trifinder(nodes, elems)
    Mu_hw = _grid_const_per_element(trifinder, elem_mu, centroids, Xg, Yg).reshape(n_r, n_a)
    M_hw = _grid_const_per_element(trifinder, elem_M, centroids, Xg, Yg).reshape(n_r, n_a)
    A_hw = _grid_barycentric(tri, node_A, Xg, Yg).reshape(n_r, n_a)

    return {
        'node_x': node_x, 'node_y': node_y,
        'edge_index': edge_index, 'edge_attr': edge_attr,
        'elem_x': elem_x,
        'cross_edge_index': cross_edge_index, 'cross_edge_attr': cross_edge_attr,
        'x_hw': np.stack([Mu_hw, M_hw], axis=0),
        'y_hw': A_hw[None, :, :],
        'L': np.array([n_nodes], dtype=np.int64),
        'elem_L': np.array([n_elems], dtype=np.int64),
        'E_L': np.array([edge_index.shape[1]], dtype=np.int64),
        'C_L': np.array([cross_edge_index.shape[1]], dtype=np.int64),
        'dim_H': np.array(n_r, dtype=np.int64),
        'dim_W': np.array(n_a, dtype=np.int64),
    }
