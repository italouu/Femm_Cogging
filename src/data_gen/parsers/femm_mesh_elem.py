"""
parsers/femm_mesh_elem.py
-------------------------
PROTÓTIPO (2026-10-07) -- grafo duplo do FNO_BipartiteGNN com os PAPÉIS
TROCADOS: o grafo de ELEMENTOS vira o principal (iterado, saída Bx,By por
elemento) e o grafo de VÉRTICES vira o auxiliar (estático, injetado nos
elementos por arestas cruzadas vértice -> elemento). Arch:
src/neural_op/archs/femm_mesh_elem_gnn.py::FNO_BipartiteGNN_Elem.

Mesmo raw da bateria (data/raw/mesh_ans_138x276/), mesmas regras de
amostragem: x_hw/y_hw/elem_x/arestas cruzadas saem de
parse_ans_gzip_sample(target_field='B') SEM mudança (y_hw = interpolação
baricêntrica do B médio nodal, idêntico ao da bateria). Só o alvo do grafo
muda: B = curl(A) EXATO por elemento P1 (_element_b_from_A -- a mesma
função que gera o gabarito da bateria, que é a média simples desses valores
nos nós), sem média nodal.

Layout -- as CHAVES do chunk são as do femm_mesh_v2 (loader/collate/
Normalizer/_detect_chunk_dims/step_fn/metric_fn da bipartite reaproveitados
sem mudança), mas com o significado trocado:

  grafo PRINCIPAL = elementos (chaves node_*):
    node_x     [M,5]  r_base_c, c_base_c (centróide, CRUS -- interpolação do
                      FNO), mu_r, M, area
    node_y     [M,2]  Bx, By por elemento (T)
    edge_index [2,F]  dual bidirecional: elementos que compartilham uma aresta
                      da malha + wrap periódico θ=0/120° (elementos cuja aresta
                      está no corte, casados pelo pareamento de vértices de
                      _wrap_edge_pairs)
    edge_attr  [F,3]  delta_r, delta_c, center_dist entre centróides (mesma
                      convenção do grafo de vértices: j−i, em células da grade;
                      no wrap, centróide j girado de −120° -- distância real
                      através do corte, não zero como no wrap de vértices,
                      porque aqui os dois lados NÃO são o mesmo grau de
                      liberdade)
  grafo AUXILIAR = vértices (chaves elem_*):
    elem_x     [S,2]  r_base, c_base dos vértices (CRUS)
  arestas cruzadas (vértice -> elemento):
    cross_edge_index [2,C]  linha 0 = vértice (auxiliar), linha 1 = elemento
                            (principal) -- mesma convenção "linha 0 = auxiliar,
                            linha 1 = principal" do v2, então os offsets de
                            build_unified_ans_chunks_direct._bufs_v2 e de
                            femm_mesh_v2_collate continuam corretos
    cross_edge_attr  [C,1]  distância centróide↔vértice (mm)
  contagens: L = nº de elementos, elem_L = nº de vértices, E_L, C_L.
"""
from pathlib import Path

import numpy as np

from src.data_gen.parsers.ans_parsing import _element_b_from_A, _wrap_edge_pairs
from src.data_gen.parsers.femm_mesh_v2 import parse_ans_gzip_sample
from src.data_gen.parsers.femm_mesh_unified import _read_ans_gz


def _build_dual_edges(nodes: np.ndarray, elems: np.ndarray, ang_1: float, ang_2: float):
    """Pares (i,j) de elementos vizinhos (compartilham uma aresta) + pares de
    wrap através do corte periódico. Retorna (i_int, j_int, i_wrap, j_wrap),
    com i_wrap sempre do lado θ=ang_1 e j_wrap do lado θ=ang_2."""
    n = nodes.shape[0]
    tri = elems[:, :3].astype(np.int64)
    m = tri.shape[0]
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    owner = np.tile(np.arange(m, dtype=np.int64), 3)
    key = e[:, 0] * n + e[:, 1]

    order = np.argsort(key, kind='stable')
    ks, ow = key[order], owner[order]
    dup = ks[1:] == ks[:-1]
    if (dup[1:] & dup[:-1]).any():
        raise ValueError("aresta da malha compartilhada por >2 elementos (malha não conforme)")
    i_int, j_int = ow[:-1][dup], ow[1:][dup]

    # arestas de contorno (aparecem 1x) -- as do corte θ=ang_1 casam com as de θ=ang_2
    bmask = np.ones(len(ks), dtype=bool)
    bmask[:-1][dup] = False
    bmask[1:][dup] = False
    bkey, bown = ks[bmask], ow[bmask]
    be = np.stack([bkey // n, bkey % n], axis=1)

    idx_1, idx_2 = _wrap_edge_pairs(nodes, ang_1, ang_2)
    in1 = np.zeros(n, dtype=bool); in1[idx_1] = True
    in2 = np.zeros(n, dtype=bool); in2[idx_2] = True
    m12 = np.full(n, -1, dtype=np.int64); m12[idx_1] = idx_2

    cut1 = in1[be[:, 0]] & in1[be[:, 1]]
    cut2 = in2[be[:, 0]] & in2[be[:, 1]]
    if cut1.sum() != cut2.sum():
        raise ValueError(f"corte periódico com nº de arestas diferente: {cut1.sum()} x {cut2.sum()}")
    mapped = np.sort(m12[be[cut1]], axis=1)
    mkey = mapped[:, 0] * n + mapped[:, 1]
    pos = np.searchsorted(bkey, mkey)
    pos = np.clip(pos, 0, len(bkey) - 1)
    if not np.array_equal(bkey[pos], mkey):
        raise ValueError("aresta do corte θ=ang_1 sem par no corte θ=ang_2")
    return i_int, j_int, bown[cut1], bown[pos]


def parse_ans_gzip_sample_elem(ans_gz_path: Path, r_in: float, r_ext: float,
                                ang_1: float = 0.0, ang_2: float = 120.0,
                                n_r: int = 138, n_a: int = 276,
                                tmp_dir: Path = None) -> dict:
    """1 `.ans.gz` -> layout com papéis trocados (ver docstring do módulo).
    Mesmo formato de dict de parse_ans_gzip_sample (dim_H/dim_W escalares)."""
    ans_gz_path = Path(ans_gz_path)
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else ans_gz_path.parent

    bip = parse_ans_gzip_sample(ans_gz_path, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                                n_r=n_r, n_a=n_a, tmp_dir=tmp_dir, target_field='B')
    _, nodes, elems = _read_ans_gz(ans_gz_path, tmp_dir)
    assert nodes.shape[0] == bip['node_x'].shape[0], "contagem de nós divergente"
    assert elems.shape[0] == bip['elem_x'].shape[0], "contagem de elementos divergente"

    # --- principal: elementos ---
    ex = bip['elem_x']                     # mu_r, M, area, r_base_c, c_base_c
    node_x = ex[:, [3, 4, 0, 1, 2]].astype(np.float32)
    rc, cc = ex[:, 3].astype(np.float64), ex[:, 4].astype(np.float64)
    # A em float32, como em parse_ans_gzip_sample (node_A) -- média nodal deste
    # B por elemento reproduz exatamente o node_y da bateria
    elem_Bx, elem_By = _element_b_from_A(nodes, elems, nodes[:, 2].astype(np.float32))
    node_y = np.stack([elem_Bx, elem_By], axis=1).astype(np.float32)

    centroids = nodes[elems[:, :3], :2].mean(axis=1)
    i_int, j_int, i_wrap, j_wrap = _build_dual_edges(nodes, elems, ang_1, ang_2)

    # interior: deltas diretos entre centróides
    dx = centroids[j_int] - centroids[i_int]
    dist_int = np.hypot(dx[:, 0], dx[:, 1])
    dr_int = (rc[j_int] - rc[i_int]) * n_r
    dc_int = (cc[j_int] - cc[i_int]) * n_a

    # wrap: centróide do lado θ=ang_2 girado de −(ang_2−ang_1) pra perto de θ=ang_1
    phi = -np.deg2rad(ang_2 - ang_1)
    cj = centroids[j_wrap]
    cj_rot = np.stack([cj[:, 0] * np.cos(phi) - cj[:, 1] * np.sin(phi),
                       cj[:, 0] * np.sin(phi) + cj[:, 1] * np.cos(phi)], axis=1)
    dw = cj_rot - centroids[i_wrap]
    dist_wrap = np.hypot(dw[:, 0], dw[:, 1])
    dr_wrap = (rc[j_wrap] - rc[i_wrap]) * n_r
    dc_wrap = (cc[j_wrap] - 1.0 - cc[i_wrap]) * n_a

    i = np.concatenate([i_int, i_wrap])
    j = np.concatenate([j_int, j_wrap])
    delta_r = np.concatenate([dr_int, dr_wrap]).astype(np.float32)
    delta_c = np.concatenate([dc_int, dc_wrap]).astype(np.float32)
    center_dist = np.concatenate([dist_int, dist_wrap]).astype(np.float32)

    edge_attr = np.concatenate([
        np.stack([delta_r, delta_c, center_dist], axis=1),
        np.stack([-delta_r, -delta_c, center_dist], axis=1),
    ], axis=0).astype(np.float32)
    edge_index = np.stack([np.concatenate([i, j]), np.concatenate([j, i])], axis=0).astype(np.int64)

    # --- auxiliar: vértices; arestas cruzadas invertidas (vértice -> elemento) ---
    cross_edge_index = bip['cross_edge_index'][[1, 0]].copy()

    return {
        'node_x': node_x, 'node_y': node_y,
        'edge_index': edge_index, 'edge_attr': edge_attr,
        'elem_x': bip['node_x'],
        'cross_edge_index': cross_edge_index, 'cross_edge_attr': bip['cross_edge_attr'],
        'x_hw': bip['x_hw'], 'y_hw': bip['y_hw'],
        'L': np.array([node_x.shape[0]], dtype=np.int64),
        'elem_L': np.array([bip['node_x'].shape[0]], dtype=np.int64),
        'E_L': np.array([edge_index.shape[1]], dtype=np.int64),
        'C_L': np.array([cross_edge_index.shape[1]], dtype=np.int64),
        'dim_H': bip['dim_H'], 'dim_W': bip['dim_W'],
    }
