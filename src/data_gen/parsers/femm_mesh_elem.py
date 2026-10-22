"""
parsers/femm_mesh_elem.py
-------------------------
Grafo duplo do FNO_BipartiteGNN com os PAPÉIS TROCADOS (criado em 2026-10-07):
o grafo de ELEMENTOS vira o principal (iterado, saída B por elemento) e o grafo de
VÉRTICES vira o auxiliar (estático, injetado nos elementos por arestas cruzadas
vértice -> elemento). Arch: src/neural_op/archs/femm_mesh_elem_gnn.py::
FNO_BipartiteGNN_Elem (mesma rede da FNO_BipartiteGNN; muda só o layout).

Raw: data/raw/<DatagenConfig.dataset>/. Estrutura (x_hw, elem_x, arestas
cruzadas) do layout bipartite (parse_ans_gzip_sample), reorganizada.
Gabarito por elemento:
  - dataset (femm_mesh_elem_smooth, target_field=None aqui): (Br, Bθ) do B
    suavizado do FEMM no centróide; y_hw = B suavizado no centro do pixel;
  - target_field='B' (gabarito do torque, postprocess/torque.py): (Bx, By) do
    curl(A) EXATO por elemento P1 (_element_b_from_A), sem média nodal; y_hw = None.

=============================================================================
CONTRATO DO LAYOUT Elem (FNO_BipartiteGNN_Elem) -- fonte única (2026-10-21).
Citado por: neural_op/archs/femm_mesh_elem_gnn.py, neural_op/normalization.py
(_LAYOUTS['FNO_BipartiteGNN_Elem']), postprocess/inference.predict,
postprocess/fields.elem_to_nodes, parsers/femm_mesh_elem_smooth.py.

As CHAVES do chunk são as do femm_mesh_v2 (loader/collate/Normalizer/
_detect_chunk_dims/step_fn/metric_fn da bipartite reaproveitados sem mudança),
mas com o SIGNIFICADO TROCADO -- node_* = elementos, elem_* = vértices:

  chave             shape  significado
  ----------------  -----  ----------------------------------------------------
  node_x            [M,6]  ELEMENTOS (grafo principal): r_base_c, c_base_c
                           (centróide, CRUS -- lidos pela interpolação grade ->
                           pontos), mu_r, M, area [mm²], J [A/mm²] (2026-10-14)
  node_y            [M,2]  B por ELEMENTO [T]: no dataset (femm_mesh_elem_smooth)
                           = (Br, Bθ) do B suavizado no centróide; com
                           target_field='B' (torque) = (Bx, By) do curl(A) exato
  edge_index        [2,F]  dual bidirecional: elementos que compartilham uma
                           aresta da malha + wrap periódico θ=0/120° (elementos
                           cuja aresta está no corte, casados pelo pareamento de
                           vértices de _wrap_edge_pairs)
  edge_attr         [F,3]  delta_r, delta_c, center_dist entre centróides (j−i,
                           em células da grade; no wrap, centróide j girado de
                           −120° -- distância real através do corte, não zero
                           como no wrap de vértices, porque aqui os dois lados
                           NÃO são o mesmo grau de liberdade)
  elem_x            [S,2]  VÉRTICES (grafo auxiliar): r_base, c_base (CRUS)
  cross_edge_index  [2,C]  vértice -> elemento: linha 0 = vértice (auxiliar),
                           linha 1 = elemento (principal); C = 3M (cada
                           elemento liga seus 3 vértices). Mesma convenção
                           "linha 0 = auxiliar, linha 1 = principal" do v2, então
                           os offsets de src/data_gen/chunks.py ('bipartite') e
                           de femm_mesh_v2_collate continuam corretos
  cross_edge_attr   [C,1]  distância centróide↔vértice (mm)
  x_hw / y_hw    [3|2,H,W] grade, idêntica à das demais archs do mesmo gabarito
  L / elem_L / E_L / C_L   M (nº de elementos) / S (nº de vértices) / F / C

Saída do modelo: y_nodes [M,2] POR ELEMENTO (levar aos nós com
postprocess/fields.elem_to_nodes). Checagem barata do contrato:
check_elem_layout (chamada no parser e em inference.predict).
=============================================================================
"""
from pathlib import Path

import numpy as np

from src.data_gen.parsers.ans_parsing import AnsData, read_ans, _element_b_from_A, _wrap_edge_pairs
from src.data_gen.parsers.femm_mesh_v2 import parse_ans_gzip_sample


def check_elem_layout(d: dict, n_elems: int = None):
    """Falha alto se `d` (dict de arrays numpy ou de tensores de 1 amostra) não seguir o
    contrato Elem (docstring do módulo): node_x com uma linha por ELEMENTO. n_elems = nº de
    elementos da malha, se conhecido; senão, deduzido das arestas cruzadas (C = 3M)."""
    n_rows = d['node_x'].shape[0]
    n_cross = d['cross_edge_index'].shape[1]
    if n_elems is None:
        if n_cross % 3:
            raise ValueError(f"layout Elem: {n_cross} arestas cruzadas, não múltiplo de 3")
        n_elems = n_cross // 3
    if n_rows != n_elems or n_cross != 3 * n_rows:
        raise ValueError(f"layout Elem: node_x tem {n_rows} linhas, esperado 1 por elemento "
                         f"({n_elems}; {n_cross} arestas cruzadas) -- layout bipartite no "
                         f"lugar do Elem?")


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
                                tmp_dir: Path = None, target_field: str | None = 'B',
                                ans: AnsData = None, bip: dict = None) -> dict:
    """1 `.ans.gz` -> layout com papéis trocados (ver docstring do módulo).
    Mesmo formato de dict de parse_ans_gzip_sample (dim_H/dim_W escalares).

    target_field='B' (padrão): node_y = curl(A) exato por elemento -- é o gabarito usado
    pelo postprocess/torque.py; y_hw = None (até 2026-10-22, a interpolação baricêntrica
    do B médio nodal do femm_mesh_v2, sem leitor). None (2026-10-21): sem gabarito
    (node_y/y_hw = None), para o parser smooth (femm_mesh_elem_smooth) preencher.

    `ans` (2026-10-22): `.ans` já lido (read_ans); None = lê de `ans_gz_path`. `bip`: layout
    bipartite da mesma amostra já calculado (parse_ans_gzip_sample, mesma janela) -- a Elem
    só reorganiza as chaves dele; None = calcula aqui."""
    assert target_field in ('B', None), f"target_field inválido: {target_field!r}"
    if ans is None:
        ans = read_ans(ans_gz_path)
    if bip is None:
        bip = parse_ans_gzip_sample(None, r_in, r_ext, ang_1=ang_1, ang_2=ang_2,
                                    n_r=n_r, n_a=n_a, ans=ans)
    nodes, elems = ans.nodes, ans.elems
    assert nodes.shape[0] == bip['node_x'].shape[0], "contagem de nós divergente"
    assert elems.shape[0] == bip['elem_x'].shape[0], "contagem de elementos divergente"

    # --- principal: elementos ---
    ex = bip['elem_x']                     # mu_r, M, area, r_base_c, c_base_c, J
    node_x = ex[:, [3, 4, 0, 1, 2, 5]].astype(np.float32)
    rc, cc = ex[:, 3].astype(np.float64), ex[:, 4].astype(np.float64)
    # A em float32 (como era o node_A do femm_mesh_v2 até 2026-10-22)
    node_y = None   # target_field=None: preenchido pelo chamador
    if target_field == 'B':
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

    out = {
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
    check_elem_layout(out, n_elems=elems.shape[0])
    return out
