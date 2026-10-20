"""
Geometria das fronteiras entre materiais na malha do FEMM (2026-10-20).

  interface_edges — arestas compartilhadas por elementos de materiais diferentes (P4:
                    contornos nos mapas de erro; P5: estratificação espacial).

Material por elemento = material do rótulo (samples.parse_smooth(..., with_elem=True)
-> lay['elem_mat'], via ans_parsing._parse_label_materials).
"""
import numpy as np


def _edge_owners(tri):
    """Arestas não orientadas da malha: (edges [E,2] ordenadas, owners [E,2] elementos;
    -1 na 2ª coluna = aresta de contorno do domínio)."""
    tri = np.asarray(tri, np.int64)
    m = len(tri)
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(axis=1)
    own = np.tile(np.arange(m, dtype=np.int64), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, own = e[order], own[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    first = np.ones(len(e), bool)
    first[1:] = ~same
    idx = np.flatnonzero(first)
    owners = np.full((len(idx), 2), -1, np.int64)
    owners[:, 0] = own[idx]
    # segundo dono: a entrada seguinte, quando é a mesma aresta
    nxt = idx + 1
    ok = nxt < len(e)
    sec = np.zeros(len(idx), bool)
    sec[ok] = np.all(e[nxt[ok]] == e[idx[ok]], axis=1)
    owners[sec, 1] = own[nxt[sec]]
    return e[idx], owners


def interface_edges(tri, elem_mat):
    """Arestas [K,2] (índices de nó) entre 2 elementos de materiais diferentes."""
    edges, own = _edge_owners(tri)
    elem_mat = np.asarray(elem_mat)
    inner = own[:, 1] >= 0
    diff = np.zeros(len(edges), bool)
    diff[inner] = elem_mat[own[inner, 0]] != elem_mat[own[inner, 1]]
    return edges[diff]
