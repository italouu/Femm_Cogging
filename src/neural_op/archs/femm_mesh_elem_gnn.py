"""
femm_mesh_elem_gnn.py
---------------------
PROTÓTIPO (2026-10-07) -- FNO_BipartiteGNN com os papéis dos grafos
TROCADOS: o grafo de ELEMENTOS é o principal (iterado n_layers, saída Bx,By
por elemento) e o de VÉRTICES é o auxiliar (estático, injetado a cada camada
por arestas cruzadas vértice -> elemento). Contrato de dados: bloco
"CONTRATO DO LAYOUT Elem" da docstring de src/data_gen/parsers/femm_mesh_elem.py
(chaves node_* = elementos, elem_* = vértices).

Mesmo BipartiteGNN (_blocks.py) da bipartite, sem mudança -- só o que entra
em cada papel muda:

    x_hw  →  FNO2d  →  y_hw_fno
               ↓ interpolado nos CENTRÓIDES (node_x[:,0:2])
    [node_x | y_fno@centróides]  →  BipartiteGNN  →  Δ
               ↑ auxiliar: elem_x (vértices, só posição)
    y_elem = y_fno@centróides + Δ

O auxiliar (vértices) recebe só a posição -- troca "pura" (aux_fno=False,
decisão do usuário 2026-10-07; a opção de concatenar o FNO interpolado no
vértice saiu em 2026-10-21, sem uso).

Mesma assinatura de forward de FNO_BipartiteGNN -- make_fno_bipartite_gnn_step
e fno_bipartite_gnn_metric_fn reaproveitados sem mudança (mae_graph aqui é o
MAE por ELEMENTO).
"""
import torch

from src.neural_op.archs.femm_mesh_v2_gnn import (
    FNO_BipartiteGNN, _interpolate_fno_to_nodes_v2,
)
from src.neural_op.archs.fno_gnn import _rescale_fno_to_node_space


class FNO_BipartiteGNN_Elem(FNO_BipartiteGNN):

    def _fno_at(self, y_hw_fno, pos, counts):
        out = _interpolate_fno_to_nodes_v2(y_hw_fno, pos, counts)
        return _rescale_fno_to_node_space(out, self.normalizer)

    def forward(self, x_hw, node_x, elem_x, edge_index, edge_attr,
                cross_edge_index, cross_edge_attr, L, return_components=False):
        y_hw_fno = self.fno(x_hw)
        fno_at_main = self._fno_at(y_hw_fno, node_x, L)
        delta = self.gnn(torch.cat([node_x, fno_at_main], dim=-1), elem_x,
                         edge_index, edge_attr, cross_edge_index, cross_edge_attr)
        if return_components:
            return y_hw_fno, fno_at_main, delta
        return y_hw_fno, fno_at_main + delta
