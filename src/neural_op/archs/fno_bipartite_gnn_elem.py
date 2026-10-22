"""
fno_bipartite_gnn_elem.py
-------------------------
FNO_BipartiteGNN com os papéis dos grafos TROCADOS: o
grafo de ELEMENTOS é o principal (iterado n_layers, saída (Br, Bθ) por elemento
no dataset; (Bx, By) só no gabarito target_field='B' do torque) e o de VÉRTICES
é o auxiliar (estático, injetado a cada camada por arestas cruzadas vértice ->
elemento). Contrato de dados: bloco "CONTRATO DO LAYOUT Elem" da docstring de
src/data_gen/parsers/elem_structure.py (chaves node_* = elementos, elem_* = vértices).

A REDE É A MESMA da FNO_BipartiteGNN (subclasse; mesmo FNO2d + BipartiteGNN de
_blocks.py, mesmos hiperparâmetros) -- a diferença está toda no layout dos dados,
isto é, no que entra em cada papel:

    x_hw  →  FNO2d  →  y_hw_fno
               ↓ interpolado nos CENTRÓIDES (node_x[:,0:2])
    [node_x | y_fno@centróides]  →  BipartiteGNN  →  Δ
               ↑ auxiliar: elem_x (vértices, só posição)
    y_elem = y_fno@centróides + Δ

O auxiliar (vértices) recebe só a posição (o FNO interpolado entra só no
grafo principal).

Mesma assinatura de forward de FNO_BipartiteGNN -- make_fno_bipartite_gnn_step
e fno_bipartite_gnn_metric_fn reaproveitados sem mudança (mae_graph aqui é o
MAE por ELEMENTO).
"""
import torch

from src.neural_op.archs.fno_bipartite_gnn import (
    FNO_BipartiteGNN, _interpolate_fno_to_bipartite_nodes,
)
from src.neural_op.archs.fno_gnn import _rescale_fno_to_node_space


class FNO_BipartiteGNN_Elem(FNO_BipartiteGNN):

    def _fno_at(self, y_hw_fno, pos, counts):
        out = _interpolate_fno_to_bipartite_nodes(y_hw_fno, pos, counts)
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
