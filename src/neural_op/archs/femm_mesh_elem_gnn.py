"""
femm_mesh_elem_gnn.py
---------------------
PROTÓTIPO (2026-10-07) -- FNO_BipartiteGNN com os papéis dos grafos
TROCADOS: o grafo de ELEMENTOS é o principal (iterado n_layers, saída Bx,By
por elemento) e o de VÉRTICES é o auxiliar (estático, injetado a cada camada
por arestas cruzadas vértice -> elemento). Contrato de dados:
src/data_gen/parsers/femm_mesh_elem.py (chaves node_* = elementos,
elem_* = vértices).

Mesmo BipartiteGNN (_blocks.py) da bipartite, sem mudança -- só o que entra
em cada papel muda:

    x_hw  →  FNO2d  →  y_hw_fno
               ↓ interpolado nos CENTRÓIDES (node_x[:,0:2])
    [node_x | y_fno@centróides]  →  BipartiteGNN  →  Δ
               ↑ auxiliar: [elem_x | y_fno@vértices]  (aux_fno=True)
    y_elem = y_fno@centróides + Δ

`aux_fno`: no grafo original o auxiliar (elementos) carregava o material; no
trocado o auxiliar (vértices) só teria posição -- aux_fno=True concatena a
predição do FNO interpolada no vértice, dando aos elementos o FNO nos 3
cantos. aux_fno=False = troca "pura".

Mesma assinatura de forward de FNO_BipartiteGNN -- make_fno_bipartite_gnn_step
e fno_bipartite_gnn_metric_fn reaproveitados sem mudança (mae_graph aqui é o
MAE por ELEMENTO). O nº de vértices por amostra (elem_L, necessário pra
interpolar o FNO nos vértices) é deduzido das arestas cruzadas, já que o
step_fn genérico não o repassa: todo vértice pertence a algum elemento, e os
vértices de cada amostra são contíguos no batch.
"""
import torch

from src.neural_op.archs.femm_mesh_v2_gnn import (
    FNO_BipartiteGNN, _interpolate_fno_to_nodes_v2,
)
from src.neural_op.archs.fno_gnn import _rescale_fno_to_node_space
from src.neural_op.archs._blocks import BipartiteGNN


def _aux_counts(L, cross_edge_index, n_aux):
    """Nós auxiliares (vértices) por amostra, a partir das arestas cruzadas
    (linha 0 = auxiliar, linha 1 = principal)."""
    B = L.numel()
    main_batch = torch.repeat_interleave(torch.arange(B, device=L.device), L)
    aux_batch = torch.empty(n_aux, dtype=torch.long, device=L.device)
    aux_batch[cross_edge_index[0]] = main_batch[cross_edge_index[1]]
    return torch.bincount(aux_batch, minlength=B)


class FNO_BipartiteGNN_Elem(FNO_BipartiteGNN):

    def __init__(self, *args, aux_fno=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_fno = aux_fno
        if aux_fno:
            # auxiliar ganha grid_out_ch colunas (FNO@vértices) -- refaz a GNN
            # com elem_in_ch maior; resto idêntico ao FNO_BipartiteGNN
            grid_out_ch = kwargs['grid_out_ch']
            self.gnn = BipartiteGNN(
                in_node_features=kwargs['node_in_ch'] + grid_out_ch,
                out_node_features=grid_out_ch,
                edge_dim=kwargs['edge_dim'],
                elem_in_ch=kwargs['elem_in_ch'] + grid_out_ch,
                cross_edge_dim=kwargs['cross_edge_dim'],
                node_width=kwargs['gnn_node_width'],
                n_layers=kwargs['gnn_n_layers'],
            )

    def _fno_at(self, y_hw_fno, pos, counts):
        out = _interpolate_fno_to_nodes_v2(y_hw_fno, pos, counts, mode=self.interp_mode)
        if self.fno_node_rescale:
            out = _rescale_fno_to_node_space(out, self.normalizer)
        return out

    def forward(self, x_hw, node_x, elem_x, edge_index, edge_attr,
                cross_edge_index, cross_edge_attr, L, return_components=False):
        y_hw_fno = self.fno(x_hw)
        fno_at_main = self._fno_at(y_hw_fno, node_x, L)
        aux_x = elem_x
        if self.aux_fno:
            aux_L = _aux_counts(L, cross_edge_index, elem_x.size(0))
            aux_x = torch.cat([elem_x, self._fno_at(y_hw_fno, elem_x, aux_L)], dim=-1)
        delta = self.gnn(torch.cat([node_x, fno_at_main], dim=-1), aux_x,
                         edge_index, edge_attr, cross_edge_index, cross_edge_attr)
        if return_components:
            return y_hw_fno, fno_at_main, delta
        return y_hw_fno, fno_at_main + delta


def femm_mesh_elem_eval_fn(model, chunk_data, eval_cfg):
    """Protótipo: sem plot ainda (femm_mesh_v2_eval_fn presume vértices como
    grafo principal). Treino e mae_hw/mae_graph funcionam."""
    raise NotImplementedError("eval/plot do FNO_BipartiteGNN_Elem ainda não implementado (protótipo)")
