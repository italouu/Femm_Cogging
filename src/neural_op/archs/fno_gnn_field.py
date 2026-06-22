import torch
from src.neural_op.archs.fno_gnn import FNO_GNN, _interpolate_fno_to_nodes


class FNO_GNN_Field(FNO_GNN):
    """
    Variante de FNO_GNN: o GNN prediz o campo diretamente nos nós, em vez de uma
    correção residual (Δ) somada à interpolação do FNO.

    Pipeline
    --------
    x_hw  →  FNO2d  →  y_hw_fno
                ↓  _interpolate_fno_to_nodes
    [node_x | y_fno@nodes]  →  GNN  →  y_nodes   (campo novo — substitui fno_at_nodes + Δ)

    Mesmo construtor, mesma config (FNO_GNNConfig) e mesma step_fn de FNO_GNN —
    só o forward muda (sem soma residual).
    """

    def forward(self, x_hw, node_x, edge_index, edge_attr, L):
        y_hw_fno     = self.fno(x_hw)
        fno_at_nodes = _interpolate_fno_to_nodes(y_hw_fno, node_x, L)
        gnn_input    = torch.cat([node_x, fno_at_nodes], dim=-1)
        y_nodes      = self.gnn(gnn_input, edge_index, edge_attr)
        return y_hw_fno, y_nodes
