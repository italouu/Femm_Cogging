import torch
from src.neural_op.archs._blocks import BipartiteGNN
from src.neural_op.archs.fno import FNO2d
from src.neural_op.archs.interp import interpolate_grid_to_nodes
from src.neural_op.archs.fno_gnn import _rescale_fno_to_node_space


def _interpolate_fno_to_nodes_v2(fno_out, node_x, L):
    """
    Idêntica em espírito a fno_gnn.py::_interpolate_fno_to_nodes, mas node_x
    aqui tem layout [r_base, c_base] (colunas 0,1) -- não
    [mu_r, M, área dual, r_base, c_base, J] (colunas 3,4) do mesh_v1 (grafo de
    vértices). Ver src/data_gen/parsers/femm_mesh_v2.py.

    fno_out : [B, C, H, W]
    node_x  : [S_tot, 2]  colunas 0,1 = r_base, c_base ∈ [0,1]
    L       : [B]          nós por amostra
    Retorna : [S_tot, C]   (interpolação: src/neural_op/archs/interp.py)
    """
    return interpolate_grid_to_nodes(fno_out, node_x[:, 0], node_x[:, 1], L)


class FNO_BipartiteGNN(torch.nn.Module):
    """
    FNO (grade) + GNN de grafo duplo -- vértices iterados (n_layers de
    message passing) + elementos estáticos (nunca atualizados, injetados a
    cada camada via arestas cruzadas) -- ver
    src/neural_op/archs/_blocks.py::BipartiteGNN e src/data_gen/femm_mesh_v2.py
    para o contrato de dados. Decisão de arquitetura registrada na conversa
    de 2026-08-10 (ver CLAUDE.md).

    Pipeline
    --------
    x_hw  →  FNO2d  →  y_hw_fno                       (predição na grade regular)
                ↓  _interpolate_fno_to_nodes_v2
    [node_x | y_fno@nodes]  →  BipartiteGNN(+elem_x via arestas cruzadas)  →  Δ
    y_nodes = y_fno@nodes + Δ

    Parâmetros FNO  : fno_modes1/2, fno_conv_width/layers, fno_lift_width/layers,
                      fno_proj_width/layers, data_res
    Parâmetros GNN  : gnn_node_width, gnn_n_layers
    """

    def __init__(self,
                 fno_modes1, fno_modes2, fno_conv_width, fno_conv_layers,
                 fno_lift_width, fno_lift_layers, fno_proj_width, fno_proj_layers,
                 data_res,
                 gnn_node_width, gnn_n_layers,
                 edge_dim, grid_in_ch, grid_out_ch, node_in_ch,
                 elem_in_ch, cross_edge_dim):
        super().__init__()
        # B2 — FNO@nós sempre recodificado (fno_gnn.py::_rescale_fno_to_node_space)
        self.normalizer = None   # atribuído externamente (scripts/train.py, postprocess/inference.py)

        self.fno = FNO2d(
            in_channels=grid_in_ch, out_channels=grid_out_ch,
            modes1=fno_modes1,   modes2=fno_modes2,
            conv_width=fno_conv_width, conv_layers=fno_conv_layers,
            lift_width=fno_lift_width, lift_layers=fno_lift_layers,
            proj_width=fno_proj_width, proj_layers=fno_proj_layers,
            data_res=data_res,
        )
        self.gnn = BipartiteGNN(
            in_node_features=node_in_ch + grid_out_ch,
            out_node_features=grid_out_ch,
            edge_dim=edge_dim,
            elem_in_ch=elem_in_ch,
            cross_edge_dim=cross_edge_dim,
            node_width=gnn_node_width,
            n_layers=gnn_n_layers,
        )

    def forward(self, x_hw, node_x, elem_x, edge_index, edge_attr,
                cross_edge_index, cross_edge_attr, L, return_components=False):
        y_hw_fno     = self.fno(x_hw)
        fno_at_nodes = _interpolate_fno_to_nodes_v2(y_hw_fno, node_x, L)
        fno_at_nodes = _rescale_fno_to_node_space(fno_at_nodes, self.normalizer)
        gnn_input    = torch.cat([node_x, fno_at_nodes], dim=-1)
        delta        = self.gnn(gnn_input, elem_x, edge_index, edge_attr,
                                 cross_edge_index, cross_edge_attr)
        if return_components:
            return y_hw_fno, fno_at_nodes, delta
        return y_hw_fno, fno_at_nodes + delta


def fno_bipartite_gnn_step(batch, model, loss_fn, device):
    """Loss da saída final do grafo principal contra node_y -- nós (FNO_BipartiteGNN, v3)
    ou elementos (FNO_BipartiteGNN_Elem, mesma assinatura)."""
    _, y_nodes = model(
        batch['x_hw'].to(device), batch['node_x'].to(device), batch['elem_x'].to(device),
        batch['edge_index'].to(device), batch['edge_attr'].to(device),
        batch['cross_edge_index'].to(device), batch['cross_edge_attr'].to(device),
        batch['L'].to(device))
    return loss_fn(y_nodes, batch['node_y'].to(device))

def fno_bipartite_gnn_metric_fn(batch, model, device):
    """MAE bruto (sem máscara) só no grafo -- mesmo padrão de fno_gnn.py::fno_gnn_metric_fn:
    mae_graph = saída final nos nós (por ELEMENTO na Elem, que reaproveita esta função);
    mae_hw = None (grade do FNO intermediária, não supervisionada).

    batch já chega normalizado (CUDAPrefetcher.encode_batch, se normalize=True)
    -- decodifica pred/y de volta pra unidade física antes do MAE."""
    normalizer = getattr(model, 'normalizer', None)
    x_hw             = batch['x_hw'].to(device)
    node_x           = batch['node_x'].to(device)
    elem_x           = batch['elem_x'].to(device)
    edge_index       = batch['edge_index'].to(device)
    edge_attr        = batch['edge_attr'].to(device)
    cross_edge_index = batch['cross_edge_index'].to(device)
    cross_edge_attr  = batch['cross_edge_attr'].to(device)
    L                = batch['L'].to(device)
    y_node           = batch['node_y'].to(device)
    with torch.no_grad():
        _, y_nodes = model(
            x_hw, node_x, elem_x, edge_index, edge_attr,
            cross_edge_index, cross_edge_attr, L)
        if normalizer is not None:
            y_nodes  = normalizer.decode(y_nodes,  'node_y')
            y_node   = normalizer.decode(y_node,   'node_y')
        mae_hw    = None
        mae_graph = torch.mean(torch.abs(y_nodes  - y_node)).item()
    return mae_hw, mae_graph
