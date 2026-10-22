import torch
from src.neural_op.archs._blocks import GNN
from src.neural_op.archs.fno import FNO2d
from src.neural_op.archs.interp import interpolate_grid_to_nodes


def _interpolate_fno_to_nodes(fno_out, node_x, L):
    """
    Interpolação bilinear da saída do FNO (grade regular) nas posições dos nós quadtree.

    fno_out : [B, C, H, W]
    node_x  : [S_tot, 5]   colunas 3,4 = r_base, c_base ∈ [0,1]
    L       : [B]           nós por amostra
    Retorna : [S_tot, C]    (interpolação: src/neural_op/archs/interp.py)
    """
    return interpolate_grid_to_nodes(fno_out, node_x[:, 3], node_x[:, 4], L)


def _rescale_fno_to_node_space(fno_at_nodes, normalizer):
    """B2 (2026-10-03): o estágio FNO produz saída no espaço normalizado de
    y_hw, mas o resíduo da GNN é somado e comparado com node_y, normalizado
    com OUTRAS stats (σ Bx 0,391 pixels × 0,539 nós; σ By 0,420 × 0,522 no
    dataset unificado). Decodifica com as stats de y_hw e recodifica com as de
    node_y — mesmo round-trip de GNN_PostBase._base_pred_at_nodes. No-op sem
    normalizer (treino sem normalização: as duas escalas já são a física)."""
    if normalizer is None:
        return fno_at_nodes
    return normalizer.encode(normalizer.decode(fno_at_nodes, 'y_hw'), 'node_y')


class FNO_GNN(torch.nn.Module):
    """
    Arquitetura FNO + GNN para predição de B-field em resolução quadtree.

    Pipeline
    --------
    x_hw  →  FNO2d  →  y_hw_fno               (predição na grade regular)
                ↓  _interpolate_fno_to_nodes
    [node_x | y_fno@nodes]  →  GNN  →  Δ      (correção residual nos nós)
    y_nodes = y_fno@nodes + Δ

    Parâmetros FNO  : fno_modes1/2, fno_conv_width/layers, fno_lift_width/layers,
                      fno_proj_width/layers, data_res
    Parâmetros GNN  : gnn_node_width, gnn_n_layers
    """

    def __init__(self,
                 fno_modes1,
                 fno_modes2,
                 fno_conv_width,
                 fno_conv_layers,
                 fno_lift_width,
                 fno_lift_layers,
                 fno_proj_width,
                 fno_proj_layers,
                 data_res,
                 gnn_node_width,
                 gnn_n_layers,
                 edge_dim,
                 grid_in_ch,
                 grid_out_ch,
                 node_in_ch):
        super().__init__()
        # B2 — FNO@nós recodificado de stats y_hw -> node_y antes do resíduo
        # (_rescale_fno_to_node_space), sempre (fno_node_rescale fixo desde 2026-10-21)
        self.normalizer = None   # atribuído externamente (scripts/train.py/eval.py)

        self.fno = FNO2d(
            in_channels=grid_in_ch,
            out_channels=grid_out_ch,
            modes1=fno_modes1,   modes2=fno_modes2,
            conv_width=fno_conv_width, conv_layers=fno_conv_layers,
            lift_width=fno_lift_width, lift_layers=fno_lift_layers,
            proj_width=fno_proj_width, proj_layers=fno_proj_layers,
            data_res=data_res,
        )
        self.gnn = GNN(
            in_node_features=node_in_ch + grid_out_ch,
            out_node_features=grid_out_ch,
            edge_dim=edge_dim,
            node_width=gnn_node_width,
            n_layers=gnn_n_layers,
        )

    def forward(self, x_hw, node_x, edge_index, edge_attr, L, return_components=False):
        y_hw_fno     = self.fno(x_hw)
        fno_at_nodes = _interpolate_fno_to_nodes(y_hw_fno, node_x, L)
        fno_at_nodes = _rescale_fno_to_node_space(fno_at_nodes, self.normalizer)
        gnn_input    = torch.cat([node_x, fno_at_nodes], dim=-1)
        delta        = self.gnn(gnn_input, edge_index, edge_attr)
        if return_components:
            return y_hw_fno, fno_at_nodes, delta
        return y_hw_fno, fno_at_nodes + delta


def fno_gnn_step(batch, model, loss_fn, device):
    """Loss da saída final nos nós (estágio FNO interpolado + GNN) contra node_y."""
    _, y_nodes = model(batch['x_hw'].to(device), batch['node_x'].to(device),
                       batch['edge_index'].to(device), batch['edge_attr'].to(device),
                       batch['L'].to(device))
    return loss_fn(y_nodes, batch['node_y'].to(device))

def fno_gnn_metric_fn(batch, model, device):
    """
    MAE bruto (sem máscara) só no grafo: mae_graph compara a saída final do modelo
    nos nós (pós-GNN) contra node_y; mae_hw = None (a grade do FNO é intermediária,
    não supervisionada).

    batch já chega normalizado (CUDAPrefetcher.encode_batch, se normalize=True)
    — decodifica pred/y de volta pra unidade física antes do MAE, pra manter o
    significado documentado de mae_graph em metrics.jsonl.
    """
    normalizer = getattr(model, 'normalizer', None)
    x_hw       = batch['x_hw'].to(device)
    node_x     = batch['node_x'].to(device)
    edge_index = batch['edge_index'].to(device)
    edge_attr  = batch['edge_attr'].to(device)
    L          = batch['L'].to(device)
    y_node     = batch['node_y'].to(device)
    with torch.no_grad():
        _, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)
        if normalizer is not None:
            y_nodes  = normalizer.decode(y_nodes,  'node_y')
            y_node   = normalizer.decode(y_node,   'node_y')
        mae_hw    = None
        mae_graph = torch.mean(torch.abs(y_nodes  - y_node)).item()
    return mae_hw, mae_graph
