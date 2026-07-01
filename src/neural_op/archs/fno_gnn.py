import torch
import torch.nn.functional as F
from src.neural_op.archs._blocks import GNN
from src.neural_op.archs.fno import FNO2d


def _interpolate_fno_to_nodes(fno_out, node_x, L):
    """
    Interpolação bilinear da saída do FNO (grade regular) nas posições dos nós quadtree.

    fno_out : [B, C, H, W]
    node_x  : [S_tot, 5]   colunas 3,4 = r_base, c_base ∈ [0,1]
    L       : [B]           nós por amostra
    Retorna : [S_tot, C]
    """
    B, C, _, _ = fno_out.shape
    device = fno_out.device
    dtype  = fno_out.dtype

    # F.grid_sample usa coordenadas em [-1, 1]; eixo x=coluna, eixo y=linha
    r_norm = 2.0 * node_x[:, 3] - 1.0
    c_norm = 2.0 * node_x[:, 4] - 1.0

    fno_at_nodes = torch.empty(node_x.size(0), C, device=device, dtype=dtype)
    offset = 0
    for b in range(B):
        n    = int(L[b].item())
        grid = torch.stack(
            [c_norm[offset:offset + n], r_norm[offset:offset + n]], dim=-1
        ).unsqueeze(0).unsqueeze(2)                                   # [1, n, 1, 2]

        interp = F.grid_sample(
            fno_out[b:b + 1], grid,
            mode='bilinear', align_corners=True, padding_mode='border',
        )                                                              # [1, C, n, 1]
        fno_at_nodes[offset:offset + n] = interp[0, :, :, 0].T       # [n, C]
        offset += n

    return fno_at_nodes   # [S_tot, C]


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

    _GRID_IN_CH  = 2
    _GRID_OUT_CH = 2
    _NODE_IN_CH  = 5
    _EDGE_DIM    = 4
    _GNN_IN_CH   = _NODE_IN_CH + _GRID_OUT_CH    # 7

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
                 # gnn_msg_width,  # [REMOVIDO] — gate escalar não usa msg_width
                 gnn_n_layers):
        super().__init__()

        self.fno = FNO2d(
            in_channels=self._GRID_IN_CH,
            out_channels=self._GRID_OUT_CH,
            modes1=fno_modes1,   modes2=fno_modes2,
            conv_width=fno_conv_width, conv_layers=fno_conv_layers,
            lift_width=fno_lift_width, lift_layers=fno_lift_layers,
            proj_width=fno_proj_width, proj_layers=fno_proj_layers,
            data_res=data_res,
        )
        self.gnn = GNN(
            in_node_features=self._GNN_IN_CH,
            out_node_features=self._GRID_OUT_CH,
            edge_dim=self._EDGE_DIM,
            node_width=gnn_node_width,
            n_layers=gnn_n_layers,
        )

    def forward(self, x_hw, node_x, edge_index, edge_attr, L, return_components=False):
        y_hw_fno     = self.fno(x_hw)
        fno_at_nodes = _interpolate_fno_to_nodes(y_hw_fno, node_x, L)
        gnn_input    = torch.cat([node_x, fno_at_nodes], dim=-1)
        delta        = self.gnn(gnn_input, edge_index, edge_attr)
        if return_components:
            return y_hw_fno, fno_at_nodes, delta
        return y_hw_fno, fno_at_nodes + delta


def make_fno_gnn_step(lambda_loss, loss_cfg=None):
    """Retorna step_fn fechada sobre lambda_loss e loss_cfg."""
    subtract = getattr(loss_cfg, 'subtract_fno', False)

    def fno_gnn_step(batch, model, loss_fn, device):
        x_hw       = batch['x_hw'].to(device)
        node_x     = batch['node_x'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr  = batch['edge_attr'].to(device)
        L          = batch['L'].to(device)
        y_hw       = batch['y_hw'].to(device)
        y_node     = batch['node_y'][:, :2].to(device)
        if subtract:
            y_hw_fno, _, delta = model(x_hw, node_x, edge_index, edge_attr, L,
                                       return_components=True)
            loss_grid  = loss_fn(y_hw_fno, y_hw)
            loss_nodes = loss_fn(delta,    y_node)
        else:
            y_hw_fno, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)
            loss_grid  = loss_fn(y_hw_fno, y_hw)
            loss_nodes = loss_fn(y_nodes,  y_node)
        return lambda_loss * loss_grid + (1.0 - lambda_loss) * loss_nodes

    return fno_gnn_step
