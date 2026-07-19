import torch
from src.neural_op.archs._blocks  import GNN
from src.neural_op.archs.fno      import FNO2d
from src.neural_op.archs.fno_gnn  import _interpolate_fno_to_nodes
from src.neural_op.archs.fno_mat  import _make_material_masks

# [REMOVIDO] _material_ids_from_mu — material_ids por nó não é mais necessário;
# assemble agora é feito na grade antes de passar para a GNN.


class MaskedFNO_GNN(torch.nn.Module):
    """
    FNO_GNN com mascaramento por material em ambas as saídas (grade e nós).

    Pipeline
    --------
    x_hw  →  FNO2d(out=8)  →  y_hw_8               [B, 8, H, W]  (loss mascarada aqui)
                  ↓  assemble_grid (8→2)
             y_hw_2                                  [B, 2, H, W]
                  ↓  _interpolate_fno_to_nodes
    [node_x | y_fno@nodes_2]  →  GNN(in=7, out=2)  →  Δ_2
    y_nodes_2 = y_fno@nodes_2 + Δ_2                [S_tot, 2]  (loss MSE aqui)

    Canal layout (grade e nós):
        [:, 0:2] / [:, 0:2]  ferro  (Bx_ferro, By_ferro)
        [:, 2:4] / [:, 2:4]  ar     (Bx_ar,    By_ar)
        [:, 4:6] / [:, 4:6]  ima    (Bx_ima,   By_ima)
        [:, 6:8] / [:, 6:8]  cobre  (Bx_cobre, By_cobre)
    """

    _GRID_IN_CH  = 2
    _GRID_OUT_CH = 8    # 4 materiais × 2 campos (saída FNO antes do assemble)
    _NODE_IN_CH  = 5    # mu_r, M, cell_area, r_base, c_base  (MASKED_FNO_GNN_PARSER)
    _EDGE_DIM    = 4
    _GNN_IN_CH   = _NODE_IN_CH + 2   # 5 + 2 (FNO assemblado 8→2 antes da GNN)

    def __init__(self,
                 fno_modes1, fno_modes2,
                 fno_conv_width, fno_conv_layers,
                 fno_lift_width, fno_lift_layers,
                 fno_proj_width, fno_proj_layers,
                 data_res,
                 gnn_node_width, gnn_n_layers):
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
            out_node_features=2,
            edge_dim=self._EDGE_DIM,
            node_width=gnn_node_width,
            n_layers=gnn_n_layers,
        )

    def forward(self, x_hw, node_x, edge_index, edge_attr, L):
        y_hw_8       = self.fno(x_hw)                                    # [B, 8, H, W]
        masks        = _make_material_masks(x_hw[:, 0])                  # [B, 4, H, W]
        y_hw_2       = self.assemble_grid(y_hw_8, masks)                 # [B, 2, H, W]
        fno_at_nodes = _interpolate_fno_to_nodes(y_hw_2, node_x, L)     # [S_tot, 2]
        gnn_input    = torch.cat([node_x, fno_at_nodes], dim=-1)         # [S_tot, 7]
        delta        = self.gnn(gnn_input, edge_index, edge_attr)        # [S_tot, 2]
        return y_hw_8, masks, fno_at_nodes + delta                       # [B,8,H,W], [B,4,H,W], [S_tot,2]

    def assemble_grid(self, pred8, masks):
        """[B,8,H,W] + [B,4,H,W] bool → [B,2,H,W]  — idêntico a MaskedFNO2d.assemble"""
        out = pred8.new_zeros(pred8.shape[0], 2, pred8.shape[2], pred8.shape[3])
        for m in range(4):
            out += pred8[:, 2 * m: 2 * m + 2] * masks[:, m: m + 1].float()
        return out

    # [REMOVIDO] assemble_nodes — GNN agora produz 2 canais diretamente; assemble por nó desnecessário.


def make_masked_fno_gnn_step(lambda_loss):
    """Retorna step_fn fechada sobre lambda_loss (peso da loss de grade)."""
    def masked_fno_gnn_step(batch, model, loss_fn, device):
        x_hw       = batch['x_hw'].to(device)         # [B, 2, H, W]
        node_x     = batch['node_x'].to(device)        # [S_tot, 5]
        edge_index = batch['edge_index'].to(device)
        edge_attr  = batch['edge_attr'].to(device)
        L          = batch['L'].to(device)
        y_hw       = batch['y_hw'].to(device)          # [B, 2, H, W]
        y_node     = batch['node_y'][:, :2].to(device) # [S_tot, 2]

        y_hw_8, masks, y_nodes_2 = model(x_hw, node_x, edge_index, edge_attr, L)

        # loss_fn deve ser 'masked_fno_gnn_loss' — assinatura estendida com lambda_loss
        return loss_fn(y_hw_8, y_hw, masks, y_nodes_2, y_node, lambda_loss)

    return masked_fno_gnn_step


def masked_fno_gnn_metric_fn(batch, model, device):
    """
    MAE bruto: mae_hw compara o FNO assemblado por material (grade H×W) contra y_hw;
    mae_graph compara a saída final da GNN nos nós contra node_y.
    """
    x_hw       = batch['x_hw'].to(device)
    node_x     = batch['node_x'].to(device)
    edge_index = batch['edge_index'].to(device)
    edge_attr  = batch['edge_attr'].to(device)
    L          = batch['L'].to(device)
    y_hw       = batch['y_hw'].to(device)
    y_node     = batch['node_y'][:, :2].to(device)
    with torch.no_grad():
        y_hw_8, masks, y_nodes_2 = model(x_hw, node_x, edge_index, edge_attr, L)
        y_hw_assembled = model.assemble_grid(y_hw_8, masks)
        mae_hw    = torch.mean(torch.abs(y_hw_assembled - y_hw)).item()
        mae_graph = torch.mean(torch.abs(y_nodes_2 - y_node)).item()
    return mae_hw, mae_graph
