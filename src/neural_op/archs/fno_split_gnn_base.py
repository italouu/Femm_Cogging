"""
fno_split_gnn_base.py
---------------------
FNO_SplitGNN com saída e loss só em N_base (vértices divididos por material). Mesmos dados
(chunks BLDC_138x276/FNO_SplitGNN, contrato em src/data_gen/parsers/split_layout.py:
node_* = N_aux, elem_* = N_base), mesmo FNO2d, mesmas features e mesmas camadas da GNN;
N_aux continua no grafo como nó de passagem de mensagem, mas sem gabarito.

    x_hw  →  FNO2d  →  y_hw_fno
               ↓ interpolado nos centróides (node_x[:,0:2]) e nas cópias (elem_x[:,0:2])
    [x_aux | y_fno@aux], [x_base | y_fno@base]  →  SplitGNNBase  →  Δ_base
    y_base = y_fno@base + Δ_base

Loss = loss(y_base, elem_y). Independente de fno_split_gnn.py (que é pendente de exclusão).
"""
import torch

from src.neural_op.archs._blocks import SplitGNNBase
from src.neural_op.archs.fno import FNO2d
from src.neural_op.archs.interp import interpolate_grid_to_nodes


def _gnn_features(x):
    """node_x/elem_x sem a coluna c_base (1)."""
    return torch.cat([x[:, :1], x[:, 2:]], dim=-1)


class FNO_SplitGNN_Base(torch.nn.Module):

    def __init__(self,
                 fno_modes1, fno_modes2, fno_conv_width, fno_conv_layers,
                 fno_lift_width, fno_lift_layers, fno_proj_width, fno_proj_layers,
                 data_res,
                 gnn_node_width, gnn_n_layers,
                 edge_dim, grid_in_ch, grid_out_ch, node_in_ch,
                 elem_in_ch, cross_edge_dim):
        super().__init__()
        # FNO@pontos recodificado de stats y_hw -> node_y / elem_y (_fno_at)
        self.normalizer = None   # atribuído externamente (scripts/train.py)

        self.fno = FNO2d(
            in_channels=grid_in_ch, out_channels=grid_out_ch,
            modes1=fno_modes1,   modes2=fno_modes2,
            conv_width=fno_conv_width, conv_layers=fno_conv_layers,
            lift_width=fno_lift_width, lift_layers=fno_lift_layers,
            proj_width=fno_proj_width, proj_layers=fno_proj_layers,
            data_res=data_res,
        )
        self.gnn = SplitGNNBase(
            in_aux_features=node_in_ch - 1 + grid_out_ch,
            in_base_features=elem_in_ch - 1 + grid_out_ch,
            out_features=grid_out_ch,
            dual_edge_dim=edge_dim,
            aux_edge_dim=cross_edge_dim,
            node_width=gnn_node_width,
            n_layers=gnn_n_layers,
        )

    def _fno_at(self, y_hw_fno, pos, counts, key):
        """Bilinear da grade nos pontos (pos[:,0:2] = r_base, c_base) e recodificação
        das stats de y_hw para as de `key` (mesmo round-trip de
        fno_gnn._rescale_fno_to_node_space)."""
        out = interpolate_grid_to_nodes(y_hw_fno, pos[:, 0], pos[:, 1], counts)
        if self.normalizer is None:
            return out
        return self.normalizer.encode(self.normalizer.decode(out, 'y_hw'), key)

    def forward(self, x_hw, node_x, elem_x, edge_index, edge_attr,
                cross_edge_index, cross_edge_attr, L, elem_L, return_components=False):
        y_hw_fno = self.fno(x_hw)
        u_aux  = self._fno_at(y_hw_fno, node_x, L, 'node_y')
        u_base = self._fno_at(y_hw_fno, elem_x, elem_L, 'elem_y')
        d_base = self.gnn(
            torch.cat([_gnn_features(node_x), u_aux], dim=-1),
            torch.cat([_gnn_features(elem_x), u_base], dim=-1),
            edge_index, edge_attr, cross_edge_index, cross_edge_attr)
        if return_components:
            return y_hw_fno, (u_base, d_base)
        return y_hw_fno, u_base + d_base


def _forward(batch, model, device):
    keys = ('x_hw', 'node_x', 'elem_x', 'edge_index', 'edge_attr',
            'cross_edge_index', 'cross_edge_attr', 'L', 'elem_L')
    return model(*(batch[k].to(device) for k in keys))


def fno_split_gnn_base_step(batch, model, loss_fn, device):
    """loss(N_base): saída final contra elem_y."""
    _, y_base = _forward(batch, model, device)
    return loss_fn(y_base, batch['elem_y'].to(device))


def fno_split_gnn_base_metric_fn(batch, model, device):
    """MAE bruto [T] -- mae_graph = MAE de N_base; mae_hw = None (grade do FNO
    intermediária, não supervisionada)."""
    normalizer = getattr(model, 'normalizer', None)
    y_t = batch['elem_y'].to(device)
    with torch.no_grad():
        _, y = _forward(batch, model, device)
        if normalizer is not None:
            y   = normalizer.decode(y,   'elem_y')
            y_t = normalizer.decode(y_t, 'elem_y')
        mae_graph = torch.mean(torch.abs(y - y_t)).item()
    return None, mae_graph
