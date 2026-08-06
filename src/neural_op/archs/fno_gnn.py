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

    # [REMOVIDO] _GRID_IN_CH/_GRID_OUT_CH/_NODE_IN_CH/_GNN_IN_CH constantes de
    # classe hardcoded (2/2/5/7) — quebravam com datasets de parser diferente
    # (ex: FEMM_MESH_A_PARSER: y_hw/node_y de 1 canal em vez de 2). Viraram
    # parâmetros do construtor (grid_in_ch/grid_out_ch/node_in_ch), auto-
    # detectados a partir dos chunks reais em FNO_GNNConfig/NnCfg.__post_init__
    # (helper _detect_chunk_dims) — mesmo padrão já usado para edge_dim.
    #
    # _GRID_IN_CH  = 2
    # _GRID_OUT_CH = 2
    # _NODE_IN_CH  = 5
    # _GNN_IN_CH   = _NODE_IN_CH + _GRID_OUT_CH    # 7

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
                 gnn_n_layers,
                 edge_dim,
                 grid_in_ch,
                 grid_out_ch,
                 node_in_ch):
        super().__init__()

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
        # [REMOVIDO] batch['node_y'][:, :2] — fatiava 2 colunas fixas (Bx,By);
        # node_y já vem do parser só com as colunas-alvo (2 pra B, 1 pra A), o
        # slice era redundante e presumia B. grid_out_ch (auto-detectado) já
        # garante que a saída do modelo bate com node_y.shape[1] sem slicing.
        y_node     = batch['node_y'].to(device)
        if subtract:
            y_hw_fno, fno_at_nodes, delta = model(x_hw, node_x, edge_index, edge_attr, L,
                                                   return_components=True)
            loss_grid  = loss_fn(y_hw_fno, y_hw)
            # [REMOVIDO] loss_nodes = loss_fn(delta, y_node) — comparava a correção
            # bruta do GNN contra o alvo inteiro, não contra o resíduo real da
            # baseline FNO. Corrigido para residual learning de fato: delta deve
            # aprender (y_node - fno_at_nodes), não y_node diretamente.
            loss_nodes = loss_fn(delta, y_node - fno_at_nodes)
        else:
            y_hw_fno, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)
            loss_grid  = loss_fn(y_hw_fno, y_hw)
            loss_nodes = loss_fn(y_nodes,  y_node)
        return lambda_loss * loss_grid + (1.0 - lambda_loss) * loss_nodes

    return fno_gnn_step


def fno_gnn_metric_fn(batch, model, device):
    """
    MAE bruto (sem máscara): mae_hw compara a saída do FNO (grade H×W) contra
    y_hw; mae_graph compara a saída final do modelo nos nós (pós-GNN) contra node_y.

    batch já chega normalizado (CUDAPrefetcher.encode_batch, se normalize=True)
    — decodifica pred/y de volta pra unidade física antes do MAE, pra manter o
    significado documentado de mae_hw/mae_graph em metrics.jsonl.
    """
    normalizer = getattr(model, 'normalizer', None)
    x_hw       = batch['x_hw'].to(device)
    node_x     = batch['node_x'].to(device)
    edge_index = batch['edge_index'].to(device)
    edge_attr  = batch['edge_attr'].to(device)
    L          = batch['L'].to(device)
    y_hw       = batch['y_hw'].to(device)
    y_node     = batch['node_y'].to(device)   # [REMOVIDO] [:, :2] — ver fno_gnn_step
    with torch.no_grad():
        y_hw_fno, y_nodes = model(x_hw, node_x, edge_index, edge_attr, L)
        if normalizer is not None:
            y_hw_fno = normalizer.decode(y_hw_fno, 'y_hw')
            y_nodes  = normalizer.decode(y_nodes,  'node_y')
            y_hw     = normalizer.decode(y_hw,     'y_hw')
            y_node   = normalizer.decode(y_node,   'node_y')
        mae_hw    = torch.mean(torch.abs(y_hw_fno - y_hw)).item()
        mae_graph = torch.mean(torch.abs(y_nodes  - y_node)).item()
    return mae_hw, mae_graph
