import torch
import torch.nn.functional as F
from src.neural_op.archs._blocks import BipartiteGNN
from src.neural_op.archs.fno import FNO2d
from src.neural_op.losses import LOSS_REGISTRY


def _interpolate_fno_to_nodes_v2(fno_out, node_x, L):
    """
    Idêntica em espírito a fno_gnn.py::_interpolate_fno_to_nodes, mas node_x
    aqui tem layout [r_base, c_base] (colunas 0,1) -- não
    [material_id,mu_r,M,area,r_base,c_base,...] (colunas 3,4) do grafo
    qtree/femm_mesh v1. Ver src/data_gen/femm_mesh_v2.py.

    fno_out : [B, C, H, W]
    node_x  : [S_tot, 2]  colunas 0,1 = r_base, c_base ∈ [0,1]
    L       : [B]          nós por amostra
    Retorna : [S_tot, C]
    """
    B, C, _, _ = fno_out.shape
    device = fno_out.device
    dtype  = fno_out.dtype

    # F.grid_sample usa coordenadas em [-1, 1]; eixo x=coluna, eixo y=linha
    r_norm = 2.0 * node_x[:, 0] - 1.0
    c_norm = 2.0 * node_x[:, 1] - 1.0

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

    return fno_at_nodes


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
        gnn_input    = torch.cat([node_x, fno_at_nodes], dim=-1)
        delta        = self.gnn(gnn_input, elem_x, edge_index, edge_attr,
                                 cross_edge_index, cross_edge_attr)
        if return_components:
            return y_hw_fno, fno_at_nodes, delta
        return y_hw_fno, fno_at_nodes + delta


def make_fno_bipartite_gnn_step(lambda_loss, loss_cfg=None):
    """Retorna step_fn fechada sobre lambda_loss e loss_cfg -- mesmo padrão de
    fno_gnn.py::make_fno_gnn_step, adaptado pros campos extras do grafo duplo
    (elem_x, cross_edge_index, cross_edge_attr).

    graph_div_b_loss (DivBLossCfg, src/configs/loss.py) é detectada via
    `lambda_div` (só essa config tem esse campo) e tratada à parte: ela tem
    assinatura estendida (precisa de node_x/cross_edge_index pra calcular o
    divergente) e funciona EXCLUSIVAMENTE na parcela de nós -- a parcela de
    grade (FNO/H×W) não tem malha real, usa `loss_cfg.base_loss` puro (sem
    termo de div) nesse caso, em vez de `loss_fn`."""
    subtract      = getattr(loss_cfg, 'subtract_fno', False)
    div_base_loss = getattr(loss_cfg, 'base_loss', None)
    div_lambda    = getattr(loss_cfg, 'lambda_div', None)
    is_div_loss   = div_lambda is not None   # só DivBLossCfg tem lambda_div

    def step(batch, model, loss_fn, device):
        x_hw             = batch['x_hw'].to(device)
        node_x           = batch['node_x'].to(device)
        elem_x           = batch['elem_x'].to(device)
        edge_index       = batch['edge_index'].to(device)
        edge_attr        = batch['edge_attr'].to(device)
        cross_edge_index = batch['cross_edge_index'].to(device)
        cross_edge_attr  = batch['cross_edge_attr'].to(device)
        L                = batch['L'].to(device)
        y_hw             = batch['y_hw'].to(device)
        y_node           = batch['node_y'].to(device)

        # return_components=True sempre -- mesmo custo de (y_hw_fno, y_nodes)
        # direto (o forward normal só soma fno_at_nodes+delta internamente),
        # e simplifica os 3 ramos abaixo pra um único caminho de inferência.
        y_hw_fno, fno_at_nodes, delta = model(
            x_hw, node_x, elem_x, edge_index, edge_attr,
            cross_edge_index, cross_edge_attr, L, return_components=True)
        y_nodes = fno_at_nodes + delta

        grid_loss_fn = LOSS_REGISTRY[div_base_loss] if is_div_loss else loss_fn
        loss_grid = grid_loss_fn(y_hw_fno, y_hw)

        if is_div_loss:
            loss_nodes = loss_fn(y_nodes, y_node, node_x, cross_edge_index,
                                  base_loss=div_base_loss, lambda_div=div_lambda,
                                  r_in_mm=loss_cfg.r_in_mm, r_ext_mm=loss_cfg.r_ext_mm,
                                  ang_1_deg=loss_cfg.ang_1_deg, ang_2_deg=loss_cfg.ang_2_deg)
        elif subtract:
            loss_nodes = loss_fn(delta, y_node - fno_at_nodes)
        else:
            loss_nodes = loss_fn(y_nodes, y_node)

        return lambda_loss * loss_grid + (1.0 - lambda_loss) * loss_nodes

    return step


def fno_bipartite_gnn_metric_fn(batch, model, device):
    """MAE bruto (sem máscara) -- mesmo padrão de fno_gnn.py::fno_gnn_metric_fn.

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
    y_hw             = batch['y_hw'].to(device)
    y_node           = batch['node_y'].to(device)
    with torch.no_grad():
        y_hw_fno, y_nodes = model(
            x_hw, node_x, elem_x, edge_index, edge_attr,
            cross_edge_index, cross_edge_attr, L)
        if normalizer is not None:
            y_hw_fno = normalizer.decode(y_hw_fno, 'y_hw')
            y_nodes  = normalizer.decode(y_nodes,  'node_y')
            y_hw     = normalizer.decode(y_hw,     'y_hw')
            y_node   = normalizer.decode(y_node,   'node_y')
        mae_hw    = torch.mean(torch.abs(y_hw_fno - y_hw)).item()
        mae_graph = torch.mean(torch.abs(y_nodes  - y_node)).item()
    return mae_hw, mae_graph


# [REMOVIDO 2026-08-13] stub NotImplementedError -- plot/avaliação de malha
# dupla vértice+elemento implementado em src/neural_op/archs/eval.py
# (femm_mesh_v2_eval_fn + _plot_femm_mesh_v2, mesmo arquivo/padrão dos demais
# *_eval_fn do projeto). Reimportado abaixo pra ARCH_REGISTRY
# (src/neural_op/archs/__init__.py) continuar importando de
# femm_mesh_v2_gnn.py sem mudança.
# def femm_mesh_v2_eval_fn(model, chunk_data, eval_cfg):
#     """[NÃO IMPLEMENTADO] ..."""
#     raise NotImplementedError(...)
from src.neural_op.archs.eval import femm_mesh_v2_eval_fn
