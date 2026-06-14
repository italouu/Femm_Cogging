# [REMOVIDO] — conteúdo migrado para src/neural_op/archs/
#
# Mapeamento:
#   NodeMLP, MLP, SpectralConv, FNO_Blocks,
#   EdgeConvLayer, GNN          →  src/neural_op/archs/_blocks.py
#   FNO2d  + fno_step_fn        →  src/neural_op/archs/fno.py
#   FNO_GNN + make_fno_gnn_step →  src/neural_op/archs/fno_gnn.py
#   ARCH_REGISTRY               →  src/neural_op/archs/__init__.py
#
# Imports diretos:
#   from src.neural_op.archs.fno     import FNO2d
#   from src.neural_op.archs.fno_gnn import FNO_GNN
#   from src.neural_op.archs         import ARCH_REGISTRY

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # ===========================================================================
# # Blocos de baixo nível
# # ===========================================================================
#
# class NodeMLP(nn.Module):
#     def __init__(self, in_ch, out_ch, hidden_ch, n_layers, act=nn.GELU):
#         super().__init__()
#         if n_layers == 1:
#             layers = [nn.Linear(in_ch, out_ch)]
#         else:
#             layers = [nn.Linear(in_ch, hidden_ch), act()]
#             for _ in range(n_layers - 2):
#                 layers += [nn.Linear(hidden_ch, hidden_ch), act()]
#             layers += [nn.Linear(hidden_ch, out_ch)]
#         self.mlp = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.mlp(x)
#
#
# class MLP(nn.Module):
#     def __init__(self, in_ch, out_ch, layers, width, act=nn.GELU):
#         super().__init__()
#         if layers == 1:
#             seq = [nn.Linear(in_ch, out_ch), act()]
#         else:
#             seq = [nn.Linear(in_ch, width), act()]
#             for _ in range(layers - 2):
#                 seq += [nn.Linear(width, width), act()]
#             seq += [nn.Linear(width, out_ch)]
#         self.mlp = nn.Sequential(*seq)
#         self.in_ch, self.out_ch = int(in_ch), int(out_ch)
#
#     def forward(self, x):
#         B, C, res_x, res_y = x.shape
#         x = x.permute(0, 2, 3, 1).flatten(1, 2)
#         x = self.mlp(x)
#         x = x.view(B, res_x, res_y, self.out_ch).permute(0, 3, 1, 2)
#         return x
#
#
# class SpectralConv(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super().__init__()
#         self.in_channels  = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.weights1 = nn.Parameter(torch.rand(in_channels, out_channels,
#                                                  modes1, modes2, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(torch.rand(in_channels, out_channels,
#                                                  modes1, modes2, dtype=torch.cfloat))
#
#     def _compl_mul2d(self, input, weights):
#         return torch.einsum("bixy,ioxy->boxy", input, weights)
#
#     def forward(self, x):
#         batchsize = x.shape[0]
#         x_ft = torch.fft.rfft2(x)
#         out_ft = torch.zeros(batchsize, self.out_channels,
#                              x.size(-2), x.size(-1) // 2 + 1,
#                              dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1,  :self.modes2] = \
#             self._compl_mul2d(x_ft[:, :, :self.modes1,  :self.modes2], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2] = \
#             self._compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
#         return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
#
#
# class FNO_Blocks(nn.Module):
#     def __init__(self, modes1, modes2, conv_layers, conv_width, act=nn.GELU):
#         super().__init__()
#         self.act = act()
#         self.spec_convs = nn.ModuleList([
#             SpectralConv(conv_width, conv_width, modes1, modes2)
#             for _ in range(conv_layers)
#         ])
#         self.conv_bypass = nn.ModuleList([
#             MLP(in_ch=conv_width, out_ch=conv_width, layers=1, width=1, act=nn.Identity)
#             for _ in range(conv_layers)
#         ])
#
#     def forward(self, x):
#         for spec, bypass in zip(self.spec_convs, self.conv_bypass):
#             x = self.act(spec(x) + bypass(x))
#         return x
#
#
# class EdgeConvLayer(nn.Module):
#     # [REMOVIDO] msg_mlp: NodeMLP(2*node_width + edge_dim → msg_width)
#     def __init__(self, node_width, edge_dim, act=nn.GELU):
#         super().__init__()
#         self.gate_mlp = NodeMLP(in_ch=edge_dim, out_ch=1,
#                                 hidden_ch=edge_dim * 4, n_layers=2, act=act)
#         self.W = nn.Linear(node_width, node_width, bias=False)
#         self.upd_mlp = NodeMLP(in_ch=2 * node_width,  # [REMOVIDO] era node_width + msg_width
#                                out_ch=node_width, hidden_ch=node_width, n_layers=2, act=act)
#
#     def forward(self, h, edge_index, edge_attr):
#         src, dst = edge_index[0], edge_index[1]
#         gate = torch.sigmoid(self.gate_mlp(edge_attr))
#         msg  = gate * self.W(h[src])
#         agg  = torch.zeros(h.size(0), msg.size(-1), dtype=h.dtype, device=h.device)
#         agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
#         return self.upd_mlp(torch.cat([h, agg], dim=-1))
#
#
# class GNN(nn.Module):
#     # [REMOVIDO] gnn_msg_width — gate escalar não usa msg_width externo
#     def __init__(self, in_node_features, out_node_features, edge_dim,
#                  node_width, n_layers, act=nn.GELU):
#         super().__init__()
#         self.lift   = nn.Linear(in_node_features, node_width)
#         self.layers = nn.ModuleList([
#             EdgeConvLayer(node_width, edge_dim, act) for _ in range(n_layers)
#         ])
#         self.proj = nn.Linear(node_width, out_node_features)
#         self.act  = act()
#
#     def forward(self, node_features, edge_index, edge_attr):
#         h = self.act(self.lift(node_features))
#         for layer in self.layers:
#             h = layer(h, edge_index, edge_attr)
#         return self.proj(h)
#
#
# # ===========================================================================
# # Redes de alto nível
# # ===========================================================================
#
# class FNO2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, conv_width,
#                  conv_layers, lift_width, lift_layers, proj_width, proj_layers, data_res):
#         super().__init__()
#         self.modes1 = modes1 if modes1 <= data_res[0]          else data_res[0]
#         self.modes2 = modes2 if modes2 <= data_res[1] // 2 + 1 else data_res[1] // 2 + 1
#         self.lift_layer = MLP(in_ch=in_channels, out_ch=conv_width,
#                               layers=lift_layers, width=lift_width)
#         self.conv_layer = FNO_Blocks(modes1=self.modes1, modes2=self.modes2,
#                                      conv_layers=conv_layers, conv_width=conv_width)
#         self.proj_layer = MLP(in_ch=conv_width, out_ch=out_channels,
#                               layers=proj_layers, width=proj_width)
#
#     def forward(self, x):
#         return self.proj_layer(self.conv_layer(self.lift_layer(x)))
#
#
# def _interpolate_fno_to_nodes(fno_out, node_x, L):
#     B, C, _, _ = fno_out.shape
#     r_norm = 2.0 * node_x[:, 3] - 1.0
#     c_norm = 2.0 * node_x[:, 4] - 1.0
#     fno_at_nodes = torch.empty(node_x.size(0), C, device=fno_out.device, dtype=fno_out.dtype)
#     offset = 0
#     for b in range(B):
#         n    = int(L[b].item())
#         grid = torch.stack([c_norm[offset:offset+n], r_norm[offset:offset+n]], dim=-1
#                            ).unsqueeze(0).unsqueeze(2)
#         interp = F.grid_sample(fno_out[b:b+1], grid,
#                                mode='bilinear', align_corners=True, padding_mode='border')
#         fno_at_nodes[offset:offset+n] = interp[0, :, :, 0].T
#         offset += n
#     return fno_at_nodes
#
#
# class FNO_GNN(nn.Module):
#     # [REMOVIDO] gnn_msg_width — gate escalar não usa msg_width externo
#     _GRID_IN_CH  = 2
#     _GRID_OUT_CH = 2
#     _NODE_IN_CH  = 5
#     _EDGE_DIM    = 4
#     _GNN_IN_CH   = 7
#
#     def __init__(self, fno_modes1, fno_modes2, fno_conv_width, fno_conv_layers,
#                  fno_lift_width, fno_lift_layers, fno_proj_width, fno_proj_layers,
#                  data_res, gnn_node_width, gnn_n_layers):
#         super().__init__()
#         self.fno = FNO2d(in_channels=2, out_channels=2,
#                          modes1=fno_modes1, modes2=fno_modes2,
#                          conv_width=fno_conv_width, conv_layers=fno_conv_layers,
#                          lift_width=fno_lift_width, lift_layers=fno_lift_layers,
#                          proj_width=fno_proj_width, proj_layers=fno_proj_layers,
#                          data_res=data_res)
#         self.gnn = GNN(in_node_features=7, out_node_features=2,
#                        edge_dim=4, node_width=gnn_node_width, n_layers=gnn_n_layers)
#
#     def forward(self, x_hw, node_x, edge_index, edge_attr, L):
#         y_hw_fno     = self.fno(x_hw)
#         fno_at_nodes = _interpolate_fno_to_nodes(y_hw_fno, node_x, L)
#         delta        = self.gnn(torch.cat([node_x, fno_at_nodes], dim=-1), edge_index, edge_attr)
#         return y_hw_fno, fno_at_nodes + delta
