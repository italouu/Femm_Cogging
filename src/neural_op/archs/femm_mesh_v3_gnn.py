"""
femm_mesh_v3_gnn.py
--------------------
Extensão de femm_mesh_v2_gnn.py (FNO_BipartiteGNN) pedida pelo usuário
(2026-08-20), sobre o parser src/data_gen/parsers/femm_mesh_v3.py:

  1. node_x ganha uma 3ª coluna, node_cell_count -- quantos vértices da
     malha caem na mesma célula do grid base H×W que aquele nó (feature
     ESTÁTICA, calculada no parser -- pura geometria, não depende do FNO).

  2. Além do valor bilinear do FNO no ponto exato do nó (fno_at_nodes, igual
     ao v2), a GNN também recebe a projeção das 8 células vizinhas (3x3 ao
     redor da célula do nó, sem o centro) na saída do FNO -- essa parte só
     pode ser calculada em tempo de FORWARD (a saída do FNO não existe em
     tempo de parsing), daí viver aqui e não no parser. Amostragem por
     INDEXAÇÃO DIRETA da célula (nearest, não bilinear -- decisão confirmada
     com o usuário: "célula" é uma unidade discreta da grade). Wrap angular
     (coluna, c_base) -- mesma periodicidade θ=0/120° já usada no grafo
     (ans_parsing.py::_wrap_edge_pairs); ZERO na direção radial (linha,
     r_base) fora de [0,H) -- não é periódico.

Registrado como arch própria (FNO_BipartiteGNN_v2, ARCH_REGISTRY) em vez de
alterar FNO_BipartiteGNN in-place -- runs já treinadas com o v2 continuam
intactas. loader_mode continua 'femm_mesh_v2' (mesmo layout de chunk, só
node_x muda de largura -- ChunkFemmMeshV2Dataset/femm_mesh_v2_collate já são
agnósticos a isso). step_fn/metric_fn/eval_fn reaproveitados sem duplicação
de femm_mesh_v2_gnn.py/eval.py -- são genéricos (só repassam node_x pro
model, não assumem largura fixa).
"""
import torch
import torch.nn.functional as F
from src.neural_op.archs._blocks import BipartiteGNN
from src.neural_op.archs.fno import FNO2d
from src.neural_op.archs.femm_mesh_v2_gnn import (make_fno_bipartite_gnn_step,
                                                   fno_bipartite_gnn_metric_fn,
                                                   femm_mesh_v2_eval_fn)

# 8 vizinhos (3x3 ao redor da célula do nó, sem o centro) -- (delta_r, delta_c);
# linha = radial (sem wrap), coluna = angular (com wrap, periodicidade θ=0/120°)
_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                     ( 0, -1),          ( 0, 1),
                     ( 1, -1), ( 1, 0), ( 1, 1)]


def _interpolate_fno_to_nodes_v3(fno_out, node_x, L):
    """
    Estende _interpolate_fno_to_nodes_v2 (femm_mesh_v2_gnn.py) com a
    projeção das 8 células vizinhas na saída do FNO -- ver docstring do
    módulo.

    fno_out : [B, C, H, W]
    node_x  : [S_tot, 3]  colunas 0,1 = r_base, c_base ∈ [0,1] (coluna 2 =
              node_cell_count -- não usada aqui, é feature "crua" de node_x,
              consumida direto pela GNN via concatenação)
    L       : [B]          nós por amostra

    Retorna : (fno_at_nodes [S_tot, C], neighbor_proj [S_tot, 8*C])
    """
    B, C, H, W = fno_out.shape
    device = fno_out.device
    dtype  = fno_out.dtype

    r_norm = 2.0 * node_x[:, 0] - 1.0
    c_norm = 2.0 * node_x[:, 1] - 1.0

    cell_r = torch.clamp(torch.floor(node_x[:, 0] * H).long(), 0, H - 1)
    cell_c = torch.remainder(torch.floor(node_x[:, 1] * W).long(), W)

    fno_at_nodes  = torch.empty(node_x.size(0), C, device=device, dtype=dtype)
    neighbor_proj = torch.empty(node_x.size(0), 8 * C, device=device, dtype=dtype)

    offset = 0
    for b in range(B):
        n  = int(L[b].item())
        sl = slice(offset, offset + n)

        # ponto central -- bilinear, idêntico a _interpolate_fno_to_nodes_v2
        grid = torch.stack(
            [c_norm[sl], r_norm[sl]], dim=-1
        ).unsqueeze(0).unsqueeze(2)                                   # [1, n, 1, 2]
        interp = F.grid_sample(
            fno_out[b:b + 1], grid,
            mode='bilinear', align_corners=True, padding_mode='border',
        )                                                              # [1, C, n, 1]
        fno_at_nodes[sl] = interp[0, :, :, 0].T                       # [n, C]

        # 8 vizinhos -- indexação direta (nearest), wrap em c / zero em r fora da grade
        r_i, c_i = cell_r[sl], cell_c[sl]
        parts = []
        for dr, dc in _NEIGHBOR_OFFSETS:
            r_n = r_i + dr
            valid = (r_n >= 0) & (r_n < H)
            r_clamped = r_n.clamp(0, H - 1)
            c_n = torch.remainder(c_i + dc, W)
            gathered = fno_out[b, :, r_clamped, c_n].T                # [n, C]
            gathered = gathered * valid.unsqueeze(-1).to(dtype)       # zero fora da grade (radial)
            parts.append(gathered)
        neighbor_proj[sl] = torch.cat(parts, dim=-1)                  # [n, 8*C]

        offset += n

    return fno_at_nodes, neighbor_proj


class FNO_BipartiteGNN_v2(torch.nn.Module):
    """
    Igual a FNO_BipartiteGNN (femm_mesh_v2_gnn.py), com as duas extensões
    descritas na docstring do módulo -- node_x com node_cell_count e GNN
    alimentada também pela projeção das 8 células vizinhas do FNO.

    Pipeline
    --------
    x_hw  →  FNO2d  →  y_hw_fno
                ↓  _interpolate_fno_to_nodes_v3
    [node_x | y_fno@nó | y_fno@8-vizinhos]  →  BipartiteGNN(+elem_x via arestas cruzadas)  →  Δ
    y_nodes = y_fno@nó + Δ

    Parâmetros idênticos a FNO_BipartiteGNN (ver aquele módulo).
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
            in_node_features=node_in_ch + grid_out_ch + 8 * grid_out_ch,
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
        fno_at_nodes, neighbor_proj = _interpolate_fno_to_nodes_v3(y_hw_fno, node_x, L)
        gnn_input    = torch.cat([node_x, fno_at_nodes, neighbor_proj], dim=-1)
        delta        = self.gnn(gnn_input, elem_x, edge_index, edge_attr,
                                 cross_edge_index, cross_edge_attr)
        if return_components:
            return y_hw_fno, fno_at_nodes, delta
        return y_hw_fno, fno_at_nodes + delta


# make_fno_bipartite_gnn_step / fno_bipartite_gnn_metric_fn / femm_mesh_v2_eval_fn
# reaproveitados de femm_mesh_v2_gnn.py/eval.py sem alteração (reexportados
# aqui só pra ARCH_REGISTRY importar tudo deste módulo, mesmo padrão de
# femm_mesh_v2_gnn.py reimportar femm_mesh_v2_eval_fn de eval.py) -- nenhum
# deles assume largura fixa de node_x, só repassam pro forward do model.
__all__ = ['FNO_BipartiteGNN_v2', 'make_fno_bipartite_gnn_step',
           'fno_bipartite_gnn_metric_fn', 'femm_mesh_v2_eval_fn']
