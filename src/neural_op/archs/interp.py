"""
Interpolação grade H×W → nós da malha (B1, bateria definitiva 2026-10-03).

Função única usada por todos os pontos que levam a saída do FNO (grade) para
as posições dos nós: _interpolate_fno_to_nodes (fno_gnn.py),
_interpolate_fno_to_bipartite_nodes (fno_bipartite_gnn.py), GNN_PostBase e o
pós-processamento (src/postprocess/) — evita divergência entre treino e avaliação.

Convenção da grade (src/data_gen/parsers/ans_parsing.py::_grid_polar_xy): grade
CENTRADA EM CÉLULAS — o pixel i fica em r_base = (i+0,5)/H (idem colunas/c_base).

Modo (B1b, 2026-10-05 — 'cell_centered_border', o único desde 2026-10-21):
grid_sample bilinear com align_corners=False (índice r_base·H − 0,5 e
c_base·W − 0,5; x = 2·c_base − 1, y = 2·r_base − 1), SEM padding circular,
padding 'border' nos DOIS eixos — nós a menos de meio pixel do corte 0°/120°
recebem o valor da coluna de borda (0 ou W−1), sem misturar.
"""
import torch
import torch.nn.functional as F


def interpolate_grid_to_nodes(grid, r_base, c_base, L):
    """
    grid   : [B, C, H, W]  saída na grade (qualquer escala — operação linear)
    r_base : [S_tot]        posição radial normalizada ∈ [0,1]
    c_base : [S_tot]        posição angular normalizada ∈ [0,1]
    L      : [B]            nós por amostra (fatiamento de r_base/c_base)
    Retorna: [S_tot, C]
    """
    B, C, H, W = grid.shape
    x_norm, y_norm = 2.0 * c_base - 1.0, 2.0 * r_base - 1.0

    out = torch.empty(r_base.size(0), C, device=grid.device, dtype=grid.dtype)
    offset = 0
    for b in range(B):
        n = int(L[b].item())
        g = torch.stack(
            [x_norm[offset:offset + n], y_norm[offset:offset + n]], dim=-1
        ).to(grid.dtype).unsqueeze(0).unsqueeze(2)                     # [1, n, 1, 2]
        interp = F.grid_sample(
            grid[b:b + 1], g,
            mode='bilinear', align_corners=False, padding_mode='border',
        )                                                              # [1, C, n, 1]
        out[offset:offset + n] = interp[0, :, :, 0].T                  # [n, C]
        offset += n
    return out
