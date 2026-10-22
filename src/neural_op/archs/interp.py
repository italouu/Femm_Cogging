"""
Interpolação grade H×W → pontos da malha (nós ou centróides).

Função única usada por todos os pontos que levam a saída do FNO (grade) para
as posições da malha: _interpolate_fno_to_nodes (fno_gnn.py, também no
GNN_PostBase), _interpolate_fno_to_bipartite_nodes (fno_bipartite_gnn.py, também
na Elem), inference.predict/floor_nodes e timing (pós-processamento) — evita
divergência entre treino e avaliação.

Convenção da grade (src/data_gen/parsers/ans_parsing.py::_grid_polar_xy): grade
CENTRADA EM CÉLULAS — o pixel i fica em r_base = (i+0,5)/H (idem colunas/c_base).

Bilinear (grid_sample, align_corners=False) com:
  - eixo angular CIRCULAR: o alvo polar (Br, Bθ) é periódico no corte 0°/120° (setor
    com nº par de polos), então a grade ganha 1 coluna de cada lado (coluna W−1 antes
    da 0 e coluna 0 depois da W−1) e pontos a menos de meio pixel do corte interpolam
    entre as colunas 0 e W−1;
  - eixo radial 'border' (sem periodicidade em r): pontos a menos de meio pixel de
    R_in/R_out recebem a linha de borda.
Índices de pixel u = c_base·W − 0,5 (coluna) e v = r_base·H − 0,5 (linha); na grade
com padding a coluna é u + 1, e com align_corners=False
  x = (2(u + 1) + 1)/(W + 2) − 1 = (2·c_base·W + 2)/(W + 2) − 1,   y = 2·r_base − 1.
"""
import torch
import torch.nn.functional as F


def interpolate_grid_to_nodes(grid, r_base, c_base, L):
    """
    grid   : [B, C, H, W]  saída na grade (qualquer escala — operação linear)
    r_base : [S_tot]        posição radial normalizada ∈ [0,1]
    c_base : [S_tot]        posição angular normalizada ∈ [0,1]
    L      : [B]            pontos por amostra (fatiamento de r_base/c_base)
    Retorna: [S_tot, C]
    """
    B, C, H, W = grid.shape
    grid = torch.cat([grid[..., -1:], grid, grid[..., :1]], dim=-1)   # [B, C, H, W+2]
    x_norm = (2.0 * c_base * W + 2.0) / (W + 2) - 1.0
    y_norm = 2.0 * r_base - 1.0

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
