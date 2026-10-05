"""
Interpolação grade H×W → nós da malha (B1, bateria definitiva 2026-10-03).

Função única usada por todos os pontos que levam a saída do FNO (grade) para
as posições dos nós: _interpolate_fno_to_nodes (fno_gnn.py),
_interpolate_fno_to_nodes_v2 (femm_mesh_v2_gnn.py), GNN_PostBase e os
*_eval_fn — evita divergência entre treino e avaliação.

Convenção da grade (src/data_gen/parsers/ans_parsing.py::_grid_polar_xy): grade
CENTRADA EM CÉLULAS — o pixel i fica em r_base = (i+0,5)/H (idem colunas/c_base).

Modos
-----
'legacy'        : comportamento antigo — grid_sample com align_corners=True,
                  padding 'border' nos dois eixos. Índice efetivo r_base·(H−1),
                  quando o correto é r_base·H − 0,5 → desalinhamento de
                  (0,5 − r_base) pixel em cada eixo. Mantido para reproduzir
                  runs já treinadas (default dos configs — configs antigos sem o
                  campo `interp_mode` são reconstruídos com ele).
'cell_centered' : [OBSOLETO 2026-10-05 — mantido só para recarregar as runs
                  arquivadas em data/logs/_arquivo/bateria_wrap_circular/]
                  align_corners=False (índice r_base·H − 0,5) + padding
                  CIRCULAR de 1 coluna em cada lado do eixo angular (setor
                  periódico 0°/120°) — nós a menos de meio pixel de 0° ou 120°
                  misturam as colunas 0 e W−1. Eixo radial com 'border' (sem
                  periodicidade física em r).
                  Errado para Bx/By: no corte de 120° o vetor cartesiano está
                  girado de 120°, então misturar as colunas 0 e W−1 mistura
                  vetores com direções diferentes. No FNO_GNN/FNO_BipartiteGNN
                  o interpolado é um latente (lambda_loss=0), para o qual
                  nenhuma rotação faria sentido.
'cell_centered_border' : (B1b, 2026-10-05 — modo atual da bateria) align_corners=
                  False (índice r_base·H − 0,5 e c_base·W − 0,5; x = 2·c_base − 1,
                  y = 2·r_base − 1), SEM padding circular, padding 'border' nos
                  DOIS eixos — nós a menos de meio pixel do corte 0°/120°
                  recebem o valor da coluna de borda (0 ou W−1), sem misturar.
"""
import torch
import torch.nn.functional as F

# [REMOVIDO 2026-10-05, B1b] sem 'cell_centered_border'
# INTERP_MODES = ('legacy', 'cell_centered')
INTERP_MODES = ('legacy', 'cell_centered', 'cell_centered_border')


def _grid_coords(r_base, c_base, H, W, mode):
    """Coordenadas normalizadas (x=coluna, y=linha) para F.grid_sample."""
    if mode == 'legacy':
        return 2.0 * c_base - 1.0, 2.0 * r_base - 1.0
    if mode == 'cell_centered_border':
        # mesma fórmula do legacy, mas com align_corners=False (ver
        # interpolate_grid_to_nodes): índice efetivo c·W − 0,5 / r·H − 0,5
        return 2.0 * c_base - 1.0, 2.0 * r_base - 1.0
    # cell_centered — eixo angular na grade com padding circular (largura W+2):
    # índice no original u = c·W − 0,5  →  no padded u' = u + 1;
    # align_corners=False: x = (2u' + 1)/(W+2) − 1 = (2cW + 2)/(W+2) − 1
    x = (2.0 * c_base * W + 2.0) / (W + 2) - 1.0
    y = 2.0 * r_base - 1.0
    return x, y


def interpolate_grid_to_nodes(grid, r_base, c_base, L, mode='legacy'):
    """
    grid   : [B, C, H, W]  saída na grade (qualquer escala — operação linear)
    r_base : [S_tot]        posição radial normalizada ∈ [0,1]
    c_base : [S_tot]        posição angular normalizada ∈ [0,1]
    L      : [B]            nós por amostra (fatiamento de r_base/c_base)
    mode   : 'legacy' | 'cell_centered' (obsoleto) | 'cell_centered_border'
             (ver docstring do módulo)
    Retorna: [S_tot, C]
    """
    if mode not in INTERP_MODES:
        raise ValueError(f"interp_mode inválido: {mode!r} (esperado um de {INTERP_MODES})")
    B, C, H, W = grid.shape

    if mode == 'cell_centered':
        grid = torch.cat([grid[..., -1:], grid, grid[..., :1]], dim=-1)   # [B, C, H, W+2]
        align_corners = False
    elif mode == 'cell_centered_border':
        align_corners = False                       # sem padding — 'border' nos dois eixos
    else:
        align_corners = True

    x_norm, y_norm = _grid_coords(r_base, c_base, H, W, mode)

    out = torch.empty(r_base.size(0), C, device=grid.device, dtype=grid.dtype)
    offset = 0
    for b in range(B):
        n = int(L[b].item())
        g = torch.stack(
            [x_norm[offset:offset + n], y_norm[offset:offset + n]], dim=-1
        ).to(grid.dtype).unsqueeze(0).unsqueeze(2)                     # [1, n, 1, 2]
        interp = F.grid_sample(
            grid[b:b + 1], g,
            mode='bilinear', align_corners=align_corners, padding_mode='border',
        )                                                              # [1, C, n, 1]
        out[offset:offset + n] = interp[0, :, :, 0].T                  # [n, C]
        offset += n
    return out
