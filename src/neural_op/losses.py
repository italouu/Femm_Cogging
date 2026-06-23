import torch


def _magnitude(t):
    """[B,C,H,W] (grade) ou [S,C] (nós) — magnitude vetorial se C>=2, abs se C==1."""
    if t.shape[1] >= 2:
        return (t[:, 0] ** 2 + t[:, 1] ** 2).sqrt()
    return t[:, 0].abs()


def topk_tail_term(pred, y, k_frac):
    """
    Termo de cauda reutilizável: erro quadrático relativo médio dos k_frac
    elementos (pixels/nós) com maior erro de magnitude, normalizado pelo RMS
    global de |y| (mesma referência relativa — B_ref — usada em
    src/bench/metrics.py) para ficar em escala comparável à MSE.

    pred, y : [B, C, H, W] (grade) ou [S, C] (nós), mesmo shape
    k_frac  : fração (0,1] dos elementos com maior erro a penalizar
    """
    mag_true = _magnitude(y)
    mag_pred = _magnitude(pred)
    b_ref_sq = mag_true.pow(2).mean().clamp(min=1e-8)
    err_rel  = ((mag_pred - mag_true) ** 2 / b_ref_sq).reshape(-1)
    k = max(1, int(k_frac * err_rel.numel()))
    return torch.topk(err_rel, k).values.mean()


def mse_loss(out, y, tail_alpha=0.0, tail_k_frac=0.05):
    base = torch.mean((out - y) ** 2)
    if tail_alpha == 0.0:
        return base
    tail = topk_tail_term(out, y, tail_k_frac)
    return (1.0 - tail_alpha) * base + tail_alpha * tail


def mae_loss(out, y):
    return torch.mean(torch.abs(out - y))


def relative_l2_loss(out, y):
    # Normaliza por amostra (dim 0) — funciona para grade [B,C,H,W] e nós [S,C]
    diff = (out - y).reshape(out.shape[0], -1)
    norm = y.reshape(y.shape[0], -1)
    return (diff.norm(dim=1) / (norm.norm(dim=1) + 1e-8)).mean()


def masked_fno_loss(pred, y, masks):
    """
    Loss mascarada por material — usada por MaskedFNO2d.

    Assinatura estendida: (pred, y, masks) — step_fn é responsável por passar masks.

    pred  : [B, 8, H, W]  — 4 pares (Bx_m, By_m), um por material (ferro=0, ar=1, ima=2, cobre=3)
    y     : [B, 2, H, W]  — alvo (Bx, By)
    masks : [B, 4, H, W]  — bool, partição do domínio sem sobreposição

    Peso de cada material proporcional à sua contagem de pixels no batch (normalizado).
    Bx e By contribuem igualmente (0.5 cada).
    Loss por material: média sobre os pixels do material (evita viés por volume).
    """
    masks_f      = masks.float()
    pixel_counts = masks_f.sum(dim=(0, 2, 3))       # [4] — pixels por material no batch
    total_pixels = pixel_counts.sum().clamp(min=1.0)
    weights      = pixel_counts / total_pixels       # [4], soma 1

    total = pred.new_zeros(())
    for m in range(4):
        mask_m = masks_f[:, m]                                    # [B, H, W]
        n_m    = pixel_counts[m].clamp(min=1.0)
        err_bx = (pred[:, 2 * m    ] - y[:, 0]) ** 2             # [B, H, W]
        err_by = (pred[:, 2 * m + 1] - y[:, 1]) ** 2
        loss_m = ((0.5 * err_bx + 0.5 * err_by) * mask_m).sum() / n_m
        total  = total + weights[m] * loss_m

    return total


def single_material_fno_loss(pred, y, mask_m):
    """
    Loss MSE restrita ao material alvo.

    pred   : [B, 2, H, W]
    y      : [B, 2, H, W]
    mask_m : [B, H, W] bool — pixels do material alvo
    """
    mask_f = mask_m.float().unsqueeze(1)   # [B, 1, H, W]
    n      = mask_f.sum().clamp(min=1.0)
    return ((pred - y) ** 2 * mask_f).sum() / n


# [REMOVIDO] masked_gnn_node_loss — GNN agora produz 2 canais diretamente;
# parcela de nós é MSE simples em masked_fno_gnn_loss.


def masked_fno_gnn_loss(y_hw_8, y_hw, masks, y_nodes_2, node_y, lambda_loss):
    """
    Loss combinada de MaskedFNO_GNN — grade (mascarada) + nós (MSE simples).

    Parcela de grade : masked_fno_loss(y_hw_8, y_hw, masks) — ponderada por material
    Parcela de nós   : mse_loss(y_nodes_2, node_y)          — saída 2-canal da GNN
    Combinação       : lambda_loss * grade + (1 - lambda_loss) * nós

    y_hw_8     : [B, 8, H, W]  — pred FNO 8-canal na grade
    y_hw       : [B, 2, H, W]  — alvo na grade
    masks      : [B, 4, H, W]  — bool, derivado de x_hw[:,0] (Mu_r)
    y_nodes_2  : [S_tot, 2]    — pred GNN 2-canal nos nós (após assemble + delta)
    node_y     : [S_tot, 2]    — alvo nos nós
    lambda_loss: float         — peso da parcela de grade (cfg.lambda_loss)
    """
    loss_grid  = masked_fno_loss(y_hw_8, y_hw, masks)
    loss_nodes = mse_loss(y_nodes_2, node_y)
    return lambda_loss * loss_grid + (1.0 - lambda_loss) * loss_nodes


LOSS_REGISTRY: dict = {
    'mse':                     mse_loss,
    'mae':                     mae_loss,
    'relative_l2':             relative_l2_loss,
    # assinatura estendida (pred, y, masks) — requer step_fn compatível (MaskedFNO2d)
    'masked_fno_loss':         masked_fno_loss,
    # assinatura estendida (pred, y, mask_m) — requer step_fn compatível (FNO2d_SingleMat)
    'single_material_fno_loss': single_material_fno_loss,
    # assinatura estendida (y_hw_8, y_hw, masks, y_nodes_2, node_y, lambda_loss)
    'masked_fno_gnn_loss':     masked_fno_gnn_loss,
}
