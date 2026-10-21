import torch


def mse_loss(out, y):
    return torch.mean((out - y) ** 2)


def mae_loss(out, y):
    return torch.mean(torch.abs(out - y))


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


class BaseLoss:
    """
    Wrapper extensível em volta de uma função de loss pura (fn) -- __call__
    só repassa pra fn (mesma assinatura de sempre, nenhuma mudança em nenhum
    step_fn de arch) e acrescenta log_epoch como PROPRIEDADE da loss: quem
    decide o que aparece no console a cada época é a própria loss, não o
    loop de treino (fit(), src/neural_op/training_utils.py).

    Por ora toda entrada de LOSS_REGISTRY usa o log_epoch padrão abaixo
    (mesmo texto que já era hardcoded em fit()). Uma loss que precise
    destacar termos próprios (ex: erro de ajuste/"data" vs penalidade
    física -- caso de graph_div_b_loss, que já calcula `fit` e `div`
    separadamente) sobrescreve log_epoch numa subclasse; nada mais no
    projeto precisa mudar pra isso (scripts/train.py já extrai
    `loss_obj.log_epoch` do objeto, antes de qualquer functools.partial).
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def log_epoch(self, ep, train_loss, test_loss, train_time_s, eval_time_s, samples_per_s):
        print(f"epoch {ep:>4d}  train {train_loss:.4e}  test {test_loss:.4e}"
              f"  [{train_time_s:.1f}s + {eval_time_s:.1f}s eval]  {samples_per_s:.0f} samp/s")


LOSS_REGISTRY: dict = {
    'mse':                      BaseLoss(mse_loss),
    'mae':                      BaseLoss(mae_loss),
    'masked_fno_loss':          BaseLoss(masked_fno_loss),
    # assinatura estendida (pred, y, mask_m) — requer step_fn compatível (FNO2d_SingleMat)
    'single_material_fno_loss': BaseLoss(single_material_fno_loss),
    # assinatura estendida (y_hw_8, y_hw, masks, y_nodes_2, node_y, lambda_loss)
    'masked_fno_gnn_loss':      BaseLoss(masked_fno_gnn_loss),
    # 'graph_div_b_loss':         DivBLoss(graph_div_b_loss),
}
