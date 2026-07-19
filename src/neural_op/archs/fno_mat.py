import torch
import torch.nn as nn

from src.neural_op.archs.fno import FNO2d

# Thresholds idênticos a data_utils.derive_material_id e _MU_THRESHOLDS
_MU_IRON   = 10.0     # ferro: mu_r = 5000
_MU_MAG_LO = 1.01     # ima:   mu_r = 1.05  (limite inferior)
_MU_COP_HI = 0.9995   # cobre: mu_r = 0.999


def _make_material_masks(mu_r):
    """
    mu_r : [B, H, W]  — canal Mu_r de x_hw
    Retorna [B, 4, H, W] bool — partição do domínio sem sobreposição.
    Ordem: ferro=0, ar=1, ima=2, cobre=3  (mesmos ids de MATERIAL_ID em motor_model.py)
    """
    iron   = mu_r > _MU_IRON
    mag    = (mu_r > _MU_MAG_LO) & ~iron
    copper = mu_r < _MU_COP_HI
    air    = ~(iron | mag | copper)
    return torch.stack([iron, air, mag, copper], dim=1)   # [B, 4, H, W]


class FNO2d_SingleMat(FNO2d):
    """FNO2d treinado em um único material; armazena material_id para a eval_fn."""

    def __init__(self, material_id, in_channels, out_channels, modes1, modes2,
                 conv_width, conv_layers, lift_width, lift_layers,
                 proj_width, proj_layers, data_res):
        super().__init__(in_channels, out_channels, modes1, modes2,
                         conv_width, conv_layers, lift_width, lift_layers,
                         proj_width, proj_layers, data_res)
        self._single_mat_id = material_id


class MaskedFNO2d(FNO2d):
    """
    FNO2d com 8 canais de saída (4 materiais × Bx, By).

    forward(x) → pred [B, 8, H, W]  — usado na loss durante treino.
    assemble(pred, masks) → [B, 2, H, W]  — campo final na inferência.

    Canal layout em pred:
        [:, 0:2]  ferro  (Bx_ferro, By_ferro)
        [:, 2:4]  ar     (Bx_ar,    By_ar)
        [:, 4:6]  ima    (Bx_ima,   By_ima)
        [:, 6:8]  cobre  (Bx_cobre, By_cobre)
    """

    def assemble(self, pred, masks):
        """
        Combina pred [B, 8, H, W] nas máscaras [B, 4, H, W] → [B, 2, H, W].
        Cada pixel recebe os campos do seu material; sem sobreposição.
        """
        out = pred.new_zeros(pred.shape[0], 2, pred.shape[2], pred.shape[3])
        for m in range(4):
            out += pred[:, 2 * m: 2 * m + 2] * masks[:, m: m + 1].float()
        return out


def make_single_mat_step_fn(material_id):
    """
    Factory de step_fn para FNO2d treinado em um único material.

    material_id : int — 0=ferro, 1=ar, 2=ima, 3=cobre
                        (mesmo índice de _make_material_masks)
    """
    def step_fn(batch, model, loss_fn, device):
        x, y   = batch
        x_d    = x.to(device)
        y_d    = y.to(device)
        masks  = _make_material_masks(x_d[:, 0])  # [B, 4, H, W]
        mask_m = masks[:, material_id]             # [B, H, W] bool
        pred   = model(x_d)                        # [B, 2, H, W]
        return loss_fn(pred, y_d, mask_m)
    return step_fn


def make_single_mat_metric_fn(material_id):
    """MAE bruto restrito à região do material alvo. Sem grafo — mae_graph=None."""
    def metric_fn(batch, model, device):
        x, y   = batch
        x_d    = x.to(device)
        y_d    = y.to(device)
        with torch.no_grad():
            masks    = _make_material_masks(x_d[:, 0])
            mask_m   = masks[:, material_id]                       # [B, H, W] bool
            pred     = model(x_d)                                   # [B, 2, H, W]
            mask_exp = mask_m.unsqueeze(1).expand_as(pred)
            diff     = torch.abs(pred - y_d)[mask_exp]
            mae_hw   = diff.mean().item() if diff.numel() > 0 else 0.0
        return mae_hw, None
    return metric_fn


def masked_fno_step_fn(batch, model, loss_fn, device):
    x, y  = batch
    x_d   = x.to(device)
    y_d   = y.to(device)
    masks = _make_material_masks(x_d[:, 0])   # [B, 4, H, W] — on-the-fly, ~0ms
    pred  = model(x_d)                         # [B, 8, H, W]
    return loss_fn(pred, y_d, masks)


def masked_fno_metric_fn(batch, model, device):
    """MAE bruto (sem máscara) na grade H×W, após assemble por material. Sem mae_graph."""
    x, y = batch
    x_d  = x.to(device)
    y_d  = y.to(device)
    with torch.no_grad():
        masks     = _make_material_masks(x_d[:, 0])
        pred8     = model(x_d)
        assembled = model.assemble(pred8, masks)
        mae_hw    = torch.mean(torch.abs(assembled - y_d)).item()
    return mae_hw, None
