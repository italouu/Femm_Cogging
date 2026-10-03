import warnings

import torch
import torch.nn as nn
from src.neural_op.archs._blocks import MLP, FNO_Blocks


def full_spectrum_modes(H, W):
    """B3 (2026-10-03): (modes1, modes2) que cobrem o espectro completo de uma
    grade H×W sem sobreposição dos blocos weights1/weights2 de SpectralConv:
    weights1 cobre as linhas [0, modes1) e weights2 as linhas [H−modes1, H) do
    rfft2 (frequências positivas/negativas do eixo 0); modes2 = W//2+1 colunas
    do rfft. modes1 = ceil(H/2) → as duas faixas somam H linhas exatas
    (138×276 → (69, 139))."""
    return (H + 1) // 2, W // 2 + 1


class FNO2d(nn.Module):
    """
    Fourier Neural Operator 2D.
    Lift(MLP) → FNO_Blocks (SpectralConv + bypass) → Proj(MLP).
    Entrada/saída: [B, C, H, W].
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 modes1,
                 modes2,
                 conv_width,
                 conv_layers,
                 lift_width,
                 lift_layers,
                 proj_width,
                 proj_layers,
                 data_res,
                 interp_mode='legacy'):
        super().__init__()
        # interp_mode não é usado no forward (FNO2d só produz grade) — fica no
        # modelo pra que avaliações que interpolam a saída nos nós da malha
        # (eval.py, scripts/eval_surface_integral_table.py) usem a mesma
        # interpolação registrada no config da run (B1, src/neural_op/archs/interp.py).
        self.interp_mode = interp_mode

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1       = modes1 if modes1 <= data_res[0]          else data_res[0]
        self.modes2       = modes2 if modes2 <= data_res[1] // 2 + 1 else data_res[1] // 2 + 1
        self.conv_width   = conv_width
        self.conv_layers  = conv_layers
        self.lift_width   = lift_width
        self.lift_layers  = lift_layers
        self.proj_width   = proj_width
        self.proj_layers  = proj_layers
        self.data_res     = data_res
        # B3 — aviso (não erro, runs antigas continuam carregando): blocos
        # weights1/weights2 sobrepostos no eixo radial — a faixa de weights2
        # sobrescreve parte (ou toda) a de weights1, que fica sem gradiente.
        if 2 * self.modes1 > data_res[0]:
            warnings.warn(
                f"FNO2d: modes1={self.modes1} > data_res[0]/2={data_res[0] / 2:g} — "
                f"weights1/weights2 sobrepostos em {2 * self.modes1 - data_res[0]} linha(s) "
                f"do espectro (sem sobreposição: modes1=ceil(H/2), ver full_spectrum_modes)",
                stacklevel=2)

        self.lift_layer = MLP(in_ch=in_channels,  out_ch=conv_width,
                              layers=lift_layers,  width=lift_width)
        self.conv_layer = FNO_Blocks(modes1=self.modes1, modes2=self.modes2,
                                     conv_layers=conv_layers, conv_width=conv_width)
        self.proj_layer = MLP(in_ch=conv_width,   out_ch=out_channels,
                              layers=proj_layers,  width=proj_width)

    def forward(self, x):
        x = self.lift_layer(x)
        x = self.conv_layer(x)
        x = self.proj_layer(x)
        return x


def fno_step_fn(batch, model, loss_fn, device):
    x, y = batch
    return loss_fn(model(x.to(device)), y.to(device))


def fno_metric_fn(batch, model, device):
    """
    MAE bruto (sem máscara) na grade H×W. Sem estrutura de grafo — mae_graph=None.

    batch já chega normalizado (CUDAPrefetcher.encode_batch, se normalize=True)
    — decodifica pred/y de volta pra unidade física antes do MAE, pra manter o
    significado documentado de mae_hw em metrics.jsonl.
    """
    normalizer = getattr(model, 'normalizer', None)
    x, y = batch
    with torch.no_grad():
        pred = model(x.to(device))
        y_d  = y.to(device)
        if normalizer is not None:
            pred = normalizer.decode(pred, 'y_hw')
            y_d  = normalizer.decode(y_d,  'y_hw')
        mae_hw = torch.mean(torch.abs(pred - y_d)).item()
    return mae_hw, None
