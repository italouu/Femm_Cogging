import torch
import torch.nn as nn
from src.neural_op.archs._blocks import MLP, FNO_Blocks


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
                 data_res):
        super().__init__()

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
    """MAE bruto (sem máscara) na grade H×W. Sem estrutura de grafo — mae_graph=None."""
    x, y = batch
    with torch.no_grad():
        pred   = model(x.to(device))
        mae_hw = torch.mean(torch.abs(pred - y.to(device))).item()
    return mae_hw, None
