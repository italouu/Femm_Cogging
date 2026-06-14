import torch.nn as nn
from neuralop.models import FNO  # module-level — ImportError imediato se neuraloperator não instalado


class FNORef(nn.Module):
    """Wrapper sobre neuralop.FNO para integração com ARCH_REGISTRY. Interface idêntica ao FNO2d."""

    def __init__(self, in_channels, out_channels, n_modes, hidden_channels,
                 n_layers, lifting_channel_ratio, projection_channel_ratio):
        super().__init__()
        self.model = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
        )

    def forward(self, x):
        return self.model(x)
