import torch.nn.functional as F
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_ch, out_ch, layers, width, act=nn.GELU):
        super().__init__()
        if layers == 1:
            seq = [nn.Linear(in_ch, out_ch), act()]
        else:
            seq = [nn.Linear(in_ch, width), act()]      # [batch, (flatten_xy), in_ch] -> [batch, (flatten_xy), width]
            for _ in range(layers - 2):
                seq += [nn.Linear(width, width), act()] # [batch, (flatten_xy), width] -> [batch, (flatten_xy), width]
            seq += [nn.Linear(width, out_ch)]           # [batch, (flatten_xy), width] -> [batch, out_ch,       (flatten_xy)]
        self.mlp = nn.Sequential(*seq)
        self.in_ch, self.out_ch = int(in_ch), int(out_ch)

    def forward(self, x):
        B, C, res_x, res_y = x.shape
        x = x.permute(0, 2, 3, 1).flatten(1, 2)   # [B, data, in_ch] nn.linear format
        x = self.mlp(x)
        x = x.view(B, res_x, res_y, self.out_ch).permute(0, 3, 1, 2)
        return x

class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.in_channels = in_channels # input width ( R^ in_ch x resolution )
        self.out_channels = out_channels # output width ( R^ out_ch x resolution )
        self.modes1 = modes1 # modes in axis 1 (x)
        self.modes2 = modes2 # modes in axis 2 (y)

        # these weights have more dofs then needed, possible breaking hermitian matrix after conv
        # however, rfft2 return only the real part
        self.weights1 = nn.Parameter(torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def _compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # out[b,i,x,y] = sum_i ( in[b,i,x,y] * weights[i,o,x,y] )
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        x_ft = torch.fft.rfft2(x) # rff2 takes the last two dimensions by def

        # Spectral mult
        out_ft = torch.zeros(batchsize, self.out_channels, 
                             x.size(-2),          # dim X
                             x.size(-1) // 2 + 1, # dim Y (rfft2 only returns the positive frequencies in last axis)
                             dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self._compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self._compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) # s (Tuple[int], optional): Signal size in the transformed dimensions
        return x

class FNO_Blocks(nn.Module):
    def __init__(self, modes1, modes2, conv_layers, conv_width, act=nn.GELU): # TODO: add args
        super().__init__()
        self.modes1 = modes1            # modes to keep is axis 1
        self.modes2 = modes2            # modes to keep is axis 2
        self.conv_layers = conv_layers  # number of frequency convolution layers
        self.conv_width = conv_width    # number of frequency convolution blocks per layer
        self.act = act()

        self.spec_convs = nn.ModuleList()
        for _ in range(self.conv_layers):
            self.spec_convs.append(
                SpectralConv(in_channels=self.conv_width,   # [batch, conv_width, res_x, res_y] -> [batch, conv_width, res_x, res_y]
                            out_channels=self.conv_width,
                            modes1=self.modes1,
                            modes2=self.modes2)
                )
            
        self.conv_bypass = nn.ModuleList()
        for _ in range(self.conv_layers):
            self.conv_bypass.append(
                MLP(in_ch=self.conv_width,
                    out_ch=self.conv_width,
                    layers=1,
                    width=1,
                    act=nn.Identity)
                )
    
    def forward(self, x):
        for i in range(self.conv_layers):
            x0 = self.spec_convs[i](x)
            x1 = self.conv_bypass[i](x)
            x = self.act(x0 + x1)
        return x

class FNO2d(nn.Module):
    def __init__(self,
        in_channels,    # number of input channels
        out_channels,   # number of output channels
        modes1,         # modes to keep is axis 1
        modes2,         # modes to keep is axis 2
        conv_width,     # number of frequency convolution blocks per layer
        conv_layers,    # number of frequency convolution layers
        lift_width,     # width of lift mlp hidden layers
        lift_layers,    # number of lift mlp layers
        proj_width,     # width of proj mlp hidden layers
        proj_layers,    # number of proj mlp layers
        data_res,       # [res_r, res_th]
        ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 if modes1 <= data_res[0] else data_res[0]              # check max axis modes
        self.modes2 = modes2 if modes2 <= data_res[1]//2 + 1 else data_res[1]//2 +1 # check max axis modes
        self.conv_width = conv_width
        self.conv_layers = conv_layers
        self.lift_width = lift_width
        self.lift_layers = lift_layers
        self.proj_width = proj_width
        self.proj_layers = proj_layers
        self.data_res = data_res

        self._create_net()

    def _create_net(self):
        self.lift_layer = self._lift_layer()
        self.proj_layer = self._proj_layer()
        self.conv_layer = self._conv_layer()

    def _lift_layer(self):
        return MLP(in_ch=self.in_channels, 
                        out_ch=self.conv_width, 
                        layers=self.lift_layers, 
                        width=self.lift_width)
    
    def _proj_layer(self):
        return MLP(in_ch=self.conv_width, 
                        out_ch=self.out_channels, 
                        layers=self.proj_layers, 
                        width=self.proj_width)

    def _conv_layer(self):
        return FNO_Blocks(modes1=self.modes1, 
                        modes2=self.modes2, 
                        conv_layers=self.conv_layers, 
                        conv_width=self.conv_width)
    
    def forward(self, x):
        x = self.lift_layer(x)
        x = self.conv_layer(x)
        x = self.proj_layer(x)
        return x