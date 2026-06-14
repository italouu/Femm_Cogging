import torch
import torch.nn as nn


class NodeMLP(nn.Module):
    """MLP que opera em features de nó [N, in_ch] → [N, out_ch]."""

    def __init__(self, in_ch, out_ch, hidden_ch, n_layers, act=nn.GELU):
        super().__init__()
        if n_layers == 1:
            layers = [nn.Linear(in_ch, out_ch)]
        else:
            layers = [nn.Linear(in_ch, hidden_ch), act()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_ch, hidden_ch), act()]
            layers += [nn.Linear(hidden_ch, out_ch)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    """
    MLP que opera em tensores de grade [B, C, H, W] → [B, C', H, W].
    Usado pelo FNO para lift e projeção.
    """

    def __init__(self, in_ch, out_ch, layers, width, act=nn.GELU):
        super().__init__()
        if layers == 1:
            seq = [nn.Linear(in_ch, out_ch), act()]
        else:
            seq = [nn.Linear(in_ch, width), act()]
            for _ in range(layers - 2):
                seq += [nn.Linear(width, width), act()]
            seq += [nn.Linear(width, out_ch)]
        self.mlp = nn.Sequential(*seq)
        self.in_ch, self.out_ch = int(in_ch), int(out_ch)

    def forward(self, x):
        B, C, res_x, res_y = x.shape
        x = x.permute(0, 2, 3, 1).flatten(1, 2)            # [B, H*W, C]
        x = self.mlp(x)
        x = x.view(B, res_x, res_y, self.out_ch).permute(0, 3, 1, 2)
        return x


class SpectralConv(nn.Module):
    """Convolução espectral 2D via rfft2/irfft2."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.weights1 = nn.Parameter(torch.rand(in_channels, out_channels,
                                                 modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.rand(in_channels, out_channels,
                                                 modes1, modes2, dtype=torch.cfloat))

    def _compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels,
                             x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1,  :self.modes2] = \
            self._compl_mul2d(x_ft[:, :, :self.modes1,  :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self._compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO_Blocks(nn.Module):
    """Sequência de blocos FNO: SpectralConv + bypass linear com ativação após cada bloco."""

    def __init__(self, modes1, modes2, conv_layers, conv_width, act=nn.GELU):
        super().__init__()
        self.modes1      = modes1
        self.modes2      = modes2
        self.conv_layers = conv_layers
        self.conv_width  = conv_width
        self.act         = act()

        self.spec_convs = nn.ModuleList([
            SpectralConv(conv_width, conv_width, modes1, modes2)
            for _ in range(conv_layers)
        ])
        self.conv_bypass = nn.ModuleList([
            MLP(in_ch=conv_width, out_ch=conv_width, layers=1, width=1, act=nn.Identity)
            for _ in range(conv_layers)
        ])

    def forward(self, x):
        for spec, bypass in zip(self.spec_convs, self.conv_bypass):
            x = self.act(spec(x) + bypass(x))
        return x


class EdgeConvLayer(nn.Module):
    """
    Message passing com gate escalar por aresta.

    gate_ij = sigmoid( gate_mlp(edge_attr_ij) )
    m_ij    = gate_ij * W( h[src] )
    agg[dst] += m_ij
    h'[dst] = upd_mlp( cat(h[dst], agg[dst]) )
    """

    def __init__(self, node_width, edge_dim, act=nn.GELU):
        super().__init__()

        # [REMOVIDO] msg_mlp: NodeMLP(2*node_width + edge_dim → msg_width)
        # custo alto por aresta — gather de src+dst para cada aresta em MLP grande

        self.gate_mlp = NodeMLP(
            in_ch=edge_dim, out_ch=1, hidden_ch=edge_dim * 4, n_layers=2, act=act,
        )
        self.W = nn.Linear(node_width, node_width, bias=False)
        self.upd_mlp = NodeMLP(
            in_ch=2 * node_width,   # [REMOVIDO] era node_width + msg_width
            out_ch=node_width, hidden_ch=node_width, n_layers=2, act=act,
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]

        # [REMOVIDO] msg_in = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        # [REMOVIDO] msg    = self.msg_mlp(msg_in)

        gate = torch.sigmoid(self.gate_mlp(edge_attr))   # [E_tot, 1]
        msg  = gate * self.W(h[src])                      # [E_tot, node_width]

        agg = torch.zeros(h.size(0), msg.size(-1), dtype=h.dtype, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        return self.upd_mlp(torch.cat([h, agg], dim=-1))


class GNN(nn.Module):
    """
    GNN edge-conditioned com N camadas EdgeConvLayer.

    h = act( lift(node_features) )
    for layer in layers: h = layer(h, edge_index, edge_attr)
    out = proj(h)
    """

    def __init__(self, in_node_features, out_node_features, edge_dim,
                 node_width, n_layers, act=nn.GELU):
        super().__init__()

        self.lift   = nn.Linear(in_node_features, node_width)
        self.layers = nn.ModuleList([
            EdgeConvLayer(node_width, edge_dim, act) for _ in range(n_layers)
        ])
        self.proj = nn.Linear(node_width, out_node_features)
        self.act  = act()

    def forward(self, node_features, edge_index, edge_attr):
        h = self.act(self.lift(node_features))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.proj(h)
