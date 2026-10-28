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

        self.gate_mlp = NodeMLP(
            in_ch=edge_dim, out_ch=1, hidden_ch=edge_dim * 4, n_layers=2, act=act,
        )
        self.W = nn.Linear(node_width, node_width, bias=False)
        self.upd_mlp = NodeMLP(
            in_ch=2 * node_width,
            out_ch=node_width, hidden_ch=node_width, n_layers=2, act=act,
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]

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


# ---------------------------------------------------------------------------
# Grafo duplo (mode='bipartite', ver src/data_gen/parsers/bipartite_structure.py e
# docs/claude/modelos.md §3): vértices (iterado) + elementos (estático,
# nunca atualizado, só injeta via arestas cruzadas). EdgeMessageBlock isola
# só a parte de "mensagem" do EdgeConvLayer (gate × transform + soma) — sem
# a atualização de nó — pra ser reaproveitada tanto nas arestas vértice-
# vértice quanto nas cruzadas elemento->vértice, com pesos independentes.
# ---------------------------------------------------------------------------

class EdgeMessageBlock(nn.Module):
    """
    Só a parte de mensagem de um EdgeConvLayer (gate × transform, agregada
    por soma) — sem MLP de atualização de nó. src e dst podem ter dimensões
    e até "tipos" de nó diferentes (ex: elemento -> vértice).

    gate_ij = sigmoid( gate_mlp(edge_attr_ij) )
    m_ij    = gate_ij * W( h_src[src] )
    out[dst] = Σ_i m_ij
    """

    def __init__(self, src_width, dst_width, edge_dim, act=nn.GELU):
        super().__init__()
        self.gate_mlp = NodeMLP(
            in_ch=edge_dim, out_ch=1, hidden_ch=edge_dim * 4, n_layers=2, act=act,
        )
        self.W = nn.Linear(src_width, dst_width, bias=False)

    def forward(self, h_src, edge_index, edge_attr, num_dst):
        src, dst = edge_index[0], edge_index[1]
        gate = torch.sigmoid(self.gate_mlp(edge_attr))          # [E, 1]
        msg  = gate * self.W(h_src[src])                         # [E, dst_width]
        out  = torch.zeros(num_dst, msg.size(-1), dtype=msg.dtype, device=msg.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out


class BipartiteEdgeConvLayer(nn.Module):
    """
    Uma camada do stack de vértices que recebe DUAS fontes de mensagem:
    F1 (vizinhos vértice-vértice, grafo iterado) e F2 (vizinhos elemento,
    grafo cruzado — elementos nunca são atualizados, mas F2 tem pesos
    PRÓPRIOS por camada, lendo sempre o mesmo elem_x estático).

    h_j' = upd_mlp( [ h_j | F1_j | F2_j ] )
    F1_j = Σ_{i∈viz_vértice(j)} gate1(e_ij)  · W1(h_i)
    F2_j = Σ_{e∈viz_elem(j)}    gate2(d_je)  · W2(elem_x_e)
    """

    def __init__(self, node_width, edge_dim, elem_in_ch, cross_edge_dim, act=nn.GELU):
        super().__init__()
        self.msg_vertex = EdgeMessageBlock(node_width, node_width, edge_dim, act)
        self.msg_elem   = EdgeMessageBlock(elem_in_ch, node_width, cross_edge_dim, act)
        self.upd_mlp = NodeMLP(
            in_ch=3 * node_width, out_ch=node_width, hidden_ch=node_width, n_layers=2, act=act,
        )

    def forward(self, h, elem_x, edge_index, edge_attr, cross_edge_index, cross_edge_attr):
        n = h.size(0)
        f1 = self.msg_vertex(h,      edge_index,       edge_attr,       n)
        f2 = self.msg_elem(elem_x,   cross_edge_index,  cross_edge_attr, n)
        return self.upd_mlp(torch.cat([h, f1, f2], dim=-1))


class BipartiteGNN(nn.Module):
    """
    GNN com grafo de vértices (iterado, n_layers) + grafo de elementos
    (estático, injetado a cada camada via BipartiteEdgeConvLayer, nunca
    atualizado — sem arestas internas entre elementos).

    h = act( lift(node_features) )
    for layer in layers: h = layer(h, elem_x, edge_index, edge_attr, cross_edge_index, cross_edge_attr)
    out = proj(h)
    """

    def __init__(self, in_node_features, out_node_features, edge_dim,
                 elem_in_ch, cross_edge_dim, node_width, n_layers, act=nn.GELU):
        super().__init__()
        self.lift   = nn.Linear(in_node_features, node_width)
        self.layers = nn.ModuleList([
            BipartiteEdgeConvLayer(node_width, edge_dim, elem_in_ch, cross_edge_dim, act)
            for _ in range(n_layers)
        ])
        self.proj = nn.Linear(node_width, out_node_features)
        self.act  = act()

    def forward(self, node_features, elem_x, edge_index, edge_attr, cross_edge_index, cross_edge_attr):
        h = self.act(self.lift(node_features))
        for layer in self.layers:
            h = layer(h, elem_x, edge_index, edge_attr, cross_edge_index, cross_edge_attr)
        return self.proj(h)


# ---------------------------------------------------------------------------
# Grafo dual + vértices por material (FNO_SplitGNN, ver
# src/data_gen/parsers/split_layout.py e docs/claude/modelos.md §3): os DOIS tipos de nó
# têm estado e rede próprios. N_aux = elementos (Edge_dual entre si), N_base = vértices
# divididos por material; Edge_aux liga os dois tipos e é usada nos dois sentidos.
# Arestas estáticas (só os atributos do chunk), mensagens de EdgeMessageBlock.
# ---------------------------------------------------------------------------

class SplitGNNLayer(nn.Module):
    """
    Uma camada: os dois tipos de nó leem só o estado da camada k (atualização simultânea).

    N_aux'  = upd_aux( [ h_aux  | Σ msg_dual(h_aux) | Σ msg_base→aux(h_base) ] )
    N_base' = upd_base( [ h_base | Σ msg_aux→base(h_aux) ] )
    """

    def __init__(self, node_width, dual_edge_dim, aux_edge_dim, act=nn.GELU):
        super().__init__()
        self.msg_dual     = EdgeMessageBlock(node_width, node_width, dual_edge_dim, act)
        self.msg_base_aux = EdgeMessageBlock(node_width, node_width, aux_edge_dim, act)
        self.msg_aux_base = EdgeMessageBlock(node_width, node_width, aux_edge_dim, act)
        self.upd_aux = NodeMLP(
            in_ch=3 * node_width, out_ch=node_width, hidden_ch=node_width, n_layers=2, act=act,
        )
        self.upd_base = NodeMLP(
            in_ch=2 * node_width, out_ch=node_width, hidden_ch=node_width, n_layers=2, act=act,
        )

    def forward(self, h_aux, h_base, edge_index, edge_attr,
                base_to_aux, aux_to_base, cross_edge_attr):
        n_aux, n_base = h_aux.size(0), h_base.size(0)
        f_dual = self.msg_dual(h_aux, edge_index, edge_attr, n_aux)
        f_ba   = self.msg_base_aux(h_base, base_to_aux, cross_edge_attr, n_aux)
        f_ab   = self.msg_aux_base(h_aux, aux_to_base, cross_edge_attr, n_base)
        return (self.upd_aux(torch.cat([h_aux, f_dual, f_ba], dim=-1)),
                self.upd_base(torch.cat([h_base, f_ab], dim=-1)))


class SplitGNN(nn.Module):
    """
    lift por tipo de nó -> n_layers SplitGNNLayer -> projeção por tipo de nó.

    h_aux  = act( lift_aux(aux_features) ),  h_base = act( lift_base(base_features) )
    for layer in layers: h_aux, h_base = layer(...)
    out = proj_aux(h_aux), proj_base(h_base)

    cross_edge_index: linha 0 = N_base, linha 1 = N_aux (convenção do layout).
    """

    def __init__(self, in_aux_features, in_base_features, out_features,
                 dual_edge_dim, aux_edge_dim, node_width, n_layers, act=nn.GELU):
        super().__init__()
        self.lift_aux  = nn.Linear(in_aux_features, node_width)
        self.lift_base = nn.Linear(in_base_features, node_width)
        self.layers = nn.ModuleList([
            SplitGNNLayer(node_width, dual_edge_dim, aux_edge_dim, act) for _ in range(n_layers)
        ])
        self.proj_aux  = nn.Linear(node_width, out_features)
        self.proj_base = nn.Linear(node_width, out_features)
        self.act = act()

    def forward(self, aux_features, base_features, edge_index, edge_attr,
                cross_edge_index, cross_edge_attr):
        h_aux  = self.act(self.lift_aux(aux_features))
        h_base = self.act(self.lift_base(base_features))
        aux_to_base = cross_edge_index.flip(0)
        for layer in self.layers:
            h_aux, h_base = layer(h_aux, h_base, edge_index, edge_attr,
                                  cross_edge_index, aux_to_base, cross_edge_attr)
        return self.proj_aux(h_aux), self.proj_base(h_base)
