"""Figuras do artigo: diagrama de blocos de cada arquitetura (FNO2d, FNO_GNN, FNO_Bipartite,
FNO_SplitGNN), sem detalhar hiperparâmetros (o texto explica), mostrando as conexões.

Gramática comum:
  - linha principal: entrada → FNO → interpolação e reescala grade → grafo (bilinear +
    recodificação das estatísticas da grade para as do grafo) → concatenação com as features
    do grafo → GNN → soma residual com o FNO interpolado → saída;
  - camada da GNN aberta (notação do artigo, aresta i → j, relação r): g_ij = sig(MLP_g,r(e_ij)),
    m_ij = g_ij W_r h_i, F_j,r = Σ_i m_ij, h_j^(l+1) = MLP_u,τ([h_j ‖ F_j,r1 ‖ ...]);
    tipos de nó V (vértice), E (elemento), C (cópia de vértice por material, só no Split);
  - cores por tipo de bloco: dados (branco, canto reto), FNO (azul), interpolação (verde),
    GNN (laranja), gate (rosa).

Padrão: quadrados de 3,5 in (7 pt), fluxo de cima para baixo à esquerda e a camada aberta à
direita; FNO2d sem detalhe. --estilo largo: versões de 7,16 in (fluxo horizontal + detalhe embaixo).
Saída: docs/figures/arq_{fno2d,fno_gnn,fno_bipartite,fno_splitgnn}.{pdf,png}.

    python -m scripts.fig_arquiteturas [--only fno2d fno_gnn fno_bipartite fno_splitgnn]
                                       [--estilo quadrado|largo]
"""
import argparse

import numpy as np

from scripts.postprocess import ARTICLE_FIG_DIR
from src.postprocess.plots.primitives import apply_article_style, apply_style

FIG_WIDTH_IN = 7.16
FONT_PT = 8
W_UNITS = 100.0                                   # largura do desenho em unidades
COLORS = dict(data='white', fno='#DCEAF5', proj='#E2F0E6', gnn='#FCEBD5', gate='#F7DDEA')
EDGE = '0.25'
ARROW = '0.15'
FRAME = '0.45'


class Node:
    """Retângulo (ou círculo) centrado em (x, y) com âncoras nas bordas."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    l = property(lambda s: (s.x - s.w / 2, s.y))
    r = property(lambda s: (s.x + s.w / 2, s.y))
    t = property(lambda s: (s.x, s.y + s.h / 2))
    b = property(lambda s: (s.x, s.y - s.h / 2))

    def at(self, side, d=0.0):
        """Ponto na borda `side` deslocado de d (ao longo da borda)."""
        x, y = getattr(self, side)
        return (x, y + d) if side in 'lr' else (x + d, y)


class Diagram:
    # [REMOVIDO] __init__(self, H): só figura própria de 7,16 in e 8 pt -- agora aceita um eixo
    # já existente (painel da figura 2×2), outra largura e outra fonte
    def __init__(self, H, width_in=FIG_WIDTH_IN, fs=FONT_PT, ax=None):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.H = H
        self.fs = fs
        if ax is None:
            self.fig = plt.figure(figsize=(width_in, width_in * H / W_UNITS))
            ax = self.fig.add_axes([0, 0, 1, 1])
        else:
            self.fig = ax.figure
        ax.set_xlim(0, W_UNITS)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.axis('off')
        self.ax = ax

    def box(self, x, y, w, h, txt, kind='data', fs=None, lw=0.8):
        from matplotlib.patches import FancyBboxPatch
        style = 'square,pad=0' if kind == 'data' else 'round,pad=0,rounding_size=1.0'
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                         fc=COLORS[kind], ec=EDGE, lw=lw, zorder=3))
        self.ax.text(x, y, txt, ha='center', va='center', fontsize=fs or self.fs, zorder=4,
                     linespacing=1.15)
        return Node(x, y, w, h)

    def op(self, x, y, sym, r=1.6):
        from matplotlib.patches import Circle
        self.ax.add_patch(Circle((x, y), r, fc='white', ec=EDGE, lw=0.8, zorder=3))
        self.ax.text(x, y, sym, ha='center', va='center', fontsize=self.fs + 1, zorder=4)
        return Node(x, y, 2 * r, 2 * r)

    def bar(self, x, y0, y1, w=2.2):
        """Concatenação: barra vertical estreita (entradas pela esquerda/topo/base)."""
        from matplotlib.patches import Rectangle
        self.ax.add_patch(Rectangle((x - w / 2, y0), w, y1 - y0, fc='0.85', ec=EDGE, lw=0.8,
                                    zorder=3))
        self.ax.text(x, (y0 + y1) / 2, r'$\|$', ha='center', va='center', fontsize=self.fs,
                     zorder=4)
        return Node(x, (y0 + y1) / 2, w, y1 - y0)

    def bar_h(self, x0, x1, y, h=2.6):
        """Concatenação horizontal (entradas por cima, saída por baixo)."""
        from matplotlib.patches import Rectangle
        self.ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, fc='0.85', ec=EDGE, lw=0.8,
                                    zorder=3))
        self.ax.text((x0 + x1) / 2, y, 'concatenação', ha='center', va='center',
                     fontsize=self.fs - 1.5, zorder=4)
        return Node((x0 + x1) / 2, y, x1 - x0, h)

    def stack(self, x, y, w, h, txt, kind, n=3, d=1.2):
        """Bloco repetido: n-1 cópias deslocadas atrás do bloco principal."""
        from matplotlib.patches import FancyBboxPatch
        for k in range(n - 1, 0, -1):
            self.ax.add_patch(FancyBboxPatch((x - w / 2 + k * d, y - h / 2 + k * d), w, h,
                                             boxstyle='round,pad=0,rounding_size=1.0',
                                             fc=COLORS[kind], ec=EDGE, lw=0.6, zorder=2.5))
        return self.box(x, y, w, h, txt, kind)

    def arrow(self, *pts, ls='-', color=ARROW, lw=0.8, head=True):
        from matplotlib.patches import FancyArrowPatch
        P = np.asarray(pts, float)
        if len(P) > 2:
            # [REMOVIDO] zorder=2 (ficava atrás das cópias do bloco empilhado, zorder 2.5)
            self.ax.plot(P[:-1, 0], P[:-1, 1], color=color, lw=lw, ls=ls, zorder=2.8,
                         solid_capstyle='butt')
        self.ax.add_patch(FancyArrowPatch(P[-2], P[-1], arrowstyle='-|>' if head else '-',
                                          mutation_scale=7, lw=lw, color=color, ls=ls,
                                          shrinkA=0, shrinkB=0, zorder=2.8))

    def text(self, x, y, s, fs=None, **kw):
        kw.setdefault('ha', 'center')
        kw.setdefault('va', 'center')
        self.ax.text(x, y, s, fontsize=fs or self.fs - 1, zorder=5, **kw)

    def frame(self, x0, y0, x1, y1, label, bottom=False):
        from matplotlib.patches import FancyBboxPatch
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                         boxstyle='round,pad=0,rounding_size=1.5', fc='none',
                                         ec=FRAME, lw=0.8, ls=(0, (4, 2.5)), zorder=1))
        self.ax.text(x0 + 1.2, (y0 + 1.2) if bottom else (y1 - 1.2), label, ha='left',
                     va='bottom' if bottom else 'top', fontsize=self.fs - 0.5,
                     style='italic', color='0.25', zorder=5)

    def zoom(self, node, x0, x1, y1):
        """Linhas finas tracejadas do bloco até o topo da moldura do detalhe."""
        for (xa, ya), xb in ((node.at('b', -node.w / 2), x0), (node.at('b', node.w / 2), x1)):
            self.ax.plot([xa, xb], [ya, y1], color='0.6', lw=0.5, ls=(0, (2, 2)), zorder=0)

    def save(self, name, out_dir=ARTICLE_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out_dir / f'{name}.pdf')
        self.fig.savefig(out_dir / f'{name}.png', dpi=300)
        self.plt.close(self.fig)
        return str(out_dir / f'{name}.pdf')


# --------------------------------------------------------------------------------------------
# peças reutilizadas
# --------------------------------------------------------------------------------------------
def message(D, y, src, edge, W, x0=12.0):
    """Mensagem com gate (aberta): e → gate sig(MLP(e)); h → W; × ; Σ nos vizinhos.
    Linha da mensagem em y, gate em y + 6. Devolve o nó Σ."""
    e = D.box(x0, y + 6, 8, 4.5, edge)
    s = D.box(x0, y, 8, 4.5, src)
    g = D.box(x0 + 18, y + 6, 18, 4.5, r'gate: sig(MLP($e$))', 'gate')
    w = D.box(x0 + 18, y, 8, 4.5, W, 'gnn')
    mul = D.op(x0 + 32, y, r'$\times$')
    sm = D.op(x0 + 40, y, r'$\Sigma$')
    D.arrow(e.r, g.l)
    D.arrow(s.r, w.l)
    D.arrow(g.r, (mul.x, g.y), mul.t)
    D.arrow(w.r, mul.l)
    D.arrow(mul.r, sm.l)
    return sm


def hybrid_top(D, yc, proj_txt, x_txt, gnn_txt, y_txt, below_txt=None):
    """Linha principal das archs híbridas de 1 tipo de nó. Devolve o bloco da GNN."""
    xin = D.box(7, yc, 12, 7, r'$x_{hw}$' + '\n' + r'$(\mu_r, M, J)$')
    fno = D.box(23.5, yc, 15, 7, 'FNO\n' + r'($P$, 4$\times$Fourier, $Q$)', 'fno')
    pr = D.box(41.5, yc, 14, 7, proj_txt, 'proj')
    u = D.box(53, yc, 5, 5, r'$\tilde u$')
    cat = D.op(60.5, yc, r'$\|$')
    xn = D.box(60.5, yc + 9, 12, 5, x_txt)
    g = D.box(75, yc, 16, 7, gnn_txt, 'gnn')
    plus = D.op(87, yc, r'$+$')
    yo = D.box(95, yc, 8, 7, y_txt)
    D.arrow(xin.r, fno.l)
    D.arrow(fno.r, pr.l)
    D.text((fno.r[0] + pr.l[0]) / 2, yc + 2.2, r'$\hat y_{hw}$')
    D.arrow(pr.r, u.l)
    D.arrow(u.r, cat.l)
    D.arrow(xn.b, cat.t)
    D.arrow(cat.r, g.l)
    D.arrow(g.r, plus.l)
    D.text((g.r[0] + plus.l[0]) / 2, yc + 2.0, r'$\Delta$')
    D.arrow(plus.r, yo.l)
    D.arrow(u.t, (u.x, yc + 14), (plus.x, yc + 14), plus.t)       # resíduo
    D.text((u.x + plus.x) / 2 + 8, yc + 12.6, 'resíduo')
    if below_txt:
        bl = D.box(g.x, yc - 9, 16, 5, below_txt)
        D.arrow(bl.t, g.b)
    return g


# --------------------------------------------------------------------------------------------
# diagramas
# --------------------------------------------------------------------------------------------
def fig_fno2d():
    D = Diagram(40)
    yc = 32
    xin = D.box(9, yc, 14, 7, r'$x_{hw}$' + '\n' + r'$(\mu_r, M, J)$')
    P = D.box(29, yc, 15, 7, 'Lifting $P$\n(MLP pontual)', 'fno')
    F = D.box(51, yc, 17, 7, 'Camada de\nFourier ' + r'$\times 4$', 'fno')
    Q = D.box(73, yc, 15, 7, 'Projeção $Q$\n(MLP pontual)', 'fno')
    yo = D.box(92, yc, 13, 7, r'$\hat y_{hw}$' + '\n' + r'$(B_r, B_\theta)$')
    for a, b in ((xin, P), (P, F), (F, Q), (Q, yo)):
        D.arrow(a.r, b.l)
    # detalhe: camada de Fourier
    D.frame(8, 1, 92, 22, r'Camada de Fourier $\ell$')
    D.zoom(F, 8, 92, 22)
    yd = 10.5
    v = D.box(15, yd, 6, 5, r'$v_\ell$')
    fft = D.box(33, yd + 5, 11, 4.5, 'FFT 2D', 'fno')
    R = D.box(48, yd + 5, 14, 4.5, r'$R_\ell$ (modos)', 'fno')
    ifft = D.box(63, yd + 5, 11, 4.5, r'FFT$^{-1}$', 'fno')
    Wl = D.box(48, yd - 5, 14, 4.5, r'$W_\ell$ (linear)', 'fno')
    plus = D.op(72, yd, r'$+$')
    gelu = D.box(79.5, yd, 8, 4.5, 'GELU', 'fno')
    vo = D.box(88, yd, 6, 5, r'$v_{\ell+1}$')
    D.arrow(v.r, (22, yd), (22, fft.y), fft.l)
    D.arrow((22, yd), (22, Wl.y), Wl.l)
    D.arrow(fft.r, R.l)
    D.arrow(R.r, ifft.l)
    D.arrow(ifft.r, (plus.x, ifft.y), plus.t)
    D.arrow(Wl.r, (plus.x, Wl.y), plus.b)
    D.arrow(plus.r, gelu.l)
    D.arrow(gelu.r, vo.l)
    return D.save('arq_fno2d')


def fig_fno_gnn():
    D = Diagram(56)
    yc = 40
    g = hybrid_top(D, yc, 'Projeção\ngrade ' + r'$\to$' + ' nós', r'$x^{\mathrm{nó}}$',
                   'GNN\n' + r'3 camadas', r'$\hat y^{\mathrm{nó}}$', r'arestas $e_{ij}$')
    D.frame(4, 1, 96, 26, r'Camada de mensagens ($\times 3$)')
    D.zoom(g, 4, 96, 26)
    sm = message(D, 13, r'$h_i$', r'$e_{ij}$', r'$W$')
    D.text(sm.x + 2.5, sm.y + 1.7, 'mensagem', ha='left')
    hj = D.box(12, 4.5, 8, 4.5, r'$h_j$')
    cat = D.bar(62, 2.5, 15.5)
    D.arrow(sm.r, (cat.l[0], sm.y))
    D.arrow(hj.r, (cat.l[0], hj.y))
    upd = D.box(75, 9, 14, 6, 'MLP do nó', 'gnn')
    out = D.box(90, 9, 8, 4.5, r'$h_j^{\ell+1}$')
    D.arrow(cat.r, upd.l)
    D.arrow(upd.r, out.l)
    return D.save('arq_fno_gnn')


def fig_fno_bipartite():
    D = Diagram(65)
    yc = 49
    g = hybrid_top(D, yc, 'Projeção\ngrade ' + r'$\to$' + ' centróides', r'$x^{\mathrm{elem}}$',
                   'GNN bipartida\n3 camadas', r'$\hat y^{\mathrm{elem}}$',
                   r'$e^{\mathrm{dual}}$, $x^{\mathrm{vert}}$, $d$')
    D.frame(4, 1, 96, 35, r'Camada bipartida ($\times 3$)')
    D.zoom(g, 4, 96, 35)
    s1 = message(D, 24, r'$h_i$', r'$e^{\mathrm{dual}}_{ij}$', r'$W_1$')
    s2 = message(D, 12, r'$x^{\mathrm{vert}}_a$', r'$d_{aj}$', r'$W_2$')
    D.text(s1.x + 2.5, s1.y + 1.7, 'elem–elem', ha='left')
    D.text(s2.x + 2.5, s2.y + 1.7, 'vért' + r'$\to$' + 'elem', ha='left', fs=FONT_PT - 1.5)
    hj = D.box(12, 4, 8, 4.5, r'$h_j$')
    cat = D.bar(62, 2, 27)
    D.arrow(s1.r, (cat.l[0], s1.y))
    D.arrow(s2.r, (cat.l[0], s2.y))
    D.arrow(hj.r, (cat.l[0], hj.y))
    upd = D.box(75, 14.5, 14, 6, 'MLP do nó', 'gnn')
    out = D.box(90, 14.5, 8, 4.5, r'$h_j^{\ell+1}$')
    D.arrow(cat.r, upd.l)
    D.arrow(upd.r, out.l)
    return D.save('arq_fno_bipartite')


def fig_fno_splitgnn():
    D = Diagram(103)
    yc = 80
    ya, yb = yc + 8, yc - 8
    xin = D.box(7, yc, 12, 7, r'$x_{hw}$' + '\n' + r'$(\mu_r, M, J)$')
    fno = D.box(23.5, yc, 15, 7, 'FNO\n' + r'($P$, 4$\times$Fourier, $Q$)', 'fno')
    pr = D.box(40.5, yc, 13, 9, 'Projeção\ngrade ' + r'$\to$' + ' grafo', 'proj')
    ua = D.box(53, ya, 7, 5, r'$\tilde u^{\mathrm{aux}}$')
    ub = D.box(53, yb, 7, 5, r'$\tilde u^{\mathrm{base}}$')
    ca = D.op(62, ya, r'$\|$')
    cb = D.op(62, yb, r'$\|$')
    xa = D.box(62, ya + 8, 12, 5, r'$x^{\mathrm{aux}}$ (elementos)', fs=FONT_PT - 1)
    xb = D.box(62, yb - 8, 12, 5, r'$x^{\mathrm{base}}$ (cópias)', fs=FONT_PT - 1)
    g = D.box(75.5, yc, 13, ya - yb + 7, 'SplitGNN\n3 camadas\n' + r'($e^{\mathrm{dual}}$, $e^{\mathrm{aux}}$)',
              'gnn')
    pa = D.op(87, ya, r'$+$')
    pb = D.op(87, yb, r'$+$')
    oa = D.box(95, ya, 8, 5, r'$\hat y^{\mathrm{aux}}$')
    ob = D.box(95, yb, 8, 5, r'$\hat y^{\mathrm{base}}$')
    D.arrow(xin.r, fno.l)
    D.arrow(fno.r, pr.l)
    D.text((fno.r[0] + pr.l[0]) / 2, yc + 2.2, r'$\hat y_{hw}$')
    D.arrow(pr.at('r', 2.5), (48, yc + 2.5), (48, ya), ua.l)
    D.arrow(pr.at('r', -2.5), (48, yc - 2.5), (48, yb), ub.l)
    for u, c, x, p, o, sgn in ((ua, ca, xa, pa, oa, 1), (ub, cb, xb, pb, ob, -1)):
        D.arrow(u.r, c.l)
        D.arrow(x.b if sgn > 0 else x.t, c.t if sgn > 0 else c.b)
        D.arrow(c.r, (g.l[0], c.y))
        D.arrow((g.r[0], c.y), p.l)
        D.arrow(p.r, o.l)
        yy = u.y + sgn * 13
        D.arrow(u.t if sgn > 0 else u.b, (u.x, yy), (p.x, yy), p.t if sgn > 0 else p.b)
    D.text(84.6, ya + 2, r'$\Delta^{\mathrm{aux}}$', fs=FONT_PT - 1.5)
    D.text(84.6, yb + 2, r'$\Delta^{\mathrm{base}}$', fs=FONT_PT - 1.5)
    D.text(78, ya + 11.8, 'resíduo')
    D.text(78, yb - 11.8, 'resíduo')
    # detalhe: 3 mensagens + 2 atualizações
    D.frame(4, 1, 96, 56, r'Camada SplitGNN ($\times 3$)')
    # [REMOVIDO] D.zoom(g, 4, 96, 56)  -- cruzava o resíduo e o x^base
    s1 = message(D, 39, r'$h^{\mathrm{aux}}_j$', r'$e^{\mathrm{dual}}$', r'$W_1$')
    s2 = message(D, 27, r'$h^{\mathrm{base}}_c$', r'$e^{\mathrm{aux}}$', r'$W_2$')
    s3 = message(D, 11, r'$h^{\mathrm{aux}}_i$', r'$e^{\mathrm{aux}}$', r'$W_3$')
    D.text(s1.x + 2.5, s1.y + 1.7, 'aux' + r'$\to$' + 'aux', ha='left')
    D.text(s2.x + 2.5, s2.y + 1.7, 'base' + r'$\to$' + 'aux', ha='left')
    D.text(s3.x + 2.5, s3.y + 1.7, 'aux' + r'$\to$' + 'base', ha='left')
    ha = D.box(12, 50.5, 8, 4.5, r'$h^{\mathrm{aux}}_i$')
    hb = D.box(12, 3.5, 8, 4.5, r'$h^{\mathrm{base}}_c$')
    cat_a = D.bar(62, 25, 44)
    cat_b = D.bar(62, 6, 16)
    D.arrow(s1.r, (cat_a.l[0], s1.y))
    D.arrow(s2.r, (cat_a.l[0], s2.y))
    D.arrow(ha.r, (cat_a.x, ha.y), cat_a.t)
    D.arrow(s3.r, (cat_b.l[0], s3.y))
    D.arrow(hb.r, (cat_b.x, hb.y), cat_b.b)
    ua_ = D.box(76, cat_a.y, 15, 6, r'MLP$^{\mathrm{aux}}$', 'gnn')
    ub_ = D.box(76, cat_b.y, 15, 6, r'MLP$^{\mathrm{base}}$', 'gnn')
    oa_ = D.box(90.5, cat_a.y, 8, 4.5, r'$h_i^{\mathrm{aux},\,\ell+1}$')
    ob_ = D.box(90.5, cat_b.y, 8, 4.5, r'$h_c^{\mathrm{base},\,\ell+1}$')
    D.arrow(cat_a.r, ua_.l)
    D.arrow(ua_.r, oa_.l)
    D.arrow(cat_b.r, ub_.l)
    D.arrow(ub_.r, ob_.l)
    return D.save('arq_fno_splitgnn')


# [REMOVIDO] bloco principal anterior (só as figuras largas) -- substituído pela versão quadrada abaixo:
# [REMOVIDO] FIGS = dict(fno2d=fig_fno2d, fno_gnn=fig_fno_gnn, fno_bipartite=fig_fno_bipartite,
# [REMOVIDO]             fno_splitgnn=fig_fno_splitgnn)
#
#
# [REMOVIDO] if __name__ == '__main__':
# [REMOVIDO]     ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
# [REMOVIDO]     ap.add_argument('--only', nargs='+', choices=list(FIGS), default=list(FIGS))
# [REMOVIDO]     args = ap.parse_args()
# [REMOVIDO]     import matplotlib.pyplot as plt
# [REMOVIDO]     with plt.rc_context():
# [REMOVIDO]         apply_style()
# [REMOVIDO]         apply_article_style(FONT_PT)
# [REMOVIDO]         for k in args.only:
# [REMOVIDO]             print('figura:', FIGS[k]())


# (versões largas mantidas acima; acessíveis por --estilo largo)
FIGS_LARGO = dict(fno2d=fig_fno2d, fno_gnn=fig_fno_gnn, fno_bipartite=fig_fno_bipartite,
                  fno_splitgnn=fig_fno_splitgnn)


# --------------------------------------------------------------------------------------------
# versão quadrada (padrão): fluxo de cima para baixo à esquerda, camada aberta à direita;
# FNO2d sem detalhe (já é bem documentado)
# --------------------------------------------------------------------------------------------
SQ_IN = 3.5                       # lado do quadrado [in] (uma coluna; 2×2 em duas colunas)
SQ_FONT = 7
FX = 48                           # x da moldura do detalhe (esquerda)


# [REMOVIDO] versão quadrada com notação por arch (nó/elem, aux/base) e informação de
# instância (nº de camadas, atributos de entrada) -- substituída pela versão com notação
# padronizada abaixo.
# def _gate_txt():
#     return 'gate\nsig(MLP)'
#
#
# def sq_fno2d(D):
#     x = 50
#     xin = D.box(x, 89, 36, 11, r'$x_{hw}$' + '\n' + r'$(\mu_r, M, J)$')
#     P = D.box(x, 70, 44, 9, 'Lifting $P$ (MLP)', 'fno')
#     F = D.stack(x, 49.5, 44, 12, 'Camada de Fourier\n' + r'$\times 4$', 'fno')
#     Q = D.box(x, 29, 44, 9, 'Projeção $Q$ (MLP)', 'fno')
#     yo = D.box(x, 10, 36, 11, r'$\hat y_{hw}$' + '\n' + r'$(B_r, B_\theta)$')
#     for a, b in ((xin, P), (P, F), (F, Q), (Q, yo)):
#         D.arrow(a.b, b.t)
#
#
# def sq_left_single(D, proj_txt, x_txt, gnn_txt, y_txt, side_txt):
#     """Coluna esquerda das híbridas de 1 tipo de nó (fluxo de cima para baixo)."""
#     cx = 21
#     xin = D.box(cx, 92, 30, 7, r'$x_{hw}$ $(\mu_r, M, J)$')
#     fno = D.box(cx, 80.5, 30, 7, 'FNO', 'fno')
#     pr = D.box(cx, 67, 30, 10, proj_txt, 'proj')
#     u = D.box(cx, 54, 10, 6, r'$\tilde u$')
#     cat = D.op(cx, 44, r'$\|$', r=2.4)
#     xn = D.box(39, 44, 12, 6, x_txt)
#     g = D.box(cx, 31, 24, 10, gnn_txt, 'gnn')
#     side = D.box(40.5, 31, 13, 7, side_txt, fs=SQ_FONT - 1)
#     plus = D.op(cx, 17.5, r'$+$', r=2.4)
#     yo = D.box(cx, 6, 14, 7, y_txt)
#     D.arrow(xin.b, fno.t)
#     D.arrow(fno.b, pr.t)
#     D.text(cx + 1.5, 75.5, r'$\hat y_{hw}$', ha='left')
#     D.arrow(pr.b, u.t)
#     D.arrow(u.b, cat.t)
#     D.arrow(xn.l, cat.r)
#     D.arrow(cat.b, g.t)
#     D.arrow(side.l, g.r)
#     D.arrow(g.b, plus.t)
#     D.text(cx + 1.5, 23, r'$\Delta$', ha='left')
#     D.arrow(plus.b, yo.t)
#     D.arrow(u.l, (4, u.y), (4, plus.y), plus.l)                     # resíduo
#     D.text(2.6, (u.y + plus.y) / 2, 'resíduo', rotation=90)
#     return g
#
#
# def sq_edgeconv(D):
#     D.frame(FX, 2, 98, 98, r'Camada de mensagens ($\times 3$)', bottom=True)
#     e = D.box(57, 90, 12, 6, r'$e_{ij}$')
#     hi = D.box(76, 90, 10, 6, r'$h_i$')
#     hj = D.box(92, 90, 9, 6, r'$h_j$')
#     gt = D.box(57, 76, 14, 9, _gate_txt(), 'gate')
#     W = D.box(76, 76, 10, 7, r'$W$', 'gnn')
#     mul = D.op(76, 63, r'$\times$', r=2.4)
#     sm = D.op(76, 51, r'$\Sigma$', r=2.4)
#     cat = D.bar_h(70, 96, 40)
#     upd = D.box(83, 27, 24, 8, 'MLP do nó', 'gnn')
#     out = D.box(83, 13, 14, 7, r'$h_j^{\ell+1}$')
#     D.arrow(e.b, gt.t)
#     D.arrow(hi.b, W.t)
#     D.arrow(gt.b, (gt.x, mul.y), mul.l)
#     D.arrow(W.b, mul.t)
#     D.arrow(mul.b, sm.t)
#     D.text(78.6, 57, 'mensagem', ha='left', fs=SQ_FONT - 1.5)
#     D.arrow(sm.b, (sm.x, cat.t[1]))
#     D.arrow(hj.b, (hj.x, cat.t[1]))
#     D.arrow(cat.b, upd.t)
#     D.arrow(upd.b, out.t)
#
#
# def _msg_col(D, cx, e_txt, h_txt, W_txt, top=90, label=None, ew=8, hw=8, dx=5, gw=9,
#              wW=8, gate_txt=None):
#     """Mensagem em coluna: e e h no topo; gate e W; ×; Σ. Devolve o Σ."""
#     e = D.box(cx - dx, top, ew, 6, e_txt, fs=SQ_FONT - 0.5)
#     h = D.box(cx + dx, top, hw, 6, h_txt, fs=SQ_FONT - 0.5)
#     gt = D.box(cx - dx, top - 13, gw, 8, gate_txt or _gate_txt(), 'gate', fs=SQ_FONT - 1.5)
#     W = D.box(cx + dx, top - 13, wW, 7, W_txt, 'gnn')
#     mul = D.op(cx + dx, top - 25, r'$\times$', r=2.2)
#     sm = D.op(cx + dx, top - 35, r'$\Sigma$', r=2.2)
#     D.arrow(e.b, gt.t)
#     D.arrow(h.b, W.t)
#     D.arrow(gt.b, (gt.x, mul.y), mul.l)
#     D.arrow(W.b, mul.t)
#     D.arrow(mul.b, sm.t)
#     if label:
#         D.text(cx, top + 5.6, label, fs=SQ_FONT - 1.5)
#     return sm
#
#
# def sq_bipartite_detail(D):
#     D.frame(FX, 2, 98, 98, r'Camada bipartida ($\times 3$)', bottom=True)
#     s1 = _msg_col(D, 61, r'$e^{\mathrm{dual}}$', r'$h_i$', r'$W_1$', label='elem–elem', ew=9)
#     s2 = _msg_col(D, 85, r'$d_{aj}$', r'$x^{\mathrm{vert}}_a$', r'$W_2$',
#                   label='vért' + r'$\to$' + 'elem', hw=9)
#     hj = D.box(54, 55, 9, 6, r'$h_j$')
#     cat = D.bar_h(50, 96, 44)
#     upd = D.box(73, 30, 26, 8, 'MLP do nó', 'gnn')
#     out = D.box(73, 15, 14, 7, r'$h_j^{\ell+1}$')
#     for n in (s1, s2, hj):
#         D.arrow(n.b, (n.x, cat.t[1]))
#     D.arrow(cat.b, upd.t)
#     D.arrow(upd.b, out.t)
#
#
# def sq_split(D):
#     xin = D.box(24, 93, 34, 7, r'$x_{hw}$ $(\mu_r, M, J)$')
#     fno = D.box(24, 82, 30, 7, 'FNO', 'fno')
#     pr = D.box(24, 69.5, 30, 10, 'Projeção\ngrade ' + r'$\to$' + ' grafo', 'proj')
#     ua = D.box(14, 56, 12, 6, r'$\tilde u^{\mathrm{aux}}$')
#     ub = D.box(34, 56, 12, 6, r'$\tilde u^{\mathrm{base}}$')
#     ca = D.op(14, 45.5, r'$\|$', r=2.4)
#     cb = D.op(34, 45.5, r'$\|$', r=2.4)
#     xs = D.box(24, 45.5, 9, 9, r'$x^{\mathrm{aux}}$' + '\n' + r'$x^{\mathrm{base}}$', fs=SQ_FONT - 0.5)
#     g = D.box(24, 32, 34, 9, 'SplitGNN (3 camadas)', 'gnn')
#     pa = D.op(14, 19, r'$+$', r=2.4)
#     pb = D.op(34, 19, r'$+$', r=2.4)
#     oa = D.box(14, 7, 13, 7, r'$\hat y^{\mathrm{aux}}$')
#     ob = D.box(34, 7, 13, 7, r'$\hat y^{\mathrm{base}}$')
#     D.arrow(xin.b, fno.t)
#     D.arrow(fno.b, pr.t)
#     D.text(25.5, 77, r'$\hat y_{hw}$', ha='left')
#     D.arrow(pr.at('b', -10), ua.t)
#     D.arrow(pr.at('b', 10), ub.t)
#     D.arrow(ua.b, ca.t)
#     D.arrow(ub.b, cb.t)
#     D.arrow(xs.l, ca.r)
#     D.arrow(xs.r, cb.l)
#     D.arrow(ca.b, (ca.x, g.t[1]))
#     D.arrow(cb.b, (cb.x, g.t[1]))
#     D.arrow((pa.x, g.b[1]), pa.t)
#     D.arrow((pb.x, g.b[1]), pb.t)
#     D.arrow(pa.b, oa.t)
#     D.arrow(pb.b, ob.t)
#     D.arrow(ua.l, (3, ua.y), (3, pa.y), pa.l)
#     D.arrow(ub.r, (45, ub.y), (45, pb.y), pb.r)
#     D.text(1.8, 38, 'resíduo', rotation=90, fs=SQ_FONT - 1.5)
#     D.text(46.4, 38, 'resíduo', rotation=90, fs=SQ_FONT - 1.5)
#     # detalhe: 3 mensagens + 2 atualizações
#     D.frame(FX, 2, 98, 99, r'Camada SplitGNN ($\times 3$)', bottom=True)
#     kw = dict(top=89, ew=7, hw=7, dx=4, gw=7.5, wW=7, gate_txt='gate')
#     s1 = _msg_col(D, 57, r'$e^{\mathrm{dual}}$', r'$h^{\mathrm{aux}}_j$', r'$W_1$',
#                   label='aux' + r'$\to$' + 'aux', **kw)
#     s2 = _msg_col(D, 73, r'$e^{\mathrm{aux}}$', r'$h^{\mathrm{base}}_c$', r'$W_2$',
#                   label='base' + r'$\to$' + 'aux', **kw)
#     s3 = _msg_col(D, 89, r'$e^{\mathrm{aux}}$', r'$h^{\mathrm{aux}}_i$', r'$W_3$',
#                   label='aux' + r'$\to$' + 'base', **kw)
#     ha = D.box(53, 54, 8, 6, r'$h^{\mathrm{aux}}_i$', fs=SQ_FONT - 0.5)
#     hb = D.box(85, 54, 8, 6, r'$h^{\mathrm{base}}_c$', fs=SQ_FONT - 0.5)
#     cat_a = D.bar_h(50, 81, 44)
#     cat_b = D.bar_h(83, 97, 44)
#     for n in (ha, s1, s2):
#         D.arrow(n.b, (n.x, cat_a.t[1]))
#     for n in (hb, s3):
#         D.arrow(n.b, (n.x, cat_b.t[1]))
#     ma = D.box(65.5, 30, 22, 8, r'MLP$^{\mathrm{aux}}$', 'gnn')
#     mb = D.box(90, 30, 13, 8, r'MLP$^{\mathrm{base}}$', 'gnn', fs=SQ_FONT - 0.5)
#     oa_ = D.box(65.5, 15, 14, 7, r'$h_i^{\mathrm{aux},\,\ell+1}$')
#     ob_ = D.box(90, 15, 13, 7, r'$h_c^{\mathrm{base},\,\ell+1}$', fs=SQ_FONT - 0.5)
#     D.arrow((ma.x, cat_a.b[1]), ma.t)
#     D.arrow((mb.x, cat_b.b[1]), mb.t)
#     D.arrow(ma.b, oa_.t)
#     D.arrow(mb.b, ob_.t)
#
#
# def sq_fno_gnn(D):
#     sq_left_single(D, 'Projeção\ngrade ' + r'$\to$' + ' nós', r'$x^{\mathrm{nó}}$',
#                    'GNN\n3 camadas', r'$\hat y^{\mathrm{nó}}$', 'arestas\n' + r'$e_{ij}$')
#     sq_edgeconv(D)
#
#
# def sq_fno_bipartite(D):
#     sq_left_single(D, 'Projeção\ngrade ' + r'$\to$' + ' centr.', r'$x^{\mathrm{elem}}$',
#                    'GNN bipartida\n3 camadas', r'$\hat y^{\mathrm{elem}}$',
#                    r'$e^{\mathrm{dual}}$, $d$' + '\n' + r'$x^{\mathrm{vert}}$')
#     sq_bipartite_detail(D)
#
#
# SQ = dict(fno2d=sq_fno2d, fno_gnn=sq_fno_gnn, fno_bipartite=sq_fno_bipartite,
#           fno_splitgnn=sq_split)


# [REMOVIDO] versão quadrada com notação u→v, gate/W_k/MLP^τ, V no Split e 'Projeção grade → grafo'
# -- substituída pela versão alinhada à notação do artigo (i→j, g_ij, W_r, MLP_u^τ, cópias C,
# 'Interpolação e reescala').
# # Notação padronizada (todas as figuras):
# #   domínios: grade e grafo; tipos de nó do grafo: V (vértice) e E (elemento);
# #   x^grade, ŷ^grade: entrada e saída na grade; x^V, x^E, ũ^V, ũ^E, ŷ^V, ŷ^E: entrada, FNO
# #   projetado e saída nos nós de cada tipo; e^{VV}, e^{EE}, e^{VE}, e^{EV}: arestas entre tipos;
# #   h_u (origem), h_v (destino), h' = estado atualizado; gate = sig(MLP(e)).
# GATE = 'gate'
# XG, YG = r'$x^{\mathrm{grade}}$', r'$\hat y^{\mathrm{grade}}$'
# PROJ = 'Projeção\ngrade ' + r'$\to$' + ' grafo'
#
#
# def sq_fno2d(D):
#     x = 50
#     xin = D.box(x, 89, 30, 9, XG)
#     P = D.box(x, 70, 40, 9, 'Lifting $P$', 'fno')
#     F = D.stack(x, 49.5, 40, 11, 'Camadas de Fourier', 'fno')
#     Q = D.box(x, 29, 40, 9, 'Projeção $Q$', 'fno')
#     yo = D.box(x, 10, 30, 9, YG)
#     for a, b in ((xin, P), (P, F), (F, Q), (Q, yo)):
#         D.arrow(a.b, b.t)
#
#
# def sq_left_single(D, t):
#     """Coluna esquerda das híbridas de 1 tipo de nó t (V ou E), de cima para baixo."""
#     cx = 21
#     xin = D.box(cx, 92, 22, 7, XG)
#     fno = D.box(cx, 80.5, 30, 7, 'FNO', 'fno')
#     pr = D.box(cx, 67, 30, 10, PROJ, 'proj')
#     u = D.box(cx, 54, 10, 6, r'$\tilde u^{%s}$' % t)
#     cat = D.op(cx, 44, r'$\|$', r=2.4)
#     xn = D.box(38, 44, 10, 6, r'$x^{%s}$' % t)
#     g = D.box(cx, 31, 24, 9, 'GNN', 'gnn')
#     plus = D.op(cx, 17.5, r'$+$', r=2.4)
#     yo = D.box(cx, 6, 12, 7, r'$\hat y^{%s}$' % t)
#     D.arrow(xin.b, fno.t)
#     D.arrow(fno.b, pr.t)
#     D.text(cx + 1.5, 75.5, YG, ha='left')
#     D.arrow(pr.b, u.t)
#     D.arrow(u.b, cat.t)
#     D.arrow(xn.l, cat.r)
#     D.arrow(cat.b, g.t)
#     D.arrow(g.b, plus.t)
#     D.text(cx + 1.5, 23, r'$\Delta^{%s}$' % t, ha='left')
#     D.arrow(plus.b, yo.t)
#     D.arrow(u.l, (4, u.y), (4, plus.y), plus.l)                     # resíduo
#     D.text(2.6, (u.y + plus.y) / 2, 'resíduo', rotation=90)
#     return g
#
#
# def _msg_col(D, cx, e_txt, h_txt, W_txt, top=90, label=None, ew=8, hw=8, dx=5, gw=8.5,
#              wW=8):
#     """Mensagem em coluna: e e h_u no topo; gate e W; ×; Σ. Devolve o Σ."""
#     e = D.box(cx - dx, top, ew, 6, e_txt, fs=SQ_FONT - 0.5)
#     h = D.box(cx + dx, top, hw, 6, h_txt, fs=SQ_FONT - 0.5)
#     gt = D.box(cx - dx, top - 13, gw, 7, GATE, 'gate', fs=SQ_FONT - 1)
#     W = D.box(cx + dx, top - 13, wW, 7, W_txt, 'gnn')
#     mul = D.op(cx + dx, top - 25, r'$\times$', r=2.2)
#     sm = D.op(cx + dx, top - 35, r'$\Sigma$', r=2.2)
#     D.arrow(e.b, gt.t)
#     D.arrow(h.b, W.t)
#     D.arrow(gt.b, (gt.x, mul.y), mul.l)
#     D.arrow(W.b, mul.t)
#     D.arrow(mul.b, sm.t)
#     if label:
#         D.text(cx, top + 5.6, label, fs=SQ_FONT - 1)
#     return sm
#
#
# def _arrow_lbl(a, b):
#     return r'$%s \to %s$' % (a, b)
#
#
# def sq_gnn_detail(D):
#     """FNO_GNN: grafo de vértices, 1 mensagem V→V."""
#     D.frame(FX, 2, 98, 98, 'Camada da GNN', bottom=True)
#     s1 = _msg_col(D, 66, r'$e^{VV}_{uv}$', r'$h^V_u$', r'$W$', label=_arrow_lbl('V', 'V'),
#                   ew=10, hw=9, dx=6)
#     hv = D.box(90, 90, 9, 6, r'$h^V_v$', fs=SQ_FONT - 0.5)
#     cat = D.bar_h(62, 96, 44)
#     upd = D.box(79, 30, 24, 8, r'MLP$^V$', 'gnn')
#     out = D.box(79, 15, 12, 7, r"$h^{V\prime}_v$")
#     D.arrow(s1.b, (s1.x, cat.t[1]))
#     D.arrow(hv.b, (hv.x, cat.t[1]))
#     D.arrow(cat.b, (upd.x, upd.t[1]))
#     D.arrow(upd.b, out.t)
#
#
# def sq_bipartite_detail(D):
#     """Bipartite: nós E atualizados; mensagens E→E e V→E (V só com x^V, sem estado)."""
#     D.frame(FX, 2, 98, 98, 'Camada da GNN bipartida', bottom=True)
#     s1 = _msg_col(D, 61, r'$e^{EE}_{uv}$', r'$h^E_u$', r'$W_1$', label=_arrow_lbl('E', 'E'),
#                   ew=9, hw=8)
#     s2 = _msg_col(D, 85, r'$e^{VE}_{uv}$', r'$x^V_u$', r'$W_2$', label=_arrow_lbl('V', 'E'),
#                   ew=9, hw=8)
#     hv = D.box(54, 55, 9, 6, r'$h^E_v$', fs=SQ_FONT - 0.5)
#     cat = D.bar_h(50, 96, 44)
#     upd = D.box(73, 30, 24, 8, r'MLP$^E$', 'gnn')
#     out = D.box(73, 15, 12, 7, r"$h^{E\prime}_v$")
#     for n in (s1, s2, hv):
#         D.arrow(n.b, (n.x, cat.t[1]))
#     D.arrow(cat.b, upd.t)
#     D.arrow(upd.b, out.t)
#
#
# def sq_split(D):
#     xin = D.box(24, 93, 22, 7, XG)
#     fno = D.box(24, 82, 30, 7, 'FNO', 'fno')
#     pr = D.box(24, 69.5, 30, 10, PROJ, 'proj')
#     ue = D.box(14, 56, 10, 6, r'$\tilde u^{E}$')
#     uv = D.box(34, 56, 10, 6, r'$\tilde u^{V}$')
#     ce = D.op(14, 45.5, r'$\|$', r=2.4)
#     cv = D.op(34, 45.5, r'$\|$', r=2.4)
#     xs = D.box(24, 45.5, 8, 9, r'$x^{E}$' + '\n' + r'$x^{V}$')
#     g = D.box(24, 32, 34, 9, 'GNN', 'gnn')
#     pe = D.op(14, 19, r'$+$', r=2.4)
#     pv = D.op(34, 19, r'$+$', r=2.4)
#     oe = D.box(14, 7, 11, 7, r'$\hat y^{E}$')
#     ov = D.box(34, 7, 11, 7, r'$\hat y^{V}$')
#     D.arrow(xin.b, fno.t)
#     D.arrow(fno.b, pr.t)
#     D.text(25.5, 77, YG, ha='left')
#     D.arrow(pr.at('b', -10), ue.t)
#     D.arrow(pr.at('b', 10), uv.t)
#     D.arrow(ue.b, ce.t)
#     D.arrow(uv.b, cv.t)
#     D.arrow(xs.l, ce.r)
#     D.arrow(xs.r, cv.l)
#     D.arrow(ce.b, (ce.x, g.t[1]))
#     D.arrow(cv.b, (cv.x, g.t[1]))
#     D.arrow((pe.x, g.b[1]), pe.t)
#     D.arrow((pv.x, g.b[1]), pv.t)
#     D.text(15.5, 24, r'$\Delta^{E}$', ha='left')
#     D.text(35.5, 24, r'$\Delta^{V}$', ha='left')
#     D.arrow(pe.b, oe.t)
#     D.arrow(pv.b, ov.t)
#     D.arrow(ue.l, (3, ue.y), (3, pe.y), pe.l)
#     D.arrow(uv.r, (45, uv.y), (45, pv.y), pv.r)
#     D.text(1.8, 38, 'resíduo', rotation=90, fs=SQ_FONT - 1.5)
#     D.text(46.4, 38, 'resíduo', rotation=90, fs=SQ_FONT - 1.5)
#     # detalhe: 3 mensagens + 2 atualizações
#     D.frame(FX, 2, 98, 99, 'Camada da GNN por material', bottom=True)
#     kw = dict(top=89, ew=7, hw=7, dx=4, gw=7.5, wW=7)
#     s1 = _msg_col(D, 57, r'$e^{EE}_{uv}$', r'$h^E_u$', r'$W_1$', label=_arrow_lbl('E', 'E'), **kw)
#     s2 = _msg_col(D, 73, r'$e^{VE}_{uv}$', r'$h^V_u$', r'$W_2$', label=_arrow_lbl('V', 'E'), **kw)
#     s3 = _msg_col(D, 89, r'$e^{EV}_{uv}$', r'$h^E_u$', r'$W_3$', label=_arrow_lbl('E', 'V'), **kw)
#     he = D.box(53, 54, 8, 6, r'$h^E_v$', fs=SQ_FONT - 0.5)
#     hv = D.box(85, 54, 8, 6, r'$h^V_v$', fs=SQ_FONT - 0.5)
#     cat_e = D.bar_h(50, 81, 44)
#     cat_v = D.bar_h(83, 97, 44)
#     for n in (he, s1, s2):
#         D.arrow(n.b, (n.x, cat_e.t[1]))
#     for n in (hv, s3):
#         D.arrow(n.b, (n.x, cat_v.t[1]))
#     me = D.box(65.5, 30, 22, 8, r'MLP$^E$', 'gnn')
#     mv = D.box(90, 30, 13, 8, r'MLP$^V$', 'gnn')
#     oe_ = D.box(65.5, 15, 12, 7, r"$h^{E\prime}_v$")
#     ov_ = D.box(90, 15, 12, 7, r"$h^{V\prime}_v$")
#     D.arrow((me.x, cat_e.b[1]), me.t)
#     D.arrow((mv.x, cat_v.b[1]), mv.t)
#     D.arrow(me.b, oe_.t)
#     D.arrow(mv.b, ov_.t)
#
#
# def sq_fno_gnn(D):
#     sq_left_single(D, 'V')
#     sq_gnn_detail(D)
#
#
# def sq_fno_bipartite(D):
#     g = sq_left_single(D, 'E')
#     xv = D.box(39, g.y, 10, 6, r'$x^{V}$')                         # nós V auxiliares (sem estado)
#     D.arrow(xv.l, g.r)
#     sq_bipartite_detail(D)
#
#
# SQ = dict(fno2d=sq_fno2d, fno_gnn=sq_fno_gnn, fno_bipartite=sq_fno_bipartite,
#           fno_splitgnn=sq_split)


# Notação do artigo (todas as figuras):
#   domínios: grade e grafo; tipos de nó: V (vértice), E (elemento), C (cópia de vértice por
#   material, só no Split); r = relação (tipo de origem, tipo de destino);
#   aresta i → j (i = origem, j = destino); g_ij = sig(MLP_{g,r}(e_ij)); m_ij = g_ij W_r h_i;
#   F_{j,r} = Σ_i m_ij; h_j^{(l+1)} = MLP_{u,τ(j)}([h_j ‖ F_{j,r1} ‖ ...]);
#   x^τ, ũ^τ, Δ^τ, ŷ^τ: entrada, FNO interpolado e reescalado, correção e saída nos nós do tipo τ.
GATE = r'$g_{ij}$'
XG, YG = r'$x^{\mathrm{grade}}$', r'$\hat y^{\mathrm{grade}}$'
PROJ = 'Interpolação e\nreescala\ngrade ' + r'$\to$' + ' grafo'


def _h_next(t):
    return r'$h^{%s,(l+1)}_j$' % t


def _mlp(t):
    return r'MLP$_u^{%s}$' % t


def sq_fno2d(D):
    x = 50
    xin = D.box(x, 89, 30, 9, XG)
    P = D.box(x, 70, 40, 9, 'Elevação $P$', 'fno')
    F = D.stack(x, 49.5, 40, 11, 'Camadas de Fourier', 'fno')
    Q = D.box(x, 29, 40, 9, 'Projeção $Q$', 'fno')
    yo = D.box(x, 10, 30, 9, YG)
    for a, b in ((xin, P), (P, F), (F, Q), (Q, yo)):
        D.arrow(a.b, b.t)


def sq_left_single(D, t):
    """Coluna esquerda das híbridas de 1 tipo de nó t (V ou E), de cima para baixo."""
    cx = 21
    xin = D.box(cx, 92, 22, 7, XG)
    fno = D.box(cx, 80.5, 30, 7, 'FNO', 'fno')
    pr = D.box(cx, 67, 30, 12, PROJ, 'proj')
    u = D.box(cx, 54, 10, 6, r'$\tilde u^{%s}$' % t)
    cat = D.op(cx, 44, r'$\|$', r=2.4)
    xn = D.box(38, 44, 10, 6, r'$x^{%s}$' % t)
    g = D.box(cx, 31, 24, 9, 'GNN', 'gnn')
    plus = D.op(cx, 17.5, r'$+$', r=2.4)
    yo = D.box(cx, 6, 12, 7, r'$\hat y^{%s}$' % t)
    D.arrow(xin.b, fno.t)
    D.arrow(fno.b, pr.t)
    D.text(cx + 1.5, 75.5, YG, ha='left')
    D.arrow(pr.b, u.t)
    D.arrow(u.b, cat.t)
    D.arrow(xn.l, cat.r)
    D.arrow(cat.b, g.t)
    D.arrow(g.b, plus.t)
    D.text(cx + 1.5, 23, r'$\Delta^{%s}$' % t, ha='left')
    D.arrow(plus.b, yo.t)
    D.arrow(u.l, (4, u.y), (4, plus.y), plus.l)                     # resíduo
    D.text(2.6, (u.y + plus.y) / 2, 'resíduo', rotation=90)
    return g


def _msg_col(D, cx, e_txt, h_txt, W_txt, top=90, label=None, F_txt=None, cat_y=44, ew=8, hw=8,
             dx=5, gw=8.5, wW=8):
    """Mensagem em coluna: e_ij e h_i no topo; g_ij e W_r; ×; Σ (→ F_{j,r}). Devolve o Σ."""
    e = D.box(cx - dx, top, ew, 6, e_txt, fs=SQ_FONT - 0.5)
    h = D.box(cx + dx, top, hw, 6, h_txt, fs=SQ_FONT - 0.5)
    gt = D.box(cx - dx, top - 13, gw, 7, GATE, 'gate')
    W = D.box(cx + dx, top - 13, wW, 7, W_txt, 'gnn')
    mul = D.op(cx + dx, top - 25, r'$\times$', r=2.2)
    sm = D.op(cx + dx, top - 35, r'$\Sigma$', r=2.2)
    D.arrow(e.b, gt.t)
    D.arrow(h.b, W.t)
    D.arrow(gt.b, (gt.x, mul.y), mul.l)
    D.arrow(W.b, mul.t)
    D.arrow(mul.b, sm.t)
    if label:
        D.text(cx, top + 5.6, label, fs=SQ_FONT - 1)
    if F_txt:
        D.text(sm.x - 1.0, (sm.y - 2.2 + cat_y + 1.3) / 2, F_txt, ha='right', fs=SQ_FONT - 1)
    return sm


def _arrow_lbl(a, b):
    return r'$%s \to %s$' % (a, b)


def _F(r):
    return r'$F_{j,%s}$' % r


def sq_gnn_detail(D):
    """FNO_GNN: grafo de vértices, 1 mensagem V→V (EdgeConvLayer)."""
    D.frame(FX, 2, 98, 98, 'Camada da GNN', bottom=True)
    s1 = _msg_col(D, 66, r'$e^{VV}_{ij}$', r'$h^V_i$', r'$W_{VV}$', label=_arrow_lbl('V', 'V'),
                  F_txt=_F('VV'), ew=10, hw=9, dx=6, wW=10)
    hv = D.box(90, 90, 9, 6, r'$h^V_j$', fs=SQ_FONT - 0.5)
    cat = D.bar_h(62, 96, 44)
    upd = D.box(79, 30, 24, 8, _mlp('V'), 'gnn')
    out = D.box(79, 15, 16, 7, _h_next('V'))
    D.arrow(s1.b, (s1.x, cat.t[1]))
    D.arrow(hv.b, (hv.x, cat.t[1]))
    D.arrow(cat.b, (upd.x, upd.t[1]))
    D.arrow(upd.b, out.t)


def sq_bipartite_detail(D):
    """Bipartite (Elem): nós E atualizados; E→E sobre h^E e V→E sobre x^V (BipartiteEdgeConvLayer)."""
    D.frame(FX, 2, 98, 98, 'Camada da GNN bipartida', bottom=True)
    s1 = _msg_col(D, 61, r'$e^{EE}_{ij}$', r'$h^E_i$', r'$W_{EE}$', label=_arrow_lbl('E', 'E'),
                  F_txt=_F('EE'), ew=9, hw=8, wW=9)
    s2 = _msg_col(D, 85, r'$e^{VE}_{ij}$', r'$x^V_i$', r'$W_{VE}$', label=_arrow_lbl('V', 'E'),
                  F_txt=_F('VE'), ew=9, hw=8, wW=9)
    hv = D.box(54, 55, 9, 6, r'$h^E_j$', fs=SQ_FONT - 0.5)
    cat = D.bar_h(50, 96, 44)
    upd = D.box(73, 30, 24, 8, _mlp('E'), 'gnn')
    out = D.box(73, 15, 16, 7, _h_next('E'))
    for n in (s1, s2, hv):
        D.arrow(n.b, (n.x, cat.t[1]))
    D.arrow(cat.b, upd.t)
    D.arrow(upd.b, out.t)


def sq_split(D):
    cE, cC = 11, 37                                                 # colunas dos ramos E e C
    xin = D.box(24, 93, 22, 7, XG)
    fno = D.box(24, 82, 30, 7, 'FNO', 'fno')
    pr = D.box(24, 69, 30, 12, PROJ, 'proj')
    ue = D.box(cE, 55, 10, 6, r'$\tilde u^{E}$')
    uc = D.box(cC, 55, 10, 6, r'$\tilde u^{C}$')
    ce = D.op(cE, 45.5, r'$\|$', r=2.4)
    cc = D.op(cC, 45.5, r'$\|$', r=2.4)
    xe = D.box(19, 45.5, 6, 6, r'$x^{E}$')
    xc = D.box(29, 45.5, 6, 6, r'$x^{C}$')
    g = D.box(24, 32, 34, 9, 'GNN', 'gnn')
    pe = D.op(cE, 19, r'$+$', r=2.4)
    pc = D.op(cC, 19, r'$+$', r=2.4)
    oe = D.box(cE, 7, 11, 7, r'$\hat y^{E}$')
    oc = D.box(cC, 7, 11, 7, r'$\hat y^{C}$')
    D.arrow(xin.b, fno.t)
    D.arrow(fno.b, pr.t)
    D.text(25.5, 77, YG, ha='left')
    D.arrow(pr.at('b', cE - 24), ue.t)
    D.arrow(pr.at('b', cC - 24), uc.t)
    D.arrow(ue.b, ce.t)
    D.arrow(uc.b, cc.t)
    D.arrow(xe.l, ce.r)
    D.arrow(xc.r, cc.l)
    D.arrow(ce.b, (ce.x, g.t[1]))
    D.arrow(cc.b, (cc.x, g.t[1]))
    D.arrow((pe.x, g.b[1]), pe.t)
    D.arrow((pc.x, g.b[1]), pc.t)
    D.text(cE + 1.5, 24, r'$\Delta^{E}$', ha='left')
    D.text(cC + 1.5, 24, r'$\Delta^{C}$', ha='left')
    D.arrow(pe.b, oe.t)
    D.arrow(pc.b, oc.t)
    D.arrow(ue.l, (3, ue.y), (3, pe.y), pe.l)
    D.arrow(uc.r, (45, uc.y), (45, pc.y), pc.r)
    D.text(1.8, 38, 'resíduo', rotation=90, fs=SQ_FONT - 1.5)
    D.text(46.4, 38, 'resíduo', rotation=90, fs=SQ_FONT - 1.5)
    # detalhe: 3 mensagens + 2 atualizações (SplitGNNLayer); C→E e E→C leem o mesmo
    # cross_edge_attr (índice invertido): e^{CE}_{ij} e e^{CE}_{ji}
    D.frame(FX, 2, 98, 99, 'Camada da GNN por material', bottom=True)
    kw = dict(top=89, ew=7, hw=7, dx=4, gw=7.5, wW=7.5)
    s1 = _msg_col(D, 57, r'$e^{EE}_{ij}$', r'$h^E_i$', r'$W_{EE}$', label=_arrow_lbl('E', 'E'),
                  F_txt=_F('EE'), **kw)
    s2 = _msg_col(D, 73, r'$e^{CE}_{ij}$', r'$h^C_i$', r'$W_{CE}$', label=_arrow_lbl('C', 'E'),
                  F_txt=_F('CE'), **kw)
    s3 = _msg_col(D, 89, r'$e^{CE}_{ji}$', r'$h^E_i$', r'$W_{EC}$', label=_arrow_lbl('E', 'C'),
                  F_txt=_F('EC'), **kw)
    he = D.box(53, 54, 8, 6, r'$h^E_j$', fs=SQ_FONT - 0.5)
    hc = D.box(85, 54, 8, 6, r'$h^C_j$', fs=SQ_FONT - 0.5)
    cat_e = D.bar_h(50, 81, 44)
    cat_c = D.bar_h(83, 97, 44)
    for n in (he, s1, s2):
        D.arrow(n.b, (n.x, cat_e.t[1]))
    for n in (hc, s3):
        D.arrow(n.b, (n.x, cat_c.t[1]))
    me = D.box(65.5, 30, 22, 8, _mlp('E'), 'gnn')
    mc = D.box(90, 30, 13, 8, _mlp('C'), 'gnn')
    oe_ = D.box(65.5, 15, 15, 7, _h_next('E'))
    oc_ = D.box(90, 15, 15, 7, _h_next('C'))
    D.arrow((me.x, cat_e.b[1]), me.t)
    D.arrow((mc.x, cat_c.b[1]), mc.t)
    D.arrow(me.b, oe_.t)
    D.arrow(mc.b, oc_.t)


def sq_fno_gnn(D):
    sq_left_single(D, 'V')
    sq_gnn_detail(D)


def sq_fno_bipartite(D):
    g = sq_left_single(D, 'E')
    xv = D.box(39, g.y, 10, 6, r'$x^{V}$')                         # nós V auxiliares (sem estado)
    D.arrow(xv.l, g.r)
    sq_bipartite_detail(D)


SQ = dict(fno2d=sq_fno2d, fno_gnn=sq_fno_gnn, fno_bipartite=sq_fno_bipartite,
          fno_splitgnn=sq_split)


def render_square(key):
    D = Diagram(100, width_in=SQ_IN, fs=SQ_FONT)
    SQ[key](D)
    return D.save(f'arq_{key}')


FIGS = {k: (lambda k=k: render_square(k)) for k in SQ}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', nargs='+', choices=list(SQ), default=list(SQ))
    ap.add_argument('--estilo', choices=('quadrado', 'largo'), default='quadrado')
    args = ap.parse_args()
    import matplotlib.pyplot as plt
    figs = FIGS if args.estilo == 'quadrado' else FIGS_LARGO
    with plt.rc_context():
        apply_style()
        apply_article_style(FONT_PT)
        for k in args.only:
            print('figura:', figs[k]())
