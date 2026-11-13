"""Figuras do artigo: diagrama de blocos de cada arquitetura (FNO2d, FNO_GNN, FNO_Bipartite,
FNO_SplitGNN), sem detalhar hiperparâmetros (o texto explica), mostrando as conexões.

Gramática comum:
  - linha principal: entrada → FNO → projeção grade → grafo (bilinear + recodificação) →
    concatenação com as features do grafo → GNN → soma residual com o FNO projetado → saída;
  - moldura tracejada abaixo: a camada repetida aberta (Fourier no FNO2d; gate e mensagem
    nas GNNs: gate = sigmoide(MLP(e)), mensagem = gate × W h, Σ nos vizinhos, MLP do nó);
  - cores por tipo de bloco: dados (branco, canto reto), FNO (azul), projeção (verde),
    GNN (laranja), gate (rosa).

Saída: docs/figures/arq_{fno2d,fno_gnn,fno_bipartite,fno_splitgnn}.{pdf,png} (7,16 in, 8 pt).

    python -m scripts.fig_arquiteturas [--only fno2d fno_gnn fno_bipartite fno_splitgnn]
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
    def __init__(self, H):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.H = H
        self.fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * H / W_UNITS))
        ax = self.fig.add_axes([0, 0, 1, 1])
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
        self.ax.text(x, y, txt, ha='center', va='center', fontsize=fs or FONT_PT, zorder=4,
                     linespacing=1.15)
        return Node(x, y, w, h)

    def op(self, x, y, sym, r=1.6):
        from matplotlib.patches import Circle
        self.ax.add_patch(Circle((x, y), r, fc='white', ec=EDGE, lw=0.8, zorder=3))
        self.ax.text(x, y, sym, ha='center', va='center', fontsize=FONT_PT + 1, zorder=4)
        return Node(x, y, 2 * r, 2 * r)

    def bar(self, x, y0, y1, w=2.2):
        """Concatenação: barra vertical estreita (entradas pela esquerda/topo/base)."""
        from matplotlib.patches import Rectangle
        self.ax.add_patch(Rectangle((x - w / 2, y0), w, y1 - y0, fc='0.85', ec=EDGE, lw=0.8,
                                    zorder=3))
        self.ax.text(x, (y0 + y1) / 2, r'$\|$', ha='center', va='center', fontsize=FONT_PT,
                     zorder=4)
        return Node(x, (y0 + y1) / 2, w, y1 - y0)

    def arrow(self, *pts, ls='-', color=ARROW, lw=0.8, head=True):
        from matplotlib.patches import FancyArrowPatch
        P = np.asarray(pts, float)
        if len(P) > 2:
            self.ax.plot(P[:-1, 0], P[:-1, 1], color=color, lw=lw, ls=ls, zorder=2,
                         solid_capstyle='butt')
        self.ax.add_patch(FancyArrowPatch(P[-2], P[-1], arrowstyle='-|>' if head else '-',
                                          mutation_scale=7, lw=lw, color=color, ls=ls,
                                          shrinkA=0, shrinkB=0, zorder=2))

    def text(self, x, y, s, fs=None, **kw):
        kw.setdefault('ha', 'center')
        kw.setdefault('va', 'center')
        self.ax.text(x, y, s, fontsize=fs or FONT_PT - 1, zorder=5, **kw)

    def frame(self, x0, y0, x1, y1, label):
        from matplotlib.patches import FancyBboxPatch
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                         boxstyle='round,pad=0,rounding_size=1.5', fc='none',
                                         ec=FRAME, lw=0.8, ls=(0, (4, 2.5)), zorder=1))
        self.ax.text(x0 + 1.2, y1 - 1.2, label, ha='left', va='top', fontsize=FONT_PT - 0.5,
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


FIGS = dict(fno2d=fig_fno2d, fno_gnn=fig_fno_gnn, fno_bipartite=fig_fno_bipartite,
            fno_splitgnn=fig_fno_splitgnn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', nargs='+', choices=list(FIGS), default=list(FIGS))
    args = ap.parse_args()
    import matplotlib.pyplot as plt
    with plt.rc_context():
        apply_style()
        apply_article_style(FONT_PT)
        for k in args.only:
            print('figura:', FIGS[k]())
