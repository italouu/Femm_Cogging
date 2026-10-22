"""
Figuras de saturação do aço: curva B-H × pontos de operação e mapa de |B|
no aço com contornos no joelho.
"""
import numpy as np

from src.data_gen.motor_constants import PERMEABILITY
from src.postprocess.plots.primitives import INK, INK2

MU_LABEL = PERMEABILITY['iron_1008']      # μ_r do aço usado como rótulo de entrada (5000)


def fig_bh(bh, hist_edges, hist_w, p5, p95, knee, n_samples, n_above):
    """Figura A: H(B) da curva (pontos do .ans + interpolação do FEMM) e μ_eff(B) no eixo
    secundário; embaixo, histograma ponderado por área do |B| dos elementos de aço do
    conjunto avaliado, com a faixa p5–p95; μ_r = 5000 (rótulo de entrada) e joelho marcados."""
    import matplotlib.pyplot as plt
    b = np.linspace(1e-3, hist_edges[-1], 2000)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    ax.semilogy(b, bh.h(b), color=INK, lw=1.4, label='H(B) — interpolação do FEMM')
    ax.semilogy(bh.B_in[1:], bh.H_in[1:], 'o', color=INK, ms=3.5, label='pontos B-H do .ans')
    ax.set_ylabel('H [A/m]')
    ax2 = ax.twinx()
    ax2.plot(b, bh.mu_eff(b), color='#2a78d6', lw=1.4, label='μ_eff = B / (μ0·H)')
    ax2.axhline(MU_LABEL, color='#eb6834', ls='--', lw=1.2, label='μ_r = 5000 (rótulo de entrada)')
    ax2.set_ylabel('μ_eff', color='#2a78d6')
    ax2.spines['right'].set_visible(True)
    ax2.grid(False)
    for a in (ax, axh):
        a.axvspan(p5, p95, color='#e4e3df', alpha=0.6, lw=0, zorder=0)
        if knee.get('B_knee_T'):
            a.axvline(knee['B_knee_T'], color=INK2, ls=':', lw=1.2)
    if bh.B_in[-1] < hist_edges[-1]:
        ax.axvline(bh.B_in[-1], color=INK2, lw=0.6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, framealpha=0.9, loc='center right', fontsize=7.5)
    centers = 0.5 * (hist_edges[1:] + hist_edges[:-1])
    axh.bar(centers, 100 * hist_w / hist_w.sum(), width=np.diff(hist_edges), color='#2a78d6',
            edgecolor='none')
    axh.set(xlabel='|B| [T]', ylabel='% da área de aço')
    kt = (f"joelho (μ_eff = {100 * knee['frac']:.0f}% do máx.) = {knee['B_knee_T']:.3f} T"
          if knee.get('B_knee_T') else 'joelho não atingido')
    axh.set_title(f'Pontos de operação do aço: |B| suavizado no centróide, {n_samples} amostras, '
                  f'ponderado por área;\nfaixa cinza = p5–p95 ({p5:.2f}–{p95:.2f} T); {kt};\n'
                  f'{n_above} elementos acima do último ponto da curva', fontsize=7.5, loc='left')
    fig.suptitle('Curva B-H do aço (iron_1008) e pontos de operação no conjunto avaliado', y=0.93)
    return fig


def fig_steel_map(panels, knee_levels, vmax):
    """Figura B: |B| por elemento no aço (outros materiais em cinza) com contornos nos níveis
    do joelho. panels = [(título, xy [N,2], tri [M,3], steel [Ms], bmag [Ms])]. Contorno:
    |B| por elemento levado aos nós (média dos elementos de aço incidentes) -- só visual."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(7 * n, 6.6), squeeze=False)
    for ax, (title, xy, tri, steel, bmag) in zip(axs[0], panels):
        full = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=tri)
        ax.tripcolor(full, facecolors=np.zeros(len(tri)), cmap='Greys', vmin=0, vmax=8,
                     rasterized=True)
        st = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=tri[steel])
        m = ax.tripcolor(st, facecolors=bmag, cmap='magma', vmin=0, vmax=vmax, rasterized=True)
        num = np.bincount(tri[steel].ravel(), np.repeat(bmag, 3), minlength=len(xy))
        den = np.bincount(tri[steel].ravel(), minlength=len(xy))
        nodal = np.divide(num, den, out=np.zeros(len(xy)), where=den > 0)
        cs = ax.tricontour(st, nodal, levels=sorted(knee_levels), colors=['#00e5ff', '#7dff6b'][:len(knee_levels)],
                           linewidths=1.4)
        ax.clabel(cs, fmt='%.2f T', fontsize=7, colors='k')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=9)
    cb = fig.colorbar(m, ax=axs[0].tolist(), orientation='horizontal', fraction=0.04, pad=0.03)
    cb.set_label('|B| suavizado no centróide do elemento de aço [T]; linhas = níveis do joelho')
    return fig
