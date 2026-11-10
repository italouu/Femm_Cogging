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


def fig_sat_error(rep, bh, knee, title, archs, colors, labels=None):
    """L1 (% do B_ref em Ω) × |B| da referência no aço, por intervalos de mesma área
    (saturation.sat_report): um painel por conjunto (aço ∩ Ω_int, aço inteiro), uma linha
    por arch (MAE contínuo, MSE tracejado), pisos em cinza; μ_eff(|B|) da curva B-H no eixo
    secundário e o joelho marcado. rep = {chave: sat_report}; chaves '<arch>_<loss>' e
    'piso_<tipo>'."""
    import matplotlib.pyplot as plt
    labels = labels or {}
    sets = (('aco_int', 'aço ∩ Ω_int (longe das interfaces)'), ('aco', 'aço inteiro (em Ω)'))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    b = np.linspace(0.02, 2.6, 600)
    mu = bh.mu_eff(b)
    for ax, (k, ttl) in zip(axs, sets):
        for a in archs:
            for loss, ls in (('mae', '-'), ('mse', '--')):
                r = rep.get(f'{a}_{loss}', {}).get(k)
                if not r or not r['intervals']:
                    continue
                x = [i['B_median_T'] for i in r['intervals']]
                y = [i['L1_pct'] for i in r['intervals']]
                ax.plot(x, y, ls, color=colors[a], lw=1.5 if loss == 'mae' else 1.0, marker='o', ms=2.5,
                        label=f"{labels.get(a, a)} {loss.upper()}")
        for kd, r_ in rep.items():
            if kd.startswith('piso_') and r_.get(k) and r_[k]['intervals'] and kd in ('piso_P1_mat', 'piso_P0_elem'):
                ax.plot([i['B_median_T'] for i in r_[k]['intervals']], [i['L1_pct'] for i in r_[k]['intervals']],
                        ':', color='#8c8c8c', lw=1.0, label=kd.replace('_', ' '))
        if knee and knee.get('B_knee_T'):
            ax.axvline(knee['B_knee_T'], color=INK2, lw=0.9, ls='-.')
            ax.text(knee['B_knee_T'], ax.get_ylim()[1] * 0.97, f" joelho {knee['B_knee_T']:.2f} T",
                    fontsize=7.5, color=INK2, va='top')
        a2 = ax.twinx()
        a2.plot(b, mu, color='#b0b0b0', lw=2.0, alpha=0.6)
        a2.set_ylabel('μ_eff (curva B-H)', color='#8c8c8c')
        a2.tick_params(axis='y', colors='#8c8c8c')
        a2.grid(False)
        ax.set_zorder(a2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set(xlabel='|B| da referência suavizada no ponto [T]', title=ttl, xlim=(0, 2.6))
    axs[0].set_ylabel('L1 [% do B_ref em Ω]')
    axs[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def fig_sat_error_bh(bh, knee, series, colors, labels=None):
    """Erro × ponto de operação sobre a curva B-H. Eixo x: H [A/m] (log). Eixo esquerdo:
    B(H) do aço (interpolação do FEMM, pontos da curva marcados). Eixo direito: L1 (% do B_ref
    em Ω) por arch, cada intervalo de mesma área em H = H(|B| mediano). Terceiro eixo
    (deslocado à direita): μ_eff = B/(μ0·H) ao longo da curva, pontilhado. Verticais no H do
    pico de μ_eff e do joelho. Legenda única, sem título.
    series = {arch: (H [A/m], L1 [%])}; knee = FemmBH.knee()."""
    import matplotlib.pyplot as plt
    labels = labels or {}
    c_bh, c_mu = '#3d3d3a', '#7f7f7a'
    b = np.geomspace(bh.B_in[1] * 1e-2, bh.B_in[-1], 1500)
    h = bh.h(b)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(h, b, color=c_bh, lw=1.6, label='curva B-H (iron_1008, interpolação do FEMM)', zorder=3)
    ax.plot(bh.H_in[1:], bh.B_in[1:], 'o', color=c_bh, ms=3.5, label='pontos da curva B-H', zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('H [A/m]')
    ax.set_ylabel('B [T]', color=c_bh)
    ax.set_ylim(0, bh.B_in[-1] * 1.05)

    ax2 = ax.twinx()
    for a, (hx, l1) in series.items():
        ax2.plot(hx, l1, '-', color=colors[a], lw=1.5, marker='o', ms=2.5, label=f'L1 {labels.get(a, a)}')
    ax2.set_ylabel('L1 [% do B_ref em Ω]')
    ax2.set_ylim(0, None)
    ax2.spines['right'].set_visible(True)
    ax2.grid(False)

    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('axes', 1.12))
    ax3.spines['right'].set_visible(True)
    ax3.plot(h, bh.mu_eff(b), ':', color=c_mu, lw=1.6, label='μ_eff = B/(μ0·H) ao longo da curva')
    ax3.set_ylabel('μ_eff', color=c_mu)
    ax3.tick_params(axis='y', colors=c_mu)
    ax3.set_ylim(0, None)
    ax3.grid(False)

    vl = []
    for B_, H_, ls, nome in ((knee['B_mu_max_T'], float(bh.h(np.array([knee['B_mu_max_T']]))[0]), '--',
                              'pico de μ_eff'),
                             (knee.get('B_knee_T'), knee.get('H_knee_A_m'), '-.',
                              f"joelho (μ_eff = {100 * knee['frac']:.0f}% do máx.)")):
        if B_ is None:
            continue
        vl.append(ax.axvline(H_, color='#52514e', ls=ls, lw=1.0,
                             label=f'{nome}: B = {B_:.2f} T, H = {H_:.0f} A/m'))
        ax.text(H_ * 1.06, 0.03, f'{B_:.2f} T', color='#52514e', fontsize=7.5,
                transform=ax.get_xaxis_transform())

    ax.set_xlim(min(h[0], min(min(s[0]) for s in series.values()) / 1.3), bh.H_in[-1] * 1.3)
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)
    hs, ls_ = [], []
    for a_ in (ax, ax2, ax3):
        hh, ll = a_.get_legend_handles_labels()
        hs += hh
        ls_ += ll
    fig.legend(hs, ls_, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=7.5)
    fig.tight_layout()
    return fig
