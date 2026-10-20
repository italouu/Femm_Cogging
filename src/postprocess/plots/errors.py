"""
Figuras de erro: barras por arch/loss, mapas de erro de superfície.

Origem: tests/resultados/figures_smooth.fig_bars e surface_error_smooth.plot
(2026-10-20). Mudanças: rótulos de subset genéricos ('teste'/'fora', 'fora' pode
estar vazio), N archs.
"""
import numpy as np

from src.postprocess.plots.primitives import INK2, MODEL_COLOR


# [REMOVIDO 2026-10-20] barras do erro nodal (pontual / L1_A / L2_A nos nós + piso da grade)
# -- substituídas pelo erro de superfície oficial.
# def fig_bars(res, run_order):
#     """Pontual / L1_A / L2_A nos nós (eval_smooth.json) + piso da grade; hachurado = mse."""
#     import matplotlib.pyplot as plt
#     metrics = (('MAE_pt_pct', 'Pontual (% B_ref)'), ('L1_area_pct', 'L1_A — superfície (% B_ref)'),
#                ('L2_area_pct', 'L2_A — superfície (% B_ref)'))
#     labels = [f"{a.replace('FNO_BipartiteGNN_', 'Bipartite_')}\n{l}" for a, l in run_order]
#     x = np.arange(len(run_order))
#     fig, axs = plt.subplots(3, 1, figsize=(1.2 * len(run_order) + 1, 9), sharex=True)
#     for ax, (mk, title) in zip(axs, metrics):
#         v = [res['runs'][f'{a}_{l}']['mesh']['total'][mk] for a, l in run_order]
#         cols = [MODEL_COLOR[a] for a, _ in run_order]
#         bars = ax.bar(x, v, 0.6, color=cols, zorder=3, edgecolor=cols)
#         for bb, (_, l) in zip(bars, run_order):
#             if l == 'mse':
#                 bb.set_facecolor('white')
#                 bb.set_hatch('////')
#         for bb, vv in zip(bars, v):
#             ax.text(bb.get_x() + bb.get_width() / 2, vv, f'{vv:.1f}', ha='center',
#                     va='bottom', fontsize=6.5, color=INK2)
#         ax.axhline(res['floor']['total'][mk], color=INK2, ls='-', lw=1,
#                    label='piso da grade (gabarito em H×W interpolado nos nós)', zorder=2)
#         ax.set_title(title, loc='left')
#         ax.grid(axis='x', visible=False)
#     axs[0].legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.12))
#     axs[-1].set_xticks(x)
#     axs[-1].set_xticklabels(labels, fontsize=7.5)
#     subs = ' + '.join(f'{n} {s}' for s, n in res['n_by_subset'].items() if n)
#     fig.suptitle(f"Erro de |B| nos nós da malha, gabarito smooth — {res['n_samples']} amostras "
#                  f"({subs}); hachurado = mse", y=1.0)
#     fig.tight_layout()
#     return fig


def fig_bars(res, run_order):
    """L1_A / L2_A do erro de superfície (eval_smooth.json) dos runs; traço = piso da
    representação de cada run (o gabarito na mesma superfície); hachurado = mse."""
    import matplotlib.pyplot as plt
    metrics = (('L1_area_pct', 'L1_A — erro de superfície (% B_ref)'),
               ('L2_area_pct', 'L2_A — erro de superfície (% B_ref)'))
    labels = [f"{a.replace('FNO_BipartiteGNN_', 'Bipartite_')}\n{l}" for a, l in run_order]
    x = np.arange(len(run_order))
    fig, axs = plt.subplots(2, 1, figsize=(1.2 * len(run_order) + 1, 6.5), sharex=True)
    for ax, (mk, title) in zip(axs, metrics):
        runs = [res['runs'][f'{a}_{l}'] for a, l in run_order]
        v = [r['surface']['total'][mk] for r in runs]
        cols = [MODEL_COLOR[a] for a, _ in run_order]
        bars = ax.bar(x, v, 0.6, color=cols, zorder=3, edgecolor=cols)
        for bb, (_, l) in zip(bars, run_order):
            if l == 'mse':
                bb.set_facecolor('white')
                bb.set_hatch('////')
        for bb, vv, r in zip(bars, v, runs):
            ax.text(bb.get_x() + bb.get_width() / 2, vv, f'{vv:.1f}', ha='center',
                    va='bottom', fontsize=6.5, color=INK2)
            fl = res['floor_surface'][r['surface_kind']]['total'][mk]
            ax.hlines(fl, bb.get_x() - 0.08, bb.get_x() + bb.get_width() + 0.08,
                      color=INK2, lw=1.6, zorder=4)
        ax.set_title(title, loc='left')
        ax.grid(axis='x', visible=False)
    axs[0].plot([], [], color=INK2, lw=1.6, label='piso: o gabarito na mesma representação '
                '(grade / P1 nos nós / P0 por elemento)')
    axs[0].legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.12))
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(labels, fontsize=7.5)
    subs = ' + '.join(f'{n} {s}' for s, n in res['n_by_subset'].items() if n)
    fig.suptitle(f"Erro de superfície de |B| (modelo × FEMM P1) — {res['n_samples']} amostras "
                 f"({subs}); hachurado = mse", y=1.0)
    fig.tight_layout()
    return fig


# [REMOVIDO 2026-10-20] mapa do erro nodal (gouraud por nó) + barras com o pontual --
# substituído pelo mapa por triângulo do erro de superfície oficial.
# def fig_surface_error(res, xy, elems, E, archs):
#     """Mapa do |erro| por nó de cada arch (escala comum, corte no p99) + barras."""
#     import matplotlib.pyplot as plt
#     import matplotlib.tri as mtri
#     n = len(archs)
#     tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=elems)
#     vmax = float(np.percentile(np.concatenate(list(E.values())), 99))
#     fig = plt.figure(figsize=(4.5 * n, 9.4))
#     gs = fig.add_gridspec(3, n, height_ratios=[1, 0.045, 0.95], hspace=0.55, wspace=0.06)
#     axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
#     cax = fig.add_subplot(gs[1, 1:n - 1])
#     bar_ax = fig.add_subplot(gs[2, 0:n])
#     for a, arch in zip(axes, archs):
#         r = res['erros'][arch]
#         m = a.tripcolor(tri, E[arch], shading='gouraud', cmap='Blues', vmin=0, vmax=vmax,
#                         rasterized=True)
#         a.set_aspect('equal')
#         a.axis('off')
#         a.set_title(f'{arch}\nL1 = {r["L1_A_T"]:.4f} T ({r["L1_A_pct"]:.2f}%)   '
#                     f'L2 = {r["L2_A_T"]:.4f} T ({r["L2_A_pct"]:.2f}%)'.replace('.', ','), fontsize=9.5)
#     cb = fig.colorbar(m, cax=cax, orientation='horizontal')
#     cb.set_label(f'| |B|pred − |B|gt | (T) — escala comum, corte no p99 = {vmax:.3f} T'.replace('.', ','))
#
#     x = np.arange(3)
#     w = 0.8 / n
#     for k, arch in enumerate(archs):
#         r = res['erros'][arch]
#         v = [r['L1_pt_pct'], r['L1_A_pct'], r['L2_A_pct']]
#         bars = bar_ax.bar(x + (k - (n - 1) / 2) * w, v, w * 0.92, color=MODEL_COLOR[arch], label=arch)
#         for b_, val in zip(bars, v):
#             bar_ax.text(b_.get_x() + b_.get_width() / 2, val, f'{val:.2f}'.replace('.', ','),
#                         ha='center', va='bottom', fontsize=8, color='#2c2c2a')
#     bar_ax.set_xticks(x, ['L1 ponto a ponto', 'L1 de superfície', 'L2 de superfície'])
#     b_ref = next(iter(res['erros'].values()))['B_ref_T']
#     bar_ax.set_ylabel(f'% de B_ref ({b_ref:.3f} T)'.replace('.', ','))
#     bar_ax.spines[['top', 'right']].set_visible(False)
#     bar_ax.grid(axis='y', color='#eeedea', lw=0.6)
#     bar_ax.set_axisbelow(True)
#     bar_ax.legend(fontsize=8, frameon=False, loc='upper left')
#     bar_ax.set_title('Erro de |B| nos nós (superfície: peso = área dual)', fontsize=9.5, pad=10)
#
#     fig.suptitle(f'Erro de |B| contra o gabarito smooth — amostra {res["sample_idx"]}, '
#                  f'loss {res["loss"]}, best.pth', fontsize=11)
#     return fig


def fig_surface_error(res, xy, elems, E, archs):
    """Mapa da densidade do erro de superfície por triângulo do FEMM (E[arch] [M], T;
    escala comum, corte no p99) + barras de L1_A / L2_A de superfície."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    n = len(archs)
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=elems)
    vmax = float(np.percentile(np.concatenate(list(E.values())), 99))
    fig = plt.figure(figsize=(4.5 * n, 9.4))
    gs = fig.add_gridspec(3, n, height_ratios=[1, 0.045, 0.95], hspace=0.55, wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax = fig.add_subplot(gs[1, 1:n - 1])
    bar_ax = fig.add_subplot(gs[2, 0:n])
    for a, arch in zip(axes, archs):
        r = res['erros'][arch]
        m = a.tripcolor(tri, facecolors=E[arch], cmap='Blues', vmin=0, vmax=vmax,
                        rasterized=True)
        a.set_aspect('equal')
        a.axis('off')
        a.set_title(f'{arch} [{r["surface_kind"]}]\nL1 = {r["L1_A_T"]:.4f} T ({r["L1_A_pct"]:.2f}%)   '
                    f'L2 = {r["L2_A_T"]:.4f} T ({r["L2_A_pct"]:.2f}%)'.replace('.', ','), fontsize=9.5)
    cb = fig.colorbar(m, cax=cax, orientation='horizontal')
    cb.set_label(f'erro de superfície médio no triângulo (T) — escala comum, corte no p99 = '
                 f'{vmax:.3f} T'.replace('.', ','))

    x = np.arange(2)
    w = 0.8 / n
    for k, arch in enumerate(archs):
        r = res['erros'][arch]
        v = [r['L1_A_pct'], r['L2_A_pct']]
        bars = bar_ax.bar(x + (k - (n - 1) / 2) * w, v, w * 0.92, color=MODEL_COLOR[arch], label=arch)
        for b_, val in zip(bars, v):
            bar_ax.text(b_.get_x() + b_.get_width() / 2, val, f'{val:.2f}'.replace('.', ','),
                        ha='center', va='bottom', fontsize=8, color='#2c2c2a')
    bar_ax.set_xticks(x, ['L1 de superfície', 'L2 de superfície'])
    b_ref = next(iter(res['erros'].values()))['B_ref_T']
    bar_ax.set_ylabel(f'% de B_ref ({b_ref:.3f} T)'.replace('.', ','))
    bar_ax.spines[['top', 'right']].set_visible(False)
    bar_ax.grid(axis='y', color='#eeedea', lw=0.6)
    bar_ax.set_axisbelow(True)
    bar_ax.legend(fontsize=8, frameon=False, loc='upper left')
    bar_ax.set_title('Erro de superfície de |B| (superfície do modelo × FEMM P1)', fontsize=9.5, pad=10)

    fig.suptitle(f'Erro de superfície de |B| — amostra {res["sample_idx"]}, '
                 f'loss {res["loss"]}, best.pth', fontsize=11)
    return fig
