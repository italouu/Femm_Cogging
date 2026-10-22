"""
Figuras de erro: barras por arch/loss, mapas de erro de superfície.

fig_bars: rótulo com o nº de amostras de cada subset avaliado ('test' ou 'val').
fig_surface_error_map: escala comum, contornos dos materiais, painel do piso.
"""
import numpy as np

from src.postprocess.plots.primitives import INK2, MODEL_COLOR


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


def fig_surface_error_map(res, xy, elems, E, archs, iface, floor):
    """Mapa por triângulo do erro de superfície: 1 painel por arch +
    1 painel com o piso da representação em grade (floor = (rótulo, E_piso [M], dict com
    L1_A/L2_A)); MESMA escala de cor em todos (corte no p99 de todos os painéis, piso
    incluído); contornos das fronteiras entre materiais (iface [K,2], regions.interface_edges)
    sobrepostos; barras de L1_A / L2_A embaixo."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    f_lab, f_E, f_r = floor
    panels = [(a, E[a], res['erros'][a]) for a in archs] + [(f_lab, f_E, f_r)]
    n = len(panels)
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=elems)
    vmax = float(np.percentile(np.concatenate([p[1] for p in panels]), 99))
    seg = xy[iface]
    fig = plt.figure(figsize=(4.2 * n, 9.4))
    gs = fig.add_gridspec(3, n, height_ratios=[1, 0.045, 0.95], hspace=0.55, wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax = fig.add_subplot(gs[1, 1:n - 1])
    bar_ax = fig.add_subplot(gs[2, 0:n])
    for a, (lab, e, r) in zip(axes, panels):
        m = a.tripcolor(tri, facecolors=e, cmap='Blues', vmin=0, vmax=vmax, rasterized=True)
        a.add_collection(LineCollection(seg, colors=INK2, linewidths=0.25, alpha=0.8))
        a.set_aspect('equal')
        a.axis('off')
        a.set_title(f'{lab} [{r["surface_kind"]}]\nL1 = {r["L1_A_T"]:.4f} T ({r["L1_A_pct"]:.2f}%)   '
                    f'L2 = {r["L2_A_T"]:.4f} T ({r["L2_A_pct"]:.2f}%)'.replace('.', ','), fontsize=9)
    cb = fig.colorbar(m, cax=cax, orientation='horizontal')
    cb.set_label(f'erro de superfície médio no triângulo (T) — mesma escala em todos os painéis, '
                 f'corte no p99 = {vmax:.3f} T; linhas = fronteiras entre materiais'.replace('.', ','))

    x = np.arange(2)
    w = 0.8 / n
    for k, (lab, _, r) in enumerate(panels):
        v = [r['L1_A_pct'], r['L2_A_pct']]
        col = MODEL_COLOR.get(lab, INK2)
        bars = bar_ax.bar(x + (k - (n - 1) / 2) * w, v, w * 0.92, color=col, label=lab,
                          hatch=None if lab in MODEL_COLOR else '////',
                          fill=lab in MODEL_COLOR, edgecolor=col)
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
    sel = res.get('selection')
    how = (f" — P{sel['pct']:g} do L1_A de superfície de {sel['ref_run']} em {sel['subset']}"
           if sel else '')
    fig.suptitle(f'Erro de superfície de |B| — amostra {res["sample_idx"]}{how}, '
                 f'loss {res["loss"]}, best.pth', fontsize=11)
    return fig


def fig_regions(xy, elems, region, iface, corners, names, title, zoom=None):
    """Verificação da estratificação espacial: triângulos coloridos pela região
    (names[código]), fronteiras entre materiais e vértices de canto; zoom = (x, y, meia
    largura) mm opcional (2º painel)."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    cols = ['#e4e3df', '#2a78d6', '#eb6834'][:len(names)]
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=elems)
    n = 2 if zoom else 1
    fig, axs = plt.subplots(1, n, figsize=(7.5 * n, 7.5), squeeze=False)
    for k, ax in enumerate(axs[0]):
        ax.tripcolor(tri, facecolors=region.astype(float), cmap=ListedColormap(cols),
                     vmin=-0.5, vmax=len(names) - 0.5, edgecolors='none', rasterized=True)
        ax.add_collection(LineCollection(xy[iface], colors=INK2, linewidths=0.4 if k else 0.2))
        ax.plot(xy[corners, 0], xy[corners, 1], '.', color='k', ms=3 if k else 1)
        ax.set_aspect('equal')
        ax.axis('off')
        if k:
            x, y, h = zoom
            ax.set(xlim=(x - h, x + h), ylim=(y - h, y + h))
    for c, nm in zip(cols, names):
        axs[0, 0].plot([], [], 's', color=c, label=nm)
    axs[0, 0].plot([], [], '.', color='k', label='vértice de canto')
    axs[0, 0].legend(frameon=False, loc='lower left')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
