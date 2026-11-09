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


REGION_COLOR = ('#d62728', '#2ca02c', '#1f77b4', '#dddddd')   # quina, fronteira, interior, fora de Ω


def _ref_subtris(n_sub):
    """Subtriângulos (baricêntricas dos vértices [S,3,3]) do triângulo de referência, na
    ordem de metrics._ref_quadrature (pontos = 3 por subtriângulo, arestas 01, 12, 20)."""
    tris = [np.eye(3)]
    for _ in range(n_sub):
        new = []
        for v in tris:
            m01, m12, m20 = (v[0] + v[1]) / 2, (v[1] + v[2]) / 2, (v[2] + v[0]) / 2
            new += [np.stack(z) for z in ((v[0], m01, m20), (m01, v[1], m12),
                                          (m20, m12, v[2]), (m01, m12, m20))]
        tris = new
    return np.array(tris)


def fig_regions_map(xy, elems, code, iface, corners, d, title, zoom=None):
    """Verificação das regiões numa amostra: os subtriângulos da quadratura da malha
    (os que de fato são integrados, metrics.mesh_surface_quadrature) pintados pela região
    dos seus 3 pontos com prioridade quina > fronteira sem quinas > interior; fora de Ω em
    cinza claro. code [M*K] por ponto (0 quina, 1 fronteira sem quinas, 2 interior, -1 fora
    de Ω), na ordem dos pontos da malha. Γ em preto, 𝒱_c em pontos pretos; zoom =
    (x, y, meia largura) mm (2º painel, com a malha do FEMM)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import to_rgba
    M = len(elems)
    K = len(code) // M
    n_sub = int(round(np.log(K / 3) / np.log(4)))
    sb = _ref_subtris(n_sub)
    S = len(sb)
    c = np.asarray(code).reshape(M, S, 3).copy()
    c[c < 0] = 3
    c = c.min(axis=2).ravel()
    V = np.einsum('svj,mjd->msvd', sb, xy[elems]).reshape(-1, 3, 2)
    rgba = np.array([to_rgba(x) for x in REGION_COLOR])
    n = 2 if zoom else 1
    fig, axs = plt.subplots(1, n, figsize=(7.5 * n, 7.0), squeeze=False)
    for k, ax in enumerate(axs[0]):
        if k:
            x, y, h = zoom
            cen = V.mean(1)
            m = (np.abs(cen[:, 0] - x) < 1.5 * h) & (np.abs(cen[:, 1] - y) < 1.5 * h)
            ax.add_collection(PolyCollection(V[m], facecolors=rgba[c[m]], edgecolors='none'))
            P = xy[elems]
            pm = (np.abs(P[:, :, 0].mean(1) - x) < 1.5 * h) & (np.abs(P[:, :, 1].mean(1) - y) < 1.5 * h)
            ax.add_collection(PolyCollection(P[pm], facecolors='none', edgecolors='#8c8c8c',
                                             linewidths=0.3))
            ax.set(xlim=(x - h, x + h), ylim=(y - h, y + h))
        else:
            ax.add_collection(PolyCollection(V, facecolors=rgba[c], edgecolors='none',
                                             rasterized=True))
            ax.autoscale()
        ax.add_collection(LineCollection(xy[iface], colors='k', linewidths=1.0 if k else 0.2))
        ax.plot(xy[corners, 0], xy[corners, 1], 'o', color='k', ms=3 if k else 0.8)
        ax.set_aspect('equal')
        ax.axis('off')
    for col, lab in zip(REGION_COLOR, ('quina (Ω_q)', 'fronteira sem quinas (Ω_f ∖ Ω_q)', 'interior',
                                       'fora de Ω (anéis de ar)')):
        axs[0, 0].plot([], [], 's', color=col, label=lab)
    axs[0, 0].plot([], [], '-', color='k', label='Γ (interface entre materiais)')
    axs[0, 0].plot([], [], 'o', color='k', ms=3, label='V_c (vértices de canto)')
    axs[0, 0].legend(frameon=False, loc='lower left', title=f'd = {d:.4f} mm')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def fig_region_curves(curves, keys, d_marks, title):
    """Curvas acumuladas por distância: colunas = fronteira Ω_f(d) e quinas Ω_q(d); linhas =
    loss (keys = {loss: [(rótulo, arch, chave em curves)]}). Em cada painel, a fração da
    área (cinza tracejado; pontos da malha) e a fração do erro total η (uma curva por
    modelo); d da tabela marcados. Eixo d em escala log."""
    import matplotlib.pyplot as plt
    losses = list(keys)
    fig, axs = plt.subplots(len(losses), 2, figsize=(11, 3.6 * len(losses)),
                            squeeze=False, sharex=True, sharey=True)
    for i, loss in enumerate(losses):
        for j, (band, name) in enumerate((('f', 'fronteira Ω_f(d)'), ('q', 'quinas Ω_q(d)'))):
            ax = axs[i, j]
            area_done = False
            for lab, arch, ck in keys[loss]:
                c = curves[ck]
                d = c['d'][1:]
                if not area_done and arch != 'FNO2d':
                    ax.plot(d, 100 * c[f'{band}_area_frac'][1:], '--', color=INK2, lw=1.2,
                            label='área')
                    area_done = True
                ax.plot(d, 100 * c[f'{band}_eta'][1:], '-', color=MODEL_COLOR.get(arch, INK2),
                        lw=1.5, label=f'η {lab}')
            for dm in d_marks:
                ax.axvline(dm, color='#9a9994', lw=0.7, ls=':')
            ax.set_xscale('log')
            ax.set_title(f'{name} — loss {loss}')
            if i == len(losses) - 1:
                ax.set_xlabel('d [mm]')
            if j == 0:
                ax.set_ylabel('% da área / % do ∫e dA total')
    axs[0, 0].legend(frameon=False, loc='upper left')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
