"""
pos_bateria_figures.py
----------------------
Figuras das Tarefas 4b e 5 do pedido pós-bateria (depois de
scripts/pos_bateria_eval.py, que grava eval_per_sample.npz).

  - escolha das amostras: percentis 5/50/95 do L1_A por amostra da
    FNO_BipartiteGNN mae (amostra de valor mais próximo do percentil) +
    amostra de maior saturação (maior fração de área de ferro com |B|>1,6 T);
  - 4b: B_r(θ), B_θ(θ), erro ao longo de θ e espectro de B_r -- gabarito vs os
    4 modelos com loss mae;
  - 5a: |B| no ferro (contornos 1,6/1,8 T) e mu_r efetivo (log, ref. 5000);
  - 5b: histograma de mu_r efetivo ponderado por área + dispersão
    fração saturada x L1_A (Spearman).
PNG 300 dpi + PDF em pos_bateria/figuras/; números em pos_bateria/figures.json.

Execução:  python -m scripts.pos_bateria_figures
"""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm, ListedColormap
from scipy.stats import spearmanr

from scripts.pos_bateria_common import (
    OUT_DIR, FIG_DIR, IRON_ID, N_HARM, FUND, run_key, design_rows, test_samples,
    parse_worker, to_torch_sample, predict, load_runs, arc_values_from_nodes,
    arc_values_fno, to_polar, harmonics, arc_theta_deg,
)

# slots categóricos 1-4 (paleta de referência, modo claro); gabarito em tinta escura
MODEL_STYLE = {
    'FNO2d':            ('#2a78d6', 'FNO2d'),
    'FNO_GNN':          ('#eb6834', 'FNO_GNN'),
    'GNN_PostBase':     ('#1baf7a', 'GNN_PostBase'),
    'FNO_BipartiteGNN': ('#eda100', 'FNO_BipartiteGNN'),
}
GT_COLOR = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
OTHER_MAT = '#d9d9d6'
B_VMAX = 2.5            # T -- mesma escala em todas as figuras de |B|
MU_LIM = (3.0, 1e4)     # mu_r efetivo, escala log comum
# Purples invertido, truncado antes do branco (mu ~1000-2200 continua legível)
MU_CMAP = ListedColormap(plt.get_cmap('Purples_r')(np.linspace(0.0, 0.78, 256)))

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': '#0b0b0b', 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def nearest_to(values, q):
    target = np.percentile(values, q)
    return int(np.argmin(np.abs(values - target))), float(target)


# --------------------------------------------------------------------------- #
def fig_arc(tag, idx, s, preds, l1_bip):
    th = arc_theta_deg()
    gt = arc_values_from_nodes(s['bip']['node_y'].astype(np.float64), s['arc_idx'], s['arc_w'])
    br_t, bt_t = to_polar(gt)
    amp_t, _ = harmonics(br_t)

    fig, ax = plt.subplots(5, 1, figsize=(7.2, 11.0), gridspec_kw=dict(height_ratios=[1, 1, .8, .8, 1.1]))
    ax[0].plot(th, br_t, color=GT_COLOR, lw=2.2, label='gabarito (FEMM)', zorder=5)
    ax[1].plot(th, bt_t, color=GT_COLOR, lw=2.2, label='gabarito (FEMM)', zorder=5)
    k = np.arange(N_HARM + 1)
    ax[4].bar(k[1:], amp_t[1:], width=0.8, color='#bdbcb6', label='gabarito (FEMM)', zorder=2)
    offs = np.linspace(-0.3, 0.3, len(preds))
    for o, (arch, bxy) in zip(offs, preds.items()):
        c, lab = MODEL_STYLE[arch]
        br, bt = to_polar(bxy)
        ax[0].plot(th, br, color=c, lw=1.2, label=lab)
        ax[1].plot(th, bt, color=c, lw=1.2, label=lab)
        ax[2].plot(th, br - br_t, color=c, lw=1.0, label=lab)
        ax[3].plot(th, bt - bt_t, color=c, lw=1.0, label=lab)
        amp, _ = harmonics(br)
        ax[4].plot(k[1:] + o, amp[1:], 'o', ms=2.6, color=c, label=lab, zorder=3)
    ax[0].set_ylabel('$B_r$ (T)')
    ax[1].set_ylabel(r'$B_\theta$ (T)')
    ax[2].set_ylabel(r'$B_r^{pred}-B_r^{FEMM}$ (T)')
    ax[3].set_ylabel(r'$B_\theta^{pred}-B_\theta^{FEMM}$ (T)')
    lim = max(np.abs(ax[2].get_ylim()).max(), np.abs(ax[3].get_ylim()).max())
    for a in ax[:4]:
        a.set_xlim(0, 120)
        a.set_xticks(np.arange(0, 121, 15))
    for a in ax[2:4]:
        a.set_ylim(-lim, lim)
        a.axhline(0, color=INK2, lw=0.6)
    ax[3].set_xlabel(r'$\theta$ (°)')
    ax[4].set_yscale('log')
    ax[4].set_ylim(1e-4, 2)
    ax[4].set_xlim(0, N_HARM + 1)
    ax[4].set_xlabel('ordem harmônica (período base = setor de 120°; fundamental = 7)')
    ax[4].set_ylabel('amplitude de $B_r$ (T)')
    ax[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.42), frameon=False)
    fig.suptitle(f'Entreferro, raio médio r$_m$ = {s["r_m"]:.2f} mm (gap {s["gap"]:.2f} mm) — '
                 f'amostra {idx} ({tag}; L1$_A$ Bipartite mae = {l1_bip*1e3:.1f} mT) — modelos com loss mae',
                 y=1.03, fontsize=9.5)
    fig.tight_layout()
    save(fig, f'4b_entreferro_{tag}_amostra{idx}')


def fig_saturation(tag, idx, s):
    nodes, elems, ef = s['nodes'], s['elems'], s['elem']
    iron = ef['mat'] == IRON_ID
    x, y = nodes[:, 0], nodes[:, 1]

    # |B| nodal no ferro (média simples dos elementos de ferro incidentes) p/ contornos
    acc = np.zeros(len(nodes))
    cnt = np.zeros(len(nodes))
    for c in range(3):
        np.add.at(acc, elems[iron, c], ef['bmag'][iron])
        np.add.at(cnt, elems[iron, c], 1.0)
    bnode = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    tri_iron = mtri.Triangulation(x, y, elems)
    tri_iron.set_mask(~iron)
    tri_other = mtri.Triangulation(x, y, elems)
    tri_other.set_mask(iron)

    frac16 = float(ef['area'][iron & (ef['bmag'] > 1.6)].sum() / ef['area'][iron].sum())
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 9.4))
    for a in ax:
        a.tripcolor(tri_other, np.zeros(len(elems)), cmap=ListedColormap([OTHER_MAT]),
                    shading='flat', rasterized=True)
        a.set_aspect('equal')
        a.set_xlabel('x (mm)')
        a.set_ylabel('y (mm)')
        a.grid(False)

    pc = ax[0].tripcolor(tri_iron, facecolors=np.where(iron, ef['bmag'], 0.0), cmap='Blues',
                         vmin=0, vmax=B_VMAX, shading='flat', rasterized=True)
    cs = ax[0].tricontour(tri_iron, bnode, levels=[1.6, 1.8], colors=['#eb6834', '#e34948'],
                          linewidths=[0.8, 0.8])
    cb = fig.colorbar(pc, ax=ax[0], shrink=0.85)
    cb.set_label('|B| no ferro (T)')
    for lv, col in ((1.6, '#eb6834'), (1.8, '#e34948')):
        cb.ax.axhline(lv, color=col, lw=1.5)
    ax[0].set_title(f'|B| por elemento (curl(A) P1) — contornos 1,6 T e 1,8 T; '
                    f'fração do ferro >1,6 T = {100*frac16:.2f}%'.replace('.', ','))

    mu = np.where(iron, np.clip(ef['mu_eff'], *MU_LIM), MU_LIM[0])   # clip só p/ cor
    pc2 = ax[1].tripcolor(tri_iron, facecolors=mu, cmap=MU_CMAP,
                          norm=LogNorm(*MU_LIM), shading='flat', rasterized=True)
    cb2 = fig.colorbar(pc2, ax=ax[1], shrink=0.85)
    cb2.set_label(r'$\mu_r$ efetivo = |B| / ($\mu_0$ H(|B|)) (adim.)')
    cb2.ax.axhline(5000, color=GT_COLOR, lw=1.5)
    cb2.ax.text(1.6, 5000, ' 5000\n (constante\n do dataset)', transform=cb2.ax.get_yaxis_transform(),
                va='center', fontsize=7, color=GT_COLOR)
    ax[1].set_title(r'$\mu_r$ efetivo no ferro (curva BH iron_1008, 19 pontos)')
    fig.suptitle(f'Saturação do ferro — amostra {idx} ({tag}); demais materiais em cinza', y=0.995)
    fig.tight_layout()
    save(fig, f'5a_saturacao_{tag}_amostra{idx}')
    return frac16


# --------------------------------------------------------------------------- #
def main():
    d = np.load(OUT_DIR / 'eval_per_sample.npz')
    ids = d['sample_idx']
    l1_bip = d[f'{run_key("FNO_BipartiteGNN", "mae")}__L1']
    l1_fno = d[f'{run_key("FNO2d", "mae")}__L1']
    frac16 = d['frac_1.6']

    chosen = {}
    for q in (5, 50, 95):
        i, target = nearest_to(l1_bip, q)
        chosen[f'P{q}'] = dict(pos=i, sample_idx=int(ids[i]), L1_A_bip_mae=float(l1_bip[i]),
                               percentile_value=target)
    i = int(np.argmax(frac16))
    chosen['maxsat'] = dict(pos=i, sample_idx=int(ids[i]), frac_iron_gt_1_6T=float(frac16[i]),
                            L1_A_bip_mae=float(l1_bip[i]))
    print('amostras:', {k: v['sample_idx'] for k, v in chosen.items()})

    rows = design_rows()
    path_of = {idx: p for _, idx, p in test_samples()}
    runs = {k: r for k, r in load_runs().items() if r['loss'] == 'mae'}

    for tag, c in chosen.items():
        idx = c['sample_idx']
        s = parse_worker(path_of[idx], rows[idx], full=True)
        if tag != 'maxsat':
            ts = to_torch_sample(s)
            preds = {}
            for k, r in runs.items():
                out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
                if r['arch'] == 'FNO2d':
                    preds[r['arch']] = arc_values_fno(out_hw, s['r_m'], s['r_in'], s['r_ext'])
                else:
                    preds[r['arch']] = arc_values_from_nodes(yn.double().cpu().numpy(),
                                                             s['arc_idx'], s['arc_w'])
            fig_arc(tag, idx, s, preds, c['L1_A_bip_mae'])
        c['frac_iron_gt_1_6T'] = fig_saturation(tag, idx, s)
        c['r_m_mm'], c['gap_mm'] = s['r_m'], s['gap']
        print(f'  {tag}: amostra {idx} ok')

    # --- 5b: histograma de mu_r efetivo ponderado por área ---
    bins = d['mu_bins_log10']
    h = d['mu_hist']
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(10 ** bins[:-1], h / h.sum() * 100, width=np.diff(10 ** bins), align='edge',
           color='#2a78d6', linewidth=0)
    ax.set_xscale('log')
    ax.set_yscale('log')      # cauda saturada (mu baixo) tem área pequena
    ax.set_ylim(1e-5, 100)
    ax.axvline(5000, color=GT_COLOR, lw=1.2, ls='--')
    ax.text(5000, 30, ' 5000 (constante\n do dataset)', fontsize=7.5, va='top')
    ax.set_xlabel(r'$\mu_r$ efetivo (adim., escala log)')
    ax.set_ylabel('% da área de ferro do teste')
    ax.set_title(r'$\mu_r$ efetivo por elemento de ferro, ponderado por área — '
                 f'{len(ids)} amostras de teste')
    fig.tight_layout()
    save(fig, '5b_histograma_mu_efetivo')
    cdf = np.cumsum(h) / h.sum()
    mu_q = {f'p{q}': float(10 ** bins[1:][np.searchsorted(cdf, q / 100)]) for q in (5, 25, 50, 75, 95)}

    # --- 5b: dispersão fração saturada x L1_A ---
    sp = {}
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for a, (key, lab, col) in zip(ax, ((l1_bip, 'FNO_BipartiteGNN mae', MODEL_STYLE['FNO_BipartiteGNN'][0]),
                                        (l1_fno, 'FNO2d mae', MODEL_STYLE['FNO2d'][0]))):
        rho, p = spearmanr(frac16, key)
        sp[lab] = dict(spearman_rho=float(rho), p_value=float(p), n=int(len(key)))
        a.scatter(100 * frac16, 1e3 * key, s=6, color=col, alpha=0.45, linewidths=0)
        a.set_xscale('symlog', linthresh=1e-2)
        a.set_xlim(left=0)
        a.set_xlabel('% da área de ferro com |B| > 1,6 T (symlog)')
        a.set_title(f'{lab} — Spearman ρ = {rho:.3f}')
    ax[0].set_ylabel('L1$_A$ por amostra (mT)')
    fig.tight_layout()
    save(fig, '5b_dispersao_saturacao_L1A')

    out = dict(chosen_samples=chosen, spearman_frac16_vs_L1A=sp, mu_eff_area_quantiles=mu_q,
               mu_eff_area_frac_above_5000=float(h[10 ** bins[:-1] >= 5000].sum() / h.sum()),
               figures=sorted(p.name for p in FIG_DIR.glob('*.png')))
    (OUT_DIR / 'figures.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
