"""
pos_bateria_smooth_figures.py
-----------------------------
Figuras da bateria smooth (depois de scripts/pos_bateria_smooth_eval.py, que
grava pos_bateria/eval_smooth.json + eval_smooth_per_sample.npz):

  1_curvas_treino   -- erro no teste (mae_graph; FNO2d: mae_hw) por época,
                       mse x mae, best.pth e ponto de parada marcados;
  2_barras_erro     -- pontual / L1_A / L2_A nos 51 chunks fora do treino e do
                       teste: smooth (vs gabarito smooth) x oficial (vs gabarito
                       unificado) x oficial vs gabarito smooth; piso da grade;
  3_campo_<tag>     -- |B| na malha (gabarito smooth, 4 modelos smooth mae) e
                       |erro| por nó, setor inteiro e zoom no entreferro, para
                       as amostras P50/P95 do L1_A da Bipartite mae (fora51).
PNG 300 dpi + PDF em <bateria smooth>/pos_bateria/figuras/.

Execução (raiz do projeto):  python -m scripts.pos_bateria_smooth_figures
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from scripts.pos_bateria_common import RUN_ORDER, run_key, design_rows, to_torch_sample, predict
from scripts.pos_bateria_smooth_eval import (
    ROOT_S, OUT_DIR, TMP_DIR, find_runs, load_set, samples_with_subset, parse_both,
)
from scripts.pos_bateria_common import gap_radius_mm
from src.data_gen.parsers.femm_mesh_unified import _read_ans_gz

FIG_DIR = OUT_DIR / 'figuras'
ARCHS = ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN')
MODEL_COLOR = {'FNO2d': '#2a78d6', 'FNO_GNN': '#eb6834', 'GNN_PostBase': '#1baf7a',
               'FNO_BipartiteGNN': '#eda100'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
B_VMAX = 2.5     # T -- mesma escala das figuras da bateria oficial

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_training():
    runs = find_runs(ROOT_S)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, arch in zip(axs, ARCHS):
        for loss, ls in (('mse', '--'), ('mae', '-')):
            d = runs[(arch, loss)]
            m = [json.loads(l) for l in (d / 'metrics.jsonl').read_text().splitlines() if l]
            summ = json.loads((d / 'run_summary.json').read_text())
            key = 'mae_hw' if arch == 'FNO2d' else 'mae_graph'
            ep = np.array([x['epoch'] for x in m])
            v = np.array([x[key] for x in m]) * 1e3
            ax.plot(ep, v, ls, color=MODEL_COLOR[arch], lw=1.6,
                    label=f"{loss} — best ép. {summ['best_epoch']}, parou ép. {summ['last_epoch']} "
                          f"({summ['stop_reason'].replace('early_stop_patience', 'paciência')})")
            b = summ['best_epoch']
            if b in ep:
                ax.plot(b, v[ep == b][0], 'o', ms=6, mfc='white', mec=MODEL_COLOR[arch], mew=1.6)
        ax.set_title(arch)
        ax.set_xlabel('época')
        ax.set_ylim(bottom=0, top=np.percentile(np.concatenate([l.get_ydata() for l in ax.get_lines()]), 97) * 1.15)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=7.2)
    axs[0].set_ylabel('MAE no teste (mT)\nFNO2d: grade; demais: nós')
    fig.suptitle('Bateria smooth — erro no teste (37 chunks) por época (pico inicial cortado); ○ = best.pth', y=1.02)
    fig.tight_layout()
    save(fig, '1_curvas_treino')


def fig_bars(res):
    sub = 'total88'   # todas as amostras avaliadas (teste37 + fora51)
    metrics = (('MAE_pt_pct', 'Pontual (% B_ref)'), ('L1_area_pct', 'L1_A — superfície (% B_ref)'),
               ('L2_area_pct', 'L2_A — superfície (% B_ref)'))
    series = (('S', 'smooth (vs gabarito smooth)', INK, None),
              ('U', 'oficial (vs gabarito unificado)', '#a8a7a2', None),
              ('UxS', 'oficial vs gabarito smooth', 'white', '////'))
    labels = [f'{a}\n{l}' for a, l in RUN_ORDER]
    x = np.arange(len(RUN_ORDER))
    w = 0.27
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (mk, title) in zip(axs, metrics):
        for i, (ek, lab, c, hatch) in enumerate(series):
            v = [res['evals'][ek][run_key(a, l)]['mesh'][sub][mk] for a, l in RUN_ORDER]
            bars = ax.bar(x + (i - 1) * w, v, w * 0.92, color=c, edgecolor=INK2 if hatch else c,
                          hatch=hatch, lw=0.6, label=lab, zorder=3)
            if ek != 'UxS':
                for bb, vv in zip(bars, v):
                    ax.text(bb.get_x() + bb.get_width() / 2, vv, f'{vv:.1f}', ha='center',
                            va='bottom', fontsize=6.5, color=INK2)
        for g, ls, lab in (('S', '-', 'piso grade (smooth)'), ('U', ':', 'piso grade (unificado)')):
            ax.axhline(res['floor'][g][sub][mk], color=INK2, ls=ls, lw=1, label=lab, zorder=2)
        ax.set_title(title, loc='left')
        ax.grid(axis='x', visible=False)
    axs[0].legend(ncol=5, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.12))
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(labels, fontsize=7.5)
    fig.suptitle(f"Erro de |B| nos nós da malha — {res['n_samples']} amostras "
                 f"({res['n_by_subset']['teste37']} do teste smooth + "
                 f"{res['n_by_subset']['fora51']} fora do treino/teste)", y=1.0)
    fig.tight_layout()
    save(fig, '2_barras_erro')


def fig_field(tag, idx, path, row, runs, l1):
    lay = parse_both(path, row)
    lines, nodes, elems = _read_ans_gz(path, TMP_DIR)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elems[:, :3])
    ts = to_torch_sample(lay['S'])
    gt = np.hypot(*lay['S']['v1']['node_y'].T.astype(np.float64))
    gt_u = np.hypot(*lay['U']['v1']['node_y'].T.astype(np.float64))
    preds = {}
    for r in runs.values():
        if r['loss'] != 'mae':
            continue
        _, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
        yn = yn.double().cpu().numpy()
        preds[r['arch']] = np.hypot(yn[:, 0], yn[:, 1])

    r_m, gap = gap_radius_mm(row)
    th0 = np.deg2rad(60.0)
    cx, cy, half = r_m * np.cos(th0), r_m * np.sin(th0), 4.5
    err_vmax = 0.5
    for zoom in (False, True):
        fig, axs = plt.subplots(2, 5, figsize=(17, 6.6) if not zoom else (17, 7.4))
        cols = [('gabarito smooth (FEMM)', gt)] + [(a, preds[a]) for a in ARCHS]
        for j, (name, v) in enumerate(cols):
            ax = axs[0, j]
            pc = ax.tripcolor(tri, v, shading='gouraud', cmap='Blues', vmin=0, vmax=B_VMAX,
                              rasterized=True)
            ax.set_title(f'|B| — {name}\n')
            if j == 0:
                ax2 = axs[1, 0]
                pe0 = ax2.tripcolor(tri, np.abs(gt_u - gt), shading='gouraud', cmap='Reds',
                                    vmin=0, vmax=err_vmax, rasterized=True)
                ax2.set_title('|gabarito unificado − smooth|\n(mudança do gabarito)')
            else:
                ax2 = axs[1, j]
                e = np.abs(v - gt)
                pe = ax2.tripcolor(tri, e, shading='gouraud', cmap='Reds', vmin=0,
                                   vmax=err_vmax, rasterized=True)
                ax2.set_title(f'|erro| — {name}\nmédia {e.mean() * 1e3:.0f} mT, '
                              f'p95 {np.percentile(e, 95) * 1e3:.0f} mT')
        for ax in axs.ravel():
            ax.set_aspect('equal')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if zoom:
                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)
        fig.colorbar(pc, ax=axs[0, :].tolist(), shrink=0.85, pad=0.01, label='|B| (T)')
        fig.colorbar(pe, ax=axs[1, :].tolist(), shrink=0.85, pad=0.01, label='|erro| (T)',
                     extend='max')
        where = (f'zoom no entreferro (r≈{r_m:.1f} mm, θ≈60°, janela {2 * half:.0f} mm)'
                 if zoom else 'setor de 120°')
        fig.suptitle(f'Amostra {idx} ({tag} do L1_A da Bipartite mae smooth, {l1 * 1e3:.1f} mT) — '
                     f'modelos smooth com loss mae — {where}', y=0.99)
        save(fig, f"3_campo_{tag}_amostra{idx}{'_zoom' if zoom else ''}")


def main():
    res = json.loads((OUT_DIR / 'eval_smooth.json').read_text())
    per = np.load(OUT_DIR / 'eval_smooth_per_sample.npz')
    fig_training()
    fig_bars(res)

    sub = per['subset'] == 'fora51'
    l1 = per['S__FNO_BipartiteGNN_mae'][:, 0]
    rows = design_rows()
    path_of = {idx: p for _, _, idx, p in samples_with_subset()}
    runs = load_set(ROOT_S)
    chosen = {}
    for tag, q in (('P50', 50), ('P95', 95)):
        cand = np.flatnonzero(sub)
        target = np.percentile(l1[cand], q)
        pos = cand[np.argmin(np.abs(l1[cand] - target))]
        idx = int(per['sample_idx'][pos])
        chosen[tag] = dict(sample_idx=idx, L1_A_bip_mae_smooth=float(l1[pos]))
        fig_field(tag, idx, path_of[idx], rows[idx], runs, float(l1[pos]))
        print(f'  {tag}: amostra {idx}')
    (OUT_DIR / 'figures_smooth.json').write_text(json.dumps(chosen, indent=2))


if __name__ == '__main__':
    main()
