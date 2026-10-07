"""
pos_bateria_figures_dente.py
----------------------------
Variante da figura 4b (scripts/pos_bateria_figures.py::fig_arc) num arco no
MEIO DOS DENTES do estator, em vez do raio médio do entreferro.

Motor de rotor externo: os dentes/ranhuras são do estator (o rotor só tem ímãs
+ coroa). Raio do arco: r_d = D_est_ext/2 - (Hs0+Hs1+Hs2)/2 (meia altura da
ranhura, contada a partir da face do entreferro).

Ao longo dos 120° o arco alterna ferro (dente) e ranhura (cobre/ar) -- o fundo
de cada painel é pintado com a cor do material do elemento que contém o ponto
(transparência alta); campo e direção (setas do gabarito no referencial
polar: horizontal = +θ, vertical = +r) por cima.

Mesmas amostras (P5/P50/P95 do L1_A da FNO_BipartiteGNN mae, lidas de
pos_bateria/figures.json) e mesmos 4 modelos com loss mae.
PNG 300 dpi + PDF em pos_bateria/figuras/ (4c_dente_<tag>_amostra<idx>).

Execução:  python -m scripts.pos_bateria_figures_dente
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch

from scripts.pos_bateria_common import (
    OUT_DIR, design_rows, test_samples, parse_worker, to_torch_sample, predict, load_runs,
    arc_weights, arc_values_from_nodes, arc_values_fno, to_polar, arc_theta_deg,
)
from scripts.pos_bateria_figures import MODEL_STYLE, GT_COLOR, INK2, save
from src.data_gen.motor_constants import MATERIAL_ID

# material_id -> (rótulo, cor do fundo); alpha baixo pra não competir com as curvas
MAT_BG = {
    MATERIAL_ID['iron_1008']: ('ferro (dente)', '#6b6b6b'),
    MATERIAL_ID['vacuum']:    ('ar',            '#5fb4e8'),
    MATERIAL_ID['copper']:    ('cobre',         '#c8742c'),
    MATERIAL_ID['N35p']:      ('ímã',           '#9b59b6'),
}
BG_ALPHA = 0.16
N_ARROWS = 120            # setas de direção ao longo dos 120° (1 a cada 1°)


def tooth_radius_mm(row):
    so = float(row['stator_outer_diameter [mm]'])
    hs = sum(float(row[f'slot_Hs{k} [mm]']) for k in range(3))
    return so / 2.0 - hs / 2.0, hs


def arc_material(nodes_xy, elems, elem_mat, r):
    """material_id do elemento que contém cada ponto do arco (mesmo arco de arc_theta_deg)."""
    th = np.deg2rad(arc_theta_deg())
    tri = mtri.Triangulation(nodes_xy[:, 0], nodes_xy[:, 1], triangles=elems[:, :3])
    tf = tri.get_trifinder()
    tidx = tf(r * np.cos(th), r * np.sin(th))
    bad = tidx < 0
    if bad.any():   # pontos sobre o corte θ=0 (mesmo deslocamento de arc_weights)
        th2 = th[bad] + 1e-9
        tidx[bad] = tf(r * np.cos(th2), r * np.sin(th2))
    assert (tidx >= 0).all(), 'ponto do arco fora da malha'
    return elem_mat[tidx]


def paint_materials(ax, th, mat):
    """Faixas de fundo: um axvspan por trecho contínuo do mesmo material."""
    dth = th[1] - th[0]
    cut = np.flatnonzero(np.diff(mat)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(mat)]
    for a, b in zip(starts, ends):
        _, col = MAT_BG[int(mat[a])]
        ax.axvspan(th[a] - dth / 2, min(th[b - 1] + dth / 2, 120), color=col, alpha=BG_ALPHA,
                   lw=0, zorder=0)


def fig_tooth(tag, idx, s, r_d, hs, mat, gt, preds, l1_bip):
    th = arc_theta_deg()
    br_t, bt_t = to_polar(gt)
    bm_t = np.hypot(br_t, bt_t)

    fig, ax_all = plt.subplots(6, 1, figsize=(7.2, 12.6), sharex=True,
                               gridspec_kw=dict(height_ratios=[.55, 1, 1, 1, .8, .8]))
    for a in ax_all:
        paint_materials(a, th, mat)

    # --- direção de B (gabarito): faixa própria, comprimento da seta ∝ |B| ---
    step = max(1, len(th) // N_ARROWS)
    sl = slice(step // 2, None, step)
    ad = ax_all[0]
    ad.quiver(th[sl], np.zeros_like(th[sl]), bt_t[sl], br_t[sl], angles='uv', pivot='middle',
              scale_units='inches', scale=1.6 * bm_t.max(), width=0.0022, headwidth=3.5,
              headlength=4, headaxislength=3.6, color=GT_COLOR, zorder=6)
    ad.set_ylim(-1, 1)
    ad.set_yticks([])
    ad.grid(False)
    ad.set_ylabel('direção\nde B', rotation=0, ha='right', va='center')
    ad.set_title(r'direção de B do gabarito: $\rightarrow$ = +θ, $\uparrow$ = +r (para fora); '
                 'comprimento ∝ |B|', fontsize=7.5, color=INK2, loc='right')

    ax = ax_all[1:]
    ax[0].plot(th, bm_t, color=GT_COLOR, lw=2.2, label='gabarito (FEMM)', zorder=5)
    ax[1].plot(th, br_t, color=GT_COLOR, lw=2.2, zorder=5)
    ax[2].plot(th, bt_t, color=GT_COLOR, lw=2.2, zorder=5)
    for arch, bxy in preds.items():
        c, lab = MODEL_STYLE[arch]
        br, bt = to_polar(bxy)
        ax[0].plot(th, np.hypot(br, bt), color=c, lw=1.1, label=lab, zorder=4)
        ax[1].plot(th, br, color=c, lw=1.1, zorder=4)
        ax[2].plot(th, bt, color=c, lw=1.1, zorder=4)
        ax[3].plot(th, br - br_t, color=c, lw=0.9, zorder=4)
        ax[4].plot(th, bt - bt_t, color=c, lw=0.9, zorder=4)

    ax[0].set_ylabel('|B| (T)')
    ax[0].set_ylim(bottom=0, top=1.12 * max(bm_t.max(), ax[0].get_ylim()[1]))
    ax[1].set_ylabel('$B_r$ (T)')
    ax[2].set_ylabel(r'$B_\theta$ (T)')
    ax[3].set_ylabel(r'$B_r^{pred}-B_r^{FEMM}$ (T)')
    ax[4].set_ylabel(r'$B_\theta^{pred}-B_\theta^{FEMM}$ (T)')
    lim = max(np.abs(ax[3].get_ylim()).max(), np.abs(ax[4].get_ylim()).max())
    for a in ax[1:]:
        a.axhline(0, color=INK2, lw=0.6, zorder=1)
    for a in ax[3:]:
        a.set_ylim(-lim, lim)
    for a in ax_all:
        a.set_xlim(0, 120)
        a.set_xticks(np.arange(0, 121, 10))
        a.grid(axis='x', visible=False)
    ax[4].set_xlabel(r'$\theta$ (°)')

    lines, labels = ax[0].get_legend_handles_labels()
    present = [m for m in MAT_BG if m in set(mat.tolist())]
    patches = [Patch(facecolor=MAT_BG[m][1], alpha=BG_ALPHA * 2.2, label=MAT_BG[m][0]) for m in present]
    fig.legend(lines + patches, labels + [p.get_label() for p in patches], ncol=4,
               loc='upper center', bbox_to_anchor=(0.5, 0.975), frameon=False)

    fig.suptitle(f'Meio dos dentes do estator, r = {r_d:.2f} mm (altura do dente {hs:.2f} mm) — '
                 f'amostra {idx} ({tag}; L1$_A$ Bipartite mae = {l1_bip*1e3:.1f} mT) — '
                 'modelos com loss mae', y=0.995, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, f'4c_dente_{tag}_amostra{idx}')


def main():
    chosen = json.loads((OUT_DIR / 'figures.json').read_text())['chosen_samples']
    rows = design_rows()
    path_of = {idx: p for _, idx, p in test_samples()}
    runs = {k: r for k, r in load_runs().items() if r['loss'] == 'mae'}

    out = {}
    for tag in ('P5', 'P50', 'P95'):
        c = chosen[tag]
        idx = c['sample_idx']
        s = parse_worker(path_of[idx], rows[idx], full=True)
        r_d, hs = tooth_radius_mm(rows[idx])
        nodes, elems = s['nodes'], s['elems']
        a_idx, a_w, _ = arc_weights(nodes[:, :2], elems, r_d)
        mat = arc_material(nodes[:, :2], elems, s['elem']['mat'], r_d)
        gt = arc_values_from_nodes(s['bip']['node_y'].astype(np.float64), a_idx, a_w)

        ts = to_torch_sample(s)
        preds = {}
        for r in runs.values():
            out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
            if r['arch'] == 'FNO2d':
                preds[r['arch']] = arc_values_fno(out_hw, r_d, s['r_in'], s['r_ext'])
            else:
                preds[r['arch']] = arc_values_from_nodes(yn.double().cpu().numpy(), a_idx, a_w)
        fig_tooth(tag, idx, s, r_d, hs, mat, gt, preds, c['L1_A_bip_mae'])

        frac = {MAT_BG[m][0]: float((mat == m).mean()) for m in np.unique(mat)}
        mae = {a: float(np.abs(np.hypot(*to_polar(b)) - np.hypot(*to_polar(gt))).mean())
               for a, b in preds.items()}
        out[tag] = dict(sample_idx=idx, r_tooth_mm=r_d, slot_depth_mm=hs,
                        arc_material_frac=frac, mae_absB_T=mae)
        print(f'  {tag}: amostra {idx}  r={r_d:.2f} mm  materiais={frac}')

    (OUT_DIR / 'figures_dente.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
