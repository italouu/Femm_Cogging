"""
pos_bateria_smooth_bip_compare.py
---------------------------------
FNO_BipartiteGNN best (loss mae) da bateria oficial (gabarito unificado) x da
bateria smooth, cada uma medida contra OS DOIS gabaritos (matriz 2x2), com o
erro de superfície (L1_A/L2_A, peso = área dual do nó) e o pontual.

  - tabela: mesmas amostras de pos_bateria_smooth_eval (N_AMOSTRAS, metade de
    cada subset), salva em pos_bateria/bip_compare.json;
  - figura: amostra SAMPLE (setor inteiro + zoom), pos_bateria/figuras/4_bip_*.

Execução (raiz do projeto):
    python -m scripts.pos_bateria_smooth_bip_compare [N_AMOSTRAS] [SAMPLE]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from scripts.eval_surface_integral_table import Accum, _mag
from scripts.pos_bateria_common import design_rows, to_torch_sample, predict, gap_radius_mm
from scripts.pos_bateria_smooth_eval import (
    ROOT_S, ROOT_U, OUT_DIR, TMP_DIR, load_set, samples_with_subset, parse_both, node_err,
)
from scripts.pos_bateria_smooth_figures import save, B_VMAX
from src.data_gen.parsers.femm_mesh_unified import _read_ans_gz

_args = [a for a in sys.argv[1:] if not a.startswith('--')]
N_SAMPLES = int(_args[0]) if len(_args) > 0 else 64
SAMPLE = int(_args[1]) if len(_args) > 1 else 1296
KEY = 'FNO_BipartiteGNN_mae'
MODELS = {'U': 'Bipartite original\n(treino: gabarito unificado)',
          'S': 'Bipartite smooth\n(treino: gabarito smooth)'}
GTS = {'U': 'gabarito unificado', 'S': 'gabarito smooth (FEMM)'}


def predict_both(lay, models):
    out = {}
    for m, r in models.items():
        _, yn = predict(r['arch'], r['model'], r['normalizer'], to_torch_sample(lay[m]))
        out[m] = _mag(yn.float().cpu())
    return out


def table(models):
    rows = design_rows()
    samples = samples_with_subset()
    samples = ([x for x in samples if x[1] == 'teste37'][:N_SAMPLES // 2]
               + [x for x in samples if x[1] == 'fora51'][:N_SAMPLES - N_SAMPLES // 2])
    acc = {(m, g): Accum() for m in models for g in GTS}
    for _, _, idx, p in samples:
        lay = parse_both(p, rows[idx])
        area = lay['S']['v1']['node_x'][:, 2]
        pred = predict_both(lay, models)
        for g in GTS:
            t = np.hypot(*lay[g]['v1']['node_y'].T)
            for m in models:
                acc[(m, g)].add(pred[m], t, area)
    res = {f'modelo_{m}__vs_gabarito_{g}': a.report() for (m, g), a in acc.items()}
    res['n_samples'] = len(samples)
    (OUT_DIR / 'bip_compare.json').write_text(json.dumps(res, indent=2))
    return res


def figure(models):
    rows = design_rows()
    path = {idx: p for _, _, idx, p in samples_with_subset()}[SAMPLE]
    row = rows[SAMPLE]
    lay = parse_both(path, row)
    _, nodes, elems = _read_ans_gz(path, TMP_DIR)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elems[:, :3])
    area = lay['S']['v1']['node_x'][:, 2].astype(np.float64)
    gt = {g: np.hypot(*lay[g]['v1']['node_y'].T.astype(np.float64)) for g in GTS}
    pred = {m: v.astype(np.float64) for m, v in predict_both(lay, models).items()}

    r_m, _ = gap_radius_mm(row)
    th0 = np.deg2rad(60.0)
    cx, cy, half = r_m * np.cos(th0), r_m * np.sin(th0), 4.5
    for zoom in (False, True):
        fig, axs = plt.subplots(3, 3, figsize=(12.5, 12.5) if zoom else (13, 10.5))
        # linha 0: campos
        tops = [(GTS['S'], gt['S']), (MODELS['U'], pred['U']), (MODELS['S'], pred['S'])]
        for ax, (name, v) in zip(axs[0], tops):
            pc = ax.tripcolor(tri, v, shading='gouraud', cmap='Blues', vmin=0, vmax=B_VMAX,
                              rasterized=True)
            ax.set_title(f'|B| — {name}' + ('' if '\n' in name else '\n'), fontsize=8.5)
        # linhas 1-2: erro de cada modelo contra cada gabarito
        for i, g in enumerate(('S', 'U')):
            ax = axs[i + 1, 0]
            if g == 'S':
                ax.tripcolor(tri, np.abs(gt['U'] - gt['S']), shading='gouraud', cmap='Reds',
                             vmin=0, vmax=0.5, rasterized=True)
                ax.set_title('|gabarito unificado − smooth|', fontsize=8.5)
            else:
                ax.tripcolor(tri, gt['U'], shading='gouraud', cmap='Blues', vmin=0,
                             vmax=B_VMAX, rasterized=True)
                ax.set_title(f"|B| — {GTS['U']}", fontsize=8.5)
            for j, m in enumerate(('U', 'S')):
                e = np.abs(pred[m] - gt[g])
                l1 = (e * area).sum() / area.sum()
                l2 = np.sqrt((e ** 2 * area).sum() / area.sum())
                ax = axs[i + 1, j + 1]
                pe = ax.tripcolor(tri, e, shading='gouraud', cmap='Reds', vmin=0, vmax=0.5,
                                  rasterized=True)
                best = (l1 <= ((np.abs(pred['S' if m == 'U' else 'U'] - gt[g]) * area).sum()
                               / area.sum()))
                ax.set_title(f"|erro| {'original' if m == 'U' else 'smooth'} vs {GTS[g]}\n"
                             f"L1_A {l1 * 1e3:.1f} · L2_A {l2 * 1e3:.1f} · "
                             f"pontual {e.mean() * 1e3:.1f} mT{'  ★' if best else ''}",
                             fontsize=8.5)
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
        fig.colorbar(pe, ax=axs[1:, :].ravel().tolist(), shrink=0.6, pad=0.01,
                     label='|erro| (T)', extend='max')
        where = (f'zoom no entreferro (r≈{r_m:.1f} mm, θ≈60°, janela {2 * half:.0f} mm)'
                 if zoom else 'setor de 120°')
        fig.suptitle(f'FNO_BipartiteGNN best (mae): original × smooth — amostra {SAMPLE} — {where}\n'
                     f'linha 2: erro contra o gabarito smooth · linha 3: contra o gabarito '
                     f'unificado · ★ = menor L1_A contra aquele gabarito', fontsize=10)
        save(fig, f"4_bip_original_x_smooth_amostra{SAMPLE}{'_zoom' if zoom else ''}")


def main():
    models = {'U': load_set(ROOT_U)[KEY], 'S': load_set(ROOT_S)[KEY]}
    for m, r in models.items():
        print(f"  {m}: {r['run']} best ép.{r['epoch']}")
    figure(models)
    if '--so-figura' in sys.argv:
        return
    res = table(models)
    print(f"\n{res['n_samples']} amostras — erro de |B| nos nós (mT e % do B_ref do gabarito)")
    for g in GTS:
        for m in MODELS:
            x = res[f'modelo_{m}__vs_gabarito_{g}']
            print(f"  modelo {m} vs gabarito {g}: pontual {x['MAE_pt'] * 1e3:5.1f} ({x['MAE_pt_pct']:5.2f}%)"
                  f"  L1_A {x['L1_area'] * 1e3:5.1f} ({x['L1_area_pct']:5.2f}%)"
                  f"  L2_A {x['L2_area'] * 1e3:6.1f} ({x['L2_area_pct']:5.2f}%)")


if __name__ == '__main__':
    main()
