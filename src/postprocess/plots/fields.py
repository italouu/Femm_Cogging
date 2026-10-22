"""
Campo previsto × FEMM (ex.: amostras P50/P95), em malha e em grade.

O cálculo fica no executável (scripts/postprocess.py); aqui só a figura, N archs.
"""
import numpy as np

from src.postprocess.plots.primitives import B_VMAX


def fig_field(tri, gt, preds, title, zoom=None, err_vmax=0.5):
    """|B| na malha (gabarito + cada modelo) e |erro| por nó.
    tri: matplotlib Triangulation dos nós; gt [S]; preds {nome: |B| [S]};
    zoom: None (setor inteiro) ou (cx, cy, half) em mm."""
    import matplotlib.pyplot as plt
    n = len(preds) + 1
    fig, axs = plt.subplots(2, n, figsize=(3.4 * n, 6.6 if zoom is None else 7.4))
    cols = [('gabarito smooth (FEMM)', gt)] + list(preds.items())
    for j, (name, v) in enumerate(cols):
        pc = axs[0, j].tripcolor(tri, v, shading='gouraud', cmap='Blues', vmin=0, vmax=B_VMAX,
                                 rasterized=True)
        axs[0, j].set_title(f'|B| — {name}\n')
        if j == 0:
            axs[1, 0].set_visible(False)
            continue
        e = np.abs(v - gt)
        pe = axs[1, j].tripcolor(tri, e, shading='gouraud', cmap='Reds', vmin=0,
                                 vmax=err_vmax, rasterized=True)
        axs[1, j].set_title(f'|erro| — {name}\nmédia {e.mean() * 1e3:.0f} mT, '
                            f'p95 {np.percentile(e, 95) * 1e3:.0f} mT')
    for ax in axs.ravel():
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if zoom is not None:
            cx, cy, half = zoom
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
    fig.colorbar(pc, ax=axs[0, :].tolist(), shrink=0.85, pad=0.01, label='|B| (T)')
    fig.colorbar(pe, ax=axs[1, :].tolist(), shrink=0.85, pad=0.01, label='|erro| (T)',
                 extend='max')
    fig.suptitle(title, y=0.99)
    return fig
