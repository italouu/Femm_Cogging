"""
Curvas de treino a partir de data/logs/.../run_XXXX/metrics.jsonl (MAE na validação,
o conjunto que escolhe o best.pth e a parada).
"""
import json

import numpy as np

from src.postprocess.plots.primitives import MODEL_COLOR


def fig_training(runs, archs):
    """runs: {(arch, loss): run_dir} (inference.find_runs). Erro na validação por época
    (mae_graph; FNO2d: mae_hw), mse × mae, best.pth (○) e parada na legenda."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(archs), figsize=(3.25 * len(archs), 3.6))
    for ax, arch in zip(axs, archs):
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
    axs[0].set_ylabel('MAE na validação (mT)\nFNO2d: grade; Elem: elementos; demais: nós')
    n_val = len(json.loads((next(iter(runs.values())) / 'split.json').read_text())['val'])
    fig.suptitle(f'Erro na validação ({n_val} chunks) por época (pico inicial cortado); ○ = best.pth',
                 y=1.02)
    fig.tight_layout()
    return fig
