"""
Primitivas de plot reutilizadas por todas as figuras (recebem `ax`, não chamam plt.show).

save_fig: origem tests/resultados/common.py (2026-10-13).
Estilo (cores, B_VMAX, rcParams): origem tests/resultados/common.py e figures_smooth.py
(2026-10-20); cor nova para a FNO_BipartiteGNN_Elem.
A migrar, se usados: src/neural_op/archs/eval.py (_imshow, _scatter, _trisurf,
_err_*), src/utils/visualization.py (plot_ang_grid, adaptada para receber ax).
"""
from pathlib import Path

# Estilo das figuras (mesmo das baterias anteriores)
MODEL_COLOR = {'FNO2d': '#2a78d6', 'FNO_GNN': '#eb6834', 'GNN_PostBase': '#1baf7a',
               'FNO_BipartiteGNN': '#eda100', 'FNO_BipartiteGNN_Elem': '#8b4fc8'}
GT_COLOR, INK, INK2, GRID = '#0b0b0b', '#0b0b0b', '#52514e', '#e4e3df'
B_VMAX = 2.5              # T -- mesma escala em todas as figuras de |B|


def apply_style():
    """Backend Agg + rcParams das figuras do pós-processamento."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
        'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100,
    })


def save_fig(fig, path: Path, pdf=True, dpi=300):
    """PNG (+ PDF) e fecha a figura."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    if pdf:
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
