"""
Primitivas de plot reutilizadas por todas as figuras (recebem `ax`, não chamam plt.show).

save_fig: origem tests/resultados/common.py (2026-10-13).
A migrar, se usados: src/neural_op/archs/eval.py (_imshow, _scatter, _trisurf,
_err_*), src/utils/visualization.py (plot_ang_grid, adaptada para receber ax).
"""
from pathlib import Path


def save_fig(fig, path: Path, pdf=True, dpi=300):
    """PNG (+ PDF) e fecha a figura."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    if pdf:
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
