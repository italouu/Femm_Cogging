"""
Figuras de torque no domínio da frequência: harmônicas de T(θe) (ordem n por 360° elétricos,
torque.theta_e_harmonics) da referência (smooth) × FNO_SplitGNN.
"""
import numpy as np

from src.postprocess.plots.primitives import GT_COLOR, INK2, MODEL_COLOR


C_SP = MODEL_COLOR['FNO_SplitGNN']
LOSS_STYLE = {'mae': dict(ls='-', mfc=C_SP), 'mse': dict(ls='--', mfc='white')}
SERIES_LABEL = {'ref': 'referência (B suavizado do FEMM)', 'mae': 'FNO_SplitGNN MAE',
                'mse': 'FNO_SplitGNN MSE'}


def _style(s):
    if s == 'ref':
        return dict(color=GT_COLOR, marker='s', mfc=GT_COLOR, ls='-')
    return dict(color=C_SP, marker='o', **LOSS_STYLE[s])


def spectrum_stems(ax, n, ex, title=None, floor_y=1e-3, ms=5):
    """Espectro de 1 amostra: |X_n| da referência e do Split por loss, hastes lado a lado
    (valores abaixo de floor_y desenhados em floor_y). ex = {série: |X_n| [n]}."""
    series = ['ref'] + [k for k in LOSS_STYLE if k in ex]
    off = dict(zip(series, np.linspace(-0.22, 0.22, len(series))))
    for s in series:
        st = _style(s)
        y = np.maximum(ex[s], floor_y)
        ax.vlines(n + off[s], floor_y, y, color=st['color'], ls=st['ls'], lw=1.3 if ms >= 5 else 1.0)
        ax.plot(n + off[s], y, st['marker'], color=st['color'], mfc=st['mfc'], ms=ms, ls='none',
                label=SERIES_LABEL[s])
    ax.set_yscale('log')
    ax.set_xticks(n)
    if title:
        ax.set_title(title, fontsize=8.5, loc='left')


def fig_torque_freq_sample(n, ex, title, floor_y=1e-3):
    """Figura de 1 amostra (o painel (a) de fig_torque_freq sozinho)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    spectrum_stems(ax, n, ex, title, floor_y)
    ax.set_ylim(floor_y * 0.8, None)
    ax.set_xlabel('ordem harmônica n (ciclos por 360° elétricos de θe)')
    ax.set_ylabel('amplitude [mN·m]')
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    fig.tight_layout()
    return fig


def fig_torque_freq_grid(n, exs, titles, ncols=5, floor_y=1e-3):
    """Grade de espectros de várias amostras (spectrum_stems), eixos y compartilhados."""
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(exs) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.9 * nrows), sharey=True, squeeze=False)
    for ax, ex, t in zip(axs.flat, exs, titles):
        spectrum_stems(ax, n, ex, t, floor_y, ms=3.5)
        ax.set_ylim(floor_y * 0.8, None)
        ax.title.set_fontsize(7.5)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[len(exs):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel('ordem n', fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel('amplitude [mN·m]', fontsize=8)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_torque_freq(n, ex, amp, err, ex_title, loss_style=None, floor_y=1e-3):
    """1×3 painéis, eixo y em log (mN·m), eixo x = ordem harmônica n.
      (a) uma amostra: |X_n| da referência e do Split por loss (hastes lado a lado);
      (b) 80 amostras: mediana de |X_n| (marcador) e IQR (barra), referência e Split;
      (c) 80 amostras: mediana e IQR de |X_n(Split) − X_n(ref)| (erro complexo por ordem),
          com a mediana de |X_n| da referência em cinza para escala.
    ex = {série: |X_n| [n]}; amp/err = {série: [S, n]} (série: 'ref' ou a loss)."""
    import matplotlib.pyplot as plt
    c_sp = MODEL_COLOR['FNO_SplitGNN']
    loss_style = loss_style or {'mae': dict(ls='-', mfc=c_sp), 'mse': dict(ls='--', mfc='white')}
    lab = {'ref': 'referência (B suavizado do FEMM)', 'mae': 'FNO_SplitGNN MAE', 'mse': 'FNO_SplitGNN MSE'}
    series = ['ref'] + [k for k in loss_style if k in amp]
    off = dict(zip(series, np.linspace(-0.22, 0.22, len(series))))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)

    def style(s):
        if s == 'ref':
            return dict(color=GT_COLOR, marker='s', mfc=GT_COLOR, ls='-')
        return dict(color=c_sp, marker='o', **loss_style[s])

    spectrum_stems(axs[0], n, ex, ex_title, floor_y)

    ax = axs[1]
    for s in series:
        st = style(s)
        q25, q50, q75 = np.percentile(amp[s], [25, 50, 75], axis=0)
        ax.errorbar(n + off[s], np.maximum(q50, floor_y), yerr=[q50 - np.maximum(q25, floor_y * 0.999), q75 - q50],
                    fmt=st['marker'], color=st['color'], mfc=st['mfc'], ms=5, capsize=2.5, lw=1.1)
    ax.set_title(f'mediana e IQR de |X_n| ({len(amp["ref"])} amostras)', fontsize=8.5, loc='left')

    ax = axs[2]
    ax.plot(n, np.median(amp['ref'], axis=0), '-', color='#b0b0ac', lw=2.2, zorder=0,
            label='mediana de |X_n| da referência')
    for s in series[1:]:
        st = style(s)
        q25, q50, q75 = np.percentile(err[s], [25, 50, 75], axis=0)
        ax.errorbar(n + off[s], q50, yerr=[q50 - q25, q75 - q50], fmt=st['marker'], color=st['color'],
                    mfc=st['mfc'], ms=5, capsize=2.5, lw=1.1)
        ax.plot(n + off[s], q50, ls=st['ls'], color=st['color'], lw=1.0)
    ax.set_title('mediana e IQR do erro |X_n(Split) − X_n(ref)|', fontsize=8.5, loc='left')

    for i, ax in enumerate(axs):
        ax.set_yscale('log')
        ax.set_xticks(n)
        ax.set_xlabel('ordem harmônica n (ciclos por 360° elétricos de θe)')
        ax.text(0.0, 1.16, f'({"abc"[i]})', transform=ax.transAxes, fontsize=9, color=INK2)
    axs[0].set_ylabel('amplitude [mN·m]')
    axs[0].set_ylim(floor_y * 0.8, None)
    h0, l0 = axs[0].get_legend_handles_labels()
    h2, l2 = axs[2].get_legend_handles_labels()
    fig.legend(h0 + h2, l0 + l2, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
               fontsize=8)
    fig.tight_layout()
    return fig
