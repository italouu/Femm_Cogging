"""
Figura do artigo (seção de saturação): permeabilidade do ferro na amostra mediana do teste.

    python -m scripts.fig_permeabilidade [--sample 2184]

Painel esquerdo: Ω_fe (iron_1008, material_id 0) colorido por s(B) do campo de referência;
o resto de Ω (ar, ímãs, cobre) em cinza; contorno das interfaces entre materiais.
Painel direito: curva B-H do iron_1008 como escala de cores (B vertical, H horizontal em log),
faixa de cores com |B| de um lado e μ_r do outro, B* tracejado e histograma de área de Ω_fe
da amostra em função de |B|.

Coordenada monótona ao longo da curva (μ_r(B) não é monótona):
    s(B) = μ_r(B)/μ_max − 1  (B ≤ B*),   s(B) = 1 − μ_r(B)/μ_max  (B > B*),   μ_max = μ_r(B*).

Fontes (nada de inferência; nenhuma métrica, Ω ou curva alterada):
  - curva B-H: motor_constants.FEMM_MATERIAL_DEFS['iron_1008']['bh'] (= a do .ans, conferida
    contra dados/saturation.json) com a interpolação do FEMM (saturation.FemmBH);
  - μ_r(B) = μ_eff secante = |B|/(μ0·H(|B|)) (saturation.FemmBH.mu_eff). O canal μ_r da entrada é
    constante (PERMEABILITY['iron_1008'] = 5000) e não depende de B;
  - |B| por elemento: b_tri_ref de dados/surface_error/sample_XXXXXX_mae.npz (|B| do FEMM
    suavizado, média por triângulo nos pontos de quadratura da malha -- o campo das métricas e
    da figura campo_referencia), com xy, elems, iface_edges e in_omega (Ω do regions);
  - material por elemento: μ_r cru do chunk da FNO_BipartiteGNN_Elem (node_x[:, 2]);
  - faixa de B: 0 até o máximo de |B| no ferro nos chunks de teste da Elem (centróide),
    arredondado para cima em 0,1 T.
Saída: docs/figures/permeabilidade_ferro.{pdf,png} + dados/fig_permeabilidade.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data_gen.motor_constants import FEMM_MATERIAL_DEFS, MATERIAL_ID, PERMEABILITY
from src.postprocess.plots.primitives import apply_article_style, apply_style
from src.postprocess.samples import samples_with_subset
from src.postprocess.saturation import FemmBH

ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / 'resultados' / 'BLDC_138x276' / 'dados'
FIG_DIR = ROOT / 'docs' / 'figures'
FIG_NAME = 'permeabilidade_ferro'
ELEM_RUN = ROOT / 'data' / 'logs' / 'BLDC_138x276' / 'FNO_BipartiteGNN_Elem' / 'run_0002'
CHUNK_ROOT = ROOT / 'data' / 'torch' / 'data_chunks'
FIG_WIDTH_IN = 7.16               # duas colunas IEEE
FONT_PT = 8
CMAP = 'coolwarm'
GRAY = '#8a8a8a'                  # resto de Ω (contraste com o centro claro do coolwarm)
HIST_GRAY = '#dcdcdc'
B_STEP_ROUND = 0.1                # T, arredondamento da faixa de B
HIST_BIN = 0.05                   # T
MU_TICKS_LO = (1100, 1500, 1900)                  # ramo linear (B < B*)
MU_TICKS_HI = (1900, 1500, 1000, 400, 100, 10)    # ramo saturado (B > B*)
MIN_TICK_GAP_T = 0.18


def _comma(v, nd=None):
    s = f'{v:g}' if nd is None else f'{v:.{nd}f}'
    return s.replace('.', ',')


def bh_curve():
    B, H = map(np.array, zip(*FEMM_MATERIAL_DEFS['iron_1008']['bh']))
    sat = json.loads((DADOS / 'saturation.json').read_text(encoding='utf-8'))['bh_curve']
    assert np.array_equal(B, sat['B_T']) and np.array_equal(H, sat['H_A_m']), 'curva B-H divergente'
    return FemmBH(B, H)


def chunk_iron_stats():
    """Máximo de |B| (centróide) no ferro em todo o teste + (μ_r, |B|) por elemento de cada
    amostra pedida depois. Lê os 19 chunks de teste da Elem."""
    cfg = json.loads((ELEM_RUN / 'config.json').read_text())
    by = {}
    for name, _, idx, _ in samples_with_subset(ELEM_RUN, 'test'):
        by.setdefault(name, []).append(idx)
    return cfg['dataset'], by


def load_chunk_sample(dataset, by, sample_idx, want_max=True):
    gmax, gidx, mu_s, n_s = 0.0, None, None, 0
    for name, ids in by.items():
        if not want_max and sample_idx not in ids:
            continue
        d = torch.load(CHUNK_ROOT / dataset / name, map_location='cpu', weights_only=False)
        off = np.concatenate([[0], d['L'].numpy().cumsum()])
        mu = d['node_x'][:, 2].numpy()
        b = torch.linalg.norm(d['node_y'], dim=1).numpy()
        for k, idx in enumerate(ids):
            s = slice(off[k], off[k + 1])
            fe = mu[s] == PERMEABILITY['iron_1008']
            m = float(b[s][fe].max())
            if m > gmax:
                gmax, gidx = m, idx
            if idx == sample_idx:
                mu_s = mu[s].copy()
        n_s += len(ids)
        del d
    return mu_s, gmax, gidx, n_s


PRED_RUN = 'FNO_SplitGNN_mae'     # campo previsto da figura (pedido 2026-11-14); 'ref' = FEMM


def predicted_b_tri(sample_idx, b_tri_ref):
    """|B| previsto pelo FNO_SplitGNN mae (N_base, P1 por material) por triângulo: média nos
    pontos de quadratura da malha (pesos iguais por triângulo), mesma superfície da métrica
    (postprocess._split_points). Predição salva em data/temp/inference (chunks); nada é inferido.
    Confere que |B|_FEMM nos mesmos pontos reproduz b_tri_ref."""
    import scripts.postprocess as pp
    from src.data_gen.parsers.ans_parsing import read_ans
    from src.data_gen.parsers.split_layout import parse_split_smooth
    from src.postprocess.chunk_inference import load_prediction
    from src.postprocess.inference import find_split_runs
    from src.postprocess.samples import parse_smooth
    run_dir = find_split_runs(ROOT / 'data' / 'logs' / 'BLDC_138x276')['mae']
    p = load_prediction(run_dir, sample_idx)
    assert p is not None, f'sem predição salva de {run_dir} para a amostra {sample_idx}'
    row = pp.design_rows()[sample_idx]
    path = pp.sample_path(sample_idx)
    ans = read_ans(path)
    lay = parse_smooth(path, row, with_elem=True, with_overlay=True, ans=ans)
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    lay['split'] = parse_split_smooth(path, r_in, r_ext, ang_1=pp.ANG_1, ang_2=pp.ANG_2,
                                      n_r=pp.N_R, n_a=pp.N_A, ans=ans)
    mm, mf, _, _ = pp._split_points(lay, p['y_base'].astype(np.float64))
    M, K = lay['mesh_quad']['mag'].shape
    b_pred, b_ref = mm.reshape(M, K).mean(1), mf.reshape(M, K).mean(1)
    chk = float(np.max(np.abs(b_ref - b_tri_ref)))
    assert chk < 1e-5, f'|B|_FEMM refeito difere do b_tri_ref salvo ({chk})'
    meta = dict(run=PRED_RUN, run_dir=str(run_dir), epoch=int(p['meta_epoch']),
                fonte='data/temp/inference (scripts/infer_chunks.py), y_base = N_base',
                superficie='P1 por material (postprocess._split_points), média por triângulo nos '
                           'pontos de quadratura da malha',
                max_abs_diff_ref_refeito_vs_salvo_T=chk)
    return b_pred, meta


def s_of(mu, b, mu_max, b_star):
    r = mu / mu_max
    return np.where(b <= b_star, r - 1.0, 1.0 - r)


# Escala de μ_r (log) do mínimo (em B = b_max) ao máximo (μ_max): azul -> amarelo -> laranja -> vermelho.
# Nós em fração do log; laranja perto de μ_r(0) ≈ 1065 (0,895 do log) e vermelho no pico.
MU_CMAP_NODES = ((0.00, '#313695'), (0.35, '#74add1'), (0.65, '#fee090'), (0.86, '#fdae61'),
                 (0.94, '#f46d43'), (1.00, '#a50026'))
MU_CB_TICKS = (3, 10, 30, 100, 300, 1000, 2171)


def mu_norm_cmap(bh, mu_max, b_max):
    """Range de permeabilidade [μ_r(b_max), μ_max] (μ_r(B) é mínima no fim da faixa: depois do
    pico só cai) e o mapa de cores em log μ_r."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    mu_min = float(bh.mu_eff(np.array([b_max]))[0])
    return (LogNorm(vmin=mu_min, vmax=mu_max),
            LinearSegmentedColormap.from_list('mu_r', list(MU_CMAP_NODES)))


def make_figure(sample_idx, bh, b_star, mu_max, b_max, z, fe, b_el, s_el, area):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    from matplotlib.transforms import blended_transform_factory

    xy, tri, om = z['xy'], z['elems'], z['in_omega']
    # [REMOVIDO] cor pela coordenada s(B) com TwoSlopeNorm (pedido 2026-11-14: a escala é o próprio
    # μ_r, do mínimo ao máximo; a curva é pintada por μ_r(B))
    # s0 = float(s_of(bh.mu_eff(np.array([1e-4])), np.array([1e-4]), mu_max, b_star)[0])
    # s1 = float(s_of(bh.mu_eff(np.array([b_max])), np.array([b_max]), mu_max, b_star)[0])
    # norm = TwoSlopeNorm(vcenter=0.0, vmin=s0, vmax=s1)
    # cmap = plt.get_cmap(CMAP)
    norm, cmap = mu_norm_cmap(bh, mu_max, b_max)

    # --- geometria da figura (polegadas) ---
    P = xy[tri[om]]
    x0, x1 = P[..., 0].min(), P[..., 0].max()
    y0, y1 = P[..., 1].min(), P[..., 1].max()
    ph = 2.55                                         # altura dos painéis
    pw = ph * (x1 - x0) / (y1 - y0)
    # [REMOVIDO] margens com eixos x/y no motor e μ_r entre a faixa e a curva (pedido 2026-11-14):
    # m_l, m_b, m_t = 0.42, 0.42, 0.20
    # gap_dom, w_strip, gap_mu, w_bh, m_r = 0.40, 0.11, 0.36, None, 0.08
    m_l, m_b, m_t = 0.02, 0.42, 0.20
    gap_dom, w_strip, gap_mu, w_bh, m_r = 0.45, 0.11, 0.04, None, 0.42
    w_bh = FIG_WIDTH_IN - m_l - pw - gap_dom - w_strip - gap_mu - m_r
    if w_bh < 1.4:                                    # domínio largo demais: encolhe a altura
        pw = FIG_WIDTH_IN - m_l - gap_dom - w_strip - gap_mu - m_r - 1.4
        ph = pw * (y1 - y0) / (x1 - x0)
        w_bh = 1.4
    H_in = m_b + ph + m_t
    fig = plt.figure(figsize=(FIG_WIDTH_IN, H_in))
    fx = lambda v: v / FIG_WIDTH_IN
    fy = lambda v: v / H_in
    # [REMOVIDO] motor à esquerda e curva à direita (pedido 2026-11-14: primeiro a curva, depois o motor)
    # ax = fig.add_axes([fx(m_l), fy(m_b), fx(pw), fy(ph)])
    # xs = m_l + pw + gap_dom
    # axs = fig.add_axes([fx(xs), fy(m_b), fx(w_strip), fy(ph)])
    # axb = fig.add_axes([fx(xs + w_strip + gap_mu), fy(m_b), fx(w_bh), fy(ph)])
    # mesma largura total: rótulos de |B| à esquerda (gap_dom), μ_r entre a curva e o motor (m_r)
    xs = gap_dom
    # [REMOVIDO] faixa de cores colada ao eixo de |B| (pedido 2026-11-14: cores só na curva B-H);
    # o eixo da curva ocupa a largura da faixa
    # axs = fig.add_axes([fx(xs), fy(m_b), fx(w_strip), fy(ph)])
    # axb = fig.add_axes([fx(xs + w_strip + gap_mu), fy(m_b), fx(w_bh), fy(ph)])
    # [REMOVIDO] curva ocupando toda a largura até a escala de μ_r (pedido 2026-11-14: coluna separada)
    # axb = fig.add_axes([fx(xs), fy(m_b), fx(w_strip + gap_mu + w_bh), fy(ph)])
    gap_c = 0.10                                      # vão curva -> coluna de μ_r
    w_bh_c = w_strip + gap_mu + w_bh - gap_c - w_strip
    axb = fig.add_axes([fx(xs), fy(m_b), fx(w_bh_c), fy(ph)])
    axc = fig.add_axes([fx(xs + w_bh_c + gap_c), fy(m_b), fx(w_strip), fy(ph)])
    ax = fig.add_axes([fx(xs + w_strip + gap_mu + w_bh + m_r + m_l), fy(m_b),
                       fx(pw), fy(ph)])

    # --- painel esquerdo ---
    other = om & ~fe
    T_o = mtri.Triangulation(xy[:, 0], xy[:, 1], tri[other])
    T_f = mtri.Triangulation(xy[:, 0], xy[:, 1], tri[fe])
    ax.tripcolor(T_o, facecolors=np.zeros(int(other.sum())), cmap=ListedColormap([GRAY]),
                 vmin=-1, vmax=1, edgecolors='face', linewidth=0.05)
    # [REMOVIDO] ferro pintado por s(B) (pedido 2026-11-14: pintar por μ_r)
    # ax.tripcolor(T_f, facecolors=s_el[fe], cmap=cmap, norm=norm, edgecolors='face', linewidth=0.05)
    ax.tripcolor(T_f, facecolors=bh.mu_eff(b_el), cmap=cmap, norm=norm, edgecolors='face',
                 linewidth=0.05)
    seg = xy[z['iface_edges']]
    ax.add_collection(LineCollection(seg, colors='k', linewidths=0.25))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.grid(False)
    # [REMOVIDO] eixos em mm no motor (pedido 2026-11-14: sem eixos x, y)
    # ax.set_xlabel('$x$ [mm]')
    # ax.set_ylabel('$y$ [mm]', labelpad=1)
    ax.set_axis_off()

    # [REMOVIDO] faixa de cores de |B| (pedido 2026-11-14: cores só na curva B-H)
    # # --- faixa de cores (B à esquerda, μ_r à direita) ---
    # bb = np.linspace(1e-4, b_max, 2000)
    # sb = s_of(bh.mu_eff(bb), bb, mu_max, b_star)
    # axs.imshow(sb[:, None], origin='lower', aspect='auto', extent=(0, 1, 0, b_max),
    #            cmap=cmap, norm=norm, interpolation='bilinear')
    # axs.set_xticks([])
    # axs.set_ylim(0, b_max)
    # axs.grid(False)
    # for sp in axs.spines.values():
    #     sp.set_visible(True)
    #     sp.set_color('k')
    #     sp.set_linewidth(0.5)
    # axs.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _comma(round(v, 2))))
    # axs.set_yticks(np.arange(0, b_max + 1e-9, 0.5))
    # axs.set_ylabel(r'$|\mathbf{B}|$ [T]', labelpad=2)
    # axs.axhline(b_star, color='k', lw=0.6, ls='--')
    # μ_r -> B em cada ramo (malha fina)
    bf = np.linspace(1e-4, b_max, 400001)
    mf = bh.mu_eff(bf)
    k = int(np.nanargmax(mf))
    ticks = []
    for mu_t in MU_TICKS_LO:
        if mf[0] < mu_t < mu_max:
            j = np.searchsorted(mf[:k + 1], mu_t)
            ticks.append((float(bf[j]), mu_t))
    for mu_t in MU_TICKS_HI:
        if mf[-1] < mu_t < mu_max:
            j = k + int(np.argmax(mf[k:] <= mu_t))
            ticks.append((float(bf[j]), mu_t))
    ticks.append((b_star, int(round(mu_max))))
    ticks.sort()
    kept = []
    for b, m in ticks:
        if m == int(round(mu_max)) or all(abs(b - b2) >= MIN_TICK_GAP_T for b2, _ in kept):
            kept.append((b, m))
    kept = [t for t in kept if t[1] == int(round(mu_max))
            or abs(t[0] - b_star) >= MIN_TICK_GAP_T]
    # [REMOVIDO] μ_r no lado direito da faixa (pedido 2026-11-14: escala de μ_r à direita da curva)
    # axm = axs.twinx()
    # axm.set_ylim(0, b_max)
    # axm.set_yticks([b for b, _ in kept])
    # axm.set_yticklabels([f'{m}' for _, m in kept])
    # axm.grid(False)
    # for sp in axm.spines.values():
    #     sp.set_visible(False)
    # axm.tick_params(axis='y', length=2.5, pad=1.5)
    # axm.set_title(r'$\mu_r$', fontsize=FONT_PT, pad=3, loc='right')

    # --- painel direito: curva B-H ---
    axb.set_ylim(0, b_max)
    axb.grid(False)
    # [REMOVIDO] ticks à esquerda da curva, sem rótulo (repetiam a faixa colada; pedido 2026-11-14)
    # axb.tick_params(axis='y', labelleft=False)
    # [REMOVIDO] eixo esquerdo escondido (a faixa fazia o papel dele; faixa removida em 2026-11-14)
    # axb.tick_params(axis='y', left=False, labelleft=False)
    # axb.spines['left'].set_visible(False)
    axb.set_yticks(np.arange(0, b_max + 1e-9, 0.5))
    axb.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _comma(round(v, 2))))
    axb.set_ylabel(r'$|\mathbf{B}|$ [T]', labelpad=2)
    # [REMOVIDO] histograma de área de Ω_fe da amostra atrás da curva (pedido 2026-11-14)
    # edges = np.arange(0, b_max + HIST_BIN / 2, HIST_BIN)
    # h, _ = np.histogram(np.clip(b_el, 0, b_max), bins=edges, weights=area)
    # h = h / h.sum()
    tr = blended_transform_factory(axb.transAxes, axb.transData)
    # xh = 0.92 * h / h.max()
    # axb.fill_betweenx(np.repeat(edges, 2)[1:-1], 0, np.repeat(xh, 2), transform=tr,
    #                   color=HIST_GRAY, lw=0, zorder=0, step=None)
    # axb.text(0.03, edges[int(np.argmax(h))] + HIST_BIN / 2, ' área de $\\Omega_{\\mathrm{fe}}$',
    #          transform=tr, color='#8c8c8c', fontsize=FONT_PT - 1, va='center', ha='left', zorder=1)
    # curva colorida por s
    bc = np.linspace(0.005, b_max, 1500)
    hc = bh.h(bc)
    pts = np.stack([hc, bc], 1)
    segs = np.stack([pts[:-1], pts[1:]], 1)
    # [REMOVIDO] curva pintada por s(B) (pedido 2026-11-14: pintar por μ_r(B))
    # sc = s_of(bh.mu_eff(0.5 * (bc[:-1] + bc[1:])), 0.5 * (bc[:-1] + bc[1:]), mu_max, b_star)
    sc = bh.mu_eff(0.5 * (bc[:-1] + bc[1:]))
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.0, zorder=3, capstyle='round')
    lc.set_array(sc)
    axb.add_collection(lc)
    Bm, Hm = bh.B_in[1:], bh.H_in[1:]
    # [REMOVIDO] pontos medidos da curva sobre a linha (pedido 2026-11-14)
    # axb.plot(Hm, Bm, 'o', ms=2.2, mfc='k', mec='none', zorder=4)
    axb.set_xscale('log')
    axb.set_xlim(hc[0] * 0.7, hc[-1] * 1.4)
    axb.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    axb.xaxis.set_minor_formatter(NullFormatter())
    axb.set_xlabel('$H$ [A/m]')
    axb.axhline(b_star, color='k', lw=0.6, ls='--', zorder=2)
    # [REMOVIDO] μ_max acima e B* abaixo da linha (μ_max agora também é tick da escala à direita)
    # axb.text(0.97, b_star + 0.04, rf'$\mu_{{\max}} = {mu_max:.0f}$',
    #          transform=tr, ha='right', va='bottom', fontsize=FONT_PT)
    # axb.text(0.97, b_star - 0.04, rf'$B^* = {_comma(b_star, 2).replace(",", "{,}")}$ T',
    #          transform=tr, ha='right', va='top', fontsize=FONT_PT)
    axb.text(0.97, b_star - 0.04,
             rf'$\mu_{{\max}} = {mu_max:.0f}$ em $B^* = {_comma(b_star, 2).replace(",", "{,}")}$ T',
             transform=tr, ha='right', va='top', fontsize=FONT_PT)
    # [REMOVIDO] μ_r no eixo direito da própria curva (pedido 2026-11-14: coluna colorida separada)
    # axm = axb.twinx()
    # axm.set_ylim(0, b_max)
    # axm.set_yticks([b for b, _ in kept])
    # axm.set_yticklabels([f'{m}' for _, m in kept])
    # axm.grid(False)
    # axm.spines['right'].set_visible(True)
    # axm.spines['left'].set_visible(False)
    # axm.tick_params(axis='y', length=2.5, pad=1.5)
    # axm.set_ylabel(r'$\mu_r$', rotation=0, labelpad=6, va='center')
    # x_hi = axb.get_xlim()[1]
    # for b, m in kept:
    #     if m == int(round(mu_max)):
    #         continue                                 # já marcado pela linha de B*
    #     axb.plot([float(bh.h(np.array([b]))[0]), x_hi], [b, b], color='#9a9a9a', lw=0.5,
    #              ls=(0, (3, 2)), zorder=1)
    # [REMOVIDO] coluna em B com ticks de μ_r repetidos nos dois ramos (pedido 2026-11-14: a coluna
    # é o range de μ_r, monótono, do mínimo ao máximo)
    # # coluna de μ_r à direita da curva (mesmo eixo vertical em B, cores = as da curva)
    # bb = np.linspace(1e-4, b_max, 2000)
    # sb = s_of(bh.mu_eff(bb), bb, mu_max, b_star)
    # axc.imshow(sb[:, None], origin='lower', aspect='auto', extent=(0, 1, 0, b_max),
    #            cmap=cmap, norm=norm, interpolation='bilinear')
    # axc.set_xticks([])
    # axc.set_ylim(0, b_max)
    # axc.grid(False)
    # for sp in axc.spines.values():
    #     sp.set_visible(True)
    #     sp.set_color('k')
    #     sp.set_linewidth(0.5)
    # axc.yaxis.tick_right()
    # axc.yaxis.set_label_position('right')
    # axc.set_yticks([b for b, _ in kept])
    # axc.set_yticklabels([f'{m}' for _, m in kept])
    # axc.tick_params(axis='y', length=2.5, pad=1.5)
    # axc.set_ylabel(r'$\mu_r$', rotation=0, labelpad=8, va='center')
    # axc.axhline(b_star, color='k', lw=0.6, ls='--')
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=axc)
    cb.outline.set_linewidth(0.5)
    cb.ax.grid(False)                                 # rcParams com grade (apply_style)
    cb.set_ticks([t for t in MU_CB_TICKS if norm.vmin <= t <= norm.vmax * (1 + 1e-9)])
    cb.set_ticklabels([f'{t}' for t in MU_CB_TICKS if norm.vmin <= t <= norm.vmax * (1 + 1e-9)])
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.tick_params(axis='y', which='major', length=2.5, pad=1.5, color='k')
    cb.ax.tick_params(axis='y', which='minor', length=1.2, color='k')
    cb.set_label(r'$\mu_r$', rotation=0, labelpad=8, va='center')

    # tracejado leve do ponto da curva até a coluna (atravessa o vão entre os eixos)
    # [REMOVIDO] tracejados curva -> coluna de μ_r (pedido 2026-11-14: μ_r não é monótona, as linhas
    # sugeriam uma correspondência que não existe entre os dois eixos)
    # x_hi = axb.get_xlim()[1]
    # gap_ax = gap_c / w_bh_c                           # vão em fração da largura da curva
    # for b, m in kept:
    #     if m == int(round(mu_max)):
    #         continue                                 # já marcado pela linha de B*
    #     axb.plot([float(bh.h(np.array([b]))[0]), x_hi], [b, b], color='#9a9a9a', lw=0.5,
    #              ls=(0, (3, 2)), zorder=1)
    #     axb.plot([1.0, 1.0 + gap_ax], [b, b], transform=tr, color='#9a9a9a', lw=0.5,
    #              ls=(0, (3, 2)), clip_on=False)
    # axb.plot([1.0, 1.0 + gap_ax], [b_star, b_star], transform=tr, color='k', lw=0.6, ls='--',
    #          clip_on=False)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sample', type=int, default=2184)
    ap.add_argument('--campo', choices=('pred', 'ref'), default='pred',
                    help='pred = FNO_SplitGNN mae (inferência salva); ref = FEMM')
    a = ap.parse_args()
    fig_name = FIG_NAME if a.campo == 'pred' else f'{FIG_NAME}_ref'
    json_name = 'fig_permeabilidade.json' if a.campo == 'pred' else 'fig_permeabilidade_ref.json'
    import matplotlib.pyplot as plt

    bh = bh_curve()
    bf = np.linspace(1e-4, bh.B[-1], 400001)
    mf = bh.mu_eff(bf)
    k = int(np.nanargmax(mf))
    b_star, mu_max = float(bf[k]), float(mf[k])

    dataset, by = chunk_iron_stats()
    mu_el, gmax, gidx, n_s = load_chunk_sample(dataset, by, a.sample)
    assert mu_el is not None, f'amostra {a.sample} fora do teste'
    b_max = round(float(np.ceil(round(gmax / B_STEP_ROUND, 9)) * B_STEP_ROUND), 6)

    z = np.load(DADOS / 'surface_error' / f'sample_{a.sample:06d}_mae.npz')
    xy, tri = z['xy'], z['elems']
    assert len(tri) == len(mu_el), 'elementos do chunk × npz'
    fe = (mu_el == PERMEABILITY['iron_1008']) & z['in_omega']
    P = xy[tri]
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    # [REMOVIDO] só o campo de referência (pedido 2026-11-14: usar a inferência da amostra)
    # b_el = z['b_tri_ref'][fe].astype(np.float64)
    # assert b_el.max() <= b_max
    b_ref_el = z['b_tri_ref'][fe].astype(np.float64)
    pred_meta = None
    if a.campo == 'pred':
        b_tri, pred_meta = predicted_b_tri(a.sample, z['b_tri_ref'])
        b_el = b_tri[fe].astype(np.float64)
    else:
        b_el = b_ref_el
    n_above_range = int((b_el > b_max).sum())
    mu_e = bh.mu_eff(b_el)
    s_el = np.zeros(len(tri))
    s_el[fe] = s_of(mu_e, b_el, mu_max, b_star)
    a_fe = area[fe]
    frac_above = float(a_fe[b_el > b_star].sum() / a_fe.sum())
    frac_above_ref = float(a_fe[b_ref_el > b_star].sum() / a_fe.sum())

    sel = json.loads((DADOS / 'surface_error' / f'sample_{a.sample:06d}_mae_P50.json')
                     .read_text(encoding='utf-8')).get('selection')
    apply_style()
    with plt.rc_context():
        apply_article_style(FONT_PT)
        fig = make_figure(a.sample, bh, b_star, mu_max, b_max, z, fe, b_el, s_el, a_fe)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f'{fig_name}.pdf')
        fig.savefig(FIG_DIR / f'{fig_name}.png', dpi=300)
        plt.close(fig)

    # histograma refeito só com o ferro para o JSON
    out = dict(
        sample_idx=a.sample, selection=sel,
        mu_r_definicao='secante: mu_r(B) = |B| / (mu0 * H(|B|)) (saturation.FemmBH.mu_eff); '
                       'o canal mu_r da entrada é constante (5000) e não depende de B',
        curva_bh=dict(material='iron_1008', material_id=MATERIAL_ID['iron_1008'],
                      B_T=bh.B_in.tolist(), H_A_m=bh.H_in.tolist(),
                      interpolacao='FEMM (saturation.FemmBH: Hermite cúbico, inclinações por spline '
                                   'natural; extrapolação linear acima do último ponto)',
                      n_repair=bh.n_repair),
        B_star_T=b_star, mu_max=mu_max, malha_B=dict(B_min=1e-4, B_max=float(bh.B[-1]), n=400001),
        s_def='s = mu_r/mu_max - 1 (B <= B*); s = 1 - mu_r/mu_max (B > B*)',
        faixa_B_T=[0.0, b_max],
        faixa_B_origem=dict(max_B_ferro_teste_T=gmax, amostra=gidx, n_amostras=n_s,
                            campo='|B| do FEMM suavizado no centróide (chunks de teste da Elem)',
                            arredondamento_T=B_STEP_ROUND),
        s_range=[float(s_of(bh.mu_eff(np.array([1e-4])), np.array([1e-4]), mu_max, b_star)[0]),
                 float(s_of(bh.mu_eff(np.array([b_max])), np.array([b_max]), mu_max, b_star)[0])],
        campo=a.campo, modelo=pred_meta,
        campo_elemento=('|B| previsto (ver modelo), média por triângulo nos pontos de quadratura da malha'
                        if a.campo == 'pred' else
                        'b_tri_ref (|B| do FEMM suavizado, média por triângulo nos pontos de quadratura '
                        'da malha; mesmo campo das métricas e de campo_referencia)'),
        n_elem_fe_acima_da_faixa=n_above_range,
        area_frac_fe_B_acima_B_star_ref=frac_above_ref,
        n_elem_fe=int(fe.sum()), area_fe_mm2=float(a_fe.sum()),
        area_frac_fe_B_acima_B_star=frac_above,
        B_fe_amostra=dict(min=float(b_el.min()), max=float(b_el.max()),
                          mediana_area=float(np.interp(0.5, np.cumsum(a_fe[np.argsort(b_el)]) / a_fe.sum(),
                                                       np.sort(b_el)))),
        figura=str(FIG_DIR / f'{fig_name}.pdf'),
        # [REMOVIDO] cmap=CMAP, norma='TwoSlopeNorm(vcenter=0)' (cor por s; trocada por μ_r em 2026-11-14)
        cor='μ_r(B) em LogNorm [mu_r_min, mu_max]; mapa azul -> amarelo -> laranja -> vermelho',
        mapa_nos=MU_CMAP_NODES,
        range_mu_r=[float(bh.mu_eff(np.array([b_max]))[0]), mu_max],
    )
    (DADOS / json_name).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                   encoding='utf-8')
    print(f'amostra {a.sample}; B* = {b_star:.4f} T; mu_max = {mu_max:.1f}; faixa B = [0, {b_max}] T '
          f'(max {gmax:.4f} T na #{gidx}); campo {a.campo}: área de Ω_fe com B > B*: {100 * frac_above:.1f}% '
          f'(ref {100 * frac_above_ref:.1f}%); max |B| {b_el.max():.3f} T; acima da faixa: {n_above_range}')
    print(f'-> {FIG_DIR / fig_name}.pdf/.png; {DADOS / json_name}')


if __name__ == '__main__':
    main()
