"""
Figura do artigo: formas de onda de B_r e B_θ no arco do meio do entreferro (o mesmo arco da
tab_gap_wave: r_m = (r_1 + r_2)/2, Δθ = 0,05°, setor 0°–120°) na amostra do artigo (#2184 =
P50 do L1_A de superfície do FNO2d_mae, a mesma de erro_dominio / erro_zoom / representacoes),
para as 4 archs com MAE, e |ΔB| = √(ΔB_r² + ΔB_θ²) em % de B_ref.

    python -m scripts.fig_forma_onda_entreferro [--sample 2184] [--h-factor 3]

Campos no arco: scripts/gap_wave.arc_fields (inalterado), guardados em
data/temp/gap_wave/arc_<idx>.npz na 1ª vez (sem inferência: predições de data/temp/inference).

Fundo de cada linha (eixo próprio atrás do dos dados, sem ticks): os elementos da malha da
amostra em (θ [°], r [mm]) na faixa r_m − h ≤ r ≤ r_m + h, mapeada na altura do eixo (estator
embaixo, rotor em cima), cheios só por material (triângulos unidos por material, sem arestas,
alpha baixo), e o arco r_m tracejado. h = h_factor · gap/2, limitado a não passar do back iron
(r_2 + espessura do ímã) nem abaixo da coroa do estator.

Saída: docs/figures/forma_onda_entreferro.{pdf,png} e
resultados/BLDC_138x276/dados/fig_forma_onda_entreferro.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT   = Path(__file__).resolve().parents[1]
DADOS  = ROOT / 'resultados' / 'BLDC_138x276' / 'dados'
FIGDIR = ROOT / 'docs' / 'figures'
NAME   = 'forma_onda_entreferro'
ARCHS  = ('FNO2d', 'FNO_GNN', 'FNO_BipartiteGNN_Elem', 'FNO_SplitGNN')
LOSS   = 'mae'
FIG_WIDTH_IN, FONT_PT = 7.16, 8
BG_ALPHA = 0.13
# [REMOVIDO] ordem invertida a pedido (mais densos no fundo, logo acima da referência):
# ARCH_LS = {'FNO2d': (0, (6, 2)), 'FNO_GNN': (0, (1, 1.2)), 'FNO_BipartiteGNN_Elem': (0, (5, 1.5, 1, 1.5)),
#            'FNO_SplitGNN': '-'}     # padrão de traço por arch (além da cor)
ARCH_LS = {'FNO2d': '-', 'FNO_GNN': (0, (5, 1.5, 1, 1.5)), 'FNO_BipartiteGNN_Elem': (0, (1, 1.2)),
           'FNO_SplitGNN': (0, (6, 2))}     # padrão de traço por arch (além da cor)


def _ink(ls):
    """Fração de tinta do padrão (1 = cheio): ordem de desenho, mais denso no fundo."""
    return 1.0 if ls == '-' else sum(ls[1][::2]) / sum(ls[1])


# zorder 4, 5, ... do mais denso ao menos denso (a referência fica em 3, no fundo)
ARCH_Z = {ar: 4 + n for n, ar in enumerate(sorted(ARCH_LS, key=lambda a: -_ink(ARCH_LS[a])))}
# [REMOVIDO] cores neutras trocadas por cores destacáveis, ímã por polaridade (pedido 2026-11-14):
# MAT_STYLE = {0: ('Ferro', '#5f5f5f'), 2: ('Ímã', '#9c7a4a'), 3: ('Cobre', '#c06a2b')}   # ar: branco
# Códigos do fundo: material_id; ímãs separados pela polaridade M do rótulo (21: M = +1, radial
# para fora; 22: M = -1, para dentro). Ar (1): branco (não desenhado).
MAT_STYLE = {0: ('Ferro', '#1e9e3c'), 21: ('Ímã (M+)', '#d62728'), 22: ('Ímã (M−)', '#1f5fd6'),
             3: ('Cobre', '#f07800')}


def arc_data(idx):
    from scripts import gap_wave as gw
    cache = gw.CKPT / f'arc_{idx:06d}.npz'
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    _, _, pred, radii = gw._setup()
    r1, r2 = radii[idx]
    F = gw.arc_fields(idx, pred, r1, r2, 0.05)
    out = dict(theta=F['theta'], ref=F['ref'], r1=r1, r2=r2, rm=F['rm'],
               n_nonvac=F['checks']['n_nonvac'], **{k: F['runs'][k] for k in gw.RUN_KEYS})
    np.savez(cache, **out)
    return out


def mesh_band(idx, rlo, rhi):
    """Triângulos (θ°, r) com alguma parte na faixa, unidos por material -> {mat: geometria}."""
    import shapely
    from src.data_gen.parsers.ans_parsing import _parse_label_materials, read_ans
    from src.postprocess.samples import sample_path
    ans = read_ans(sample_path(idx))
    xy = np.stack([ans.mesh.x, ans.mesh.y], axis=1).astype(np.float64)
    tri = ans.elems[:, :3].astype(np.int64)
    lmat, _, lM = _parse_label_materials(ans.lines)
    emat = lmat[ans.elems[:, 3]].astype(np.int64)
    eM = lM[ans.elems[:, 3]]
    mag = emat == 2
    emat[mag] = np.where(eM[mag] > 0, 21, 22)                       # ímã por polaridade
    r = np.hypot(xy[:, 0], xy[:, 1])
    th = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    rt = r[tri]
    sel = (rt.max(1) >= rlo - 0.5) & (rt.min(1) <= rhi + 0.5)
    P = np.stack([th[tri[sel]], rt[sel]], axis=-1)                    # [n,3,2] (θ°, r)
    polys = shapely.polygons(P)
    out = {int(m): shapely.union_all(polys[emat[sel] == m], grid_size=1e-9)
           for m in np.unique(emat[sel])}
    return out


def _geom_patches(g):
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    import shapely
    parts = list(g.geoms) if hasattr(g, 'geoms') else [g]
    verts, codes = [], []
    for p in parts:
        if p.geom_type != 'Polygon':
            continue
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)
            verts.append(c)
            codes.append(np.r_[MPath.MOVETO, np.full(len(c) - 2, MPath.LINETO), MPath.CLOSEPOLY])
    return MPath(np.concatenate(verts), np.concatenate(codes)) if verts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=2184)
    ap.add_argument('--h-factor', type=float, default=3.0)
    ap.add_argument('--layout', choices=('arch', 'comp'), default='arch',
                    help="arch: 1 linha por arch, B_r (esq.) e B_θ (dir.); comp: 3 linhas antigas "
                         "(B_r, B_θ, |ΔB|) com as 4 archs sobrepostas")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, PathPatch
    from scripts.postprocess import ARTICLE_ARCH_NAME
    from src.postprocess.plots.primitives import MODEL_COLOR, apply_article_style, apply_style
    from src.postprocess.samples import design_rows

    idx = a.sample
    D = arc_data(idx)
    assert int(D['n_nonvac']) == 0
    reg = json.loads((DADOS / 'regions.json').read_text(encoding='utf-8'))
    b_ref = float(reg['runs']['FNO2d_mae']['B_ref_global'])
    # conferência com o checkpoint da tab_gap_wave
    from scripts import gap_wave as gw
    with np.load(gw.CKPT / f'sample_{idx:06d}.npz') as z:
        E_ck = z['E']
    E_now = np.array([np.abs(D[k] - D['ref']).mean(0) for k in gw.RUN_KEYS])
    assert np.allclose(E_now, E_ck, rtol=1e-9, atol=1e-12), np.abs(E_now - E_ck).max()

    row = design_rows()[idx]
    r1, r2, rm = float(D['r1']), float(D['r2']), float(D['rm'])
    gap = r2 - r1
    r_back = r2 + float(row['pole_thickness [mm]'])                  # início do back iron
    r_yoke = float(row['stator_outer_diameter [mm]']) / 2 - sum(
        float(row[f'slot_Hs{i} [mm]']) for i in (0, 1, 2))           # fundo da ranhura
    h = min(a.h_factor * gap / 2, r_back - rm, rm - r_yoke)
    rlo, rhi = rm - h, rm + h
    bg = mesh_band(idx, rlo, rhi)

    th = np.degrees(D['theta'])
    keys = [f'{ar}_{LOSS}' for ar in ARCHS]
    dB = {k: 100 * np.hypot(*(D[k] - D['ref']).T) / b_ref for k in keys}

    def draw_bg(fig, ax):
        """Fundo desenrolado num eixo próprio atrás de ax: faixa [rlo, rhi] = altura do eixo."""
        axb = fig.add_axes(ax.get_position(), zorder=0)
        axb.set_xlim(0, 120)
        axb.set_ylim(rlo, rhi)
        axb.axis('off')
        for m, g in bg.items():
            if m not in MAT_STYLE:
                continue
            pth = _geom_patches(g)
            if pth is not None:
                axb.add_patch(PathPatch(pth, facecolor=MAT_STYLE[m][1], edgecolor='none',
                                        alpha=BG_ALPHA, lw=0))
        axb.axhline(rm, color='0.35', lw=0.5, ls=(0, (4, 3)))

    def bg_handles():
        h_ = [Patch(facecolor=MAT_STYLE[m][1], alpha=min(1, 3 * BG_ALPHA), edgecolor='none',
                    label=MAT_STYLE[m][0]) for m in sorted(bg) if m in MAT_STYLE]
        return h_ + [Line2D([], [], color='0.35', lw=0.5, ls=(0, (4, 3)), label=r'arco $r_m$')]

    comma = plt.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))
    apply_style()
    if a.layout == 'arch':
        # [REMOVIDO] B_θ tracejado -> linhas cheias; B_θ num tom claro da mesma cor (pedido)
        # LS_T = (0, (3, 1.5))                       # B_θ tracejado; B_r cheio
        from matplotlib.colors import to_rgb
        from matplotlib.legend_handler import HandlerTuple

        def light(col, f=0.5):
            """Tom claro: mistura com branco (f = fração da cor original)."""
            return tuple(1 - f * (1 - np.array(to_rgb(col))))
        REF_T = '0.6'                              # referência de B_θ (cinza); B_r em preto
        LS_ERR = (0, (2.5, 1.5))                   # erro com sinal de cada curva
        lim_r = 1.06 * max(np.abs(D[k][:, 0]).max() for k in ['ref', *keys])
        lim_t = 1.06 * max(np.abs(D[k][:, 1]).max() for k in ['ref', *keys])
        with plt.rc_context():
            apply_article_style(FONT_PT)
            plt.rcParams.update({'axes.grid': False})
            fig, axs = plt.subplots(len(ARCHS), 1, sharex=True, figsize=(FIG_WIDTH_IN, 6.4))
            fig.subplots_adjust(left=0.075, right=0.925, top=0.875, bottom=0.065, hspace=0.32)
            for ax, k, ar in zip(axs, keys, ARCHS):
                draw_bg(fig, ax)
                axt = ax.twinx()
                for z, x in enumerate((ax, axt), 1):
                    x.set_zorder(z)
                    x.patch.set_visible(False)
                    x.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                axt.spines['left'].set_visible(False)
                axt.spines['right'].set_visible(True)
                c = MODEL_COLOR[ar]
                ax.plot(th, D['ref'][:, 0], color='k', lw=1.3, zorder=3)
                ax.plot(th, D[k][:, 0], color=c, lw=0.8, zorder=4)
                axt.plot(th, D['ref'][:, 1], color=REF_T, lw=1.3, zorder=3)
                axt.plot(th, D[k][:, 1], color=light(c), lw=0.8, zorder=4)
                # [REMOVIDO] curvas de erro tracejadas (pedido: voltar à versão anterior)
                # ax.plot(th, D[k][:, 0] - D['ref'][:, 0], color=c, lw=0.7, ls=LS_ERR, zorder=5)
                # axt.plot(th, D[k][:, 1] - D['ref'][:, 1], color=light(c), lw=0.7, ls=LS_ERR, zorder=5)
                ax.set_ylim(-lim_r, lim_r)
                axt.set_ylim(-lim_t, lim_t)
                ax.set_xlim(0, 120)
                ax.set_ylabel(r'$B_r$ (T)')
                axt.set_ylabel(r'$B_\theta$ (T)')
                ax.yaxis.set_major_formatter(comma)
                axt.yaxis.set_major_formatter(comma)
                ax.set_title(ARTICLE_ARCH_NAME[ar], loc='left', pad=2, color=c)
            axs[-1].set_xticks(np.arange(0, 121, 10))
            axs[-1].set_xlabel(r'$\theta$ mecânico (°)')
            # cada entrada de curva = par (B_r escuro, B_θ claro)
            pair = lambda c1, c2: (Line2D([], [], color=c1, lw=1.3), Line2D([], [], color=c2, lw=1.3))
            handles = [pair('k', REF_T)] + [pair(MODEL_COLOR[ar], light(MODEL_COLOR[ar])) for ar in ARCHS]
            labels = ['Referência (FEMM)'] + [ARTICLE_ARCH_NAME[ar] for ar in ARCHS]
            # [REMOVIDO] entrada do erro na legenda (curvas de erro retiradas)
            # handles.append((Line2D([], [], color='0.25', lw=1.0, ls=LS_ERR),
            #                 Line2D([], [], color='0.6', lw=1.0, ls=LS_ERR)))
            # labels.append('erro (predição − FEMM)')
            for h_ in bg_handles():
                handles.append(h_)
                labels.append(h_.get_label())
            fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False,
                       bbox_to_anchor=(0.5, 1.0), handlelength=3.0, columnspacing=1.4,
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
                       title=r'$B_r$: eixo esq., tom escuro  |  $B_\theta$: eixo dir., tom claro',
                       title_fontsize=FONT_PT)
            FIGDIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGDIR / f'{NAME}.pdf')
            fig.savefig(FIGDIR / f'{NAME}.png', dpi=300)
            plt.close(fig)
    else:
      with plt.rc_context():
        apply_article_style(FONT_PT)
        plt.rcParams.update({'axes.grid': False})
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(FIG_WIDTH_IN, 5.0))
        fig.subplots_adjust(left=0.075, right=0.99, top=0.885, bottom=0.08, hspace=0.12)
        ylab = (r'$B_r$ (T)', r'$B_\theta$ (T)', r'$|\Delta \mathbf{B}|$ (%)')
        for i, ax in enumerate(axs):
            axb = fig.add_axes(ax.get_position(), zorder=0)
            axb.set_xlim(0, 120)
            axb.set_ylim(rlo, rhi)
            axb.axis('off')
            for m, g in bg.items():
                if m not in MAT_STYLE:
                    continue
                pth = _geom_patches(g)
                if pth is not None:
                    axb.add_patch(PathPatch(pth, facecolor=MAT_STYLE[m][1], edgecolor='none',
                                            alpha=BG_ALPHA, lw=0))
            axb.axhline(rm, color='0.35', lw=0.5, ls=(0, (4, 3)))
            ax.set_zorder(1)
            ax.patch.set_visible(False)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if i < 2:
                ax.plot(th, D['ref'][:, i], color='k', lw=1.5, zorder=3)
                for k, ar in zip(keys, ARCHS):
                    ax.plot(th, D[k][:, i], color=MODEL_COLOR[ar], ls=ARCH_LS[ar], lw=0.8, zorder=ARCH_Z[ar])
            else:
                for k, ar in zip(keys, ARCHS):
                    ax.plot(th, dB[k], color=MODEL_COLOR[ar], ls=ARCH_LS[ar], lw=0.8, zorder=ARCH_Z[ar])
                ax.set_ylim(bottom=0)
            ax.set_ylabel(ylab[i])
            ax.set_xlim(0, 120)
        axs[-1].set_xticks(np.arange(0, 121, 10))
        axs[-1].set_xlabel(r'$\theta$ mecânico (°)')
        for ax in axs:   # vírgula decimal nos ticks
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
        handles = [Line2D([], [], color='k', lw=1.5, label='Referência (FEMM)')]
        handles += [Line2D([], [], color=MODEL_COLOR[ar], ls=ARCH_LS[ar], lw=1.0, label=ARTICLE_ARCH_NAME[ar])
                    for ar in ARCHS]
        handles += [Patch(facecolor=MAT_STYLE[m][1], alpha=min(1, 3 * BG_ALPHA), edgecolor='none',
                          label=MAT_STYLE[m][0]) for m in sorted(bg) if m in MAT_STYLE]
        handles += [Line2D([], [], color='0.35', lw=0.5, ls=(0, (4, 3)), label=r'arco $r_m$')]
        fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False,
                   bbox_to_anchor=(0.53, 1.0), handlelength=3.2, columnspacing=1.5)
        FIGDIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGDIR / f'{NAME}.pdf')
        fig.savefig(FIGDIR / f'{NAME}.png', dpi=300)
        plt.close(fig)

    info = dict(sample_idx=idx, criterio='P50 do L1_A de superfície do FNO2d_mae no teste '
                '(surface --pct 50; mesma amostra de erro_dominio/erro_zoom/representacoes)',
                loss=LOSS, archs=list(ARCHS), B_ref_T=b_ref, passo_deg=0.05, setor_deg=[0, 120],
                r1_mm=r1, r2_mm=r2, rm_mm=rm, gap_mm=gap, h_mm=h, h_factor=a.h_factor,
                faixa_mm=[rlo, rhi], r_back_iron_mm=r_back, r_fundo_ranhura_mm=r_yoke,
                materiais_no_fundo=[MAT_STYLE.get(m, ('Ar', 'branco'))[0] for m in sorted(bg)],
                bg_alpha=BG_ALPHA,
                eps_arco_pct={k: dict(eps_r=100 * float(np.abs(D[k] - D['ref'])[:, 0].mean()) / b_ref,
                                      eps_t=100 * float(np.abs(D[k] - D['ref'])[:, 1].mean()) / b_ref,
                                      dB_mean=float(dB[k].mean()), dB_max=float(dB[k].max()))
                              for k in keys})
    (DADOS / f'fig_{NAME}.json').write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                            encoding='utf-8')
    print(json.dumps({k: v for k, v in info.items() if k != 'eps_arco_pct'}, ensure_ascii=False, indent=1))
    for k, v in info['eps_arco_pct'].items():
        print(f"  {k:28s} ε_r {v['eps_r']:.2f}  ε_θ {v['eps_t']:.2f}  |ΔB| médio {v['dB_mean']:.2f}  máx {v['dB_max']:.1f} %")
    print(f'-> {FIGDIR / NAME}.pdf/.png')


if __name__ == '__main__':
    main()
