"""
artigo_v1.py
------------
Figuras F1–F7, tabelas LaTeX (booktabs) e resultados.md do artigo v1, a partir das saídas
de scripts/postprocess.py (resultados/BLDC_138x276/dados/) e dos testes do FEMM
(tests/check_nodal_b_femm.py, tests/check_analyze_remesh.py). Nada aqui treina, recalcula
métrica ou muda o pipeline: métricas, superfícies, regiões, torque e saturação vêm das
funções de src/postprocess e de scripts/postprocess.py; este script só agrega e desenha.

Subcomandos (raiz do projeto):
  joelho  — fração da área de aço do teste acima do joelho e do pico de μ_eff
            (saturation.steel_operating_points, joelho de dados/saturation.json)
            -> artigo_v1/dados/saturacao_joelho.json
  campos  — F2 (referência × predição × erro, zoom em interface e canto; superfície nativa
            de cada modelo, scripts.postprocess._surface_points), mapa das regiões do F3 e
            F6b (|B| no aço da amostra de maior corrente) -> figuras + artigo_v1/dados/campos.json
  render  — tabelas, F1, F3 (barras), F4, F5, F6a, F7, análises de L4/L7 e resultados.md
  all     — joelho + campos + render

Saída: resultados/BLDC_138x276/artigo_v1/{figuras,tabelas,dados}/ e resultados.md.
Imports pesados (torch, modelos) só dentro das funções: os workers do joelho não carregam
o torch (limite de memória virtual com muitos processos).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

DATASET = 'BLDC_138x276'
ROOT_S = Path('data/logs') / DATASET
SRC = Path('resultados') / DATASET / 'dados'
OUT = Path('resultados') / DATASET / 'artigo_v1'
FIG, TAB, DAT = OUT / 'figuras', OUT / 'tabelas', OUT / 'dados'

ARCHS = ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN', 'FNO_BipartiteGNN_Elem')
LOSSES = ('mse', 'mae')
REPR = {'FNO2d': 'grade', 'FNO_GNN': 'P1', 'GNN_PostBase': 'P1', 'FNO_BipartiteGNN': 'P1',
        'FNO_BipartiteGNN_Elem': 'P0'}
FLOOR_KIND = {'grade': 'grade', 'P1': 'P1_nos', 'P0': 'P0_elem'}   # chave de floor_surface do eval
CM = 1 / 2.54
W1, W2 = 8.8 * CM, 18.0 * CM          # largura de coluna IEEE / dupla (polegadas)
BANDS = [(a, a + 2) for a in range(0, 10, 2)]   # faixas de corrente do torque (A)
SUBSET = 'test'


# --------------------------------------------------------------------------- #
# utilidades
# --------------------------------------------------------------------------- #
def _j(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(v, nd=2):
    """Número com vírgula decimal (texto em português); None -> '--'."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '--'
    return f'{v:.{nd}f}'.replace('.', ',')


def _tex(name):
    return name.replace('_', r'\_')


def _short(arch):
    return arch.replace('FNO_BipartiteGNN', 'FNO_Bipartite\nGNN') if 'Bipartite' in arch else arch


def _style():
    from src.postprocess.plots.primitives import apply_style
    import matplotlib.pyplot as plt
    apply_style()
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                         'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'pdf.fonttype': 42, 'savefig.dpi': 300})


def _save(fig, name):
    """PDF vetorial (raster embutido a 300 dpi) + PNG de conferência."""
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  figura {FIG / name}.pdf')


def _write_tex(name, colspec, header, rows, source, notes=()):
    """tabular booktabs pronta para \\input{} (o table/caption fica no artigo)."""
    TAB.mkdir(parents=True, exist_ok=True)
    lines = [f'% gerado por scripts/artigo_v1.py render -- fonte: {source}']
    lines += [f'% {n}' for n in notes]
    lines += [rf'\begin{{tabular}}{{{colspec}}}', r'\toprule']
    lines += header + [r'\midrule']
    for r in rows:
        lines.append(r if isinstance(r, str) else ' & '.join(r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    (TAB / f'{name}.tex').write_text('\n'.join(lines), encoding='utf-8')
    print(f'  tabela {TAB / name}.tex')


def _run_dirs():
    from src.postprocess.inference import find_runs
    return find_runs(ROOT_S)


# --------------------------------------------------------------------------- #
# joelho: fração do aço acima do joelho (dados exatos por elemento, sem histograma)
# --------------------------------------------------------------------------- #
def _steel_one(path, row, b_knee, b_mu_max):
    from src.postprocess.saturation import steel_operating_points
    d = steel_operating_points(path, row)
    b, a = d['bmag'], d['area']
    return dict(area=float(a.sum()), above_knee=float(a[b > b_knee].sum()),
                above_mu_max=float(a[b > b_mu_max].sum()), n=int(len(b)),
                n_above_knee=int((b > b_knee).sum()))


def run_joelho(n_workers=8):
    from concurrent.futures import ProcessPoolExecutor
    from src.postprocess.samples import design_rows, samples_with_subset
    sat = _j(SRC / 'saturation.json')
    k = sat['knee'][0]
    rows = design_rows()
    samples = samples_with_subset(_run_dirs()[('FNO2d', 'mse')], SUBSET)
    with ProcessPoolExecutor(n_workers) as ex:
        res = list(ex.map(_steel_one, [p for *_, p in samples], [rows[i] for _, _, i, _ in samples],
                          [k['B_knee_T']] * len(samples), [k['B_mu_max_T']] * len(samples)))
    A = sum(r['area'] for r in res)
    per = np.array([r['above_knee'] / r['area'] for r in res])
    out = dict(subset=SUBSET, n_samples=len(res), B_knee_T=k['B_knee_T'], B_mu_max_T=k['B_mu_max_T'],
               frac_area_above_knee=sum(r['above_knee'] for r in res) / A,
               frac_area_above_mu_max=sum(r['above_mu_max'] for r in res) / A,
               frac_elems_above_knee=sum(r['n_above_knee'] for r in res) / sum(r['n'] for r in res),
               per_sample_frac_above_knee=dict(p5=float(np.percentile(per, 5)), p50=float(np.median(per)),
                                               p95=float(np.percentile(per, 95)), max=float(per.max())),
               n_samples_with_any_above_knee=int((per > 0).sum()),
               fonte='saturation.steel_operating_points (|B| suavizado no centróide, área do elemento)')
    DAT.mkdir(parents=True, exist_ok=True)
    (DAT / 'saturacao_joelho.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- #
# campos: F2, mapa do F3, F6b
# --------------------------------------------------------------------------- #
def _tri_avg(n_tri, t, v, w):
    W = np.bincount(t, w, n_tri)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.bincount(t, v * w, n_tri) / W


def _zoom_points(xy, tri, elem_mat, row, iface, corners, th_deg=60.0):
    """(canto, interface plana) perto do entreferro em θ ≈ th_deg: canto = vértice de canto
    (regions.corner_nodes) no raio externo do estator; interface = ponto médio de aresta de
    interface a > 2 mm de qualquer canto, no raio interno do rotor (fallback: qualquer)."""
    from src.postprocess.torque import gap
    r1, r2 = gap(row)
    th0 = np.deg2rad(th_deg)
    r = np.hypot(xy[:, 0], xy[:, 1])
    th = np.arctan2(xy[:, 1], xy[:, 0])
    c = np.asarray(corners)
    cand = c[(r[c] > r1 - 0.6) & (r[c] < r1 + 0.05)]
    cand = cand if len(cand) else c
    ic = cand[np.argmin(np.abs(th[cand] - th0))]
    mid = xy[iface].mean(axis=1)
    rm, tm = np.hypot(mid[:, 0], mid[:, 1]), np.arctan2(mid[:, 1], mid[:, 0])
    dc = np.min(np.hypot(mid[:, None, 0] - xy[c][None, :, 0], mid[:, None, 1] - xy[c][None, :, 1]), axis=1)
    ok = (dc > 2.0) & (np.abs(rm - r2) < 0.05)
    if not ok.any():
        ok = dc > 2.0
    ii = np.flatnonzero(ok)[np.argmin(np.abs(tm[ok] - th0))]
    return (float(xy[ic, 0]), float(xy[ic, 1])), (float(mid[ii, 0]), float(mid[ii, 1]))


def _fig_f2(case, idx, loss, tri, R, P, E, iface_xy, zooms, evmax, info):
    """Linhas: referência + 5 modelos; colunas: setor |B|, interface |B|, interface erro,
    canto |B|, canto erro. Erro com sinal (|B|_mod − |B|_ref) por triângulo, RdBu centrado
    em zero, mesma escala em todos os modelos."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from src.postprocess.plots.primitives import B_VMAX
    names = ['referência (FEMM suavizado)'] + list(ARCHS)
    fig, axs = plt.subplots(len(names), 5, figsize=(W2, W2 * 1.18))
    h = zooms['half_mm']
    win = [None, zooms['interface'], zooms['interface'], zooms['canto'], zooms['canto']]
    for i, nm in enumerate(names):
        vals = R if i == 0 else P[nm]
        for j in range(5):
            ax = axs[i, j]
            is_err = j in (2, 4)
            if is_err and i == 0:
                ax.axis('off')
                continue
            if is_err:
                m_e = ax.tripcolor(tri, facecolors=E[nm], cmap='RdBu_r', vmin=-evmax, vmax=evmax,
                                   rasterized=True)
            else:
                m_b = ax.tripcolor(tri, facecolors=vals, cmap='viridis', vmin=0, vmax=B_VMAX,
                                   rasterized=True)
            ax.add_collection(LineCollection(iface_xy, colors='k', linewidths=0.15 if j == 0 else 0.4))
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if win[j] is not None:
                x, y = win[j]
                ax.set(xlim=(x - h, x + h), ylim=(y - h, y + h))
            elif i == 0:
                for key, col in (('interface', '#eb6834'), ('canto', '#e4e3df')):
                    x, y = zooms[key]
                    ax.add_patch(plt.Rectangle((x - h, y - h), 2 * h, 2 * h, fill=False, ec=col, lw=0.8))
        lab = nm if i == 0 else f"{_short(nm)}\nL1={_fmt(info[nm]['L1_pct'], 1)}%"
        axs[i, 0].set_ylabel(lab, fontsize=6, rotation=0, ha='right', va='center')
        axs[i, 0].yaxis.set_label_coords(-0.05, 0.5)
    for j, t in enumerate(('setor de 120°', '|B| — interface', 'erro — interface', '|B| — canto',
                           'erro — canto')):
        axs[1 if j in (2, 4) else 0, j].set_title(t)   # linha 0 das colunas de erro fica vazia
    # barras de cor em eixos próprios (colorbar(ax=...) com duas listas se sobrepunha)
    fig.subplots_adjust(left=0.15, right=0.995, top=0.95, bottom=0.075, wspace=0.05, hspace=0.06)
    cb1 = fig.colorbar(m_b, cax=fig.add_axes([0.15, 0.04, 0.38, 0.011]), orientation='horizontal')
    cb1.set_label('|B| [T]')
    cb2 = fig.colorbar(m_e, cax=fig.add_axes([0.61, 0.04, 0.38, 0.011]), orientation='horizontal',
                       extend='both')
    cb2.set_label('|B|$_{modelo}$ − |B|$_{ref}$ [T]')
    fig.suptitle(f'Amostra {idx} ({case}), perda {loss}; janela de zoom {2 * h:.1f} mm', y=0.99)
    return fig


def _fig_regions_map(xy, tri, region, iface_xy, corners, zoom, half):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from src.postprocess.regions import REGIONS
    cols = ['#e4e3df', '#2a78d6', '#eb6834']
    T = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=tri)
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.45))
    for k, ax in enumerate(axs):
        ax.tripcolor(T, facecolors=region.astype(float), cmap=ListedColormap(cols), vmin=-0.5,
                     vmax=len(REGIONS) - 0.5, rasterized=True)
        ax.add_collection(LineCollection(iface_xy, colors='k', linewidths=0.4 if k else 0.15))
        ax.plot(xy[corners, 0], xy[corners, 1], '.', color='k', ms=2.5 if k else 0.6)
        ax.set_aspect('equal'); ax.axis('off')
        if k:
            ax.set(xlim=(zoom[0] - half, zoom[0] + half), ylim=(zoom[1] - half, zoom[1] + half))
    for c, nm in zip(cols, REGIONS):
        axs[0].plot([], [], 's', color=c, label=nm)
    axs[0].plot([], [], '.', color='k', label='vértice de canto')
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    axs[0].set_title('setor de 120°')
    axs[1].set_title(f'zoom no entreferro ({2 * half:.0f} mm)')
    return fig


def _fig_f6b(xy, tri, steel, bmag, b_knee, title, vmax):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    full = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=tri)
    st = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=tri[steel])
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.95))
    ax.tripcolor(full, facecolors=np.zeros(len(tri)), cmap='Greys', vmin=0, vmax=8, rasterized=True)
    m = ax.tripcolor(st, facecolors=bmag, cmap='magma', vmin=0, vmax=vmax, rasterized=True)
    # contorno: |B| por elemento levado aos nós (média dos elementos de aço incidentes) -- visual
    num = np.bincount(tri[steel].ravel(), np.repeat(bmag, 3), minlength=len(xy))
    den = np.bincount(tri[steel].ravel(), minlength=len(xy))
    nodal = np.divide(num, den, out=np.zeros(len(xy)), where=den > 0)
    ax.tricontour(st, nodal, levels=[b_knee], colors=['#00e5ff'], linewidths=0.8)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title)
    cb = fig.colorbar(m, ax=ax, orientation='horizontal', fraction=0.05, pad=0.02, extend='max')
    cb.set_label(f'|B| no aço [T]; contorno = joelho ({_fmt(b_knee, 2)} T)')
    return fig


def run_campos(losses=LOSSES, half_mm=1.2):
    import matplotlib.tri as mtri
    from scripts.postprocess import _mesh, _pick_by_pct, _surface_points
    from src.postprocess.inference import load_set, predict, to_torch_sample
    from src.postprocess.metrics import Accum
    from src.postprocess.regions import classify_triangles
    from src.postprocess.samples import N_R, design_rows, parse_smooth, sample_path
    from src.postprocess.metrics import R_EXT_MM, R_IN_MM
    from src.postprocess.saturation import steel_operating_points
    _style()
    per = np.load(SRC / 'eval_smooth_per_sample.npz')
    rows = design_rows()
    i50, s50 = _pick_by_pct(per, 50)
    imax, smax = _pick_by_pct(per, 100)
    ids = per['sample_idx'][per['subset'] == SUBSET]
    cur = np.array([float(rows[i]['current_amplitude [A]']) for i in ids])
    iI = int(ids[np.argmax(cur)])
    cases = {'mediano': (int(i50), s50), 'pior caso': (int(imax), smax),
             'corrente alta': (iI, dict(rule='maior current_amplitude [A] do teste',
                                        current_amplitude_A=float(cur.max())))}
    dr = (R_EXT_MM - R_IN_MM) / N_R
    runs = {l: load_set(ROOT_S, losses=(l,)) for l in losses}
    meta = dict(cases={}, half_mm=half_mm, regions_params=dict(d_int_mm=dr, d_canto_mm=2 * dr, ang_deg=30.0))
    for case, (idx, sel) in cases.items():
        row = rows[idx]
        lay = parse_smooth(sample_path(idx), row, with_elem=True, with_overlay=True)
        ts = to_torch_sample(lay)
        nodes, elems = _mesh(idx)
        xy, t3 = nodes[:, :2], lay['tri']
        n_tri = len(t3)
        region, iface, corners = classify_triangles(xy, t3, lay['elem_mat'], dr, 2 * dr, 30.0)
        zc, zi = _zoom_points(xy, t3, lay['elem_mat'], row, iface, corners)
        zooms = dict(canto=zc, interface=zi, half_mm=half_mm)
        T = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=t3)
        meta['cases'][case] = dict(sample_idx=idx, selection=sel, zoom_canto_mm=zc, zoom_interface_mm=zi,
                                   current_amplitude_A=float(row['current_amplitude [A]']), losses={})
        for loss in losses:
            P, E, info, R = {}, {}, {}, None
            for r in runs[loss].values():
                a = r['arch']
                out_hw, yn = predict(a, r['model'], r['normalizer'], ts)
                mm, mf, w, t = _surface_points(a, lay, out_hw, yn)
                acc = Accum()
                acc.add_surface(mm, mf, w)
                x = acc.report()
                P[a] = _tri_avg(n_tri, t, mm, w)
                E[a] = _tri_avg(n_tri, t, mm - mf, w)
                if a != 'FNO2d':   # referência pelos pontos da malha (não pelos da grade)
                    R = _tri_avg(n_tri, t, mf, w)
                info[a] = dict(L1_pct=x['L1_area_pct'], L2_pct=x['L2_area_pct'], run=r['run'])
            # escala comum do erro: p99 de |erro| nas duas janelas, todos os modelos
            box = np.zeros(n_tri, bool)
            cen = xy[t3].mean(axis=1)
            for p in (zc, zi):
                box |= (np.abs(cen[:, 0] - p[0]) < half_mm) & (np.abs(cen[:, 1] - p[1]) < half_mm)
            evmax = float(np.nanpercentile(np.abs(np.concatenate([E[a][box] for a in ARCHS])), 99))
            evmax = max(0.01, float(np.ceil(evmax * 100) / 100))
            meta['cases'][case]['losses'][loss] = dict(err_vmax_T=evmax, runs=info)
            _save(_fig_f2(case, idx, loss, T, R, P, E, xy[iface], zooms, evmax, info),
                  f"F2_campo_{case.replace(' ', '_')}_{loss}")
        if case == 'mediano':
            _save(_fig_regions_map(xy, t3, region, xy[iface], corners, zi, 3.0), 'F3a_regioes_mapa')
            meta['F3a'] = dict(sample_idx=idx, zoom_mm=zi)
        if case == 'corrente alta':
            sat = _j(SRC / 'saturation.json')
            bk = sat['knee'][0]['B_knee_T']
            d = steel_operating_points(sample_path(idx), row)
            _save(_fig_f6b(xy, elems[:, :3].astype(np.int64), d['elem'], d['bmag'], bk,
                           f'Amostra {idx}: I = {_fmt(float(row["current_amplitude [A]"]), 2)} A',
                           vmax=max(2.2, bk * 1.3)), 'F6b_saturacao_mapa_aco')
            meta['F6b'] = dict(sample_idx=idx, B_knee_T=bk,
                               frac_area_above_knee=float(d['area'][d['bmag'] > bk].sum() / d['area'].sum()))
    DAT.mkdir(parents=True, exist_ok=True)
    (DAT / 'campos.json').write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# render: análises a partir dos JSON + tabelas + figuras
# --------------------------------------------------------------------------- #
def _stability(run_dir: Path):
    """Estabilidade da validação (epochs.csv): teto de épocas, divergência e oscilação.
    Definições deste script (documentadas no resultados.md):
      teto       = stop_reason == 'n_epochs' (rodou n_epochs_cfg sem parada antecipada);
      divergência= val_loss não finita, ou val_loss final > 10× o mínimo;
      pico_max   = max_t val_loss[t] / min(val_loss[:t]) para t ≥ 20;
      rugosidade = mediana de |Δ ln val_loss| entre épocas consecutivas (t ≥ 20);
      oscilação forte = pico_max > 2."""
    import csv
    rows = list(csv.DictReader(open(run_dir / 'epochs.csv')))
    v = np.array([float(r['val_loss']) for r in rows])
    s = _j(run_dir / 'run_summary.json')
    t0 = 20
    run_min = np.minimum.accumulate(v)
    spike = float(np.max(v[t0:] / run_min[t0 - 1:-1])) if len(v) > t0 else float('nan')
    rough = float(np.median(np.abs(np.diff(np.log(v[t0 - 1:]))))) if len(v) > t0 else float('nan')
    return dict(stop_reason=s['stop_reason'], last_epoch=s['last_epoch'], best_epoch=s['best_epoch'],
                n_epochs_cfg=s['n_epochs_cfg'], hit_ceiling=s['stop_reason'] == 'n_epochs',
                diverged=bool((~np.isfinite(v)).any() or v[-1] > 10 * np.nanmin(v)),
                spike_max=spike, roughness=rough, strong_oscillation=bool(spike > 2),
                best_in_last_5pct=bool(s['best_epoch'] >= 0.95 * s['last_epoch']),
                val_best=float(np.nanmin(v)), val_last=float(v[-1]))


def _torque_analysis(loss):
    """L7: vs referência suavizada (mesmo método da faixa do entreferro) e vs FEMM T22;
    erro da referência (suavizada × exata P0 = 'gt'). curve_row/fundamental/scatter_stats
    de src.postprocess.torque."""
    from src.postprocess.torque import curve_row, fundamental, scatter_stats
    out = dict(loss=loss)
    sc = _j(SRC / 'torque' / f'scatter_{loss}.json')
    if sc:
        S, G = np.array(sc['T_smooth']) * 1e3, np.array(sc['T_gt']) * 1e3
        out['scatter'] = dict(
            n=len(S), fonte=f'dados/torque/scatter_{loss}.json',
            vs_smooth={a: scatter_stats(S, np.array(sc[f'T_pred_{a}']) * 1e3, G) for a in ARCHS
                       if f'T_pred_{a}' in sc},
            vs_gt=sc['stats'],
            ref_smooth_vs_gt=dict(rmse_mNm=float(np.sqrt(np.mean((S - G) ** 2))),
                                  bias_mNm=float(np.mean(S - G)),
                                  rms_gt_mNm=float(np.sqrt(np.mean(G ** 2)))))
    bt = _j(SRC / 'torque' / f'theta_e_batch_{loss}.json')
    if bt:
        res = {a: {f'{lo}-{hi}A': [] for lo, hi in BANDS} for a in ARCHS}
        ref = {f'{lo}-{hi}A': [] for lo, hi in BANDS}
        bias_curves = {a: {} for a in ARCHS}
        for i, smp in bt['samples'].items():
            c, I = smp['curves'], smp['I']
            if len(c['theta_e_deg']) < 3:
                continue
            th = np.array(c['theta_e_deg'])
            S, G, F = (np.array(c[k]) * 1e3 for k in ('smooth', 'gt', 'femm'))
            band = next(f'{lo}-{hi}A' for lo, hi in BANDS if lo <= I < hi)
            ref[band].append(dict(I=I, rmse_smooth_vs_gt=float(np.sqrt(np.mean((S - G) ** 2))),
                                  rmse_smooth_vs_femm=float(np.sqrt(np.mean((S - F) ** 2))),
                                  rmse_gt_vs_femm=float(np.sqrt(np.mean((G - F) ** 2))),
                                  rms_gt=float(np.sqrt(np.mean(G ** 2))),
                                  A_smooth_over_A_gt=float(fundamental(th, S)[1] / fundamental(th, G)[1])))
            for a in ARCHS:
                k = f'pred_{a}'
                if k not in c:
                    continue
                Pp = np.array(c[k]) * 1e3
                res[a][band].append(dict(I=I, **curve_row(th, S, Pp, G)))
                bias_curves[a].setdefault(band, []).append((th, Pp - S))
        summ = {}
        for a in ARCHS:
            summ[a] = {}
            for b, lst in res[a].items():
                if not lst:
                    continue
                arr = {k: np.array([x[k] for x in lst]) for k in ('rmse', 'rms_femm', 'A_ratio', 'dphi', 'dT0')}
                bc = bias_curves[a].get(b, [])
                th = bc[0][0]
                mean_bias = np.mean([e for _, e in bc], axis=0)       # viés médio por θe (mNm)
                T0b, Ab, phb = fundamental(th, mean_bias)
                summ[a][b] = dict(n=len(lst), rmse_median_mNm=float(np.median(arr['rmse'])),
                                  rel_median_pct=float(np.median(100 * arr['rmse'] / arr['rms_femm'])),
                                  rms_ref_median_mNm=float(np.median(arr['rms_femm'])),
                                  dT0_mean_mNm=float(arr['dT0'].mean()), dT0_std_mNm=float(arr['dT0'].std()),
                                  A_ratio_mean=float(arr['A_ratio'].mean()), A_ratio_std=float(arr['A_ratio'].std()),
                                  dphi_mean_deg=float(arr['dphi'].mean()),
                                  bias_vs_theta_mNm=dict(theta_e_deg=th.tolist(), mean=mean_bias.tolist(),
                                                         min=float(mean_bias.min()), max=float(mean_bias.max()),
                                                         offset=float(T0b), first_harmonic_amp=float(Ab),
                                                         first_harmonic_phase_deg=float(phb)))
        out['batch'] = dict(fonte=f'dados/torque/theta_e_batch_{loss}.json',
                            referencia='suavizada (gt_smooth), mesmo método da faixa do entreferro',
                            por_arch=summ,
                            vs_femm_resumo_existente=(_j(SRC / 'torque' / f'theta_e_batch_{loss}_summary.json')
                                                      or {}).get('summary'),
                            ref_error={b: dict(n=len(v), rmse_smooth_vs_gt_median_mNm=float(np.median([x['rmse_smooth_vs_gt'] for x in v])),
                                               rel_smooth_vs_gt_median_pct=float(np.median([100 * x['rmse_smooth_vs_gt'] / x['rms_gt'] for x in v])),
                                               rmse_smooth_vs_femm_median_mNm=float(np.median([x['rmse_smooth_vs_femm'] for x in v])),
                                               rmse_gt_vs_femm_median_mNm=float(np.median([x['rmse_gt_vs_femm'] for x in v])),
                                               A_smooth_over_A_gt_mean=float(np.mean([x['A_smooth_over_A_gt'] for x in v])))
                                       for b, v in ref.items() if v})
    return out


def _fig_f1(ev):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fs = ev['floor_surface']
    col = {'L1': '#2a78d6', 'L2': '#eb6834'}
    fig, ax = plt.subplots(figsize=(W2, 6.0 * CM))
    x0 = np.arange(len(ARCHS)) * 5.0
    off = {('mse', 'L1'): -1.5, ('mse', 'L2'): -0.5, ('mae', 'L1'): 0.5, ('mae', 'L2'): 1.5}
    ymax = 0
    for g, a in zip(x0, ARCHS):
        kind = FLOOR_KIND[REPR[a]]
        for (loss, m), dx in off.items():
            v = ev['runs'][f'{a}_{loss}']['surface'][SUBSET][f'{m}_area_pct']
            ymax = max(ymax, v)
            ax.bar(g + dx, v, 0.9, color='white' if loss == 'mae' else col[m], edgecolor=col[m],
                   hatch='////' if loss == 'mae' else None, lw=0.8, zorder=3)
            ax.text(g + dx, v, _fmt(v, 1), ha='center', va='bottom', fontsize=5, rotation=90)
            fl = fs[kind][SUBSET][f'{m}_area_pct']
            ax.hlines(fl, g + dx - 0.5, g + dx + 0.5, color='k', lw=1.4, zorder=4)
    for m in ('L1', 'L2'):
        ax.axhline(fs['grade'][SUBSET][f'{m}_area_pct'], color=col[m], ls='--', lw=0.9, zorder=2)
    ax.set_xticks(x0)
    ax.set_xticklabels([_short(a) for a in ARCHS])
    ax.set_ylabel('erro de superfície [% de $B_{ref}$]')
    ax.set_ylim(0, ymax * 1.25)
    ax.grid(axis='x', visible=False)
    h = [Patch(fc=col['L1'], ec=col['L1'], label='$L1_A$'), Patch(fc=col['L2'], ec=col['L2'], label='$L2_A$'),
         Patch(fc='#9b9a96', ec='#9b9a96', label='MSE (cheio)'),
         Patch(fc='white', ec='k', hatch='////', label='MAE (hachurado)'),
         Line2D([], [], color='k', ls='--', lw=0.9, label='piso da grade (cor = métrica)'),
         Line2D([], [], color='k', lw=1.4, label='piso da representação (grade/P1/P0)')]
    ax.legend(handles=h, ncol=6, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              fontsize=5.5, handlelength=1.6, columnspacing=1.0)
    return fig


def _fig_f3b(rg):
    import matplotlib.pyplot as plt
    from src.postprocess.regions import REGIONS
    keys = [f'{a}_{l}' for a in ARCHS for l in LOSSES]
    cols = {'interior': '#9b9a96', 'interface': '#2a78d6', 'canto': '#eb6834'}
    fig, axs = plt.subplots(2, 1, figsize=(W2, 9 * CM), sharex=True)
    x = np.arange(len(keys))
    for j, rgn in enumerate(REGIONS):
        dx = (j - 1) * 0.27
        axs[0].bar(x + dx, [rg['runs'][k][rgn]['L1_area_pct'] for k in keys], 0.25, color=cols[rgn],
                   label=rgn, zorder=3)
        axs[1].bar(x + dx, [100 * rg['runs'][k][rgn]['L1_contrib_frac'] for k in keys], 0.25,
                   color=cols[rgn], zorder=3)
        af = 100 * rg['runs'][keys[0]][rgn]['area_frac']
        axs[1].axhline(af, color=cols[rgn], ls='--', lw=0.8)
    axs[0].set_ylabel('$L1_A$ da região\n[% de $B_{ref}$ global]')
    axs[1].set_ylabel('fração do $\\int e\\,dA$ [%]\n(tracejado = fração da área)')
    axs[0].legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([k.rsplit('_', 1)[1] for k in keys], fontsize=5.5)
    for g, a in enumerate(ARCHS):   # nome da arch centrado sob o par mse/mae
        axs[1].text(2 * g + 0.5, -0.13, _short(a), transform=axs[1].get_xaxis_transform(),
                    ha='center', va='top', fontsize=6)
    for g in range(1, len(ARCHS)):
        for ax in axs:
            ax.axvline(2 * g - 0.5, color='#d9d8d4', lw=0.6, zorder=1)
    for ax in axs:
        ax.grid(axis='x', visible=False)
    return fig


def _fig_f4(run_dirs):
    import csv
    import matplotlib.pyplot as plt
    from src.postprocess.plots.primitives import MODEL_COLOR
    fig, axs = plt.subplots(2, 5, figsize=(W2, 8.5 * CM), sharex=False)
    for j, a in enumerate(ARCHS):
        for i, loss in enumerate(LOSSES):
            ax = axs[i, j]
            d = run_dirs[(a, loss)]
            rows = list(csv.DictReader(open(d / 'epochs.csv')))
            ep = np.array([int(r['epoch']) for r in rows])
            tr = np.array([float(r['train_loss']) for r in rows])
            va = np.array([float(r['val_loss']) for r in rows])
            s = _j(d / 'run_summary.json')
            ax.semilogy(ep, tr, color=MODEL_COLOR[a], lw=0.8, ls='--', label='treino')
            ax.semilogy(ep, va, color=MODEL_COLOR[a], lw=0.9, label='validação')
            b = s['best_epoch']
            if b in ep:
                ax.plot(b, va[ep == b][0], 'o', ms=3, mfc='white', mec='k', mew=0.7)
            ax.set_title(f'{_short(a).replace(chr(10), "")} — {loss}', fontsize=6)
            if i == 1:
                ax.set_xlabel('época')
            if j == 0:
                ax.set_ylabel(f'perda {loss.upper()}\n(z-score)')
    axs[0, 0].legend(frameon=False, fontsize=5.5)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
    return fig


def _fig_f5a(sc_by_loss):
    import matplotlib.pyplot as plt
    from src.postprocess.plots.primitives import MODEL_COLOR
    fig, axs = plt.subplots(1, 2, figsize=(W2, 8.0 * CM))
    for ax, loss in zip(axs, LOSSES):
        sc = sc_by_loss.get(loss)
        if not sc:
            ax.set_title(f'{loss}: PENDENTE'); continue
        S = np.array(sc['T_smooth'])
        for a in ARCHS:
            if f'T_pred_{a}' in sc:
                ax.plot(S, np.array(sc[f'T_pred_{a}']), '.', ms=1.6, color=MODEL_COLOR[a],
                        label=_short(a).replace('\n', ''), alpha=0.7)
        lim = [min(S.min(), 0), S.max()]
        ax.plot(lim, lim, color='k', lw=0.7)
        ax.set(xlabel='torque — referência suavizada [N·m]', ylabel='torque — modelo [N·m]',
               title=f'perda {loss.upper()} ({len(S)} amostras de teste)')
        ax.set_aspect('equal', adjustable='datalim')
    axs[0].legend(frameon=False, markerscale=4)
    fig.tight_layout()
    return fig


def _fig_f5b(theta_files, loss):
    import matplotlib.pyplot as plt
    from src.postprocess.plots.primitives import MODEL_COLOR
    n = len(theta_files)
    fig, axs = plt.subplots(1, n, figsize=(W2, 5.2 * CM), squeeze=False)
    for ax, (idx, d) in zip(axs[0], theta_files):
        th = np.array(d['theta_e_deg'])
        T = {k: np.array(v) for k, v in d['T_Nm'].items()}
        ax.plot(th, T['gt_smooth'], color='k', lw=1.3, label='referência suavizada')
        ax.plot(th, T['gt'], color='#52514e', lw=0.8, ls='--', label='referência exata (P0)')
        ax.plot(th, T['femm_T22'], color='#52514e', lw=0.8, ls=':', label='FEMM (T22)')
        for a in ARCHS:
            if f'pred_{a}' in T:
                ax.plot(th, T[f'pred_{a}'], color=MODEL_COLOR[a], lw=0.8, label=_short(a).replace('\n', ''))
        ax.set_title(f'amostra {idx}, I = {_fmt(d["current_amplitude_A"], 2)} A', fontsize=6)
        ax.set_xlabel('$\\theta_e$ [°]')
        ax.set_xticks([0, 120, 240, 360])
    axs[0, 0].set_ylabel('torque [N·m]')
    fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, fontsize=5, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=8)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.9, bottom=0.33, wspace=0.42)
    return fig


def _fig_f6a(sat):
    import matplotlib.pyplot as plt
    from src.postprocess.saturation import FemmBH
    from src.postprocess.plots.saturation import MU_LABEL
    bh = FemmBH(sat['bh_curve']['B_T'], sat['bh_curve']['H_A_m'])
    k = sat['knee'][0]
    edges = np.array(sat['hist']['edges_T'])
    hw = np.array(sat['hist']['area_mm2'])
    b = np.linspace(1e-3, edges[-1], 3000)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(W1, 8.5 * CM), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    ax.plot(b, bh.h(b) / 1e3, color='k', lw=1.0, label='H(B), interpolação do FEMM')
    ax.plot(bh.B_in, bh.H_in / 1e3, 'o', color='k', ms=2, label='pontos B-H do .ans')
    ax.set_ylabel('H [kA/m]')
    ax.set_ylim(0, bh.h(np.array([edges[-1]]))[0] / 1e3 * 1.05)
    ax2 = ax.twinx()
    ax2.plot(b, bh.mu_eff(b), color='#2a78d6', lw=1.0, label='$\\mu_{eff} = B/(\\mu_0 H)$')
    ax2.axhline(MU_LABEL, color='#eb6834', ls='--', lw=0.8, label=f'$\\mu_r$ = {MU_LABEL:g} (rótulo de entrada)')
    ax2.plot(k['B_mu_max_T'], k['mu_max'], 'v', color='#2a78d6', ms=4)
    ax2.annotate(f"máx. {k['mu_max']:.0f} em {_fmt(k['B_mu_max_T'], 2)} T", (k['B_mu_max_T'], k['mu_max']),
                 textcoords='offset points', xytext=(4, 4), fontsize=5.5, color='#2a78d6')
    ax2.set_ylabel('$\\mu_{eff}$', color='#2a78d6')
    ax2.spines['right'].set_visible(True)
    ax2.grid(False)
    ax2.set_ylim(0, max(MU_LABEL, k['mu_max']) * 1.15)
    for a in (ax, axh):
        a.axvline(k['B_knee_T'], color='#52514e', ls=':', lw=0.9)
    ax.text(k['B_knee_T'], ax.get_ylim()[1] * 0.97, f" joelho {_fmt(k['B_knee_T'], 2)} T", fontsize=5.5, va='top')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axh.legend(h1 + h2, l1 + l2, frameon=False, loc='upper right', fontsize=5.2)
    c = 0.5 * (edges[1:] + edges[:-1])
    axh.bar(c, 100 * hw / hw.sum(), width=np.diff(edges), color='#2a78d6', edgecolor='none')
    axh.set(xlabel='|B| [T]', ylabel='% da área\nde aço')
    axh.set_xlim(0, edges[-1])
    return fig


def _fig_f7(tm):
    import matplotlib.pyplot as plt
    fp, inf = tm['femm_preproc'], tm['inference']['runs']
    solve = fp['femm_ms']['solve']['median']
    xhw = fp['pre_stage_ms']['x_hw']['median']
    labels, segs = ['FEMM (mi_analyze)'], [(solve, 0, 0, 0)]
    for a in ARCHS:
        pre = fp['pre_total_ms'][a]['median']
        i1 = inf[f'{a}_mae']['batch1']['ms']['median']
        labels.append(_short(a).replace('\n', ''))
        segs.append((0, xhw, pre - xhw, i1))
    fig, ax = plt.subplots(figsize=(W2, 5.5 * CM))
    y = np.arange(len(labels))[::-1]
    cols = ['#52514e', '#2a78d6', '#9bc0ef', '#eb6834']
    names = ['solução FEMM', 'entrada em grade (x_hw, trifinder)', 'demais pré-processamentos', 'inferência (lote 1)']
    left = np.zeros(len(labels))
    for k in range(4):
        v = np.array([s[k] for s in segs])
        ax.barh(y, v, left=left, color=cols[k], label=names[k], height=0.6, zorder=3)
        left += v
    for yy, tot in zip(y, left):
        ax.text(tot, yy, f' {tot:.0f} ms', va='center', fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('tempo por amostra (mediana) [ms]')
    ax.set_xlim(0, left.max() * 1.12)
    ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, ncol=2, loc='lower right', fontsize=6)
    return fig


def run_render():
    _style()
    DAT.mkdir(parents=True, exist_ok=True)
    run_dirs = _run_dirs()
    ev = _j(SRC / 'eval_smooth.json')
    tt = _j(SRC / 'train_table.json')
    rg = _j(SRC / 'regions.json')
    sat = _j(SRC / 'saturation.json')
    tm = _j(SRC / 'timing.json')
    nb = _j(SRC / 'check_nodal_b_femm.json')
    rm = _j(SRC / 'check_analyze_remesh.json')
    kn = _j(DAT / 'saturacao_joelho.json')
    cp = _j(DAT / 'campos.json')
    md = ['# Resultados do artigo v1', '',
          'Gerado por `python -m scripts.artigo_v1 render`. Fontes relativas a '
          '`resultados/BLDC_138x276/`. Conjunto de **teste** (split 60/30/10 de `split.json`, '
          'idêntico nas 10 runs). PENDENTE = arquivo de origem ausente.', '']

    # ---- L1 ----
    md += ['## L1. Normalização', '']
    if ev:
        g = ev['runs']['FNO2d_mse']
        md += [f"- $B_{{ref}}$ (superfície) = **{g['surface'][SUBSET]['B_ref']:.4f} T** "
               f"({ev['n_samples']} amostras de teste).",
               '- Definição (`src/postprocess/metrics.py::Accum.report`): '
               '$B_{ref} = \\sqrt{\\int_\\Omega |B|^2_{FEMM}\\,dA / \\int_\\Omega dA}$, |B| do FEMM '
               'suavizado (P1 descontínuo), integrais acumuladas sobre todos os pontos de quadratura de '
               'todas as amostras do subconjunto (não é média de médias); o mesmo para todos os modelos e pisos.',
               f"- O E_pt (erro nodal, secundário) usa outro $B_{{ref}}$: o RMS nodal ponderado pela área "
               f"dual, **{g['nodal'][SUBSET]['B_ref']:.4f} T**.",
               '- Fonte: `dados/eval_smooth.json`.', '']
    else:
        md += ['- PENDENTE: `dados/eval_smooth.json` ausente.', '']

    # ---- L2 ----
    md += ['## L2. Validação do B nodal', '']
    if nb:
        t = nb['total']
        md += [f"- {nb['n_samples']} amostras de teste "
               f"({', '.join(str(s['sample_idx']) for s in nb['samples'])}); "
               f"**{t['interface']['n_nodes'] + t['interior']['n_nodes']} nós**: "
               f"{t['interface']['n_nodes']} de interface ({t['interface']['n_points']} pontos elemento-vértice) "
               f"e {t['interior']['n_nodes']} de interior ({t['interior']['n_points']} pontos).",
               f"- Porte da suavização × `mo_getb` (mesmo ponto e elemento): máx. "
               f"{t['interface']['porte_vs_femm_max_T']:.1e} T (interface), {t['interior']['porte_vs_femm_max_T']:.1e} T (interior).",
               f"- `node_y` (alvo nodal) × média dos `mo_getb` dos elementos do material do nó: máx. "
               f"{t['interface']['node_y_vs_femm_max_T']:.1e} T / {100 * t['interface']['node_y_vs_femm_max_rel']:.1e} % (interface), "
               f"{t['interior']['node_y_vs_femm_max_T']:.1e} T / {100 * t['interior']['node_y_vs_femm_max_rel']:.1e} % (interior); "
               f"resíduo do deslocamento do ponto (eps = {nb['eps']:g} do caminho nó→centróide) e do float32 do alvo.",
               '- Interface = nó com elementos incidentes de mais de um material (rótulo do bloco).',
               '- Fonte: `dados/check_nodal_b_femm.json` (`python -m tests.check_nodal_b_femm`).', '']
    else:
        md += ['- PENDENTE: `dados/check_nodal_b_femm.json` ausente.', '']

    # ---- L3 ----
    md += ['## L3. Tabela principal (teste)', '']
    if ev:
        body, mdt = [], ['| Arquitetura | Repr. | Perda | Época best | L1 (%) | L2 (%) | E_pt (%) |',
                         '|---|---|---|---|---|---|---|']
        for a in ARCHS:
            for l in LOSSES:
                r = ev['runs'][f'{a}_{l}']
                s, n = r['surface'][SUBSET], r['nodal'][SUBSET]
                cells = [_tex(a), REPR[a], l.upper(), str(r['best_epoch']), _fmt(s['L1_area_pct']),
                         _fmt(s['L2_area_pct']), _fmt(n['MAE_pt_pct'])]
                body.append(cells)
                mdt.append(f"| {a} | {REPR[a]} | {l.upper()} | {r['best_epoch']} | {_fmt(s['L1_area_pct'])} | "
                           f"{_fmt(s['L2_area_pct'])} | {_fmt(n['MAE_pt_pct'])} |")
        body.append(r'\midrule')
        fs, fn = ev['floor_surface'], ev['floor_nodal'][SUBSET]
        for lab, kd, ept in (('Piso da grade', 'grade', fn['MAE_pt_pct']), ('Piso nodal', 'P1_nos', None),
                             ('Piso por elemento', 'P0_elem', None)):
            s = fs[kd][SUBSET]
            rep = {'grade': 'grade', 'P1_nos': 'P1', 'P0_elem': 'P0'}[kd]
            body.append([lab, rep, '--', '--', _fmt(s['L1_area_pct']), _fmt(s['L2_area_pct']), _fmt(ept)])
            mdt.append(f"| {lab} | {rep} | -- | -- | {_fmt(s['L1_area_pct'])} | {_fmt(s['L2_area_pct'])} | {_fmt(ept)} |")
        _write_tex('tab_principal', 'lllrrrr',
                   [r'Arquitetura & Repr. & Perda & Época & $L1_A$ (\%) & $L2_A$ (\%) & $E_{pt}$ (\%) \\'],
                   body, 'dados/eval_smooth.json',
                   (f"teste, {ev['n_samples']} amostras; L1/L2 em % do B_ref de superfície; "
                    'E_pt = MAE_pt nos vértices em % do B_ref nodal (área dual)',))
        md += mdt + ['',
                     '- L1/L2: erro de superfície de |B| (superfície nativa: grade por interseção célula ∩ '
                     'triângulo; P1 dos valores nodais; P0 por elemento) × FEMM P1 descontínuo, % de $B_{ref}$.',
                     '- E_pt = `MAE_pt_pct` do eval (média simples de $||B|_{mod}-|B|_{ref}|$ nos vértices, alvo '
                     '`node_y`; FNO2d interpolado nos nós; Elem levada aos nós por `elem_to_nodes`), em % do '
                     '$B_{ref}$ nodal. O E_pt de piso existe só para a grade (`floor_nodal`: `y_hw` do gabarito '
                     'interpolado nos nós), como pedido.',
                     '- Pisos: o gabarito levado a cada representação (`_floor_points`): grade = `y_hw` (B '
                     'suavizado no centro do pixel) constante na célula; P1 = `node_y` dos vértices; P0 = B '
                     'suavizado no centróide (alvo da Elem).',
                     '- Fonte: `dados/eval_smooth.json`; tabela `tabelas/tab_principal.tex`.', '']
    else:
        md += ['- PENDENTE: `dados/eval_smooth.json` ausente.', '']

    # ---- L4 ----
    md += ['## L4. Treinamento conjunto × base congelada', '']
    stab = {f'{a}_{l}': _stability(run_dirs[(a, l)]) for a in ARCHS for l in LOSSES}
    (DAT / 'estabilidade.json').write_text(json.dumps(stab, indent=2))
    if ev:
        for l in LOSSES:
            cmp_ = [(a, ev['runs'][f'{a}_{l}']['surface'][SUBSET]) for a in
                    ('GNN_PostBase', 'FNO_GNN', 'FNO_BipartiteGNN', 'FNO_BipartiteGNN_Elem')]
            order = sorted(cmp_, key=lambda x: x[1]['L1_area_pct'])
            md.append(f"- {l.upper()}: ordem por L1 (menor primeiro): " + '; '.join(
                f"{a} {_fmt(s['L1_area_pct'])} / {_fmt(s['L2_area_pct'])} %" for a, s in order)
                + f" (L1 / L2). À frente em L1: **{order[0][0]}**; em L2: "
                  f"**{min(cmp_, key=lambda x: x[1]['L2_area_pct'])[0]}**.")
    md += ['', '| Run | parada | última ép. | best ép. | teto | divergiu | pico máx. | rugosidade | osc. forte |',
           '|---|---|---|---|---|---|---|---|---|']
    for k, s in stab.items():
        md.append(f"| {k} | {s['stop_reason']} | {s['last_epoch']} | {s['best_epoch']} | "
                  f"{'sim' if s['hit_ceiling'] else 'não'} | {'sim' if s['diverged'] else 'não'} | "
                  f"{_fmt(s['spike_max'])} | {_fmt(s['roughness'], 3)} | {'sim' if s['strong_oscillation'] else 'não'} |")
    md += ['', '- Definições (deste script, sobre `val_loss` do `epochs.csv`): teto = `stop_reason == n_epochs`; '
           'divergência = valor não finito ou final > 10× o mínimo; pico máx. = $\\max_t v_t/\\min_{s<t} v_s$ '
           '($t \\ge 20$); rugosidade = mediana de $|\\Delta \\ln v|$ entre épocas ($t \\ge 20$); '
           'oscilação forte = pico máx. > 2.']
    for l in LOSSES:
        c = _j(run_dirs[('GNN_PostBase', l)] / 'config.json')
        pb = c.get('postbase_base', {})
        md.append(f"- GNN_PostBase {l}: base `{pb.get('base_run_dir')}` (perda da base **{pb.get('base_loss')}**, "
                  f"checkpoint {pb.get('base_checkpoint')} ép. {pb.get('base_checkpoint_epoch')}) — "
                  f"{'mesma perda: OK' if pb.get('base_loss') == l else 'PERDA DIFERENTE'}.")
    md += ['- Fontes: `dados/eval_smooth.json`, `data/logs/BLDC_138x276/*/run_*/{epochs.csv,run_summary.json,config.json}`, '
           '`artigo_v1/dados/estabilidade.json`.', '']

    # ---- L5 ----
    md += ['## L5. Hiperparâmetros e custo de treino', '']
    if tt:
        body, mdt = [], ['| Run | parâmetros | treináveis | lr | s/época (média) | s/época (mediana) | pico GPU (GiB) |',
                         '|---|---|---|---|---|---|---|']
        for a in ARCHS:
            for l in LOSSES:
                r = tt['runs'][f'{a}_{l}']
                lr = _j(run_dirs[(a, l)] / 'config.json')['lr']
                mean_ep = r['sum_epoch_time_s'] / r['n_epochs_csv']
                body.append([_tex(a), l.upper(), f"{r['n_params_real']:,}".replace(',', r'\,'),
                             f"{r['n_params_trainable_real']:,}".replace(',', r'\,'), f'{lr:g}'.replace('.', ','),
                             _fmt(mean_ep, 1), _fmt(r['gpu_peak_mem_gib'], 1)])
                mdt.append(f"| {a}_{l} | {r['n_params_real']:,} | {r['n_params_trainable_real']:,} | {lr:g} | "
                           f"{_fmt(mean_ep, 1)} | {_fmt(r['median_epoch_time_s'], 1)} | {_fmt(r['gpu_peak_mem_gib'], 2)} |")
        hw = tt['runs']['FNO2d_mse']['hardware']
        _write_tex('tab_hiper_custo', 'llrrrrr',
                   [r'Arquitetura & Perda & Parâmetros & Treináveis & $\eta$ & s/época & GPU (GiB) \\'],
                   body, 'dados/train_table.json + config.json',
                   (f"s/época = média (sum_epoch_time_s / n_epochs_csv; treino+validação+métricas); "
                    f"hardware: {hw['gpu_name']} @ {hw['hostname']}",))
        md += mdt + ['', f"- Hardware do treino: {hw['gpu_name']} ({hw['hostname']}, torch {hw['torch_version']}).",
                     f"- Pico de GPU da GNN_PostBase: **{_fmt(tt['runs']['GNN_PostBase_mse']['gpu_peak_mem_gib'], 2)} / "
                     f"{_fmt(tt['runs']['GNN_PostBase_mae']['gpu_peak_mem_gib'], 2)} GiB** (mse / mae): confirma ~34 GiB.",
                     '- Comum a todas: batch 32, Adam, StepLR (passo 100), n_epochs 500, split_seed 12.',
                     '- Fontes: `dados/train_table.json`, `config.json` das runs; `tabelas/tab_hiper_custo.tex`.', '']
    else:
        md += ['- PENDENTE: `dados/train_table.json` ausente.', '']

    # ---- L6 ----
    md += ['## L6. Distribuição espacial do erro', '']
    if rg:
        p = rg['params']
        md += [f"- Limiares do código (iguais aos provisórios): d_int = {p['d_int_mm']:.4f} mm (1 Δr), "
               f"d_canto = {p['d_canto_mm']:.4f} mm (2 Δr), ângulo {p['ang_canto_deg']:g}°; prioridade "
               f"{p['prioridade']}. O código também classifica como canto o vértice com ≥ 3 arestas de interface "
               'ou que toca ≥ 3 materiais. Distâncias exatas (shapely) do triângulo à aresta/vértice. '
               '"interior" agrega o interior de todos os materiais (uma região só).',
               '', '| Run | interior: área / L1 / L2 / fr. erro | interface: área / L1 / L2 / fr. erro | canto: área / L1 / L2 / fr. erro | canto/interface (L1) |',
               '|---|---|---|---|---|']
        body = []
        items = [(f'{a}_{l}', rg['runs'][f'{a}_{l}']) for a in ARCHS for l in LOSSES] + \
                [(f'piso {k}', v) for k, v in rg['floor_surface'].items()]
        for k, v in items:
            cells = []
            for rgn in ('interior', 'interface', 'canto'):
                x = v[rgn]
                cells.append(f"{_fmt(100 * x['area_frac'], 1)} / {_fmt(x['L1_area_pct'])} / {_fmt(x['L2_area_pct'])} / "
                             f"{_fmt(100 * x['L1_contrib_frac'], 1)}")
            ratio = v['canto']['L1_area_pct'] / v['interface']['L1_area_pct']
            md.append(f"| {k} | " + ' | '.join(cells) + f" | {_fmt(ratio)} |")
            body.append([_tex(k)] + [_fmt(v[rgn][m], nd) for rgn in ('interior', 'interface', 'canto')
                                     for m, nd in (('L1_area_pct', 2), ('L2_area_pct', 2))]
                        + [_fmt(100 * v[rgn]['L1_contrib_frac'], 1) for rgn in ('interior', 'interface', 'canto')])
        af = {rgn: 100 * rg['runs']['FNO2d_mse'][rgn]['area_frac'] for rgn in ('interior', 'interface', 'canto')}
        _write_tex('tab_regioes', 'l' + 'r' * 9,
                   [r' & \multicolumn{2}{c}{Interior} & \multicolumn{2}{c}{Interface} & \multicolumn{2}{c}{Canto} & \multicolumn{3}{c}{Fração do erro (\%)} \\',
                    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}',
                    r'Run & $L1_A$ & $L2_A$ & $L1_A$ & $L2_A$ & $L1_A$ & $L2_A$ & int. & interf. & canto \\'],
                   body, 'dados/regions.json',
                   (f"% do B_ref global; fração da área: interior {af['interior']:.1f}%, interface "
                    f"{af['interface']:.1f}%, canto {af['canto']:.1f}%",))
        md += ['', '- Área/fr. erro em %; L1/L2 da região normalizados pela área da região, em % do $B_{ref}$ global.',
               '- Fonte: `dados/regions.json`; `tabelas/tab_regioes.tex`.', '']
    else:
        md += ['- PENDENTE: `dados/regions.json` ausente.', '']

    # ---- L7 ----
    md += ['## L7. Torque', '']
    tq = {l: _torque_analysis(l) for l in LOSSES}
    (DAT / 'torque_analise.json').write_text(json.dumps(tq, indent=2))
    have_tq = any('batch' in tq[l] or 'scatter' in tq[l] for l in LOSSES)
    if have_tq:
        md += ['- Método (`src/postprocess/torque.py::band_torque`): tensor de Maxwell na faixa de ar do '
               'entreferro, B por elemento; referência suavizada = B suavizado do FEMM no centróide com o mesmo '
               'método; referência exata = curl(A) P0 (`gt`); FEMM T22 = `mo_blockintegral(22)`.', '']
        body = []
        for l in LOSSES:
            b = tq[l].get('batch')
            if not b:
                md.append(f'- {l.upper()}: PENDENTE (varredura batch ausente).')
                continue
            md += [f"### {l.upper()} — varredura θe (batch: {sum(v['n'] for v in b['ref_error'].values())} amostras × 16 ângulos)", '',
                   '| Arch | faixa | n | viés médio dT0 (mN·m) | viés vs θe min…max (mN·m) | A/A_ref | Δφ (°) | RMSE rel. (%) |',
                   '|---|---|---|---|---|---|---|---|']
            for a in ARCHS:
                for band, s in b['por_arch'].get(a, {}).items():
                    bv = s['bias_vs_theta_mNm']
                    md.append(f"| {a} | {band} | {s['n']} | {_fmt(s['dT0_mean_mNm'], 1)} ± {_fmt(s['dT0_std_mNm'], 1)} | "
                              f"{_fmt(bv['min'], 1)} … {_fmt(bv['max'], 1)} | {_fmt(s['A_ratio_mean'], 3)} ± "
                              f"{_fmt(s['A_ratio_std'], 3)} | {_fmt(s['dphi_mean_deg'], 1)} | {_fmt(s['rel_median_pct'], 1)} |")
                lo, hi = b['por_arch'].get(a, {}).get('0-2A'), b['por_arch'].get(a, {}).get('8-10A')
                if lo and hi:
                    body.append([_tex(a), l.upper(), _fmt(lo['dT0_mean_mNm'], 1),
                                 f"{_fmt(lo['bias_vs_theta_mNm']['min'], 1)} a {_fmt(lo['bias_vs_theta_mNm']['max'], 1)}",
                                 _fmt(100 * (hi['A_ratio_mean'] - 1), 1), _fmt(lo['rel_median_pct'], 1),
                                 _fmt(hi['rel_median_pct'], 1)])
            md += ['', '| faixa | n | RMSE suav.×exata (mN·m) | rel. (%) | A_suav/A_exata | RMSE suav.×FEMM | RMSE exata×FEMM |',
                   '|---|---|---|---|---|---|---|']
            for band, s in b['ref_error'].items():
                md.append(f"| {band} | {s['n']} | {_fmt(s['rmse_smooth_vs_gt_median_mNm'], 2)} | "
                          f"{_fmt(s['rel_smooth_vs_gt_median_pct'], 2)} | {_fmt(s['A_smooth_over_A_gt_mean'], 4)} | "
                          f"{_fmt(s['rmse_smooth_vs_femm_median_mNm'], 2)} | {_fmt(s['rmse_gt_vs_femm_median_mNm'], 2)} |")
            md.append('')
        for l in LOSSES:
            s = tq[l].get('scatter')
            if s:
                md += [f"- Dispersão {l.upper()} ({s['n']} amostras de teste, θe do CSV): RMSE / viés / R² vs referência suavizada: "
                       + '; '.join(f"{a} {_fmt(v['rmse_mNm'], 1)} / {_fmt(v['bias_mNm'], 1)} mN·m / {_fmt(v['r2'], 4)}"
                                   for a, v in s['vs_smooth'].items())
                       + f". Referência suavizada × exata: RMSE {_fmt(s['ref_smooth_vs_gt']['rmse_mNm'], 2)} mN·m, "
                         f"viés {_fmt(s['ref_smooth_vs_gt']['bias_mNm'], 2)} mN·m (RMS exata {_fmt(s['ref_smooth_vs_gt']['rms_gt_mNm'], 1)} mN·m)."]
        if body:
            _write_tex('tab_torque', 'llrrrrr',
                       [r' & & \multicolumn{2}{c}{0--2 A (viés, mN$\cdot$m)} & 8--10 A & \multicolumn{2}{c}{RMSE rel. (\%)} \\',
                        r'\cmidrule(lr){3-4}\cmidrule(lr){6-7}',
                        r'Arquitetura & Perda & médio & vs. $\theta_e$ & $A/A_{ref}-1$ (\%) & 0--2 A & 8--10 A \\'],
                       body, 'dados/torque/theta_e_batch_<loss>.json (artigo_v1/dados/torque_analise.json)',
                       ('referência = torque com o B suavizado pelo mesmo método (faixa do entreferro)',))
        md += ['- Viés vs θe = mínimo…máximo, sobre os 16 ângulos, da média (nas amostras da faixa) de '
               'T_modelo − T_ref; dT0 = deslocamento médio da fundamental (`curve_row`).',
               '- Fontes: `dados/torque/{theta_e_batch_<loss>.json, scatter_<loss>.json, theta_e_s*_<loss>.json}`; '
               '`artigo_v1/dados/torque_analise.json`; `tabelas/tab_torque.tex`.', '']
    else:
        md += ['- PENDENTE: saídas de `torque` ausentes.', '']

    # ---- L8 ----
    md += ['## L8. Saturação', '']
    if sat:
        k = sat['knee'][0]
        op = sat['operating_points']
        md += [f"- Curva B-H `{sat['bh_curve']['name']}` lida do .ans ({sat['bh_curve']['n_points']} pontos), "
               f"idêntica em todas as {sat['n_samples']} amostras: {sat['bh_identical_all_samples']}; "
               f"reparos de monotonicidade: {sat['n_repair']}.",
               f"- $\\mu_{{eff}}$ máximo = **{k['mu_max']:.0f}** em **{k['B_mu_max_T']:.3f} T**; joelho "
               f"(μ_eff = {100 * k['frac']:.0f}% do máx.) = **{k['B_knee_T']:.3f} T** (H = {k['H_knee_A_m']:.0f} A/m).",
               f"- Pontos de operação do aço (|B| suavizado no centróide, ponderado por área, teste): p5/p50/p95 = "
               f"{op['p5_T']:.3f} / {op['p50_T']:.3f} / {op['p95_T']:.3f} T; μ_eff nesses pontos "
               f"{op['mu_eff_at_p5']:.0f} / {op['mu_eff_at_p50']:.0f} / {op['mu_eff_at_p95']:.0f}.",
               f"- Elementos acima do último ponto da curva ({sat['above_last_bh_point']['B_last_T']} T): "
               f"{sat['above_last_bh_point']['n_elements']}."]
        if kn:
            md.append(f"- Fração da área de aço acima do joelho: **{100 * kn['frac_area_above_knee']:.2f}%** "
                      f"(acima do pico de μ_eff: {100 * kn['frac_area_above_mu_max']:.1f}%); por amostra: mediana "
                      f"{100 * kn['per_sample_frac_above_knee']['p50']:.2f}%, p95 {100 * kn['per_sample_frac_above_knee']['p95']:.2f}%, "
                      f"máx. {100 * kn['per_sample_frac_above_knee']['max']:.2f}%; "
                      f"{kn['n_samples_with_any_above_knee']} de {kn['n_samples']} amostras com algum elemento acima. "
                      'Fonte: `artigo_v1/dados/saturacao_joelho.json`.')
        else:
            md.append('- PENDENTE: fração acima do joelho (`artigo_v1/dados/saturacao_joelho.json`).')
        if 'femm_check' in sat:
            fc = sat['femm_check']
            md.append(f"- Conferência × FEMM (`mo_getpointvalues`, amostra {fc['sample_idx']}, {fc['n_points']} pontos): "
                      f"diferença relativa de μ máx. {fc['max_rel_diff']:.1e}, mediana {fc['median_rel_diff']:.1e}.")
        md += ['- O μ_r = 5000 da entrada é o rótulo do material (`motor_constants.PERMEABILITY[\'iron_1008\']`), '
               'não a permeabilidade física: o FEMM resolve com a curva B-H não linear.',
               '- Leitura qualitativa do mapa de erro nas regiões saturadas: ver F2 (corrente alta) × F6b — a '
               'redigir pelo autor a partir das figuras (sem estatística, como pedido).',
               '- Fonte: `dados/saturation.json`.', '']
    else:
        md += ['- PENDENTE: `dados/saturation.json` ausente.', '']

    # ---- L9 ----
    md += ['## L9. Custo computacional', '']
    if tm and tm.get('femm_preproc') and tm.get('inference'):
        fp, inf = tm['femm_preproc'], tm['inference']
        solve = fp['femm_ms']['solve']['median']
        xhw = fp['pre_stage_ms']['x_hw']['median']
        body = []
        md += [f"- FEMM `mi_analyze` (mediana, {fp['n_samples']} amostras de teste): **{solve:.0f} ms** "
               f"(`mi_createmesh` à parte: {fp['femm_ms']['mesh']['median']:.0f} ms); malha = raw em "
               f"{fp['n_mesh_equal_raw']}/{fp['n_samples']}; entradas = parsers: {fp['all_preproc_equal_parsers']}.",
               f"- Lote unitário: amostra {inf['batch1_sample_idx']} ({inf['batch1_n_nodes']} nós, mediana de nós entre "
               f"{inf['n_samples_cycled']} do 1º chunk de teste); {inf['n_warmup']} aquecimentos, {inf['n_reps']} repetições.",
               '', '| Run | entrada em grade (ms) | demais pré-proc. (ms) | inferência (ms) | acel. inferência | acel. total |',
               '|---|---|---|---|---|---|']
        for a in ARCHS:
            for l in LOSSES:
                k_ = f'{a}_{l}'
                pre = fp['pre_total_ms'][a]['median']
                i1 = inf['runs'][k_]['batch1']['ms']['median']
                md.append(f"| {k_} | {_fmt(xhw, 0)} | {_fmt(pre - xhw, 0)} | {_fmt(i1, 2)} | {_fmt(solve / i1, 0)} | "
                          f"{_fmt(solve / (pre + i1), 1)} |")
                body.append([_tex(a), l.upper(), _fmt(xhw, 0), _fmt(pre - xhw, 0), _fmt(i1, 2),
                             _fmt(solve / i1, 0), _fmt(solve / (pre + i1), 1)])
        hw = tm.get('hardware', {})
        _write_tex('tab_custo', 'llrrrrr',
                   [r'Arquitetura & Perda & Grade (ms) & Demais (ms) & Inferência (ms) & Acel. inf. & Acel. total \\'],
                   body, 'dados/timing.json',
                   (f"FEMM mi_analyze = {solve:.0f} ms (mediana); hardware: {hw.get('cpu')} / {hw.get('gpu')}",
                    'demais = mediana do pré-processamento total da arch − mediana do x_hw (inclui codificação CPU→GPU)'))
        md += ['', '- Entrada em grade = etapa `x_hw` (`_build_trifinder` + `_grid_const_per_element`); demais = '
               'mediana do pré-processamento total da arch (`pre_total_ms`, etapas de `ARCH_STAGES` + codificação) − '
               'mediana do `x_hw`. Acel. inferência = FEMM/inferência; total = FEMM/(pré-proc. + inferência).',
               f"- Hardware: {hw.get('cpu')}, {hw.get('gpu')}, FEMM {hw.get('femm')}.",
               '- Fonte: `dados/timing.json`; `tabelas/tab_custo.tex`.']
    else:
        md += ['- PENDENTE: `dados/timing.json` incompleto.']
    if rm:
        m = rm['mediana_ms']
        rem = m['t_analyze_apos_malha_s'] > m['t_analyze_sem_malha_s'] - 0.5 * m['t_createmesh_s']
        md += [f"- Remalhamento (`tests/check_analyze_remesh.py`, amostra {rm['sample_idx']}, {rm['reps']} repetições, "
               f"medianas): createmesh {m['t_createmesh_s']:.0f} ms; analyze sem malha prévia {m['t_analyze_sem_malha_s']:.0f} ms; "
               f"analyze após createmesh {m['t_analyze_apos_malha_s']:.0f} ms → "
               + ('**o `mi_analyze` remalha** (pré-malhar não reduz o tempo do analyze); o tempo da solução FEMM '
                  'inclui a malha.' if rem else '**o `mi_analyze` não remalha**.')
               + ' O FEMM apaga .poly/.node/.ele após o analyze, então a evidência é só pelo tempo; malha A = B = raw.',
               '- Fonte: `dados/check_analyze_remesh.json`.', '']
    else:
        md += ['- PENDENTE: teste de remalhamento.', '']

    # ---- figuras ----
    md += ['## Figuras', '']
    if ev:
        _save(_fig_f1(ev), 'F1_erro_pisos')
        md.append('- F1 `figuras/F1_erro_pisos.pdf` (dupla coluna): novo layout pedido; o `fig_bars` original '
                  '(10 runs, MSE hachurado) continua em `plots/2_barras_erro` via `postprocess figures`.')
    if rg:
        _save(_fig_f3b(rg), 'F3b_regioes_barras')
    _save(_fig_f4(run_dirs), 'F4_curvas_treino')
    md.append('- F4 `figuras/F4_curvas_treino.pdf`: perda de treino (tracejada) e validação (cheia), z-score, por run; ○ = best.pth.')
    sc = {l: _j(SRC / 'torque' / f'scatter_{l}.json') for l in LOSSES}
    if any(sc.values()):
        _save(_fig_f5a(sc), 'F5a_torque_dispersao')
    for l in LOSSES:
        files = sorted(SRC.joinpath('torque').glob(f'theta_e_s*_{l}.json'))
        if files:
            th = [(int(f.stem.split('_')[2][1:]), _j(f)) for f in files]
            th.sort(key=lambda x: x[1]['current_amplitude_A'])
            _save(_fig_f5b(th, l), f'F5b_torque_theta_e_{l}')
    if sat:
        _save(_fig_f6a(sat), 'F6a_saturacao_curva_BH')
    if tm and tm.get('femm_preproc') and tm.get('inference'):
        _save(_fig_f7(tm), 'F7_custo')
    if cp:
        md.append('- F2 `figuras/F2_campo_<caso>_<perda>.pdf`: casos ' + '; '.join(
            f"{c} = amostra {v['sample_idx']}" for c, v in cp['cases'].items())
                  + '. Mediano/pior caso = P50/máximo do L1_A de `FNO_BipartiteGNN_mae` (regra `_pick_by_pct` do '
                    'pós-processamento); corrente alta = maior amplitude de corrente do teste. Valores por triângulo '
                    '= média, nos pontos de quadratura da superfície nativa do modelo, de |B| e de |B|_mod − |B|_ref.')
    md += ['- F3 `F3a_regioes_mapa.pdf` + `F3b_regioes_barras.pdf`; F5 `F5a_torque_dispersao.pdf`, '
           '`F5b_torque_theta_e_<perda>.pdf`; F6 `F6a_saturacao_curva_BH.pdf`, `F6b_saturacao_mapa_aco.pdf`; '
           'F7 `F7_custo.pdf` (inferência do run mae de cada arch).', '']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'resultados.md').write_text('\n'.join(md), encoding='utf-8')
    print(f"resultados.md em {OUT / 'resultados.md'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='figuras, tabelas e resultados do artigo v1')
    ap.add_argument('cmd', choices=('joelho', 'campos', 'render', 'all'))
    ap.add_argument('--workers', type=int, default=8, help='joelho: processos')
    a = ap.parse_args()
    if a.cmd in ('joelho', 'all'):
        run_joelho(a.workers)
    if a.cmd in ('campos', 'all'):
        run_campos()
    if a.cmd in ('render', 'all'):
        run_render()
