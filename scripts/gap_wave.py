"""
Erro nas componentes (B_r, B_θ) da densidade de fluxo no arco do meio do entreferro, no
conjunto de teste (608 amostras), para as 8 runs do artigo.

    python -m scripts.gap_wave [--workers 4] [--n N] [--step-deg 0.05]
    python -m scripts.gap_wave --plot IDX     # conferência: B_r, B_θ no arco de 1 amostra
    python -m scripts.gap_wave --table        # só gera tab_gap_wave.tex do JSON

Nada de inferência: predições de data/temp/inference (mesmos best.pth do regions.json e do
regions_gap_fe.json); r_1, r_2 por amostra de dados/regions_gap_fe.json.

Arco: r_m = (r_1 + r_2)/2, θ_k = ANG_1 + (k + ½)Δθ, k = 0..n-1, cobrindo o setor simulado
[ANG_1, ANG_2) (a costura periódica não é contada 2×). Cada ponto é localizado no triângulo do
FEMM que o contém (baricêntricas); todos devem ser vacuum (MATERIAL_ID 1).

Campo no ponto (mesmas representações das superfícies do regions; as saídas são polares na
base do próprio ponto do modelo, levadas a (x, y) nesse ângulo antes de interpolar):
  referência   B suavizado do FEMM P1 (cartesiano) no triângulo -- igual para as 8 runs;
  FNO2d        constante na célula da grade (lados retos, como grid_mesh_overlay), girada
               no ângulo do centro do pixel;
  FNO_GNN      P1 dos valores nodais, cada nó girado no seu ângulo;
  Elem         P0 do elemento, girado no ângulo do centróide;
  Split        P1_mat: as 3 cópias do material do elemento (split_corner_values), cada uma
               girada no ângulo do seu vértice original.
Projeção: B_r = B_x cos θ + B_y sen θ, B_θ = −B_x sen θ + B_y cos θ.
Métrica: ε = média no arco de |B_pred − B_ref| em % de B_ref (o B_ref global do regions.json,
0,6829 T); por run: média e P95 dos valores por amostra.

Saída: dados/gap_wave.json, dados/tab_gap_wave.tex (+ checkpoint em data/temp/gap_wave/).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).resolve().parents[1]
DADOS    = ROOT / 'resultados' / 'BLDC_138x276' / 'dados'
OUT_JSON = DADOS / 'gap_wave.json'
OUT_TEX  = DADOS / 'tab_gap_wave.tex'
CKPT     = ROOT / 'data' / 'temp' / 'gap_wave'
RUN_KEYS = ('FNO2d_mse', 'FNO2d_mae', 'FNO_GNN_mse', 'FNO_GNN_mae',
            'FNO_BipartiteGNN_Elem_mse', 'FNO_BipartiteGNN_Elem_mae',
            'FNO_SplitGNN_mse', 'FNO_SplitGNN_mae')
TAB_ROWS = (('FNO2d', 'FNO2d'), ('FNO\\_GNN', 'FNO_GNN'),
            ('FNO\\_Bipartite', 'FNO_BipartiteGNN_Elem'), ('FNO\\_SplitGNN', 'FNO_SplitGNN'))

_G = {}


def _init(pred_dirs, radii, step_deg):
    _G.update(pred=pred_dirs, radii=radii, step=step_deg)


def _rot(br, bt, th):
    """(Br, Bθ) na base do ângulo th -> (Bx, By)."""
    c, s = np.cos(th), np.sin(th)
    return br * c - bt * s, br * s + bt * c


def _locate(xy, tri, pts):
    """Triângulo que contém cada ponto e baricêntricas [P,3]; candidatos = triângulos cuja
    faixa radial (com a flecha das arestas) cobre o raio dos pontos."""
    P = xy[tri]
    r = np.hypot(P[..., 0], P[..., 1])
    edge = np.linalg.norm(P - np.roll(P, 1, axis=1), axis=-1).max(axis=1)
    rp = np.hypot(pts[:, 0], pts[:, 1])
    tol = edge ** 2 / (8 * r.min(1)) + 1e-6
    cand = np.flatnonzero((r.min(1) - tol <= rp.max()) & (r.max(1) + tol >= rp.min()))
    A, B, C = (P[cand, k][None] for k in range(3))                   # [1,Nc,2]
    q = pts[:, None, :]
    det = (B[..., 0] - A[..., 0]) * (C[..., 1] - A[..., 1]) - (C[..., 0] - A[..., 0]) * (B[..., 1] - A[..., 1])
    l1 = ((q[..., 0] - A[..., 0]) * (C[..., 1] - A[..., 1]) - (C[..., 0] - A[..., 0]) * (q[..., 1] - A[..., 1])) / det
    l2 = ((B[..., 0] - A[..., 0]) * (q[..., 1] - A[..., 1]) - (q[..., 0] - A[..., 0]) * (B[..., 1] - A[..., 1])) / det
    lam = np.stack([1 - l1 - l2, l1, l2], axis=-1)                    # [P,Nc,3]
    best = lam.min(-1).argmax(1)
    lam_b = lam[np.arange(len(pts)), best]
    return cand[best], lam_b, lam_b.min(1)


def _grid_cell(pts, r_in, r_ext, n_r, n_a, a1, a2):
    """Célula (row-major) da grade de lados retos (_cell_corners) que contém cada ponto:
    coluna pelo ângulo (lados radiais exatos); linha pela distância à corda de cada raio da
    grade, r·cos(θ − a_meio) ≥ R_i·cos(Δa/2)."""
    da = (a2 - a1) / n_a
    th = np.arctan2(pts[:, 1], pts[:, 0])
    j = np.floor((th - a1) / da).astype(np.int64)
    assert ((j >= 0) & (j < n_a)).all()
    a_mid = a1 + (j + 0.5) * da
    proj = np.hypot(pts[:, 0], pts[:, 1]) * np.cos(th - a_mid)
    R = r_in + np.arange(1, n_r) * (r_ext - r_in) / n_r            # raios internos da grade
    i = (proj[:, None] >= R[None] * np.cos(da / 2)).sum(1)
    return i * n_a + j, a_mid


def arc_fields(idx, pred_dirs, r1, r2, step_deg):
    """Campos no arco de 1 amostra: dict(theta [P], ref [P,2] (Br,Bθ), runs {k: [P,2]},
    polar_interp {k: [P,2]} (variante das superfícies P1: interpola as componentes polares
    direto, sem girar -- só conferência), checks)."""
    from src.data_gen.motor_constants import MATERIAL_ID
    from src.data_gen.parsers.ans_b_smoothing import element_b, nodal_b
    from src.data_gen.parsers.ans_parsing import _parse_label_materials, read_ans
    from src.data_gen.parsers.split_layout import parse_split_smooth
    from src.postprocess.fields import split_corner_values
    from src.postprocess.samples import ANG_1, ANG_2, N_A, N_R, design_rows, sample_path
    row = design_rows()[idx]
    r_in, r_ext = float(row['inner_diameter [mm]']) / 2, float(row['outer_diameter [mm]']) / 2
    path = sample_path(idx)
    ans = read_ans(path)
    m = ans.mesh
    xy = np.stack([m.x, m.y], axis=1).astype(np.float64)
    tri = ans.elems[:, :3].astype(np.int64)
    assert np.array_equal(m.p, tri)
    lmat, _, _ = _parse_label_materials(ans.lines)
    emat = lmat[ans.elems[:, 3]]
    rm = 0.5 * (r1 + r2)
    n = int(round((ANG_2 - ANG_1) / step_deg))
    th = np.deg2rad(ANG_1 + (np.arange(n) + 0.5) * step_deg)
    pts = np.stack([rm * np.cos(th), rm * np.sin(th)], axis=1)
    t, lam, lam_min = _locate(xy, tri, pts)
    mat_pts = emat[t]
    B1, B2 = element_b(m)
    b1, b2 = nodal_b(m, B1, B2)                                     # [M,3] P1 descontínuo
    c, s = np.cos(th), np.sin(th)

    def proj(bx, by):
        return np.stack([bx * c + by * s, -bx * s + by * c], axis=1)

    ref = proj((lam * b1[t]).sum(1), (lam * b2[t]).sum(1))
    ang_node = np.arctan2(xy[:, 1], xy[:, 0])
    cen = xy[tri].mean(1)
    ang_cen = np.arctan2(cen[:, 1], cen[:, 0])
    cell, a_mid = _grid_cell(pts, r_in, r_ext, N_R, N_A, np.deg2rad(ANG_1), np.deg2rad(ANG_2))
    split = None
    runs, polar = {}, {}
    for k in RUN_KEYS:
        arch = k.rsplit('_', 1)[0]
        with np.load(Path(pred_dirs[k]) / f'sample_{idx:06d}.npz') as z:
            p = {nm: z[nm].astype(np.float64) for nm in ('out_hw', 'y', 'y_base') if nm in z.files}
        if arch == 'FNO2d':
            hw = p['out_hw'].reshape(2, -1)[:, cell]
            runs[k] = proj(*_rot(hw[0], hw[1], a_mid))
        elif arch == 'FNO_GNN':
            y = p['y']
            assert len(y) == len(xy)
            bx, by = _rot(y[:, 0], y[:, 1], ang_node)
            V = tri[t]
            runs[k] = proj((lam * bx[V]).sum(1), (lam * by[V]).sum(1))
            polar[k] = np.stack([(lam * y[V, 0]).sum(1), (lam * y[V, 1]).sum(1)], axis=1)
        elif arch == 'FNO_BipartiteGNN_Elem':
            y = p['y']
            assert len(y) == len(tri)
            runs[k] = proj(*_rot(y[t, 0], y[t, 1], ang_cen[t]))
        else:
            if split is None:
                split = parse_split_smooth(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
                                           n_r=N_R, n_a=N_A, ans=ans)
            Vc = split_corner_values(split, p['y_base'])[t]          # [P,3,2]
            bx, by = _rot(Vc[..., 0], Vc[..., 1], ang_node[tri[t]])
            runs[k] = proj((lam * bx).sum(1), (lam * by).sum(1))
            polar[k] = np.stack([(lam * Vc[..., 0]).sum(1), (lam * Vc[..., 1]).sum(1)], axis=1)
    _, cnt = np.unique(t, return_counts=True)
    checks = dict(n_pts=n, n_nonvac=int((mat_pts != MATERIAL_ID['vacuum']).sum()),
                  nonvac_mats=sorted(set(map(int, mat_pts[mat_pts != MATERIAL_ID['vacuum']]))),
                  lam_min=float(lam_min.min()), n_elems=int(len(cnt)),
                  pts_per_elem_min=int(cnt.min()), pts_per_elem_med=float(np.median(cnt)),
                  n_cells=int(len(np.unique(cell))))
    return dict(theta=th, ref=ref, runs=runs, polar=polar, checks=checks, rm=rm)


def _sample(idx):
    out = CKPT / f'sample_{idx:06d}.npz'
    if out.exists():
        return idx
    r1, r2 = _G['radii'][idx]
    F = arc_fields(idx, _G['pred'], r1, r2, _G['step'])
    E = np.array([np.abs(F['runs'][k] - F['ref']).mean(0) for k in RUN_KEYS])      # [8,2] T
    Ep = np.array([np.abs(F['polar'][k] - F['ref']).mean(0) if k in F['polar'] else [np.nan] * 2
                   for k in RUN_KEYS])
    ch = F['checks']
    tmp = out.with_name(f'{out.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, E=E, E_polar=Ep, r1=r1, r2=r2, rm=F['rm'],
             ref_mean_abs=np.abs(F['ref']).mean(0),
             **{k: np.asarray(v) for k, v in ch.items() if k != 'nonvac_mats'},
             nonvac_mats=np.asarray(ch['nonvac_mats'], np.int64))
    os.replace(tmp, out)
    return idx


def _setup():
    reg = json.loads((DADOS / 'regions.json').read_text(encoding='utf-8'))
    gfe = json.loads((DADOS / 'regions_gap_fe.json').read_text(encoding='utf-8'))
    from src.postprocess.chunk_inference import pred_dir
    pred = {k: str(pred_dir(ROOT / reg['runs'][k]['run_dir'].replace('\\', '/'))) for k in RUN_KEYS}
    for k in RUN_KEYS:     # mesmas runs/épocas do regions_gap_fe
        assert gfe['runs'][k]['run'] == reg['runs'][k]['run']
        assert gfe['runs'][k]['best_epoch'] == reg['runs'][k]['best_epoch']
    ps = gfe['per_sample']
    assert ps['sample_idx'] == reg['sample_idx']
    radii = {i: (a, b) for i, a, b in zip(ps['sample_idx'], ps['r1_mm'], ps['r2_mm'])}
    return reg, gfe, pred, radii


def _br(x, nd=2):
    return f'{x:.{nd}f}'.replace('.', ',')


def render_table():
    res = json.loads(OUT_JSON.read_text(encoding='utf-8'))
    L = [r'\begin{tabular}{ll cccc}', r'\toprule',
         r'Arquitetura & Perda & $\varepsilon_{r}$ (\%) & $\varepsilon_{\theta}$ (\%) & '
         r'$P95_{r}$ (\%) & $P95_{\theta}$ (\%) \\', r'\midrule']
    for a, (tex, arch) in enumerate(TAB_ROWS):
        for j, loss in enumerate(('mse', 'mae')):
            g = res['runs'][f'{arch}_{loss}']
            L.append(f"{tex if j == 0 else ''} & {loss.upper()} & {_br(g['eps_r_mean_pct'])} & "
                     f"{_br(g['eps_t_mean_pct'])} & {_br(g['eps_r_p95_pct'])} & {_br(g['eps_t_p95_pct'])} \\\\")
        if a < len(TAB_ROWS) - 1:
            L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}']
    OUT_TEX.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    print(f'-> {OUT_TEX}')


def plot_sample(idx, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, _, pred, radii = _setup()
    r1, r2 = radii[idx]
    F = arc_fields(idx, pred, r1, r2, 0.05)
    th = np.rad2deg(F['theta'])
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    cols = plt.cm.tab10(np.arange(8))
    for c, nm in enumerate(('B_r', 'B_θ')):
        ax[c].plot(th, F['ref'][:, c], 'k', lw=2.2, label='FEMM (ref.)')
        for i, k in enumerate(RUN_KEYS):
            ax[c].plot(th, F['runs'][k][:, c], color=cols[i], lw=0.8,
                       ls='-' if k.endswith('mae') else '--', label=k)
        ax[c].set_ylabel(f'{nm} (T)')
        ax[c].grid(alpha=0.3)
    ax[1].set_xlabel('θ (°)')
    ax[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f'amostra {idx}: r_1 = {r1:.3f}, r_2 = {r2:.3f}, r_m = {F["rm"]:.3f} mm '
                 f'(gap {r2 - r1:.3f} mm), Δθ = 0,05°')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f'-> {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--step-deg', type=float, default=0.05)
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--plot', type=int, default=None)
    ap.add_argument('--plot-out', default=None)
    a = ap.parse_args()
    if a.table:
        return render_table()
    if a.plot is not None:
        return plot_sample(a.plot, a.plot_out or str(CKPT / f'arc_sample_{a.plot:06d}.png'))
    reg, gfe, pred, radii = _setup()
    idxs = gfe['per_sample']['sample_idx'][:a.n]
    CKPT.mkdir(parents=True, exist_ok=True)
    todo = [i for i in idxs if not (CKPT / f'sample_{i:06d}.npz').exists()]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(pred, radii, a.step_deg)) as ex:
        for n, _ in enumerate(ex.map(_sample, todo), 1):
            if n % 16 == 0 or n == len(todo):
                el = time.perf_counter() - t0
                print(f'{n}/{len(todo)}  {el / 60:.1f} min', flush=True)
    Z = [dict(np.load(CKPT / f'sample_{i:06d}.npz')) for i in idxs]
    nonvac = [(i, int(z['n_nonvac']), z['nonvac_mats'].tolist()) for i, z in zip(idxs, Z) if z['n_nonvac'] > 0]
    print(f'pontos fora do vacuum: {sum(v[1] for v in nonvac)} em {len(nonvac)} amostras')
    for v in nonvac[:30]:
        print('  amostra', v)
    print('λ_min (localização):', min(float(z['lam_min']) for z in Z))
    ppe = np.array([float(z['pts_per_elem_med']) for z in Z])
    print(f"pontos/elemento: mediana por amostra {np.median(ppe):.1f} (min {ppe.min():.1f}); "
          f"mínimo absoluto {min(int(z['pts_per_elem_min']) for z in Z)}; "
          f"elementos no arco {np.median([int(z['n_elems']) for z in Z]):.0f} (mediana)")
    if nonvac:
        print('PARADO: há pontos fora do vacuum.')
        return
    b_ref = float(reg['runs']['FNO2d_mse']['B_ref_global'])
    assert all(abs(reg['runs'][k]['B_ref_global'] / b_ref - 1) < 1e-9 for k in RUN_KEYS)
    E = 100 * np.stack([z['E'] for z in Z]) / b_ref                  # [S,8,2] %
    Ep = 100 * np.stack([z['E_polar'] for z in Z]) / b_ref
    r1 = np.array([float(z['r1']) for z in Z]); r2 = np.array([float(z['r2']) for z in Z])
    rm = np.array([float(z['rm']) for z in Z]); gap = r2 - r1
    res = dict(n_samples=len(idxs), subset='test', B_ref_T=b_ref,
               arco=dict(extensao_deg=[0.0, 120.0], setor='setor periódico simulado (120°), sem repetir a costura',
                         passo_deg=a.step_deg, n_pontos=int(Z[0]['n_pts']),
                         theta='ANG_1 + (k + 1/2)·Δθ', r_m='(r_1 + r_2)/2'),
               definicao=dict(
                   ref='B suavizado do FEMM, P1 cartesiano no triângulo que contém o ponto',
                   FNO2d='constante na célula da grade (lados retos), girada no ângulo do centro do pixel',
                   FNO_GNN='P1 dos valores nodais, cada nó girado no seu ângulo',
                   FNO_BipartiteGNN_Elem='P0 do elemento, girado no ângulo do centróide',
                   FNO_SplitGNN='P1_mat (3 cópias do material do elemento), cada cópia girada no ângulo do vértice',
                   metrica='ε = média no arco de |B_pred − B_ref| em % de B_ref; agregados: média e P95 por amostra'),
               checks=dict(n_pontos_fora_vacuum=0,
                           lam_min=min(float(z['lam_min']) for z in Z),
                           pts_per_elem_median=float(np.median(ppe)),
                           pts_per_elem_min=min(int(z['pts_per_elem_min']) for z in Z)),
               per_sample=dict(sample_idx=list(map(int, idxs)), r1_mm=r1.tolist(), r2_mm=r2.tolist(),
                               rm_mm=rm.tolist(), gap_mm=gap.tolist(),
                               ref_mean_abs_Br_T=[float(z['ref_mean_abs'][0]) for z in Z],
                               ref_mean_abs_Bt_T=[float(z['ref_mean_abs'][1]) for z in Z],
                               eps_r_pct={k: E[:, i, 0].tolist() for i, k in enumerate(RUN_KEYS)},
                               eps_t_pct={k: E[:, i, 1].tolist() for i, k in enumerate(RUN_KEYS)}),
               runs={})
    thin = gap < 1.0
    print(f'\n{"run":28s} {"ε_r":>6s} {"ε_θ":>6s} {"P95_r":>6s} {"P95_θ":>6s} | '
          f'{"ε_r<1":>6s} {"ε_θ<1":>6s} {"ε_r≥1":>6s} {"ε_θ≥1":>6s} | polar-interp ε_r ε_θ')
    for i, k in enumerate(RUN_KEYS):
        R = reg['runs'][k]
        g = dict(run=R['run'], best_epoch=R['best_epoch'],
                 eps_r_mean_pct=float(E[:, i, 0].mean()), eps_t_mean_pct=float(E[:, i, 1].mean()),
                 eps_r_p95_pct=float(np.percentile(E[:, i, 0], 95)),
                 eps_t_p95_pct=float(np.percentile(E[:, i, 1], 95)),
                 eps_r_median_pct=float(np.median(E[:, i, 0])), eps_t_median_pct=float(np.median(E[:, i, 1])),
                 gap_lt_1mm=dict(n=int(thin.sum()), eps_r=float(E[thin, i, 0].mean()), eps_t=float(E[thin, i, 1].mean())),
                 gap_ge_1mm=dict(n=int((~thin).sum()), eps_r=float(E[~thin, i, 0].mean()), eps_t=float(E[~thin, i, 1].mean())))
        assert g['eps_r_p95_pct'] >= g['eps_r_mean_pct'] and g['eps_t_p95_pct'] >= g['eps_t_mean_pct'], k
        if not np.isnan(Ep[0, i, 0]):
            g['variante_interp_polar'] = dict(eps_r=float(Ep[:, i, 0].mean()), eps_t=float(Ep[:, i, 1].mean()))
        res['runs'][k] = g
        pv = g.get('variante_interp_polar')
        print(f"{k:28s} {g['eps_r_mean_pct']:6.2f} {g['eps_t_mean_pct']:6.2f} {g['eps_r_p95_pct']:6.2f} "
              f"{g['eps_t_p95_pct']:6.2f} | {g['gap_lt_1mm']['eps_r']:6.2f} {g['gap_lt_1mm']['eps_t']:6.2f} "
              f"{g['gap_ge_1mm']['eps_r']:6.2f} {g['gap_ge_1mm']['eps_t']:6.2f} | "
              + (f"{pv['eps_r']:.3f} {pv['eps_t']:.3f}" if pv else '-'))
    print(f'gap < 1 mm: {thin.sum()} amostras; ≥ 1 mm: {(~thin).sum()}')
    score = E.mean(axis=(1, 2))
    med = int(np.asarray(idxs)[np.argsort(score)[len(score) // 2]])
    res['amostra_mediana'] = dict(sample_idx=med, criterio='mediana da média de (ε_r, ε_θ) sobre as 8 runs')
    print(f'amostra mediana: {med}')
    print(f"|B_r| médio ref {np.mean(res['per_sample']['ref_mean_abs_Br_T']):.3f} T, "
          f"|B_θ| médio ref {np.mean(res['per_sample']['ref_mean_abs_Bt_T']):.3f} T")
    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'-> {OUT_JSON}  [{(time.perf_counter() - t0) / 60:.1f} min]')


if __name__ == '__main__':
    main()
