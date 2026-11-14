"""
Regiões de análise sobrepostas à partição do `regions` (quinas / fronteira sem quinas /
interior): entreferro Ω_gap e ferro Ω_fe, no conjunto de teste, para as 8 runs do artigo.

    python -m scripts.regions_gap_fe [--workers 3] [--n N] [--verify-run FNO_SplitGNN_mae]

Nada de inferência: as predições vêm de data/temp/inference (scripts/infer_chunks.py, mesmos
best.pth do regions.json); a referência do FEMM, Ω, os pontos de quadratura e a superfície de
cada arch são os do `regions` (scripts/postprocess.py: omega_mask, _surface_points,
_split_points, _quad_xy; mesmas chamadas de parse_smooth para overlay e mesh_quad).

Regiões (por ponto de quadratura, como o corte do regions):
  Ω_gap: pontos em Ω de elementos de ar (MATERIAL_ID vacuum) com r_1 <= r <= r_2,
         r_1 = stator_outer_diameter/2 (pontas dos dentes), r_2 = rotor_inner_diameter/2
         (face interna dos ímãs) -- por amostra, do valid_designs.csv.
  Ω_fe:  pontos em Ω de elementos de aço (MATERIAL_ID iron_1008 = 0): estator e back iron.
Métricas (iguais ao BandAccum): η = Σe·w(região)/Σe·w(Ω); L1 = Σe·w/Σw da região em % do
B_ref global em Ω; área = Σw(região)/Σw(Ω).
Verificação: as 3 regiões originais (d = REGIONS_D_MM) recalculadas nas runs de superfície
'mesh' (distâncias só dos pontos da malha) e comparadas com dados/regions.json.

Saída: dados/regions_gap_fe.json (+ checkpoint por amostra em data/temp/regions_gap_fe/).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / 'resultados' / 'BLDC_138x276' / 'dados' / 'regions_gap_fe.json'
CKPT     = ROOT / 'data' / 'temp' / 'regions_gap_fe'
RUN_KEYS = ('FNO2d_mse', 'FNO2d_mae', 'FNO_GNN_mse', 'FNO_GNN_mae',
            'FNO_BipartiteGNN_Elem_mse', 'FNO_BipartiteGNN_Elem_mae',
            'FNO_SplitGNN_mse', 'FNO_SplitGNN_mae')
REGS     = ('total', 'gap', 'fe', 'quinas', 'fronteira_sem_quinas', 'interior')
SUMS     = ('w', 'ew', 'e2w', 'mf2w')

_G = {}


def _init(pred_dirs):
    """Worker: imports pesados uma vez por processo."""
    import scripts.postprocess as pp
    _G.update(pp=pp, rows=pp.design_rows(), pred=pred_dirs,
              edges=pp.regions_setup()[1], d=float(pp.REGIONS_D_MM[0]))


def _sample(idx):
    import torch
    from src.data_gen.motor_constants import MATERIAL_ID
    from src.data_gen.parsers.ans_b_smoothing import element_b, nodal_b
    from src.data_gen.parsers.ans_parsing import _parse_label_materials, read_ans
    from src.data_gen.parsers.split_layout import parse_split_smooth
    from src.postprocess.metrics import grid_mesh_overlay, mesh_surface_quadrature
    from src.postprocess.regions import SeamGeometry, omega_mask, region_geometry
    pp = _G['pp']
    out = CKPT / f'sample_{idx:06d}.npz'
    if out.exists():
        return idx
    row, path = _G['rows'][idx], pp.sample_path(idx)
    r_in, r_ext = float(row['inner_diameter [mm]']) / 2, float(row['outer_diameter [mm]']) / 2
    r1 = float(row['stator_outer_diameter [mm]']) / 2
    r2 = float(row['rotor_inner_diameter [mm]']) / 2
    ans = read_ans(path)
    m = ans.mesh
    xy = np.stack([m.x, m.y], axis=1).astype(np.float64)
    tri = ans.elems[:, :3].astype(np.int64)
    assert np.array_equal(m.p, tri)
    lmat, _, _ = _parse_label_materials(ans.lines)
    emat = lmat[ans.elems[:, 3]]
    in_omega, _ = omega_mask(ans, row, xy, tri, r_in, r_ext, pp.SEAM_RAD)
    # overlay e mesh_quad: mesmas chamadas de samples.parse_smooth(with_overlay=True)
    B1, B2 = element_b(m)
    b1, b2 = nodal_b(m, B1, B2)
    xy0 = np.stack([m.x, m.y], axis=1)
    overlay = grid_mesh_overlay(xy0, m.p, b1, b2, r_in, r_ext, pp.N_R, pp.N_A,
                                ang_1=pp.ANG_1, ang_2=pp.ANG_2)
    mq = mesh_surface_quadrature(xy0, m.p, b1, b2)
    split = parse_split_smooth(path, r_in, r_ext, ang_1=pp.ANG_1, ang_2=pp.ANG_2, n_r=pp.N_R,
                               n_a=pp.N_A, ans=ans)
    lay = dict(overlay=overlay, mesh_quad=mq, tri=tri, split=split)
    M, K = mq['mag'].shape
    pts = pp._quad_xy(lay, xy)
    tri_of = dict(mesh=np.repeat(np.arange(M), K), overlay=overlay['tri'])
    masks = {}
    for ps in ('mesh', 'overlay'):
        t = tri_of[ps]
        keep = in_omega[t]
        r = np.hypot(pts[ps][:, 0], pts[ps][:, 1])
        mat = emat[t]
        masks[ps] = dict(total=keep,
                         gap=keep & (mat == MATERIAL_ID['vacuum']) & (r >= r1) & (r <= r2),
                         fe=keep & (mat == MATERIAL_ID['iron_1008']))
    # 3 regiões originais nos pontos da malha (mesma regra de band_points / _regions_sample)
    iface, corners = region_geometry(xy, tri, emat, 30.0)
    geo = SeamGeometry(xy, iface, corners, d_cap=_G['edges'][-1], seam_rad=pp.SEAM_RAD)
    keep = masks['mesh']['total']
    d_f, d_q = geo.dist(pts['mesh'][keep])
    code = np.full(int(keep.sum()), 2, np.int8)
    code[d_f <= _G['d']] = 1
    code[d_q <= _G['d']] = 0
    for c, name in enumerate(('quinas', 'fronteira_sem_quinas', 'interior')):
        mk = np.zeros(len(keep), bool)
        mk[np.flatnonzero(keep)[code == c]] = True
        masks['mesh'][name] = mk
    S = np.full((len(RUN_KEYS), len(REGS), len(SUMS)), np.nan)
    for i, k in enumerate(RUN_KEYS):
        arch = k.rsplit('_', 1)[0]
        with np.load(Path(_G['pred'][k]) / f'sample_{idx:06d}.npz') as z:
            p = {n: z[n] for n in z.files}
        if arch == 'FNO_SplitGNN':
            mm, mf, w, ps = pp._split_points(lay, p['y_base'].astype(np.float64))
        else:
            yn = torch.from_numpy(p['y']) if 'y' in p else None
            mm, mf, w, _, ps = pp._surface_points(arch, lay, torch.from_numpy(p['out_hw']), yn,
                                                  with_set=True)
        mm, mf, w = (np.asarray(v, np.float64) for v in (mm, mf, w))
        e = np.abs(mm - mf)
        for j, reg in enumerate(REGS):
            if reg not in masks[ps]:
                continue
            mk = masks[ps][reg]
            S[i, j] = [w[mk].sum(), (e[mk] * w[mk]).sum(), (e[mk] ** 2 * w[mk]).sum(),
                       (mf[mk] ** 2 * w[mk]).sum()]
    wm, wo = masks['mesh'], masks['overlay']
    W = np.repeat((mq['area'] / K)[:, None], K, axis=1).ravel()
    tmp = out.with_name(f'{out.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, S=S, r1=r1, r2=r2,
             area_gap_mesh=W[wm['gap']].sum(), area_gap_overlay=overlay['w'][wo['gap']].sum(),
             area_fe_mesh=W[wm['fe']].sum(), area_omega_mesh=W[wm['total']].sum(),
             area_annulus_exact=(r2 ** 2 - r1 ** 2) * pp.SEAM_RAD / 2,
             n_overlap_gap_fe=int((wm['gap'] & wm['fe']).sum() + (wo['gap'] & wo['fe']).sum()))
    os.replace(tmp, out)
    return idx


def _stats(s, s_tot, b_ref):
    w, ew = s[0], s[1]
    return dict(eta_pct=100 * ew / s_tot[1], L1_pct=100 * ew / w / b_ref,
                L2_pct=100 * np.sqrt(s[2] / w) / b_ref, area_frac_pct=100 * w / s_tot[0],
                sum_w=w, sum_ew=ew, sum_e2w=s[2])


OUT_TEX = OUT_JSON.with_name('tab_regions.tex')
TAB_ROWS = (('FNO2d', 'FNO2d'), ('FNO\\_GNN', 'FNO_GNN'),
            ('FNO\\_BipartiteGNN', 'FNO_BipartiteGNN_Elem'), ('FNO\\_SplitGNN', 'FNO_SplitGNN'))
CAPTION = (r'Estratificação do erro de superfície por região no conjunto de teste (608 amostras), '
           r'para $d = 0{,}216$\,mm. As três primeiras regiões são disjuntas e cobrem $\Omega$. '
           r'O entreferro, $\Omega_{\mathrm{gap}}$, e o ferro, $\Omega_{\mathrm{fe}}$, são recortes '
           r'sobrepostos a elas e não somam com as demais. $\eta$: fração do erro $L_{1}$ total de '
           r'$\Omega$ contida na região, em \%. $L_{1}$: erro de superfície restrito à região e '
           r'normalizado pela sua área, em \% de $B_{\text{ref}} = 0{,}683$\,T. Entre parênteses, a '
           r'fração da área de $\Omega$ ocupada pela região.')


def _br(x, nd):
    return f'{x:.{nd}f}'.replace('.', ',')


def render_table():
    """tab:regions: 3 regiões do regions.json (inalteradas) + Entreferro e Ferro deste JSON.
    Fração de área do cabeçalho: pontos da malha (as runs de superfície 'mesh')."""
    reg = json.loads((ROOT / 'resultados/BLDC_138x276/dados/regions.json').read_text(encoding='utf-8'))
    new = json.loads(OUT_JSON.read_text(encoding='utf-8'))
    dk = f"{reg['params']['d_default_mm']:g}"
    orig = ('quinas', 'fronteira_sem_quinas', 'interior')
    ref = 'FNO_GNN_mse'                                   # conjunto de pontos 'mesh'
    af = [100 * reg['runs'][ref]['by_d'][dk][g]['area_frac'] for g in orig]
    af += [new['runs'][ref][g]['area_frac_pct'] for g in ('gap', 'fe')]
    names = ('Quinas', 'Fronteira s/ quinas', 'Interior', 'Entreferro', 'Ferro')
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{n} ({_br(a, 1)}\\,\\%)}}' for n, a in zip(names, af))
    L = [r'\begin{table*}[!t]', r'\renewcommand{\arraystretch}{1.2}', f'\\caption{{{CAPTION}}}',
         r'\label{tab:regions}', r'\centering', r'\footnotesize',
         r'\begin{tabular}{ll cccccc@{\hspace{3\tabcolsep}}cccc}', r'\toprule',
         f' & & {head} \\\\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}',
         'Arquitetura & Perda & ' + ' & '.join([r'$\eta$ & $L_{1}$'] * 5) + r' \\', r'\midrule']
    for a, (tex, arch) in enumerate(TAB_ROWS):
        for j, loss in enumerate(('mse', 'mae')):
            k = f'{arch}_{loss}'
            b = reg['runs'][k]['by_d'][dk]
            cells = [c for g in orig for c in (_br(100 * b[g]['L1_contrib_frac'], 1), _br(b[g]['L1_area_pct'], 2))]
            cells += [c for g in ('gap', 'fe') for c in (_br(new['runs'][k][g]['eta_pct'], 1),
                                                          _br(new['runs'][k][g]['L1_pct'], 2))]
            L.append(f"{tex if j == 0 else ''} & {loss.upper()} & " + ' & '.join(cells) + r' \\')
        if a < len(TAB_ROWS) - 1:
            L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    OUT_TEX.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'-> {OUT_TEX}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--n', type=int, default=None, help='só as N primeiras amostras (teste)')
    ap.add_argument('--table', action='store_true', help='só gera tab_regions.tex dos JSONs')
    a = ap.parse_args()
    if a.table:
        return render_table()
    reg = json.loads((ROOT / 'resultados/BLDC_138x276/dados/regions.json').read_text(encoding='utf-8'))
    sat = json.loads((ROOT / 'resultados/BLDC_138x276/dados/saturation_error.json').read_text(encoding='utf-8'))
    from src.postprocess.chunk_inference import pred_dir
    pred = {k: str(pred_dir(ROOT / reg['runs'][k]['run_dir'].replace('\\', '/'))) for k in RUN_KEYS}
    idxs = reg['sample_idx'][:a.n]
    CKPT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    todo = [i for i in idxs if not (CKPT / f'sample_{i:06d}.npz').exists()]
    # [REMOVIDO] max_tasks_per_child=8: no Windows os workers saíam após 8 tarefas e não eram
    # repostos (pool travado em 27 amostras)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(pred,)) as ex:
        for n, _ in enumerate(ex.map(_sample, todo), 1):
            if n % 8 == 0 or n == len(todo):
                el = time.perf_counter() - t0
                print(f'{n}/{len(todo)}  {el / 60:.1f} min  (~{el / n * (len(todo) - n) / 60:.0f} min restantes)',
                      flush=True)
    Z = [np.load(CKPT / f'sample_{i:06d}.npz') for i in idxs]
    S = sum(z['S'] for z in Z)                                # [run, região, soma]
    r1 = np.array([float(z['r1']) for z in Z])
    r2 = np.array([float(z['r2']) for z in Z])
    res = dict(n_samples=len(idxs), subset='test', d_mm=float(reg['params']['d_default_mm']),
               definicao=dict(
                   gap='pontos de quadratura em Ω de elementos vacuum (MATERIAL_ID 1) com '
                       'r_1 <= r <= r_2; r_1 = stator_outer_diameter/2, r_2 = rotor_inner_diameter/2 '
                       '(por amostra); abertura das ranhuras (r < r_1) e ar entre ímãs (r > r_2) fora',
                   fe='pontos de quadratura em Ω de elementos iron_1008 (MATERIAL_ID 0): estator e '
                      'back iron do rotor; ímãs (2), cobre (3) e ar (1) fora',
                   metricas='eta = Σe·w(região)/Σe·w(Ω); L1 = Σe·w/Σw da região em % do B_ref global '
                            'em Ω; area_frac = Σw(região)/Σw(Ω); somas sobre as amostras (como BandAccum)',
                   fonte_predicoes='data/temp/inference (scripts/infer_chunks.py), mesmos best.pth do regions.json'),
               material_id=dict(gap=[1], fe=[0], nomes={'0': 'iron_1008', '1': 'vacuum',
                                                        '2': 'N35p/N35n', '3': 'copper'}),
               r1_mm=dict(min=r1.min(), max=r1.max(), mean=r1.mean()),
               r2_mm=dict(min=r2.min(), max=r2.max(), mean=r2.mean()),
               gap_mm=dict(min=(r2 - r1).min(), max=(r2 - r1).max(), mean=(r2 - r1).mean()),
               per_sample=dict(sample_idx=list(map(int, idxs)), r1_mm=r1.tolist(), r2_mm=r2.tolist(),
                               area_gap_mesh_mm2=[float(z['area_gap_mesh']) for z in Z],
                               area_annulus_exact_mm2=[float(z['area_annulus_exact']) for z in Z],
                               area_fe_mesh_mm2=[float(z['area_fe_mesh']) for z in Z]),
               runs={}, checks={})
    chk = res['checks']
    chk['n_overlap_gap_fe_points'] = int(sum(int(z['n_overlap_gap_fe']) for z in Z))
    ag, ae = (np.array(res['per_sample'][k]) for k in ('area_gap_mesh_mm2', 'area_annulus_exact_mm2'))
    chk['area_gap_vs_annulus_exact'] = dict(sum_ratio=float(ag.sum() / ae.sum()),
                                            min_ratio=float((ag / ae).min()), max_ratio=float((ag / ae).max()))
    for i, k in enumerate(RUN_KEYS):
        s_tot = S[i, 0]
        b_ref = float(np.sqrt(s_tot[3] / s_tot[0]))
        R = reg['runs'][k]
        rr = dict(run=R['run'], best_epoch=R['best_epoch'], B_ref=b_ref,
                  total_L1_pct=100 * s_tot[1] / s_tot[0] / b_ref,
                  gap=_stats(S[i, 1], s_tot, b_ref), fe=_stats(S[i, 2], s_tot, b_ref))
        c = dict(B_ref_regions_json=R['B_ref_global'],
                 total_L1_pct_regions_json=R['total']['L1_area_pct'])
        c['rel_diff_total_L1'] = rr['total_L1_pct'] / c['total_L1_pct_regions_json'] - 1
        c['rel_diff_B_ref'] = b_ref / c['B_ref_regions_json'] - 1
        aco = sat['runs'][k]['aco']['total']
        c['fe_vs_saturation_aco'] = dict(area_mm2_sat=aco['area_mm2'], area_mm2_here=rr['fe']['sum_w'],
                                         L1_pct_sat=aco['L1_pct'], L1_pct_here=rr['fe']['L1_pct'])
        if not np.isnan(S[i, 3, 0]):
            bd = R['by_d'][f"{res['d_mm']:g}"]
            c['regioes_originais'] = {
                g: dict(here=dict(eta_pct=100 * S[i, j, 1] / s_tot[1],
                                  L1_pct=100 * S[i, j, 1] / S[i, j, 0] / b_ref,
                                  area_frac_pct=100 * S[i, j, 0] / s_tot[0]),
                        regions_json=dict(eta_pct=100 * bd[g]['L1_contrib_frac'],
                                          L1_pct=bd[g]['L1_area_pct'],
                                          area_frac_pct=100 * bd[g]['area_frac']))
                for j, g in ((3, 'quinas'), (4, 'fronteira_sem_quinas'), (5, 'interior'))}
        rr['check'] = c
        res['runs'][k] = rr
    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'-> {OUT_JSON}  [{(time.perf_counter() - t0) / 60:.1f} min]')


if __name__ == '__main__':
    main()
