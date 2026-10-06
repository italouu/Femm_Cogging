"""
pos_bateria_eval.py
-------------------
Passada única sobre o conjunto de teste (88 chunks, 2816 amostras) da bateria
mesh_ans_138x276_unified_best_mse_mae -- Tarefas 2 (T0a), 4a (O1) e 5b do
pedido pós-bateria. Não treina nada.

Por amostra (parse do raw em paralelo, CPU; 8 modelos best.pth, GPU):
  - erro de |B| nos nós (L1_A, L2_A, pontual) por run -- por amostra e global
    (o global tem que reproduzir surface_integral_table.json: conferência);
  - T0a: y_hw (gabarito na grade) interpolado nos nós (cell_centered) vs node_y;
  - O1: arco no raio médio do entreferro (1200 pts, θ∈[0°,120°)), B_r/B_θ,
    erros L1/L2 e harmônicos de B_r (fundamental = ordem 7, THD até 100);
  - saturação do ferro (|B| por elemento P1, mu_r efetivo pela curva BH).

Saída: pos_bateria/eval_pass.json (agregados) + eval_per_sample.npz.

Execução (raiz do projeto):  python -m scripts.pos_bateria_eval
"""
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from scripts.eval_surface_integral_table import Accum, _mag
from scripts.pos_bateria_common import (
    DEVICE, OUT_DIR, MU_BINS, SAT_THR, FUND, RUN_ORDER, run_key,
    design_rows, test_samples, parse_worker, to_torch_sample, predict, load_runs,
    arc_values_from_nodes, arc_values_fno, to_polar, harmonics, thd, wrap_deg,
    percentile_stats, INTERP_MODE,
)
from src.neural_op.archs.interp import interpolate_grid_to_nodes

N_WORKERS = 12
import sys
MAX_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else None   # None = teste inteiro


def node_errors(mag_p, mag_t, area):
    e = np.abs(mag_p - mag_t).astype(np.float64)
    a = area.astype(np.float64)
    return dict(L1=float((e * a).sum() / a.sum()), L2=float(math.sqrt((e ** 2 * a).sum() / a.sum())),
                pt=float(e.mean()))


def arc_metrics(bxy_p, br_t, bt_t, rms_t, amp_t, ph_t, thd_t):
    br, bt = to_polar(bxy_p)
    er, et = br - br_t, bt - bt_t
    amp, ph = harmonics(br)
    return dict(
        L1_Br=np.abs(er).mean(), L2_Br=np.sqrt((er ** 2).mean()),
        L1_Bt=np.abs(et).mean(), L2_Bt=np.sqrt((et ** 2).mean()),
        rms=rms_t,
        amp7=amp[FUND], amp7_relerr=(amp[FUND] - amp_t[FUND]) / amp_t[FUND],
        ph7_err=wrap_deg(ph[FUND] - ph_t[FUND]), thd=thd(amp), thd_err=thd(amp) - thd_t,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = design_rows()
    samples = test_samples()
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]
    runs = load_runs()
    for k, r in runs.items():
        print(f"  {k:24s} {r['run']} best ép.{r['epoch']} interp={r['interp_mode']}")

    acc = {k: Accum() for k in runs}
    acc_floor = Accum()
    per = {k: {m: [] for m in ('L1', 'L2', 'pt', 'L1_Br', 'L2_Br', 'L1_Bt', 'L2_Bt',
                                'amp7', 'amp7_relerr', 'ph7_err', 'thd', 'thd_err')}
           for k in runs}
    per_floor = {m: [] for m in ('L1', 'L2', 'pt')}
    gt = {m: [] for m in ('rms_arc', 'amp7', 'ph7', 'thd', 'r_m', 'gap', 'Bref_sample')}
    sat = {f'frac_{t:.1f}': [] for t in SAT_THR}
    sat_extra = dict(iron_area=[], n_iron_bh_extrap=[], bmax_iron=[], n_bh_extrap_all=[])
    mu_hist = np.zeros(len(MU_BINS) - 1)
    ids, chunks = [], []
    n_fallback = 0

    groups = {}
    for name, idx, p in samples:
        groups.setdefault(name, []).append((idx, p))

    t0 = time.time()
    done = 0
    for gi, (name, items) in enumerate(groups.items()):
        t = time.time()
        # pool novo por chunk (reciclagem de worker -- vazamento matplotlib.tri/scipy)
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items))) as ex:
            futs = [ex.submit(parse_worker, p, rows[idx]) for idx, p in items]
            for (idx, p), f in zip(items, futs):
                s = f.result()
                ts = to_torch_sample(s)
                node_y = s['v1']['node_y']
                assert np.array_equal(node_y, s['bip']['node_y'])
                area = s['v1']['node_x'][:, 2]
                mag_t = np.hypot(node_y[:, 0], node_y[:, 1])
                ids.append(idx)
                chunks.append(name)
                n_fallback += s['arc_fallback']

                # --- T0a: piso de representação da grade ---
                yh = ts['v1']['y_hw'][None].to(DEVICE)
                nx = ts['v1']['node_x'].to(DEVICE)
                # [REMOVIDO 2026-10-06] 'cell_centered' fixo (wrap circular — obsoleto)
                # fl = interpolate_grid_to_nodes(yh, nx[:, 3], nx[:, 4], ts['v1']['L'].to(DEVICE),
                #                                mode='cell_centered').cpu()
                fl = interpolate_grid_to_nodes(yh, nx[:, 3], nx[:, 4], ts['v1']['L'].to(DEVICE),
                                               mode=INTERP_MODE).cpu()
                mag_f = _mag(fl)
                acc_floor.add(mag_f, mag_t, area)
                for m, v in node_errors(mag_f, mag_t, area).items():
                    per_floor[m].append(v)
                a64 = area.astype(np.float64)
                gt['Bref_sample'].append(math.sqrt((mag_t.astype(np.float64) ** 2 * a64).sum() / a64.sum()))

                # --- gabarito no arco ---
                bxy_t = arc_values_from_nodes(node_y.astype(np.float64), s['arc_idx'], s['arc_w'])
                br_t, bt_t = to_polar(bxy_t)
                rms_t = float(np.sqrt((np.hypot(bxy_t[:, 0], bxy_t[:, 1]) ** 2).mean()))
                amp_t, ph_t = harmonics(br_t)
                thd_t = thd(amp_t)
                gt['rms_arc'].append(rms_t)
                gt['amp7'].append(amp_t[FUND])
                gt['ph7'].append(ph_t[FUND])
                gt['thd'].append(thd_t)
                gt['r_m'].append(s['r_m'])
                gt['gap'].append(s['gap'])

                # --- saturação ---
                for th in SAT_THR:
                    sat[f'frac_{th:.1f}'].append(s['sat']['frac'][f'{th:.1f}'])
                mu_hist += s['sat']['mu_hist']
                sat_extra['iron_area'].append(s['sat']['iron_area_mm2'])
                sat_extra['n_iron_bh_extrap'].append(s['sat']['n_iron_bh_extrap'])
                sat_extra['bmax_iron'].append(s['sat']['bmax_iron'])
                sat_extra['n_bh_extrap_all'].append(s['n_bh_extrap_all'])

                # --- modelos ---
                for k, r in runs.items():
                    out_hw, y_nodes = predict(r['arch'], r['model'], r['normalizer'], ts)
                    yn = y_nodes.float().cpu()
                    mag_p = _mag(yn)
                    acc[k].add(mag_p, mag_t, area)
                    for m, v in node_errors(mag_p, mag_t, area).items():
                        per[k][m].append(v)
                    if r['arch'] == 'FNO2d':
                        bxy_p = arc_values_fno(out_hw, s['r_m'], s['r_in'], s['r_ext'])
                    else:
                        bxy_p = arc_values_from_nodes(yn.numpy().astype(np.float64),
                                                      s['arc_idx'], s['arc_w'])
                    for m, v in arc_metrics(bxy_p, br_t, bt_t, rms_t, amp_t, ph_t, thd_t).items():
                        if m != 'rms':
                            per[k][m].append(float(v))
                done += 1
        print(f'  [{gi + 1}/{len(groups)}] {name}  {time.time() - t:.0f}s  '
              f'(amostras {done}, {(time.time() - t0) / 60:.1f} min)', flush=True)

    # ------------------------------------------------------------------ #
    # Agregados
    # ------------------------------------------------------------------ #
    rms_arc = np.array(gt['rms_arc'])
    res = dict(n_samples=done, n_arc_fallback_points=n_fallback, runs={}, floor={}, arc_gt={},
               saturation={})
    ref = None
    sit_path = OUT_DIR.parent / 'surface_integral_table.json'
    if sit_path.exists():
        ref = {(x['arch'], x['loss']): x['mesh'] for x in json.loads(sit_path.read_text())}

    for k, r in runs.items():
        g = acc[k].report()
        p = {m: np.array(v) for m, v in per[k].items()}
        arc = {}
        for comp in ('Br', 'Bt'):
            for nrm in ('L1', 'L2'):
                v = p[f'{nrm}_{comp}']
                arc[f'{nrm}_{comp}_T'] = percentile_stats(v)
                arc[f'{nrm}_{comp}_pct'] = percentile_stats(100 * v / rms_arc)
        arc['amp7_relerr_pct_abs'] = percentile_stats(100 * np.abs(p['amp7_relerr']))
        arc['amp7_relerr_pct_signed_mean'] = float(100 * p['amp7_relerr'].mean())
        arc['ph7_err_deg_abs'] = percentile_stats(np.abs(p['ph7_err']))
        arc['thd_err_pp_abs'] = percentile_stats(100 * np.abs(p['thd_err']))
        arc['thd_err_pp_signed_mean'] = float(100 * p['thd_err'].mean())
        check = None
        if ref is not None and (r['arch'], r['loss']) in ref:
            rr = ref[(r['arch'], r['loss'])]
            check = {m: dict(this=g[m], surface_integral_table=rr[m], rel_diff=abs(g[m] - rr[m]) / rr[m])
                     for m in ('L1_area', 'L2_area', 'MAE_pt', 'B_ref')}
        res['runs'][k] = dict(arch=r['arch'], loss=r['loss'], run=r['run'], best_epoch=r['epoch'],
                              mesh_global=g,
                              per_sample_L1_A=percentile_stats(p['L1']),
                              arc=arc, check_vs_surface_integral_table=check)

    res['floor'] = dict(mesh_global=acc_floor.report(),
                        per_sample_L1_A=percentile_stats(per_floor['L1']),
                        descricao='y_hw (gabarito na grade, T) interpolado nos nós com '
                                  'interpolate_grid_to_nodes(cell_centered) vs node_y')
    res['arc_gt'] = dict(rms_absB=percentile_stats(rms_arc),
                         amp7_T=percentile_stats(gt['amp7']), thd_pct=percentile_stats(100 * np.array(gt['thd'])),
                         r_m_mm=percentile_stats(gt['r_m']), gap_mm=percentile_stats(gt['gap']))
    res['saturation'] = {f'frac_area_iron_B_gt_{t:.1f}T': percentile_stats(sat[f'frac_{t:.1f}'])
                         for t in SAT_THR}
    res['saturation'].update(
        n_samples_with_iron_elems_beyond_bh_curve=int((np.array(sat_extra['n_iron_bh_extrap']) > 0).sum()),
        total_iron_elems_beyond_bh_curve=int(np.sum(sat_extra['n_iron_bh_extrap'])),
        bmax_iron_T=percentile_stats(sat_extra['bmax_iron']),
    )

    np.savez_compressed(
        OUT_DIR / 'eval_per_sample.npz',
        sample_idx=np.array(ids), chunk=np.array(chunks),
        mu_hist=mu_hist, mu_bins_log10=MU_BINS,
        **{f'floor_{m}': np.array(v) for m, v in per_floor.items()},
        **{f'gt_{m}': np.array(v) for m, v in gt.items()},
        **{m: np.array(v) for m, v in sat.items()},
        **{f'sat_{m}': np.array(v) for m, v in sat_extra.items()},
        **{f'{k}__{m}': np.array(v) for k in per for m, v in per[k].items()},
    )
    (OUT_DIR / 'eval_pass.json').write_text(json.dumps(res, indent=2))
    print(f'\nsalvo em {OUT_DIR}/eval_pass.json e eval_per_sample.npz '
          f'({(time.time() - t0) / 60:.1f} min)')
    for k, v in res['runs'].items():
        g = v['mesh_global']
        c = v['check_vs_surface_integral_table']
        cs = '' if c is None else f"  Δrel L1 vs tabela={c['L1_area']['rel_diff']:.1e}"
        print(f"  {k:24s} L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%  "
              f"pt {g['MAE_pt_pct']:5.2f}%{cs}")
    f = res['floor']['mesh_global']
    print(f"  {'piso grade':24s} L1 {f['L1_area_pct']:5.2f}%  L2 {f['L2_area_pct']:5.2f}%  pt {f['MAE_pt_pct']:5.2f}%")


if __name__ == '__main__':
    main()
