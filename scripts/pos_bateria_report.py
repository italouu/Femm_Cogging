"""
pos_bateria_report.py
---------------------
Junta eval_pass.json + timing.json + figures.json (+ git_check.json) em
pos_bateria/pos_bateria.json e gera as tabelas do relatório em
pos_bateria/tabelas.md (o RELATORIO.md é escrito à mão em cima delas).

Execução:  python -m scripts.pos_bateria_report
"""
import json

import numpy as np

from scripts.pos_bateria_common import OUT_DIR, RUN_ORDER, run_key

PRE_COMMON = ('pre_read_ans', 'pre_materials', 'pre_x_hw', 'pre_node_pos')
PRE_BY_ARCH = {
    'FNO2d':            PRE_COMMON,
    'FNO_GNN':          PRE_COMMON + ('pre_graph_topology', 'pre_v1_graph'),
    'GNN_PostBase':     PRE_COMMON + ('pre_graph_topology', 'pre_v1_graph'),
    'FNO_BipartiteGNN': PRE_COMMON + ('pre_graph_topology', 'pre_bip_graph'),
}


def f(x, n=2):
    return f'{x:.{n}f}'.replace('.', ',')


def load(name):
    p = OUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def pct_stats(s, n=2):
    return f"{f(s['mean'], n)} / {f(s['median'], n)} / {f(s['p95'], n)}"


def main():
    ev, tm, fg, gc = load('eval_pass.json'), load('timing.json'), load('figures.json'), load('git_check.json')
    md = []

    # ---------------- Tarefa 2 ----------------
    if ev:
        md += ['## T2 — tabela §2.1 + piso da grade', '',
               '| Modelo | Loss | Best ép. | L1_A (T) | L1_A (%B_ref) | L2_A (T) | L2_A (%B_ref) | Pontual (T) | Pontual (%B_ref) |',
               '|---|---|---|---|---|---|---|---|---|']
        for a, l in RUN_ORDER:
            r = ev['runs'][run_key(a, l)]
            g = r['mesh_global']
            md.append(f"| {a} | {l} | {r['best_epoch']} | {f(g['L1_area'],4)} | {f(g['L1_area_pct'])} | "
                      f"{f(g['L2_area'],4)} | {f(g['L2_area_pct'])} | {f(g['MAE_pt'],4)} | {f(g['MAE_pt_pct'])} |")
        g = ev['floor']['mesh_global']
        md.append(f"| **gabarito da grade interpolado** | — | — | {f(g['L1_area'],4)} | {f(g['L1_area_pct'])} | "
                  f"{f(g['L2_area'],4)} | {f(g['L2_area_pct'])} | {f(g['MAE_pt'],4)} | {f(g['MAE_pt_pct'])} |")
        md.append(f"\nB_ref = {f(g['B_ref'],4)} T; {ev['n_samples']} amostras; {g['n_pts']:,} nós.\n")
        md += ['Conferência contra `surface_integral_table.json` (maior diferença relativa entre as 4 métricas):', '']
        for k, r in ev['runs'].items():
            c = r['check_vs_surface_integral_table']
            if c:
                md.append(f"- {k}: {max(v['rel_diff'] for v in c.values()):.1e}")
        md.append('')

    # ---------------- Tarefa 3 ----------------
    if tm:
        hw = tm.get('hardware', {})
        md += ['## T3 — custo computacional', '', '```', json.dumps(hw, indent=2, ensure_ascii=False), '```', '']
        fp = tm.get('femm_preproc')
        inf = tm.get('inference')
        if fp:
            S = fp['summary_ms']
            md += [f"### 3a — FEMM ({fp['n_samples']} amostras; nós: mediana {f(fp['n_nodes']['median'],0)})", '',
                   '| Etapa | mediana (ms) | IQR (ms) |', '|---|---|---|']
            for k, lab in (('openfemm', 'openfemm (fora do custo por amostra)'),
                           ('geometry', 'geometria (desenho + materiais + saveas)'),
                           ('mesh', 'malha (mi_createmesh)'), ('solve', 'solução (mi_analyze)'),
                           ('closefemm', 'closefemm (fora do custo por amostra)'),
                           ('post_read_B', 'leitura do .ans + B nodal')):
                md.append(f"| {lab} | {f(S[k]['median'],1)} | {f(S[k]['iqr'],1)} |")
            per = fp['per_sample']
            ms_solve = np.array([1e3 * (p['mesh'] + p['solve']) for p in per])
            md.append(f"| malha + solução (por amostra) | {f(np.median(ms_solve),1)} | "
                      f"{f(np.subtract(*np.percentile(ms_solve, [75, 25])),1)} |")
            md += ['', f"Conferências: pré-processamento idêntico ao parser oficial em todas as amostras: "
                       f"{fp['all_preproc_equal']}; B nodal idêntico: {fp['all_nodeB_equal']}; "
                       f"malha regenerada igual à do raw (nº de nós/elementos): {fp['n_mesh_equal_raw']}/{fp['n_samples']}.", '']
            md += ['### 3b — pré-processamento por etapa (mediana ms)', '', '| Etapa | mediana | IQR |', '|---|---|---|']
            for k in [k for k in S if k.startswith('pre_') or k.startswith('enc_')]:
                md.append(f"| {k} | {f(S[k]['median'],1)} | {f(S[k]['iqr'],1)} |")
            md.append('')

        if fp and inf:
            md += ['### Tabela final (ms por amostra, medianas)', '',
                   '| Arch | Loss | FEMM malha+solução | Pré-proc. | Infer. batch 1 (IQR) | Batch máx. | Infer./amostra no batch máx. | Pico GPU b1 / bmáx (GiB) | Speedup s/ solução | Speedup inferência pura |',
                   '|---|---|---|---|---|---|---|---|---|---|']
            per = fp['per_sample']
            final = {}
            for a, l in RUN_ORDER:
                k = run_key(a, l)
                ri = inf['runs'][k]
                pre = np.array([1e3 * (sum(p[x] for x in PRE_BY_ARCH[a]) + p[f'enc_{a}']) for p in per])
                mesh = np.array([1e3 * p['mesh'] for p in per])
                solve = np.array([1e3 * p['solve'] for p in per])
                b1 = ri['batch1']['ms']['median']
                bm = ri['batch_max']
                bmr = ri['batch_sweep'][str(bm)] if bm else None
                per_s = bmr['ms_per_sample_median'] if bmr else b1
                sp_sol = np.median((mesh + solve) / (mesh + pre + b1))
                sp_inf = np.median(solve / b1)
                final[k] = dict(femm_mesh_solve_ms=float(np.median(mesh + solve)), preproc_ms=float(np.median(pre)),
                                infer_b1_ms=b1, infer_b1_iqr_ms=ri['batch1']['ms']['iqr'], batch_max=bm,
                                infer_per_sample_bmax_ms=per_s, peak_gib_b1=ri['batch1']['peak_mem_gib'],
                                peak_gib_bmax=bmr['peak_mem_gib'] if bmr else None,
                                speedup_over_solution=float(sp_sol), speedup_inference_pure=float(sp_inf),
                                speedup_inference_pure_bmax=float(np.median(solve / per_s)))
                md.append(f"| {a} | {l} | {f(np.median(mesh + solve),0)} | {f(np.median(pre),0)} | "
                          f"{f(b1,2)} ({f(ri['batch1']['ms']['iqr'],2)}) | {bm} | {f(per_s,2)} | "
                          f"{f(ri['batch1']['peak_mem_gib'],2)} / {f(bmr['peak_mem_gib'],2) if bmr else '—'} | "
                          f"{f(sp_sol,2)}× | {f(sp_inf,0)}× |")
            tm['final_table'] = final
            md.append('')

    # ---------------- Tarefa 4 ----------------
    if ev:
        md += ['## T4a — entreferro (média / mediana / p95 sobre as amostras)', '',
               '| Arch | Loss | L1 B_r (%) | L2 B_r (%) | L1 B_θ (%) | L2 B_θ (%) | L1 B_r (mT) | L1 B_θ (mT) |',
               '|---|---|---|---|---|---|---|---|']
        for a, l in RUN_ORDER:
            r = ev['runs'][run_key(a, l)]['arc']
            mt = lambda s: {k: 1e3 * v for k, v in s.items()}   # noqa: E731
            md.append(f"| {a} | {l} | {pct_stats(r['L1_Br_pct'])} | {pct_stats(r['L2_Br_pct'])} | "
                      f"{pct_stats(r['L1_Bt_pct'])} | {pct_stats(r['L2_Bt_pct'])} | "
                      f"{pct_stats(mt(r['L1_Br_T']),1)} | {pct_stats(mt(r['L1_Bt_T']),1)} |")
        md += ['', '| Arch | Loss | |erro rel.| amplitude fund. (%) | viés amplitude (%) | |erro fase| fund. (°) | |erro THD| (p.p.) | viés THD (p.p.) |',
               '|---|---|---|---|---|---|---|']
        for a, l in RUN_ORDER:
            r = ev['runs'][run_key(a, l)]['arc']
            md.append(f"| {a} | {l} | {pct_stats(r['amp7_relerr_pct_abs'])} | {f(r['amp7_relerr_pct_signed_mean'])} | "
                      f"{pct_stats(r['ph7_err_deg_abs'],3)} | {pct_stats(r['thd_err_pp_abs'])} | {f(r['thd_err_pp_signed_mean'])} |")
        g = ev['arc_gt']
        md += ['', f"Gabarito no arco: RMS |B| {pct_stats(g['rms_absB'],3)} T; amplitude fundamental "
                   f"{pct_stats(g['amp7_T'],3)} T; THD {pct_stats(g['thd_pct'])} %; r_m {pct_stats(g['r_m_mm'])} mm; "
                   f"gap {pct_stats(g['gap_mm'])} mm. Pontos do arco com fallback de localização: {ev['n_arc_fallback_points']}.", '']

    # ---------------- Tarefa 5 ----------------
    if ev:
        sat = ev['saturation']
        md += ['## T5b — saturação (média / mediana / p95 entre amostras)', '',
               '| Limiar | % da área de ferro acima |', '|---|---|']
        for t in ('1.4', '1.6', '1.8'):
            s = {k: 100 * v for k, v in sat[f'frac_area_iron_B_gt_{t}T'].items()}
            md.append(f"| {t.replace('.', ',')} T | {pct_stats(s,3)} |")
        md += ['', f"|B| máximo no ferro por amostra: {pct_stats(sat['bmax_iron_T'])} T; elementos de ferro além do "
                   f"último ponto da curva BH: {sat['total_iron_elems_beyond_bh_curve']} em "
                   f"{sat['n_samples_with_iron_elems_beyond_bh_curve']} amostras.", '']
    if fg:
        md += ['Amostras escolhidas:', '', '```', json.dumps(fg['chosen_samples'], indent=2), '```', '',
               f"Spearman (fração >1,6 T × L1_A): {json.dumps(fg['spearman_frac16_vs_L1A'])}", '',
               f"Quantis de mu_r efetivo (ponderado por área): {json.dumps(fg['mu_eff_area_quantiles'])}; "
               f"fração de área com mu_r ≥ 5000: {fg['mu_eff_area_frac_above_5000']}", '']

    (OUT_DIR / 'tabelas.md').write_text('\n'.join(md), encoding='utf-8')
    (OUT_DIR / 'pos_bateria.json').write_text(json.dumps(
        dict(tarefa1_git=gc, eval=ev, timing=tm, figures=fg,
             inconsistencia_wrap_cartesiano=load('wrap_check.json')), indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
