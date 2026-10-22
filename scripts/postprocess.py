"""
postprocess.py
--------------
Executável do pós-processamento (código em src/postprocess/) — só gabarito SMOOTH.
Nada aqui treina; lê as runs em data/logs/ e grava em resultados/<dataset>/:
  dados/  — números (json/npz), versionados;
  plots/  — figuras, fora do git.

Subcomandos:
  eval [N_AMOSTRAS] [--subset test|val] — os 10 runs da bateria (5 archs × mse/mae, best.pth)
      nos chunks do subset (padrão 'test', nunca visto no treino; 'val' = conjunto da parada e
      do best.pth; N -> as N primeiras amostras). ERRO DE SUPERFÍCIE OFICIAL (2026-10-20): superfície discreta do modelo ×
      superfície do FEMM (B suavizado P1 descontínuo), L1_A / L2_A / B_ref:
        FNO2d  -> grade (constante na célula) por interseção célula ∩ triângulo;
        Elem   -> P0 (constante no elemento) na própria malha;
        demais -> P1 contínuo dos valores nodais na própria malha.
      Pisos (o gabarito em cada representação): floor_surface (grade, P1_nos, P0_elem).
      Saída: dados/eval_smooth.json + eval_smooth_per_sample.npz
      [2026-10-20: níveis antigos mesh/grid/grid_inter/elem e pisos elem<->nó removidos.]
      Secundário (P1, 2026-10-20): erro de |B| ponto a ponto nos nós (MAE_pt, L2_pt,
      L1_A, L2_A com área dual) -> runs[k]['nodal'], floor_nodal, npz nodal_*.
      Malha (P2): n_nodes / n_elems por amostra no .npz; mesh_stats no JSON.

  figures [--data DIR] [--out DIR] — depois do eval: 1_curvas_treino, 2_barras_erro
      (L1_A / L2_A de superfície dos 10 runs + piso da representação de cada um),
      3_campo_<P50|P95> (|B| nos nós dos 5 modelos mae, setor e zoom no entreferro —
      visualização; amostras P50/P95 do L1_A de superfície da REF_RUN).
      Saída: plots/ (+ dados/figures_smooth.json).

  surface IDX [LOSS] — erro de superfície dos 5 modelos (loss LOSS, default mae) em 1
      amostra: dados/surface_error/sample_XXXXXX_<loss>.{json,npz} + mapa da densidade do
      erro por triângulo em plots/surface_error/sample_XXXXXX_<loss>.png.
  surface --pct P [LOSS] [--eval-data DIR] — (P4) a amostra do percentil P do L1_A de
      superfície da REF_RUN no .npz do eval (mesma regra de figures); sufixo _P<P>.
      Mapa: mesma escala em todos os painéis, contornos dos materiais, painel extra com o
      piso da representação em grade.

  train-table [--out DIR] — tabela de treinamento dos 10 runs a partir dos logs (config.json,
      run_summary.json, epochs.csv; nada recalculado): parâmetros, pico de memória da GPU,
      tempo mediano por época (treino / avaliação / métricas) e total, melhor/última época,
      motivo da parada, hardware registrado. Saída: dados/train_table.json.

  regions [N] [--d-int MM] [--d-canto MM] [--ang-canto GRAUS] — (P5) erro de superfície
      por região do triângulo do FEMM: canto (a até d_canto de um vértice de fronteira com
      mudança de direção > ang_canto, ramificação ou ≥ 3 materiais) > interface (a até d_int
      de uma aresta entre materiais) > interior. Por run e por piso: L1_A/L2_A (% do B_ref
      global), fração da área e da contribuição ao L1_A. Padrão: d_int = 1 passo radial da
      grade, d_canto = 2 passos, 30°. Saída: dados/regions.json + plots/regions_sample_*.png.

  torque {theta-e,batch,scatter} [--loss L] [--archs ...] [--femm] — (P6) torque pelo tensor
      de Maxwell na faixa do entreferro (B por elemento) × FEMM T22, migrado dos scripts que
      geraram dados/torque_*.json, para as 5 archs da loss escolhida. Soluções FEMM em
      data/temp/torque_theta_e{,_batch}/ (--femm resolve). Saída: dados/torque/.

  saturation [N] [--knee-frac F ...] [--check-femm] — (P7) só o gabarito: curva B-H do aço
      lida do .ans (conferida igual em todas as amostras; interpolação do FEMM), |B| suavizado
      no centróide dos elementos de aço e μ_eff; figura A (curva + μ_eff + distribuição dos
      pontos de operação) e B (|B| no aço, amostras P50 e de maior |B| médio, contornos no
      joelho). Saída: dados/saturation.json + plots/saturacao_*.

  timing [femm|infer|all] [--n-femm 50] — (P8, máquina Windows com FEMM e GPU) custo
      computacional com a malha dada aos dois lados: FEMM = só o mi_analyze (redesenho com os
      parâmetros do CSV, malha fora do cronômetro; malha e A conferidos × raw); pré-
      processamento .ans -> entrada codificada por etapa (funções dos parsers smooth,
      conferido × parsers); inferência best.pth lote 1 e lote máx. (mediana/IQR, com
      desnormalização). Saída: dados/timing.json (hardware, tabelas, acelerações).

Subset: eval, regions, saturation, timing e torque usam o 'test' por padrão e aceitam
  --subset val; figures e surface --pct usam o subset gravado pelo eval.

Execução (raiz do projeto):
  python -m scripts.postprocess eval [N_AMOSTRAS] [--subset test|val] [--out DIR]
  python -m scripts.postprocess figures [--data DIR] [--out DIR]
  python -m scripts.postprocess surface IDX [LOSS]
  python -m scripts.postprocess surface --pct 50 [LOSS]
  python -m scripts.postprocess train-table
  python -m scripts.postprocess regions [N]
  python -m scripts.postprocess saturation [N]
  python -m scripts.postprocess timing [femm|infer|all]
  python -m scripts.postprocess torque theta-e|batch|scatter [--loss mae]
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from scripts.build_chunks_main import MAX_SAMPLES
from src.configs.datagen import DatagenConfig
from src.data_gen.parsers.ans_parsing import read_ans
from src.postprocess.fields import elem_to_nodes, gap_radius_mm, mag
from src.postprocess.inference import (ARCHS, INTERP_MODE, RUN_ORDER, find_runs, floor_nodes,
                                       load_set, predict, to_torch_sample)
from src.postprocess.metrics import (R_EXT_MM, R_IN_MM, Accum, merge, pstats,
                                     surface_error_per_triangle, surface_values_p1)
from src.postprocess.plots.errors import fig_bars, fig_regions, fig_surface_error_map
from src.postprocess.plots.fields import fig_field
from src.postprocess.plots.primitives import apply_style, save_fig
from src.postprocess.plots.training import fig_training
from src.postprocess.regions import REGIONS, RegionAccum, classify_triangles, interface_edges
from src.postprocess.saturation import (FemmBH, check_against_femm, steel_operating_points,
                                        weighted_percentile)
from src.postprocess.plots.saturation import MU_LABEL, fig_bh, fig_steel_map
from src.postprocess.timing import (ARCH_STAGES, DEVICE, LAYOUT, check_mesh_vs_raw,
                                    check_preprocess, collate, encode_timed, femm_solve_timed,
                                    hardware_info, iqr_stats, preprocess_timed, time_batch)
from src.postprocess.torque import (batch_summary, curve_row, femm_ref, femm_ref_ok, femm_solve, gap,
                                    parse_torque_sample, read_rows, sample_torques, scatter_ids,
                                    scatter_stats, select_batch_samples)
from src.postprocess.torque import thetas as torque_thetas
from src.postprocess.train_table import train_table
from src.postprocess.samples import (ANG_1, ANG_2, N_A, N_R, RAW_DIR, SUBSETS, design_rows,
                                     parse_smooth, sample_path, samples_with_subset)

DATASET   = DatagenConfig().dataset
ROOT_S    = Path('data/logs') / DATASET                              # logs da bateria bldc
N_SAMPLES = MAX_SAMPLES   # amostras do raw no dataset (as mesmas do build dos chunks)
OUT_ROOT  = Path('resultados') / DATASET
N_WORKERS = 12
GRID_ARCHS = ('FNO2d',)                # saída = grade H×W -> superfície por interseção
SURF_KINDS = ('grade', 'P1_nos', 'P0_elem')   # pisos: o gabarito em cada tipo de superfície
REF_RUN   = 'FNO_BipartiteGNN_mae'     # run que escolhe as amostras P50/P95 das figuras


def _mag_np(b):
    """|B| de [N,2] numpy."""
    return np.hypot(b[:, 0], b[:, 1])


def _node_mags(runs, lay, ts):
    """{arch: |B| por nó [S]} (Elem levada aos nós por elem_to_nodes)."""
    out = {}
    for r in runs.values():
        _, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
        yn = yn.double().cpu().numpy()
        if r['arch'] == 'FNO_BipartiteGNN_Elem':
            yn = elem_to_nodes(lay, yn)
        out[r['arch']] = _mag_np(yn)
    return out


def _mesh(idx):
    """(nós [N,3], elementos [M,4]) do .ans da amostra."""
    ans = read_ans(sample_path(idx))
    return ans.nodes, ans.elems


def _surface_points(arch, lay, out_hw, yn):
    """Superfície discreta do modelo nos pontos de quadratura da superfície do FEMM:
    (|B|_modelo [Q], |B|_FEMM [Q], peso [Q] mm², triângulo [Q]).
      FNO2d  -> grade (constante na célula) por interseção célula ∩ triângulo;
      Elem   -> P0 (constante no elemento) na própria malha;
      demais -> P1 contínuo dos valores nodais na própria malha."""
    if arch in GRID_ARCHS:
        ov = lay['overlay']
        return mag(out_hw.float().cpu()).ravel()[ov['cell']], ov['mag'], ov['w'], ov['tri']
    mq = lay['mesh_quad']
    M, K = mq['mag'].shape
    yp = yn.double().cpu().numpy()
    if arch == 'FNO_BipartiteGNN_Elem':
        mm = np.repeat(_mag_np(yp)[:, None], K, axis=1)
    else:
        mm = surface_values_p1(lay['tri'], yp, mq['bary'])
    w = np.repeat((mq['area'] / K)[:, None], K, axis=1)
    return mm.ravel(), mq['mag'].ravel(), w.ravel(), np.repeat(np.arange(M), K)


def _floor_points(kind, lay):
    """O gabarito na própria representação (piso de cada tipo de superfície)."""
    mq, ov = lay['mesh_quad'], lay['overlay']
    if kind == 'grade':
        yh = lay['v1']['y_hw']
        return np.hypot(yh[0], yh[1]).ravel()[ov['cell']], ov['mag'], ov['w']
    M, K = mq['mag'].shape
    if kind == 'P1_nos':
        mm = surface_values_p1(lay['tri'], lay['v1']['node_y'].astype(np.float64), mq['bary'])
    else:   # P0_elem: B suavizado no centróide (gabarito da Elem)
        mm = np.repeat(_mag_np(lay['elem']['node_y'].astype(np.float64))[:, None], K, axis=1)
    return mm.ravel(), mq['mag'].ravel(), np.repeat((mq['area'] / K)[:, None], K, axis=1).ravel()


def _surface_kind(arch):
    return 'grade' if arch in GRID_ARCHS else 'P0_elem' if arch == 'FNO_BipartiteGNN_Elem' else 'P1_nos'


NODAL_COLS = ('MAE_pt', 'L2_pt', 'L1_area', 'L2_area')   # colunas nodal_* do .npz (T)


def _count_stats(v):
    """Mediana, mínimo, p5, p95 e máximo de uma contagem por amostra (P2)."""
    v = np.asarray(v)
    return dict(median=float(np.median(v)), min=int(v.min()), p5=float(np.percentile(v, 5)),
                p95=float(np.percentile(v, 95)), max=int(v.max()), n=int(v.size))


def _nodal_mag(arch, lay, yn):
    """|B| da saída do modelo nos nós [S]: FNO2d já interpolado por predict (modo da
    bateria), Elem levada aos nós por elem_to_nodes, demais = saída nodal."""
    yp = yn.double().cpu().numpy()
    if arch == 'FNO_BipartiteGNN_Elem':
        yp = elem_to_nodes(lay, yp)
    return _mag_np(yp)


def run_eval(out_dir: Path, max_samples=None, subset='test'):
    """Erro de superfície oficial (CLAUDE.md, "Convenção de erro"): superfície discreta de
    cada modelo × superfície do FEMM (B suavizado P1 descontínuo); L1_A, L2_A, B_ref.
    Secundário (P1, 2026-10-20): erro de |B| ponto a ponto nos nós (gabarito node_y do
    layout v1, área dual) -> runs[k]['nodal'] e floor_nodal (piso da grade: y_hw
    interpolado nos nós, floor_nodes); não substitui a superfície.
    Saída: eval_smooth.json (runs + floor_surface + floor_nodal) e eval_smooth_per_sample.npz
    (por run e por piso: [n_amostras, 2] = L1_A, L2_A em T; nodal_*: [n_amostras, 4] =
    NODAL_COLS em T)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = design_rows()
    samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], subset, max_samples, N_SAMPLES)
    runs = load_set(ROOT_S)
    for k, r in runs.items():
        print(f"  {k:26s} {r['run']} best ép.{r['epoch']}  superfície {_surface_kind(r['arch'])}"
              f"  interp {INTERP_MODE}")

    acc = {(k, sub): Accum() for k in runs for sub in SUBSETS}
    acc_floor = {(kd, sub): Accum() for kd in SURF_KINDS for sub in SUBSETS}
    acc_nodal = {(k, sub): Accum() for k in runs for sub in SUBSETS}
    acc_floor_nodal = {sub: Accum() for sub in SUBSETS}
    per = {k: [] for k in runs}
    per_floor = {kd: [] for kd in SURF_KINDS}
    per_nodal = {k: [] for k in runs}
    per_floor_nodal = []
    ids, subs = [], []
    n_nodes, n_elems = [], []          # estatística da malha (P2)

    groups = {}
    for name, sub, idx, p in samples:
        groups.setdefault(name, []).append((sub, idx, p))

    t0 = time.time()
    done = 0
    for gi, (name, items) in enumerate(groups.items()):
        t = time.time()
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items))) as ex:
            futs = [ex.submit(parse_smooth, p, rows[idx], with_elem=True, with_overlay=True)
                    for _, idx, p in items]
            for (sub, idx, p), f in zip(items, futs):
                lay = f.result()
                ts = to_torch_sample(lay)
                ids.append(idx)
                subs.append(sub)
                n_nodes.append(int(lay['v1']['node_x'].shape[0]))
                n_elems.append(int(len(lay['tri'])))
                for kd in SURF_KINDS:
                    a = Accum()
                    a.add_surface(*_floor_points(kd, lay))
                    acc_floor[(kd, sub)] = merge([acc_floor[(kd, sub)], a])
                    r_ = a.report()
                    per_floor[kd].append((r_['L1_area'], r_['L2_area']))
                # nodal (secundário): gabarito e área dual do layout v1
                area_n = lay['v1']['node_x'][:, 2].astype(np.float64)
                mag_tn = _mag_np(lay['v1']['node_y'].astype(np.float64))
                a = Accum()
                a.add(mag(floor_nodes(ts)), mag_tn, area_n)
                acc_floor_nodal[sub] = merge([acc_floor_nodal[sub], a])
                r_ = a.report()
                per_floor_nodal.append(tuple(r_[c] for c in NODAL_COLS))
                for k, r in runs.items():
                    out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
                    mm, mf, w, _ = _surface_points(r['arch'], lay, out_hw, yn)
                    a = Accum()
                    a.add_surface(mm, mf, w)
                    acc[(k, sub)] = merge([acc[(k, sub)], a])
                    r_ = a.report()
                    per[k].append((r_['L1_area'], r_['L2_area']))
                    a = Accum()
                    a.add(_nodal_mag(r['arch'], lay, yn), mag_tn, area_n)
                    acc_nodal[(k, sub)] = merge([acc_nodal[(k, sub)], a])
                    r_ = a.report()
                    per_nodal[k].append(tuple(r_[c] for c in NODAL_COLS))
                done += 1
        print(f'  [{gi + 1}/{len(groups)}] {name} ({items[0][0]})  {time.time() - t:.0f}s  '
              f'(amostras {done}, {(time.time() - t0) / 60:.1f} min)', flush=True)

    subs_arr = np.array(subs)
    present = [s for s in SUBSETS if (subs_arr == s).any()]

    def reports(getter):
        r = {sub: getter([sub]).report() for sub in present}
        r['total'] = getter(list(SUBSETS)).report()
        return r

    res = dict(n_samples=done, n_by_subset={s: int((subs_arr == s).sum()) for s in SUBSETS},
               metric='erro de superfície de |B| (superfície do modelo × FEMM P1 descontínuo)',
               runs={},
               floor_surface={kd: reports(lambda ss, kd=kd: merge([acc_floor[(kd, s)] for s in ss]))
                              for kd in SURF_KINDS})
    for k, r in runs.items():
        arr = np.array(per[k])
        res['runs'][k] = dict(
            arch=r['arch'], loss=r['loss'], run=r['run'], best_epoch=r['epoch'],
            surface_kind=_surface_kind(r['arch']),
            surface=reports(lambda ss, k=k: merge([acc[(k, s)] for s in ss])),
            per_sample_L1_A={sub: pstats(arr[subs_arr == sub, 0]) for sub in present})
    # nível secundário (P1, 2026-10-20) -- chaves novas, as acima não mudam
    for k in runs:
        res['runs'][k]['nodal'] = reports(lambda ss, k=k: merge([acc_nodal[(k, s)] for s in ss]))
    res['floor_nodal'] = reports(lambda ss: merge([acc_floor_nodal[s] for s in ss]))
    res['nodal_metric'] = ('erro de |B| ponto a ponto nos nós (secundário): gabarito node_y do '
                           'layout v1, área dual do nó; FNO2d interpolado nos nós (modo da '
                           'bateria), Elem -> nós por elem_to_nodes; piso = y_hw do gabarito '
                           f'interpolado nos nós ({INTERP_MODE})')
    # estatística da malha (P2, 2026-10-20)
    res['mesh_stats'] = {name: _count_stats(v) for name, v in (('n_nodes', n_nodes),
                                                               ('n_elems', n_elems))}

    np.savez_compressed(
        out_dir / 'eval_smooth_per_sample.npz', sample_idx=np.array(ids), subset=subs_arr,
        **{k: np.array(v) for k, v in per.items()},
        **{f'floor_{kd}': np.array(v) for kd, v in per_floor.items()},
        **{f'nodal_{k}': np.array(v) for k, v in per_nodal.items()},
        nodal_floor_grade=np.array(per_floor_nodal), nodal_columns=np.array(NODAL_COLS),
        n_nodes=np.array(n_nodes), n_elems=np.array(n_elems))
    (out_dir / 'eval_smooth.json').write_text(json.dumps(res, indent=2))
    print(f'\nsalvo em {out_dir} ({(time.time() - t0) / 60:.1f} min)')
    print(f'{done} amostras — erro de superfície, % de B_ref (L1_A / L2_A)')
    for k, v in res['runs'].items():
        g = v['surface']['total']
        print(f"  {k:26s} [{v['surface_kind']:7s}] L1 {g['L1_area_pct']:5.2f}%  "
              f"L2 {g['L2_area_pct']:5.2f}%  (B_ref {g['B_ref']:.4f} T)")
    for kd in SURF_KINDS:
        g = res['floor_surface'][kd]['total']
        print(f"  {'piso ' + kd:26s} [{kd:7s}] L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%")
    print(f'{done} amostras — [secundário] erro de |B| nos nós, % de B_ref nodal '
          f'(pontual MAE / L2 / L1_A / L2_A)')
    for k, v in res['runs'].items():
        g = v['nodal']['total']
        print(f"  {k:26s} pt {g['MAE_pt_pct']:5.2f}%  L2pt {g['L2_pt_pct']:5.2f}%  "
              f"L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%  (B_ref {g['B_ref']:.4f} T)")
    g = res['floor_nodal']['total']
    print(f"  {'piso grade (nós)':26s} pt {g['MAE_pt_pct']:5.2f}%  L2pt {g['L2_pt_pct']:5.2f}%  "
          f"L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%")
    print('malha (mediana / mín / p5 / p95 / máx):')
    for name, g in res['mesh_stats'].items():
        print(f"  {name:8s} {g['median']:.0f} / {g['min']} / {g['p5']:.0f} / {g['p95']:.0f} / {g['max']}")


def run_figures(data_dir: Path, fig_dir: Path):
    """Curvas de treino, barras do erro de superfície (+ piso de cada representação) e
    campo P50/P95 do L1_A de superfície da REF_RUN no subset avaliado pelo eval.
    O mapa de campo (3_campo) é só visualização de |B| nos nós."""
    import matplotlib.tri as mtri
    apply_style()
    res = json.loads((data_dir / 'eval_smooth.json').read_text())
    per = np.load(data_dir / 'eval_smooth_per_sample.npz')
    save_fig(fig_training(find_runs(ROOT_S), ARCHS), fig_dir / '1_curvas_treino')
    save_fig(fig_bars(res, RUN_ORDER), fig_dir / '2_barras_erro')

    sub_name = _eval_subset(per)
    sub = per['subset'] == sub_name
    l1 = per[REF_RUN][:, 0]
    rows = design_rows()
    runs = load_set(ROOT_S, losses=('mae',))
    chosen = {}
    for tag, q in (('P50', 50), ('P95', 95)):
        cand = np.flatnonzero(sub)
        # method='lower': o alvo é sempre o valor de uma amostra
        target = np.percentile(l1[cand], q, method='lower')
        pos = cand[np.argmin(np.abs(l1[cand] - target))]
        idx = int(per['sample_idx'][pos])
        chosen[tag] = dict(sample_idx=idx, subset=sub_name, ref_run=REF_RUN,
                           L1_A_surface_ref=float(l1[pos]))
        lay = parse_smooth(sample_path(idx), rows[idx], with_elem=True)
        nodes, elems = _mesh(idx)
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elems[:, :3])
        gt = _mag_np(lay['v1']['node_y'].astype(np.float64))
        preds = _node_mags(runs, lay, to_torch_sample(lay))
        preds = {a: preds[a] for a in ARCHS}
        r_m, _ = gap_radius_mm(rows[idx])
        th0, half = np.deg2rad(60.0), 4.5
        for zoom in (None, (r_m * np.cos(th0), r_m * np.sin(th0), half)):
            where = (f'zoom no entreferro (r≈{r_m:.1f} mm, θ≈60°, janela {2 * half:.0f} mm)'
                     if zoom else 'setor de 120°')
            title = (f'Amostra {idx} ({tag} do L1_A de superfície de {REF_RUN} em {sub_name}, '
                     f'{l1[pos] * 1e3:.1f} mT) — |B| nos nós, modelos mae — {where}')
            save_fig(fig_field(tri, gt, preds, title, zoom=zoom),
                     fig_dir / f"3_campo_{tag}_amostra{idx}{'_zoom' if zoom else ''}")
        print(f'  {tag}: amostra {idx}')
    (data_dir / 'figures_smooth.json').write_text(json.dumps(chosen, indent=2))
    print(f'figuras em {fig_dir}')


def _floor_tri_idx(kind, lay):
    """Triângulo do FEMM de cada ponto de _floor_points(kind, lay) (mesma ordem)."""
    if kind == 'grade':
        return lay['overlay']['tri']
    M, K = lay['mesh_quad']['mag'].shape
    return np.repeat(np.arange(M), K)


def _eval_subset(per):
    """O subset ('test' ou 'val') avaliado pelo eval cujo .npz é `per` (um só por eval)."""
    subs = sorted(set(per['subset'].tolist()))
    if len(subs) != 1:
        raise ValueError(f'eval_smooth_per_sample.npz com subsets {subs} (esperado um só)')
    return subs[0]


def _pick_by_pct(per, q, ref_run=REF_RUN):
    """Amostra do percentil q do L1_A de superfície de ref_run no .npz do eval (mesma regra
    de run_figures: subset do eval; method='lower') -> (idx, seleção)."""
    sub_name = _eval_subset(per)
    cand = np.flatnonzero(per['subset'] == sub_name)
    l1 = per[ref_run][:, 0]
    target = np.percentile(l1[cand], q, method='lower')
    pos = cand[np.argmin(np.abs(l1[cand] - target))]
    idx = int(per['sample_idx'][pos])
    return idx, dict(pct=float(q), ref_run=ref_run, subset=sub_name, n_candidates=int(len(cand)),
                     L1_A_surface_ref_T=float(l1[pos]))


def run_surface(sample_idx, loss, data_dir: Path, fig_dir: Path, pct=None, eval_dir=None):
    """Erro de superfície oficial em 1 amostra: L1_A/L2_A de cada modelo e mapa da
    densidade do erro por triângulo do FEMM (surface_error_per_triangle), com o piso da
    representação em grade como painel extra e contornos dos materiais (P4, 2026-10-20).
    pct: escolhe a amostra pelo percentil do L1_A de superfície da REF_RUN no .npz do eval
    (eval_dir) -- sample_idx ignorado."""
    apply_style()
    data_dir.mkdir(parents=True, exist_ok=True)
    selection = None
    if pct is not None:
        per = np.load((eval_dir or OUT_ROOT / 'dados') / 'eval_smooth_per_sample.npz')
        sample_idx, selection = _pick_by_pct(per, pct)
        print(f"P{pct:g} do L1_A de superfície de {REF_RUN} ({selection['subset']}, "
              f"{selection['n_candidates']} amostras): amostra {sample_idx}")
    row = design_rows()[sample_idx]
    print(f'amostra {sample_idx}: {sample_path(sample_idx)}')
    lay = parse_smooth(sample_path(sample_idx), row, with_elem=True, with_overlay=True)
    ts = to_torch_sample(lay)
    n_tri = len(lay['tri'])
    runs = load_set(ROOT_S, losses=(loss,))
    res = dict(sample_idx=sample_idx, loss=loss, n_tri=int(n_tri), selection=selection,
               runs={}, erros={})
    E = {}
    for r in runs.values():
        a = r['arch']
        out_hw, yn = predict(a, r['model'], r['normalizer'], ts)
        mm, mf, w, t_idx = _surface_points(a, lay, out_hw, yn)
        acc = Accum()
        acc.add_surface(mm, mf, w)
        x = acc.report()
        E[a] = surface_error_per_triangle(n_tri, t_idx, mm, mf, w)
        res['erros'][a] = dict(surface_kind=_surface_kind(a), B_ref_T=x['B_ref'],
                               L1_A_T=x['L1_area'], L1_A_pct=x['L1_area_pct'],
                               L2_A_T=x['L2_area'], L2_A_pct=x['L2_area_pct'])
        res['runs'][a] = dict(run_dir=r['run_dir'], epoch=r['epoch'])
        print(f"  {a:22s} [{_surface_kind(a):7s}] L1_A={x['L1_area']:.4f} T ({x['L1_area_pct']:.2f}%)  "
              f"L2_A={x['L2_area']:.4f} T ({x['L2_area_pct']:.2f}%)")
    # piso da representação em grade, por triângulo
    mm, mf, w = _floor_points('grade', lay)
    acc = Accum()
    acc.add_surface(mm, mf, w)
    x = acc.report()
    E_floor = surface_error_per_triangle(n_tri, _floor_tri_idx('grade', lay), mm, mf, w)
    res['floor_grade'] = dict(surface_kind='grade', B_ref_T=x['B_ref'],
                              L1_A_T=x['L1_area'], L1_A_pct=x['L1_area_pct'],
                              L2_A_T=x['L2_area'], L2_A_pct=x['L2_area_pct'])
    print(f"  {'piso grade':22s} [grade  ] L1_A={x['L1_area']:.4f} T ({x['L1_area_pct']:.2f}%)  "
          f"L2_A={x['L2_area']:.4f} T ({x['L2_area_pct']:.2f}%)")
    nodes, elems = _mesh(sample_idx)
    iface = interface_edges(lay['tri'], lay['elem_mat'])
    stem = f'sample_{sample_idx:06d}_{loss}' + (f'_P{pct:g}' if pct is not None else '')
    (data_dir / f'{stem}.json').write_text(json.dumps(res, indent=1))
    np.savez_compressed(data_dir / f'{stem}.npz', xy=nodes[:, :2], elems=elems[:, :3],
                        iface_edges=iface, e_tri_floor_grade=E_floor,
                        **{f'e_tri_{a}': E[a] for a in E})
    save_fig(fig_surface_error_map(res, nodes[:, :2], elems[:, :3], E, [a for a in ARCHS if a in E],
                                   iface, ('piso grade', E_floor, res['floor_grade'])),
             fig_dir / stem, pdf=False, dpi=150)
    print(f'figura: {fig_dir / stem}.png')


def run_regions(out_dir: Path, fig_dir: Path, max_samples=None, d_int=None, d_canto=None,
                ang_canto=30.0, subset='test'):
    """Estratificação espacial do erro de superfície (P5): cada triângulo do FEMM em
    interior / interface / canto (regions.classify_triangles) e o erro de superfície de
    cada run e de cada piso acumulado por região (pontos de quadratura -> triângulo).
    Mesmo conjunto do eval. Padrão: d_int = 1 passo radial da grade, d_canto = 2 passos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dr = (R_EXT_MM - R_IN_MM) / N_R
    d_int = dr if d_int is None else d_int
    d_canto = 2 * dr if d_canto is None else d_canto
    params = dict(d_int_mm=d_int, d_canto_mm=d_canto, ang_canto_deg=ang_canto,
                  grid_radial_step_mm=dr, prioridade='canto > interface > interior',
                  material='por rótulo (lay[elem_mat])')
    print('parâmetros:', params)
    rows = design_rows()
    samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], subset, max_samples, N_SAMPLES)
    runs = load_set(ROOT_S)
    acc = {k: RegionAccum() for k in runs}
    acc_floor = {kd: RegionAccum() for kd in SURF_KINDS}
    n_tri_region = np.zeros(len(REGIONS), np.int64)
    fig_done = False
    subs = []
    groups = {}
    for name, sub, idx, p in samples:
        groups.setdefault(name, []).append((sub, idx, p))
    t0 = time.time()
    for gi, (name, items) in enumerate(groups.items()):
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items))) as ex:
            futs = [ex.submit(parse_smooth, p, rows[idx], with_elem=True, with_overlay=True)
                    for _, idx, p in items]
            for (sub, idx, p), f in zip(items, futs):
                lay = f.result()
                ts = to_torch_sample(lay)
                subs.append(sub)
                nodes, _ = _mesh(idx)
                xy = nodes[:, :2]
                region, iface, corners = classify_triangles(xy, lay['tri'], lay['elem_mat'],
                                                            d_int, d_canto, ang_canto)
                n_tri_region += np.bincount(region, minlength=len(REGIONS))
                if not fig_done:
                    apply_style()
                    r_m, _ = gap_radius_mm(rows[idx])
                    th0 = np.deg2rad(60.0)
                    title = (f'Amostra {idx}: regiões (d_int = {d_int:.3f} mm, d_canto = '
                             f'{d_canto:.3f} mm, ângulo de canto > {ang_canto:g}°)')
                    save_fig(fig_regions(xy, lay['tri'], region, iface, corners, REGIONS, title,
                                         zoom=(r_m * np.cos(th0), r_m * np.sin(th0), 3.0)),
                             fig_dir / f'regions_sample_{idx:06d}', pdf=False, dpi=200)
                    fig_done = True
                for kd in SURF_KINDS:
                    mm, mf, w = _floor_points(kd, lay)
                    acc_floor[kd].add(mm, mf, w, region[_floor_tri_idx(kd, lay)])
                for k, r in runs.items():
                    out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
                    mm, mf, w, t_idx = _surface_points(r['arch'], lay, out_hw, yn)
                    acc[k].add(mm, mf, w, region[t_idx])
        print(f'  [{gi + 1}/{len(groups)}] {name}  (amostras {len(subs)}, '
              f'{(time.time() - t0) / 60:.1f} min)', flush=True)

    subs_arr = np.array(subs)
    res = dict(n_samples=len(subs), n_by_subset={s: int((subs_arr == s).sum()) for s in SUBSETS},
               params=params, regions=list(REGIONS),
               n_triangles_by_region={n: int(c) for n, c in zip(REGIONS, n_tri_region)},
               note='L1_A/L2_A de cada região normalizados pela área da região, em % do B_ref '
                    'GLOBAL; area_frac = área da região / área total; L1_contrib_frac = '
                    'fração de ∫e dA total vinda da região',
               runs={k: dict(arch=r['arch'], loss=r['loss'], run=r['run'],
                             surface_kind=_surface_kind(r['arch']), **acc[k].report())
                     for k, r in runs.items()},
               floor_surface={kd: acc_floor[kd].report() for kd in SURF_KINDS})
    (out_dir / 'regions.json').write_text(json.dumps(res, indent=2))
    print(f'\n{len(subs)} amostras — erro de superfície por região, % do B_ref global '
          f'(L1_A / L2_A) [fração da área | fração do L1_A total]')
    head = ''.join(f'{n:>34s}' for n in REGIONS)
    print(f"  {'':26s}{head}")
    items_ = [(k, v) for k, v in res['runs'].items()] + \
             [('piso ' + kd, v) for kd, v in res['floor_surface'].items()]
    for k, v in items_:
        cells = ''.join(f"{v[n]['L1_area_pct']:6.2f} /{v[n]['L2_area_pct']:6.2f} "
                        f"[{v[n]['area_frac']:.2f}|{v[n]['L1_contrib_frac']:.2f}]".rjust(34)
                        for n in REGIONS)
        print(f'  {k:26s}{cells}')
    print(f"salvo em {out_dir / 'regions.json'}; figura em {fig_dir}")


TORQUE_TMP = Path('data/temp')     # soluções FEMM das varreduras (as mesmas dos JSONs antigos)
TORQUE_WORKERS, TORQUE_FEMM_WORKERS = 6, 8


def _torque_runs(loss, archs):
    runs = load_set(ROOT_S, losses=(loss,))
    return {k: r for k, r in runs.items() if archs is None or r['arch'] in archs}


def _torque_sweep_dir(idx, work_dir=TORQUE_TMP):
    """Pasta da varredura θe de 1 amostra (amostra 15 na raiz, como nos JSONs antigos)."""
    root = Path(work_dir) / 'torque_theta_e'
    return root if idx == 15 else root / f'sample_{idx:06d}'


def _femm_batch_one(idx, k, th, row, work_dir=TORQUE_TMP):
    d = Path(work_dir) / 'torque_theta_e_batch' / f'sample_{idx:06d}'
    gz, js = d / f'theta_{k:02d}.ans.gz', d / f'theta_{k:02d}.json'
    if gz.exists() and js.exists() and femm_ref_ok(json.loads(js.read_text()), row):
        return
    tmp = Path(work_dir) / 'torque_theta_e_batch' / 'tmp' / f'{idx}_{k}'
    try:
        t22 = femm_solve(row, th, tmp, gz)
        js.write_text(json.dumps(femm_ref(row, th, t22)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_torque(mode, out_dir: Path, loss='mae', archs=None, frame='polar', femm=False,
               idxs=(15,), n_theta=None, n_samples=80, n_per_chunk=8, limit=None,
               raw_dir=None, work_dir=None, subset='test'):
    """Torque (P6): theta-e (varredura θe de amostras), batch (n_samples amostras do subset
    estratificadas por corrente × n_theta ângulos, resumo da fundamental) ou scatter (torque
    das n_per_chunk primeiras amostras de cada chunk do subset). FEMM T22 × gt × smooth ×
    modelos da loss escolhida. subset: 'test' (padrão) ou 'val'.
    limit: só os primeiros `limit` ângulos (theta-e/batch) ou amostras (scatter) -- smoke.
    raw_dir: raw de onde vêm valid_designs.csv e os .ans (None: samples.RAW_DIR, o do
    DatagenConfig); work_dir: soluções FEMM das varreduras e temporários (None: TORQUE_TMP)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    work_dir = Path(work_dir) if work_dir is not None else TORQUE_TMP
    rows = read_rows(raw_dir)
    runs = _torque_runs(loss, archs)
    split_run = find_runs(ROOT_S)[('FNO_BipartiteGNN_Elem', loss)]   # mesmo split em todas
    meta = dict(loss=loss, frame=frame, split_run=str(split_run), subset=subset,
                runs={r['arch']: dict(run_dir=r['run_dir'], best_epoch=r['epoch']) for r in runs.values()})
    pk = [f"pred_{r['arch']}" for r in runs.values()]
    mNm = lambda v: np.asarray(v) * 1e3
    if mode == 'theta-e':
        n_theta = n_theta or 24
        for idx in idxs:
            row, sw = rows[idx], _torque_sweep_dir(idx, work_dir)
            th_all = torque_thetas(n_theta)
            cache = sw / 'femm_ref.json'
            stale = cache.exists() and not all(femm_ref_ok(r, row) for r in json.loads(cache.read_text()))
            if stale:
                print(f'  s{idx}: femm_ref.json com outra profundidade (ou sem stack_length_mm) '
                      f'-> descartado, ressolvendo', flush=True)
            if femm or stale:
                ref = [femm_ref(row, th, femm_solve(row, th, sw / 'tmp', sw / 'raw' / f'theta_{k:03d}.ans.gz'))
                       for k, th in enumerate(th_all)]
                cache.write_text(json.dumps(ref, indent=1))
            ref = json.loads(cache.read_text())[:limit]
            T = {'femm_T22': [3 * r['T22_sector'] for r in ref], 'gt': [], 'gt_smooth': [],
                 **{k: [] for k in pk}}
            with ProcessPoolExecutor(TORQUE_WORKERS) as ex:
                futs = [ex.submit(parse_torque_sample, sw / 'raw' / f'theta_{k:03d}.ans.gz', row)
                        for k in range(len(ref))]
                for k, f in enumerate(futs):
                    t = sample_torques(f.result(), row, runs, frame)
                    T['gt'].append(t['gt']); T['gt_smooth'].append(t['smooth'])
                    for kk in pk:
                        T[kk].append(t[kk])
                    print(f"  s{idx} θe={ref[k]['theta_e_deg']:5.1f}°  FEMM {T['femm_T22'][k] * 1e3:8.1f}  "
                          f"gt {t['gt'] * 1e3:8.1f}  " + '  '.join(f'{kk[5:]} {t[kk] * 1e3:7.1f}' for kk in pk)
                          + ' mNm', flush=True)
            T = {k: np.array(v) for k, v in T.items()}
            rms = lambda x: float(np.sqrt(np.mean(np.square(x))) * 1e3)
            res = dict(sample_idx=idx, **meta, current_amplitude_A=float(row['current_amplitude [A]']),
                       rotor_phase_deg=float(row['rotor_phase [deg]']), gap_mm=list(gap(row)),
                       theta_e_deg=[r['theta_e_deg'] for r in ref], T_Nm={k: v.tolist() for k, v in T.items()},
                       rms_mNm={k: rms(v) for k, v in T.items()},
                       max_mNm={k: float(v.max() * 1e3) for k, v in T.items()},
                       min_mNm={k: float(v.min() * 1e3) for k, v in T.items()},
                       rms_err_vs_femm_mNm={k: rms(T[k] - T['femm_T22']) for k in ['gt', 'gt_smooth'] + pk},
                       rms_err_pred_vs_gt_mNm={k: rms(T[k] - T['gt']) for k in pk})
            (out_dir / f'theta_e_s{idx}_{loss}.json').write_text(json.dumps(res, indent=2))
            print(json.dumps(res['rms_err_vs_femm_mNm'], indent=1))
    elif mode == 'batch':
        n_theta = n_theta or 16
        ids = select_batch_samples(split_run, rows, n_samples, subset=subset)
        root = work_dir / 'torque_theta_e_batch'
        # C2: com --femm, todos os ângulos (_femm_batch_one pula só os caches válidos); sem --femm,
        # só os caches existentes com outra profundidade (ou sem o campo) são descartados e ressolvidos
        tasks = [(i, k, th) for i in ids for k, th in enumerate(torque_thetas(n_theta))]
        if not femm:
            jsons = {(i, k): root / f'sample_{i:06d}' / f'theta_{k:02d}.json' for i, k, _ in tasks}
            tasks = [(i, k, th) for i, k, th in tasks if jsons[i, k].exists()
                     and not femm_ref_ok(json.loads(jsons[i, k].read_text()), rows[i])]
            if tasks:
                print(f'  {len(tasks)} cache(s) de T22 com outra profundidade -> ressolvendo', flush=True)
        if tasks:
            with ProcessPoolExecutor(TORQUE_FEMM_WORKERS) as ex:
                for f in [ex.submit(_femm_batch_one, i, k, th, rows[i], work_dir) for i, k, th in tasks]:
                    f.result()
        res, rows_ = {}, {kk: [] for kk in pk}
        with ProcessPoolExecutor(TORQUE_WORKERS) as ex:
            for n, i in enumerate(ids, 1):
                d = root / f'sample_{i:06d}'
                ks = [k for k in range(n_theta) if (d / f'theta_{k:02d}.json').exists()][:limit]
                futs = [ex.submit(parse_torque_sample, d / f'theta_{k:02d}.ans.gz', rows[i])
                        for k in ks]
                cur = dict(theta_e_deg=[], femm=[], gt=[], smooth=[], **{kk: [] for kk in pk})
                for k, f in zip(ks, futs):
                    ref = json.loads((d / f'theta_{k:02d}.json').read_text())
                    t = sample_torques(f.result(), rows[i], runs, frame)
                    cur['theta_e_deg'].append(ref['theta_e_deg'])
                    cur['femm'].append(3 * ref['T22_sector'])
                    for kk in ['gt', 'smooth'] + pk:
                        cur[kk].append(t[kk])
                row = rows[i]
                geom = {k: float(row[k]) for k in ('stator_outer_diameter [mm]', 'pole_thickness [mm]',
                                                   'pole_embrance', 'slot_Bs0 [mm]', 'slot_Bs1 [mm]',
                                                   'slot_Bs2 [mm]', 'rotor_inner_diameter [mm]')}
                res[i] = dict(curves=cur, I=float(row['current_amplitude [A]']),
                              rotor_phase=float(row['rotor_phase [deg]']), geom=geom)
                if len(ks) >= 3:
                    th = np.array(cur['theta_e_deg'])
                    for kk in pk:
                        rows_[kk].append(dict(idx=int(i), I=res[i]['I'],
                                              **curve_row(th, mNm(cur['femm']), mNm(cur[kk]), mNm(cur['gt'])),
                                              **{g.split(' [')[0]: v for g, v in geom.items()}))
                print(f'  [{n}/{len(ids)}] amostra {i}  {len(ks)} ângulos', flush=True)
                if limit and n >= limit:
                    break
        (out_dir / f'theta_e_batch_{loss}.json').write_text(json.dumps(dict(**meta, samples=res)))
        summ = {kk[5:]: batch_summary(v) for kk, v in rows_.items() if v}
        (out_dir / f'theta_e_batch_{loss}_summary.json').write_text(
            json.dumps(dict(**meta, summary=summ, per_sample={kk[5:]: v for kk, v in rows_.items()}), indent=1))
        print(json.dumps(summ, indent=1))
    elif mode == 'scatter':
        ids = scatter_ids(split_run, n_per_chunk, subset=subset)[:limit]
        out = dict(idx=[], T_gt=[], T_smooth=[], I=[], theta_e=[], rotor_phase=[], **{f'T_{kk}': [] for kk in pk})
        with ProcessPoolExecutor(TORQUE_WORKERS) as ex:
            for b in range(0, len(ids), 32):
                batch = ids[b:b + 32]
                futs = [ex.submit(parse_torque_sample, raw_dir / f'sample_{i:06d}.ans.gz', rows[i])
                        for i in batch]
                for i, f in zip(batch, futs):
                    t = sample_torques(f.result(), rows[i], runs, frame)
                    r = rows[i]
                    out['idx'].append(i); out['T_gt'].append(t['gt']); out['T_smooth'].append(t['smooth'])
                    for kk in pk:
                        out[f'T_{kk}'].append(t[kk])
                    out['I'].append(float(r['current_amplitude [A]']))
                    out['theta_e'].append(float(r['current_angle [deg]']))
                    out['rotor_phase'].append(float(r['rotor_phase [deg]']))
                print(f"  {len(out['idx'])}/{len(ids)}", flush=True)
        stats = {kk[5:]: scatter_stats(mNm(out['T_gt']), mNm(out[f'T_{kk}']), mNm(out['T_smooth']))
                 for kk in pk}
        (out_dir / f'scatter_{loss}.json').write_text(json.dumps(dict(**meta, stats=stats, **out), indent=1))
        print(json.dumps(stats, indent=1))
    print(f'salvo em {out_dir}')


def run_saturation(out_dir: Path, fig_dir: Path, max_samples=None, knee_fracs=(0.5,),
                   check_femm=False, subset='test'):
    """Saturação do aço (P7), só sobre o gabarito: curva B-H do .ans (conferida igual em todas
    as amostras), pontos de operação dos elementos de aço (|B| suavizado no centróide) e
    μ_eff; figura A (curva + distribuição) e B (|B| no aço nas amostras P50 e de maior |B|
    médio no aço, com contornos no joelho). Mesmo conjunto do eval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    rows = design_rows()
    samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], subset, max_samples, N_SAMPLES)
    edges = np.linspace(0, 2.8, 141)
    hist = np.zeros(len(edges) - 1)
    all_b, all_w = [], []
    ids, mean_b, n_above = [], [], 0
    curve0, n_diff = None, 0
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        futs = [ex.submit(steel_operating_points, p, rows[idx]) for _, _, idx, p in samples]
        for (_, _, idx, p), f in zip(samples, futs):
            d = f.result()
            curves = d['curves']
            if curve0 is None:
                curve0 = curves[0]
            n_diff += sum(not (np.array_equal(c[1], curve0[1]) and np.array_equal(c[2], curve0[2]))
                          for c in curves)
            hist += np.histogram(np.clip(d['bmag'], 0, edges[-1] - 1e-9), edges, weights=d['area'])[0]
            all_b.append(d['bmag'].astype(np.float32)); all_w.append(d['area'].astype(np.float32))
            ids.append(idx)
            mean_b.append(float((d['bmag'] * d['area']).sum() / d['area'].sum()))
            n_above += int((d['bmag'] > curve0[1][-1]).sum())
    bh = FemmBH(curve0[1], curve0[2])
    b, w = np.concatenate(all_b).astype(float), np.concatenate(all_w).astype(float)
    knees = [bh.knee(fr) for fr in knee_fracs]
    p5, p50, p95 = (weighted_percentile(b, w, q) for q in (5, 50, 95))
    mean_b = np.array(mean_b)
    i50 = int(np.flatnonzero(mean_b == np.percentile(mean_b, 50, method='lower'))[0])
    imax = int(np.argmax(mean_b))
    res = dict(n_samples=len(ids), bh_curve=dict(name=curve0[0], B_T=curve0[1].tolist(),
                                                  H_A_m=curve0[2].tolist(), n_points=len(curve0[1])),
               bh_identical_all_samples=n_diff == 0, n_bh_blocks_different=int(n_diff),
               interpolation=('FEMM (xfemm CMMaterialProp::GetSlopes/GetH): inclinações por spline '
                              'cúbica natural (sistema tridiagonal), reparo por média móvel de 3 pontos '
                              'se algum trecho não for monótono, Hermite cúbico entre pontos, '
                              'extrapolação linear com a inclinação do último ponto'),
               n_repair=int(bh.n_repair), mu_label=MU_LABEL,
               knee=knees, knee_definition='menor B acima do pico de μ_eff em que μ_eff = frac·máx',
               operating_points=dict(n_elements=int(len(b)), area_mm2=float(w.sum()),
                                     p5_T=p5, p50_T=p50, p95_T=p95,
                                     mu_eff_at_p5=float(bh.mu_eff(np.array([p5]))[0]),
                                     mu_eff_at_p50=float(bh.mu_eff(np.array([p50]))[0]),
                                     mu_eff_at_p95=float(bh.mu_eff(np.array([p95]))[0])),
               above_last_bh_point=dict(B_last_T=float(curve0[1][-1]), n_elements=int(n_above),
                                        note='contagem apenas; o máximo de |B| não é usado como resultado'),
               hist=dict(edges_T=edges.tolist(), area_mm2=hist.tolist()),
               samples_fig_b=dict(P50_mean_B_steel=dict(sample_idx=ids[i50], mean_B_T=float(mean_b[i50])),
                                  max_mean_B_steel=dict(sample_idx=ids[imax], mean_B_T=float(mean_b[imax]))))
    if check_femm:
        d = steel_operating_points(sample_path(ids[0]), rows[ids[0]])
        res['femm_check'] = dict(sample_idx=ids[0], **check_against_femm(sample_path(ids[0]), bh,
                                                                          d['cen'], d['bmag']))
    (out_dir / 'saturation.json').write_text(json.dumps(res, indent=2))
    save_fig(fig_bh(bh, edges, hist, p5, p95, knees[0], len(ids), n_above), fig_dir / 'saturacao_A_curva_BH')
    panels = []
    for tag, pos in (('P50 do |B| médio no aço', i50), ('maior |B| médio no aço', imax)):
        idx = ids[pos]
        d = steel_operating_points(sample_path(idx), rows[idx])
        nodes, elems = _mesh(idx)
        panels.append((f'Amostra {idx} ({tag}: {mean_b[pos]:.3f} T)', nodes[:, :2],
                       elems[:, :3].astype(np.int64), d['elem'], d['bmag']))
    levels = [k['B_knee_T'] for k in knees if k.get('B_knee_T')]
    save_fig(fig_steel_map(panels, levels, vmax=max(2.2, max(levels) * 1.2)),
             fig_dir / 'saturacao_B_mapa_aco', dpi=200)
    print(f"{len(ids)} amostras ({(time.time() - t0) / 60:.1f} min) — curva B-H igual em todas: "
          f"{n_diff == 0}; reparos {bh.n_repair}; joelho(s): "
          + ', '.join(f"{100 * k['frac']:.0f}% -> {k['B_knee_T']:.4f} T" for k in knees))
    print(f'aço: |B| p5/p50/p95 = {p5:.3f}/{p50:.3f}/{p95:.3f} T; acima do último ponto '
          f'({curve0[1][-1]} T): {n_above} elementos')
    if check_femm:
        print('conferência × FEMM:', res['femm_check'])
    print(f"salvo em {out_dir / 'saturation.json'}; figuras em {fig_dir}")


TIMING_WORK = Path('data/temp/postprocess/timing')
MEM_FRACTION = 0.95      # VRAM dedicada liberada ao alocador (Windows/WDDM: OOM real)


def _merge_json(path: Path, key, value):
    d = json.loads(path.read_text()) if path.exists() else {}
    d[key] = value
    path.write_text(json.dumps(d, indent=2))


def _timing_femm(out_json, n_femm, runs, subset='test'):
    """FEMM (só mi_analyze) + pré-processamento por etapa + codificação, em n_femm amostras do
    conjunto do eval igualmente espaçadas; confere malha/A × raw e entradas × parsers."""
    import gzip as _gz
    rows = design_rows()
    samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], subset, None, N_SAMPLES)
    pick = [samples[i] for i in np.linspace(0, len(samples) - 1, n_femm).round().astype(int)]
    rec, checks = [], []
    for n, (_, _, idx, raw_gz) in enumerate(pick):
        row = rows[idx]
        r_in, r_ext = float(row['inner_diameter [mm]']) / 2, float(row['outer_diameter [mm]']) / 2
        d = TIMING_WORK / f's{idx:06d}'
        tm, ans = femm_solve_timed(row, d)
        tp, arrays, (nodes, elems) = preprocess_timed(ans, r_in, r_ext, ANG_1, ANG_2, N_R, N_A)
        te = encode_timed(runs, arrays)
        # conferências (fora do cronômetro)
        gz = d / 'model.ans.gz'
        with open(ans, 'rb') as fi, _gz.open(gz, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        ref = parse_smooth(gz, row, with_elem=True)
        eq = check_preprocess(arrays, ref)
        chk = dict(sample_idx=idx, **check_mesh_vs_raw(nodes, elems, raw_gz),
                   preproc_equal_parsers=all(eq.values()), preproc_equal_detail=eq)
        checks.append(chk)
        pre = {a: sum(tp[st] for st in ARCH_STAGES[a]) + te[a] for a in te}
        rec.append(dict(sample_idx=idx, n_nodes=chk['n_nodes'], n_elems=chk['n_elems'],
                        femm=tm, pre_stage=tp, encode=te, pre_total=pre))
        shutil.rmtree(d, ignore_errors=True)
        print(f"  [{n + 1}/{len(pick)}] amostra {idx}: nós={chk['n_nodes']}  malha {tm['mesh'] * 1e3:.0f} ms "
              f"(fora)  solução {tm['solve'] * 1e3:.0f} ms  pré(Elem) {pre.get('FNO_BipartiteGNN_Elem', 0) * 1e3:.0f} ms"
              f"  malha=raw:{chk['mesh_equal_raw']} |ΔA|={chk['max_abs_dA_vs_raw']}  "
              f"entradas=parsers:{chk['preproc_equal_parsers']}", flush=True)
    ms = lambda v: iqr_stats(1e3 * np.array(v))
    res = dict(n_samples=len(rec), sample_idx=[r['sample_idx'] for r in rec],
               note='geometria e malha fora da comparação; solução = só mi_analyze',
               femm_ms={k: ms([r['femm'][k] for r in rec]) for k in rec[0]['femm']},
               pre_stage_ms={k: ms([r['pre_stage'][k] for r in rec]) for k in rec[0]['pre_stage']},
               encode_ms={a: ms([r['encode'][a] for r in rec]) for a in rec[0]['encode']},
               pre_total_ms={a: ms([r['pre_total'][a] for r in rec]) for a in rec[0]['pre_total']},
               arch_stages={a: list(v) for a, v in ARCH_STAGES.items()},
               n_mesh_equal_raw=sum(c['mesh_equal_raw'] for c in checks),
               max_abs_dA_vs_raw=max((c['max_abs_dA_vs_raw'] or 0.0) for c in checks),
               all_preproc_equal_parsers=all(c['preproc_equal_parsers'] for c in checks),
               n_nodes=iqr_stats([r['n_nodes'] for r in rec]), checks=checks, per_sample=rec)
    _merge_json(out_json, 'femm_preproc', res)


def _timing_infer(out_json, runs, n_samples, n_warmup, n_reps, max_batch, subset='test'):
    """best.pth das runs, lote 1 (amostra de nº de nós mediano) e potências de 2 até OOM ou
    max_batch (amostras do 1º chunk do subset repetidas ciclicamente)."""
    rows = design_rows()
    samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], subset, None, N_SAMPLES)
    first = [x for x in samples if x[0] == samples[0][0]][:n_samples]
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(first))) as ex:
        lays = list(ex.map(parse_smooth, [x[3] for x in first], [rows[x[2]] for x in first],
                           [True] * len(first)))
    ts = [to_torch_sample(l) for l in lays]
    del lays
    n_nodes = np.array([t['v1']['node_x'].shape[0] for t in ts])
    i_med = int(np.argsort(n_nodes)[len(n_nodes) // 2])
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    for r in runs.values():
        r['model'].to('cpu')
    torch.cuda.empty_cache()
    res = dict(batch1_sample_idx=int(first[i_med][2]), batch1_n_nodes=int(n_nodes[i_med]),
               chunk=first[0][0], n_samples_cycled=len(ts), n_warmup=n_warmup, n_reps=n_reps,
               mem_fraction_limit=MEM_FRACTION, max_batch_tried=max_batch, runs={})
    for k, r in runs.items():
        arch, lay = r['arch'], LAYOUT[r['arch']]
        r['model'].to(DEVICE).eval()
        torch.cuda.empty_cache()
        out = dict(arch=arch, loss=r['loss'], run=r['run'])
        out['batch1'] = time_batch(arch, r, collate([ts[i_med][lay]], lay), n_warmup, n_reps)
        print(f"  {k:26s} lote 1: {out['batch1']['ms']['median']:.2f} ms (IQR {out['batch1']['ms']['iqr']:.2f}), "
              f"pico {out['batch1']['peak_mem_gib']:.2f} GiB", flush=True)
        sizes, bmax, b = {}, 1, 2
        while b <= max_batch:
            try:
                rr = time_batch(arch, r, collate([ts[i % len(ts)][lay] for i in range(b)], lay),
                                n_warmup, n_reps)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
            rr['ms_per_sample_median'] = rr['ms']['median'] / b
            sizes[str(b)], bmax = rr, b
            print(f"      lote {b:3d}: {rr['ms']['median']:.1f} ms ({rr['ms_per_sample_median']:.2f} ms/amostra), "
                  f"pico {rr['peak_mem_gib']:.2f} GiB", flush=True)
            torch.cuda.empty_cache()
            b *= 2
        out.update(batch_sweep=sizes, batch_max=bmax,
                   batch_max_ms_per_sample=(sizes[str(bmax)]['ms_per_sample_median'] if bmax > 1
                                            else out['batch1']['ms']['median']))
        r['model'].to('cpu')
        torch.cuda.empty_cache()
        res['runs'][k] = out
    _merge_json(out_json, 'inference', res)


def _timing_table(out_json):
    d = json.loads(out_json.read_text())
    fp, inf = d.get('femm_preproc'), d.get('inference')
    if not (fp and inf):
        return None
    solve = fp['femm_ms']['solve']['median']
    tab = {}
    for k, r in inf['runs'].items():
        pre = fp['pre_total_ms'][r['arch']]['median']
        i1, im = r['batch1']['ms']['median'], r['batch_max_ms_per_sample']
        tab[k] = dict(femm_solve_ms=solve, preproc_ms=pre, infer_batch1_ms=i1,
                      infer_batchmax_ms_per_sample=im, batch_max=r['batch_max'],
                      speedup_batch1=solve / (pre + i1), speedup_infer_only_batch1=solve / i1,
                      speedup_batchmax=solve / (pre + im), speedup_infer_only_batchmax=solve / im)
    _merge_json(out_json, 'table', dict(
        note='medianas (ms/amostra); aceleração = solução FEMM (só mi_analyze) / (pré-proc. + '
             'inferência); aceleração da inferência pura = solução / inferência; lote máx. = '
             'tempo por amostra no maior lote que coube', rows=tab))
    print(f"{'run':26s} {'FEMM ms':>8s} {'pré ms':>8s} {'inf(1) ms':>9s} {'inf(máx)':>9s} "
          f"{'acel.(1)':>8s} {'inf.pura(1)':>11s} {'acel.(máx)':>10s} {'inf.pura(máx)':>13s}")
    for k, t in tab.items():
        print(f"{k:26s} {t['femm_solve_ms']:8.0f} {t['preproc_ms']:8.0f} {t['infer_batch1_ms']:9.2f} "
              f"{t['infer_batchmax_ms_per_sample']:9.2f} {t['speedup_batch1']:8.1f} "
              f"{t['speedup_infer_only_batch1']:11.0f} {t['speedup_batchmax']:10.1f} "
              f"{t['speedup_infer_only_batchmax']:13.0f}")
    return tab


def run_timing(what, out_dir: Path, n_femm=50, n_warmup=10, n_reps=100, max_batch=512,
               n_infer_samples=32, subset='test'):
    """Custo computacional (P8): FEMM (só mi_analyze) | pré-processamento | inferência."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / 'timing.json'
    runs = load_set(ROOT_S)
    if what in ('femm', 'all'):
        _timing_femm(out_json, n_femm, runs, subset)
    if what in ('infer', 'all'):
        _timing_infer(out_json, runs, n_infer_samples, n_warmup, n_reps, max_batch, subset)
    _merge_json(out_json, 'hardware', hardware_info())
    _timing_table(out_json)
    print(f'salvo em {out_json}')


def run_train_table(out_dir: Path):
    """Tabela de treinamento (P3): só leitura dos logs das runs de find_runs(ROOT_S)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res = train_table(find_runs(ROOT_S))
    (out_dir / 'train_table.json').write_text(json.dumps(res, indent=2))
    f = lambda v, fmt: 'null' if v is None else format(v, fmt)
    print(f"{'run':26s} {'params':>10s} {'trein.':>10s} {'GPU GiB':>7s} {'ép. s':>6s} "
          f"{'trein s':>7s} {'aval s':>6s} {'métr s':>6s} {'total h':>7s} {'best':>5s} "
          f"{'últ.':>5s}  parada  hardware")
    for k, r in res['runs'].items():
        print(f"{k:26s} {f(r['n_params_real'], ',d'):>10s} {f(r['n_params_trainable_real'], ',d'):>10s} "
              f"{f(r['gpu_peak_mem_gib'], '.2f'):>7s} {f(r['median_epoch_time_s'], '.2f'):>6s} "
              f"{f(r['median_train_time_s'], '.2f'):>7s} {f(r['median_eval_time_s'], '.2f'):>6s} "
              f"{f(r['median_metric_time_s'], '.2f'):>6s} "
              f"{f(None if r['wall_time_s'] is None else r['wall_time_s'] / 3600, '.2f'):>7s} "
              f"{f(r['best_epoch'], 'd'):>5s} {f(r['last_epoch'], 'd'):>5s}  {r['stop_reason']}  "
              f"{r['hardware']['device']}")
    if res['missing']:
        print('campos ausentes (null):')
        for k, m in res['missing'].items():
            print(f'  {k}: {", ".join(m)}')
    print(f"salvo em {out_dir / 'train_table.json'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    sp = ap.add_subparsers(dest='cmd', required=True)
    SUBSET_ARG = dict(default='test', choices=SUBSETS,
                      help="chunks avaliados: 'test' (padrão, fora do treino) ou 'val' (parada/best.pth)")
    p_eval = sp.add_parser('eval', help='bateria bldc × gabarito smooth (+ pisos)')
    p_eval.add_argument('n', nargs='?', type=int, default=None, help='nº de amostras (default: todas)')
    p_eval.add_argument('--out', type=Path, default=OUT_ROOT / 'dados', help='diretório de saída')
    p_eval.add_argument('--subset', **SUBSET_ARG)
    p_fig = sp.add_parser('figures', help='curvas de treino, barras de erro, campo P50/P95 (após eval)')
    p_fig.add_argument('--data', type=Path, default=OUT_ROOT / 'dados', help='saída do eval')
    p_fig.add_argument('--out', type=Path, default=OUT_ROOT / 'plots', help='diretório das figuras')
    p_sur = sp.add_parser('surface', help='mapa de erro de superfície dos 5 modelos em 1 amostra')
    p_sur.add_argument('idx', nargs='?', default=None,
                       help='índice da amostra no raw (com --pct: omitir; pode receber a LOSS)')
    p_sur.add_argument('loss', nargs='?', default=None, choices=('mse', 'mae'))
    p_sur.add_argument('--pct', type=float, default=None,
                       help='percentil P do L1_A de superfície da REF_RUN no .npz do eval')
    p_sur.add_argument('--eval-data', type=Path, default=OUT_ROOT / 'dados',
                       help='diretório com eval_smooth_per_sample.npz (para --pct)')
    p_sur.add_argument('--data', type=Path, default=OUT_ROOT / 'dados' / 'surface_error')
    p_sur.add_argument('--out', type=Path, default=OUT_ROOT / 'plots' / 'surface_error')
    p_rg = sp.add_parser('regions', help='erro de superfície por região (interior/interface/canto)')
    p_rg.add_argument('n', nargs='?', type=int, default=None, help='nº de amostras (default: todas)')
    p_rg.add_argument('--d-int', type=float, default=None, help='mm (default: 1 passo radial da grade)')
    p_rg.add_argument('--d-canto', type=float, default=None, help='mm (default: 2 passos radiais)')
    p_rg.add_argument('--ang-canto', type=float, default=30.0, help='graus (default: 30)')
    p_rg.add_argument('--subset', **SUBSET_ARG)
    p_rg.add_argument('--out', type=Path, default=OUT_ROOT / 'dados', help='diretório de saída')
    p_rg.add_argument('--fig', type=Path, default=OUT_ROOT / 'plots', help='diretório da figura')
    p_tq = sp.add_parser('torque', help='torque FEMM × gabarito × modelos (theta-e/batch/scatter)')
    p_tq.add_argument('mode', choices=('theta-e', 'batch', 'scatter'))
    p_tq.add_argument('--idx', type=int, nargs='+', default=[15, 128, 130, 142, 151],
                      help='theta-e: amostras (default: as dos JSONs antigos)')
    p_tq.add_argument('--loss', default='mae', choices=('mse', 'mae'))
    p_tq.add_argument('--archs', nargs='+', default=None, help='default: as 5 archs')
    p_tq.add_argument('--frame', default='polar', choices=('polar', 'xy'),
                      help='referencial da saída das runs (polar desde 2026-10-17)')
    p_tq.add_argument('--femm', action='store_true', help='resolve as varreduras no FEMM antes')
    p_tq.add_argument('--n-theta', type=int, default=None, help='default: 24 (theta-e) / 16 (batch)')
    p_tq.add_argument('--n-samples', type=int, default=80, help='batch: nº de amostras')
    p_tq.add_argument('--n-per-chunk', type=int, default=8, help='scatter: amostras por chunk do subset')
    p_tq.add_argument('--subset', **SUBSET_ARG)
    p_tq.add_argument('--limit', type=int, default=None, help='smoke: ângulos/amostras limitados')
    p_tq.add_argument('--out', type=Path, default=OUT_ROOT / 'dados' / 'torque')
    p_tq.add_argument('--raw-dir', type=Path, default=RAW_DIR,
                      help=f'raw (valid_designs.csv + .ans) (default: {RAW_DIR})')
    p_tq.add_argument('--work-dir', type=Path, default=TORQUE_TMP,
                      help=f'soluções FEMM das varreduras e temporários (default: {TORQUE_TMP})')
    p_sat = sp.add_parser('saturation', help='saturação do aço × curva B-H (só gabarito)')
    p_sat.add_argument('n', nargs='?', type=int, default=None, help='nº de amostras (default: todas)')
    p_sat.add_argument('--knee-frac', type=float, nargs='+', default=[0.5],
                       help='joelho: μ_eff = frac·máx (default 0.5; vários -> vários contornos)')
    p_sat.add_argument('--check-femm', action='store_true',
                       help='confere μ_eff × mo_getpointvalues do FEMM na 1ª amostra (Windows)')
    p_sat.add_argument('--subset', **SUBSET_ARG)
    p_sat.add_argument('--out', type=Path, default=OUT_ROOT / 'dados')
    p_sat.add_argument('--fig', type=Path, default=OUT_ROOT / 'plots')
    p_tm = sp.add_parser('timing', help='custo: FEMM (mi_analyze) × pré-processamento + inferência')
    p_tm.add_argument('what', nargs='?', default='all', choices=('femm', 'infer', 'all'))
    p_tm.add_argument('--n-femm', type=int, default=50, help='amostras do FEMM (igualmente espaçadas)')
    p_tm.add_argument('--warmup', type=int, default=10)
    p_tm.add_argument('--reps', type=int, default=100)
    p_tm.add_argument('--max-batch', type=int, default=512, help='maior lote tentado (potências de 2)')
    p_tm.add_argument('--n-infer-samples', type=int, default=32, help='amostras do 1º chunk do subset')
    p_tm.add_argument('--subset', **SUBSET_ARG)
    p_tm.add_argument('--out', type=Path, default=OUT_ROOT / 'dados')
    p_tt = sp.add_parser('train-table', help='tabela de treinamento a partir dos logs das runs')
    p_tt.add_argument('--out', type=Path, default=OUT_ROOT / 'dados', help='diretório de saída')
    a = ap.parse_args()
    if a.cmd == 'eval':
        run_eval(a.out, a.n, a.subset)
    elif a.cmd == 'figures':
        run_figures(a.data, a.out)
    elif a.cmd == 'surface':
        if a.pct is not None and a.idx in ('mse', 'mae'):
            a.idx, a.loss = None, a.idx
        if a.pct is None and a.idx is None:
            ap.error('surface: informe IDX ou --pct P')
        run_surface(None if a.idx is None else int(a.idx), a.loss or 'mae', a.data, a.out,
                    pct=a.pct, eval_dir=a.eval_data)
    elif a.cmd == 'regions':
        run_regions(a.out, a.fig, a.n, a.d_int, a.d_canto, a.ang_canto, a.subset)
    elif a.cmd == 'torque':
        run_torque(a.mode, a.out, a.loss, a.archs, a.frame, a.femm, a.idx, a.n_theta,
                   a.n_samples, a.n_per_chunk, a.limit, a.raw_dir, a.work_dir, a.subset)
    elif a.cmd == 'saturation':
        run_saturation(a.out, a.fig, a.n, tuple(a.knee_frac), a.check_femm, a.subset)
    elif a.cmd == 'timing':
        run_timing(a.what, a.out, a.n_femm, a.warmup, a.reps, a.max_batch, a.n_infer_samples,
                   a.subset)
    elif a.cmd == 'train-table':
        run_train_table(a.out)
