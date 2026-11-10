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
      do best.pth; N -> as N primeiras amostras). ERRO DE SUPERFÍCIE: superfície discreta
      do modelo × superfície do FEMM (B suavizado P1 descontínuo), L1_A / L2_A / B_ref:
        FNO2d  -> grade (constante na célula) por interseção célula ∩ triângulo;
        Elem   -> P0 (constante no elemento) na própria malha;
        demais -> P1 contínuo dos valores nodais na própria malha.
      Pisos (o gabarito em cada representação): floor_surface (grade, P1_nos, P0_elem).
      Saída: dados/eval_smooth.json + eval_smooth_per_sample.npz
      Secundário: erro de |B| ponto a ponto nos nós (MAE_pt, L2_pt, L1_A, L2_A com área
      dual) -> runs[k]['nodal'], floor_nodal, npz nodal_*.
      Malha: n_nodes / n_elems por amostra no .npz; mesh_stats no JSON.

  figures [--data DIR] [--out DIR] — depois do eval: 1_curvas_treino, 2_barras_erro
      (L1_A / L2_A de superfície dos 10 runs + piso da representação de cada um),
      3_campo_<P50|P95> (|B| nos nós dos 5 modelos mae, setor e zoom no entreferro —
      visualização; amostras P50/P95 do L1_A de superfície da REF_RUN).
      Saída: plots/ (+ dados/figures_smooth.json).

  surface IDX [LOSS] — erro de superfície dos 5 modelos (loss LOSS, default mae) em 1
      amostra: dados/surface_error/sample_XXXXXX_<loss>.{json,npz} + mapa da densidade do
      erro por triângulo em plots/surface_error/sample_XXXXXX_<loss>.png.
  surface --pct P [LOSS] [--eval-data DIR] — a amostra do percentil P do L1_A de
      superfície da REF_RUN no .npz do eval (mesma regra de figures); sufixo _P<P>.
      Mapa: mesma escala em todos os painéis, contornos dos materiais, painel extra com o
      piso da representação em grade.

  train-table [--out DIR] — tabela de treinamento dos 10 runs a partir dos logs (config.json,
      run_summary.json, epochs.csv; nada recalculado): parâmetros, pico de memória da GPU,
      tempo mediano por época (treino / avaliação / métricas) e total, melhor/última época,
      motivo da parada, hardware registrado. Saída: dados/train_table.json.

  regions [N] [--d MM ...] [--d-max MM] [--n-bins K] [--bin-scale lin|log]
      [--ang-canto GRAUS] — erro de superfície por faixas de distância, elementos cortados:
      cada ponto de quadratura (os mesmos do eval) pela SUA distância euclidiana exata a
      Γ (arestas entre materiais) e a 𝒱_c (vértices de canto: mudança de direção >
      ang_canto, ramificação ou ≥ 3 materiais), com a costura periódica de 120° (cópias
      giradas de ±120°). Ω = só o motor: sem os rótulos vacuum do furo interno (r <
      stator_inner/2) e do ar externo (r > rotor_outer/2); pontos fora de Ω não entram em
      região, total nem B_ref. Ω_f(d) = dist(x,Γ) ≤ d, Ω_q(d) = dist(x,𝒱_c) ≤ d, interior =
      Ω ∖ Ω_f(d). Runs do artigo (FNO2d, FNO_GNN, Elem, FNO_SplitGNN N_base × mse/mae) e
      pisos grade, P1_nos, P0_elem, P1_mat; para cada d de --d: quinas, fronteira sem
      quinas, interior (disjuntas) e fronteira inteira -- L1_A/L2_A (% do B_ref em Ω),
      fração da área e do ∫e dA. Curvas acumuladas em d (histograma até --d-max).
      Padrão: d = R_in·θ_setor/N_A ≈ 0,2163 mm, --d-max 2, --n-bins 400, 30°.
      Saída: dados/regions.json, dados/regions_curves.npz, plots/regions_curvas.{png,pdf},
      plots/regions_map_sample_*_d*.png.

  torque {theta-e,batch,scatter} [--loss L] [--archs ...] [--femm] [--femm-ref] [--first K]
      — torque pelo tensor de Maxwell na faixa do entreferro (B por elemento) × REFERÊNCIA
      smooth (B suavizado do .ans no centróide, mesma fórmula), com o piso de cada arch (alvo
      de treino na reconstrução dela) e o gt (efeito da suavização), para as runs da loss
      escolhida (bateria bldc + FNO_SplitGNN; --archs filtra; Split = N_base, média das 3
      cópias do material do elemento). O FEMM só gera malha e .ans, em
      data/temp/torque_theta_e{,_batch}/ (--femm resolve); --femm-ref grava também o T22
      (mo_blockintegral), fora dos resumos.
      Saída: dados/torque/.

  saturation [N] [--knee-frac F ...] [--check-femm] — só o gabarito: curva B-H do aço
      lida do .ans (conferida igual em todas as amostras; interpolação do FEMM), |B| suavizado
      no centróide dos elementos de aço e μ_eff; figura A (curva + μ_eff + distribuição dos
      pontos de operação) e B (|B| no aço, amostras P50 e de maior |B| médio, contornos no
      joelho). Saída: dados/saturation.json + plots/saturacao_*.

  sat-bh [--data DIR] [--fig DIR] [--loss mae] — sem inferência, a partir do
      saturation_error.json do regions: L1 no aço ∩ Ω_int (50 intervalos de mesma área) por
      arch em H = H(|B| mediano), sobre a curva B-H, μ_eff, pico e joelho; faixas abaixo do
      pico / pico–joelho / acima do joelho. Saída: dados/saturation_error_bh.json +
      plots/saturacao_erro_sobre_bh[_nN].

  timing [femm|infer|all] [--n-femm 50] — (máquina Windows com FEMM e GPU) custo
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
  python -m scripts.postprocess sat-bh [--data DIR] [--fig DIR]
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
from src.postprocess.fields import elem_to_nodes, gap_radius_mm, mag, split_corner_values
from src.data_gen.parsers.split_layout import parse_split_smooth
from src.postprocess.inference import (ARCHS, INTERP_MODE, RUN_ORDER, find_runs, find_split_runs,
                                       floor_nodes, load_model, load_set, predict, predict_split,
                                       to_torch_sample)
from src.postprocess.metrics import (R_EXT_MM, R_IN_MM, Accum, merge, pstats,
                                     surface_error_per_triangle, surface_values_p1)
from src.postprocess.plots.errors import (fig_bars, fig_region_curves, fig_regions_map,
                                          fig_surface_error_map)
from src.postprocess.plots.fields import fig_field
from src.postprocess.plots.primitives import MODEL_COLOR, apply_style, save_fig
from src.postprocess.plots.training import fig_training
from src.postprocess.regions import (BAND_UNIONS, BANDS, BandAccum, SeamGeometry, band_edges,
                                     band_points, interface_edges, omega_mask, region_geometry)
from src.postprocess.saturation import (SAT_EDGES, SAT_SETS, FemmBH, SatAccum, check_against_femm,
                                        parse_bh_curves, sat_bh_bands, sat_report,
                                        steel_operating_points, weighted_percentile)
from src.postprocess.plots.saturation import (MU_LABEL, fig_bh, fig_sat_error, fig_sat_error_bh,
                                              fig_steel_map)
from src.postprocess.timing import (ARCH_STAGES, DEVICE, LAYOUT, check_mesh_vs_raw,
                                    check_preprocess, collate, encode_timed, femm_solve_timed,
                                    hardware_info, iqr_stats, preprocess_timed, time_batch)
from src.postprocess.torque import (batch_summary, curve_row, femm_ref, femm_ref_ok, femm_solve, gap,
                                    summarize_batch,
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


def _surface_points(arch, lay, out_hw, yn, with_set=False):
    """Superfície discreta do modelo nos pontos de quadratura da superfície do FEMM:
    (|B|_modelo [Q], |B|_FEMM [Q], peso [Q] mm², triângulo [Q]).
      FNO2d  -> grade (constante na célula) por interseção célula ∩ triângulo;
      Elem   -> P0 (constante no elemento) na própria malha;
      demais -> P1 contínuo dos valores nodais na própria malha.
    with_set: acrescenta o conjunto de pontos ('overlay' ou 'mesh', mesma ordem de
    _quad_xy) -- coordenadas para as faixas por distância."""
    if arch in GRID_ARCHS:
        ov = lay['overlay']
        out = mag(out_hw.float().cpu()).ravel()[ov['cell']], ov['mag'], ov['w'], ov['tri']
        return (*out, 'overlay') if with_set else out
    mq = lay['mesh_quad']
    M, K = mq['mag'].shape
    yp = yn.double().cpu().numpy()
    if arch == 'FNO_BipartiteGNN_Elem':
        mm = np.repeat(_mag_np(yp)[:, None], K, axis=1)
    else:
        mm = surface_values_p1(lay['tri'], yp, mq['bary'])
    w = np.repeat((mq['area'] / K)[:, None], K, axis=1)
    out = mm.ravel(), mq['mag'].ravel(), w.ravel(), np.repeat(np.arange(M), K)
    return (*out, 'mesh') if with_set else out


def _floor_points(kind, lay, with_set=False):
    """O gabarito na própria representação (piso de cada tipo de superfície).
    with_set: acrescenta o conjunto de pontos ('overlay' ou 'mesh'), como _surface_points."""
    mq, ov = lay['mesh_quad'], lay['overlay']
    if kind == 'grade':
        yh = lay['vertex']['y_hw']
        out = np.hypot(yh[0], yh[1]).ravel()[ov['cell']], ov['mag'], ov['w']
        return (*out, 'overlay') if with_set else out
    M, K = mq['mag'].shape
    if kind == 'P1_nos':
        mm = surface_values_p1(lay['tri'], lay['vertex']['node_y'].astype(np.float64), mq['bary'])
    else:   # P0_elem: B suavizado no centróide (gabarito da Elem)
        mm = np.repeat(_mag_np(lay['elem']['node_y'].astype(np.float64))[:, None], K, axis=1)
    out = mm.ravel(), mq['mag'].ravel(), np.repeat((mq['area'] / K)[:, None], K, axis=1).ravel()
    return (*out, 'mesh') if with_set else out


def _quad_xy(lay, xy):
    """Coordenadas (mm) dos pontos de quadratura de cada conjunto, na ordem de
    _surface_points / _floor_points: 'mesh' = bary · vértices do triângulo [M*K,2]
    (mesh_surface_quadrature), 'overlay' = grid_mesh_overlay['xy']."""
    bary = lay['mesh_quad']['bary'].astype(np.float64)               # [K,3]
    P = np.asarray(xy, np.float64)[lay['tri']]                         # [M,3,2]
    return dict(mesh=np.einsum('kj,mjd->mkd', bary, P).reshape(-1, 2),
                overlay=np.asarray(lay['overlay']['xy'], np.float64))


def _surface_kind(arch):
    return 'grade' if arch in GRID_ARCHS else 'P0_elem' if arch == 'FNO_BipartiteGNN_Elem' else 'P1_nos'


NODAL_COLS = ('MAE_pt', 'L2_pt', 'L1_area', 'L2_area')   # colunas nodal_* do .npz (T)


def _count_stats(v):
    """Mediana, mínimo, p5, p95 e máximo de uma contagem por amostra."""
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
    """Erro de superfície (docs/claude/avaliacao.md §1): superfície discreta de cada
    modelo × superfície do FEMM (B suavizado P1 descontínuo); L1_A, L2_A, B_ref.
    Secundário: erro de |B| ponto a ponto nos nós (gabarito node_y do
    layout de vértices, área dual) -> runs[k]['nodal'] e floor_nodal (piso da grade: y_hw
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
    n_nodes, n_elems = [], []          # estatística da malha

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
                n_nodes.append(int(lay['vertex']['node_x'].shape[0]))
                n_elems.append(int(len(lay['tri'])))
                for kd in SURF_KINDS:
                    a = Accum()
                    a.add_surface(*_floor_points(kd, lay))
                    acc_floor[(kd, sub)] = merge([acc_floor[(kd, sub)], a])
                    r_ = a.report()
                    per_floor[kd].append((r_['L1_area'], r_['L2_area']))
                # nodal (secundário): gabarito e área dual do layout de vértices
                area_n = lay['vertex']['node_x'][:, 2].astype(np.float64)
                mag_tn = _mag_np(lay['vertex']['node_y'].astype(np.float64))
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
    # nível secundário: erro nodal
    for k in runs:
        res['runs'][k]['nodal'] = reports(lambda ss, k=k: merge([acc_nodal[(k, s)] for s in ss]))
    res['floor_nodal'] = reports(lambda ss: merge([acc_floor_nodal[s] for s in ss]))
    res['nodal_metric'] = ('erro de |B| ponto a ponto nos nós (secundário): gabarito node_y do '
                           'layout de vértices, área dual do nó; FNO2d interpolado nos nós (modo da '
                           'bateria), Elem -> nós por elem_to_nodes; piso = y_hw do gabarito '
                           f'interpolado nos nós ({INTERP_MODE})')
    # estatística da malha
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
        gt = _mag_np(lay['vertex']['node_y'].astype(np.float64))
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
    representação em grade como painel extra e contornos dos materiais.
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


SEAM_RAD = float(np.deg2rad(ANG_2 - ANG_1))                    # setor periódico (120°)
REGIONS_D_MM = (R_IN_MM * SEAM_RAD / N_A,)  # regions --d padrão: passo angular da grade no menor raio
REGIONS_D_MAX_MM, REGIONS_N_BINS = 2.0, 400  # curvas em d: bins de 0,005 mm (lin)
REGIONS_ARCHS = ('FNO2d', 'FNO_GNN', 'FNO_BipartiteGNN_Elem', 'FNO_SplitGNN')   # archs do artigo
REGIONS_FLOORS = ('grade', 'P1_nos', 'P0_elem', 'P1_mat')     # pisos dessas representações


def _split_points(lay, y_base):
    """Superfície do N_base do FNO_SplitGNN (P1 por material: em cada triângulo, interpolação
    linear das componentes das 3 cópias do material do elemento, ligadas pela Edge_aux, depois
    módulo -- regra de tests/check_split_surface.py) nos pontos da malha:
    (|B| [M*K], |B|_FEMM [M*K], peso [M*K], 'mesh'). y_base [K_base,2]: saída N_base ou o
    gabarito elem_y do layout (piso P1_mat)."""
    mq = lay['mesh_quad']
    M, K = mq['mag'].shape
    V = split_corner_values(lay['split'], y_base)
    assert V.shape[0] == M
    mm = np.hypot(V[..., 0] @ mq['bary'].T, V[..., 1] @ mq['bary'].T)
    w = np.repeat((mq['area'] / K)[:, None], K, axis=1)
    return mm.ravel(), mq['mag'].ravel(), w.ravel(), 'mesh'


def _regions_kind(arch):
    return 'P1_mat' if arch == 'FNO_SplitGNN' else _surface_kind(arch)


def _regions_sample(path, row, ang_canto, edges, d_list, mask_omega=True):
    """Worker do regions: layouts (parse_smooth com overlay + parse_split_smooth), Ω
    (regions.omega_mask), Γ e 𝒱_c da malha completa e, para cada conjunto de pontos de
    quadratura ('mesh', 'overlay'), as distâncias dos pontos em Ω com a costura periódica
    (regions.SeamGeometry) e band_points -- uma vez por amostra, para todas as runs e pisos.
    mask_omega=False: Ω = malha inteira (validação contra o eval)."""
    ans = read_ans(path)
    lay = parse_smooth(path, row, with_elem=True, with_overlay=True, ans=ans)
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    lay['split'] = parse_split_smooth(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2, n_r=N_R,
                                      n_a=N_A, ans=ans)
    tri = lay['tri']
    M, K = lay['mesh_quad']['mag'].shape
    assert (lay['split']['cross_edge_index'][1].reshape(M, 3) == np.arange(M)[:, None]).all()
    t = time.perf_counter()
    xy = np.stack([ans.mesh.x, ans.mesh.y], axis=1).astype(np.float64)
    in_omega, omega = omega_mask(ans, row, xy, tri, r_in, r_ext, SEAM_RAD)
    if not mask_omega:
        in_omega[:] = True
    iface, corners = region_geometry(xy, tri, lay['elem_mat'], ang_canto)
    geo = SeamGeometry(xy, iface, corners, d_cap=edges[-1], seam_rad=SEAM_RAD)
    tri_of = dict(mesh=np.repeat(np.arange(M), K), overlay=lay['overlay']['tri'])
    bands = {}
    for name, pts in _quad_xy(lay, xy).items():
        keep = in_omega[tri_of[name]]
        d_f, d_q = geo.dist(pts[keep])
        bands[name] = band_points(d_f, d_q, edges, d_list, keep=keep)
    lay.update(xy=xy, iface=iface, corners=corners, in_omega=in_omega, omega=omega, bands=bands,
               band_viol=geo.max_viol, band_fallback=geo.n_fallback,
               n_copies=(geo.n_copies_seg, geo.n_copies_corner), t_bands=time.perf_counter() - t)
    lay['overlay'].pop('xy')          # só serviu às distâncias (memória no processo principal)
    curves = parse_bh_curves(ans.lines)  # curva B-H do aço (saturação; mesmo .ans já lido)
    lay['bh'] = (curves[0][1], curves[0][2]) if curves else None
    lay['bh_n_diff'] = sum(not (np.array_equal(c[1], curves[0][1]) and np.array_equal(c[2], curves[0][2]))
                           for c in curves)
    return lay


def _curves_npz(curves):
    """{chave: curves()} -> dict plano para np.savez ('<chave>__<campo>')."""
    out = {}
    for k, c in curves.items():
        for f, v in c.items():
            out[f'{k}__{f}'] = np.asarray(v)
    return out


def _regions_runs():
    """Runs do artigo: FNO2d, FNO_GNN e Elem da bateria bldc + FNO_SplitGNN da bateria
    split, mse e mae (best.pth de cada)."""
    runs = {k: r for k, r in load_set(ROOT_S).items() if r['arch'] in REGIONS_ARCHS}
    for loss, d in find_split_runs(ROOT_S).items():
        model, normalizer, _, epoch = load_model(d)
        runs[f'FNO_SplitGNN_{loss}'] = dict(arch='FNO_SplitGNN', loss=loss, run=d.name,
                                            run_dir=str(d), model=model, normalizer=normalizer,
                                            epoch=int(epoch))
    order = [f'{a}_{l}' for a in REGIONS_ARCHS for l in ('mse', 'mae')]
    return {k: runs[k] for k in order}


def regions_setup(d_list=None, d_max=None, n_bins=REGIONS_N_BINS, bin_scale='lin', ang_canto=30.0,
                  subset='test', mask_omega=True):
    """Parâmetros do regions: (d_list ordenada, bordas dos bins, params do JSON)."""
    d_list = sorted({float(d) for d in (d_list or REGIONS_D_MM)})
    d_max = REGIONS_D_MAX_MM if d_max is None else float(d_max)
    edges = band_edges(d_list, d_max, n_bins, bin_scale)
    dr = (R_EXT_MM - R_IN_MM) / N_R
    params = dict(d_mm=d_list, d_default_mm=REGIONS_D_MM[0],
                  d_default_rule='R_IN_MM * (ANG_2 - ANG_1 em rad) / N_A (passo angular da grade '
                                 'no menor raio)',
                  d_max_mm=d_max, n_bins=n_bins, bin_scale=bin_scale, n_edges=int(len(edges)),
                  ang_canto_deg=ang_canto, subset=subset, grid_radial_step_mm=dr,
                  omega=('motor: malha menos os rótulos vacuum com r_rótulo < stator_inner/2 ou '
                         '> rotor_outer/2' if mask_omega else 'malha inteira (validação)'),
                  dist='euclidiana exata no plano (regions.SeamGeometry: _dist_to_segments + '
                       'cKDTree), com cópias de Γ e 𝒱_c giradas de ±120° (costura periódica)',
                  regioes='quinas: dist_q <= d; fronteira_sem_quinas: dist_f <= d < dist_q; '
                          'interior: dist_f > d; fronteira = quinas + fronteira_sem_quinas',
                  corte='por ponto de quadratura (elemento cortado pela faixa), sem subdivisão',
                  material='por rótulo (lay[elem_mat])')
    return d_list, edges, params


def regions_state(runs, edges, d_list, sat=False):
    """Estado acumulado do regions (somas + registros por amostra; pickle = checkpoint).
    sat: acumula também o erro no aço × |B| da referência (saturation.SatAccum, por run e
    piso; conjuntos SAT_SETS no 1º d da lista)."""
    return dict(acc={k: BandAccum(edges, d_list) for k in runs},
                acc_floor={kd: BandAccum(edges, d_list) for kd in REGIONS_FLOORS},
                pset={}, subs=[], idx=[], t_bands=[], t_add=[], viol=[], n_fb=0, omegas=[],
                n_pts={'mesh': 0, 'overlay': 0}, fig_done=False,
                sat=({k: SatAccum() for k in runs} | {'piso_' + kd: SatAccum() for kd in REGIONS_FLOORS}
                     if sat else None),
                bh=None, bh_n_diff=0)


def _sat_masks(lay, ps):
    """Máscaras dos conjuntos SAT_SETS nos pontos do conjunto ps ('mesh'/'overlay'), em cache
    no lay: aço (elem_mat == iron) em Ω e, para 'aco_int', na região interior (dist_f > d)
    do 1º d da lista."""
    cache = lay.setdefault('_sat_masks', {})
    if ps not in cache:
        from src.data_gen.motor_constants import MATERIAL_ID
        M, K = lay['mesh_quad']['mag'].shape
        tri_of = np.repeat(np.arange(M), K) if ps == 'mesh' else lay['overlay']['tri']
        bp = lay['bands'][ps]
        steel = (lay['elem_mat'][tri_of] == MATERIAL_ID['iron_1008']) & bp['keep']
        inter = np.zeros(len(steel), bool)
        inter[bp['keep']] = bp['reg'][0] == BANDS.index('interior')
        cache[ps] = dict(aco_int=steel & inter, aco=steel)
    return cache[ps]


def regions_track(st, sub, idx, lay):
    """Registros da amostra (saída do worker _regions_sample); retorna a posição dela."""
    st['subs'].append(sub)
    st['idx'].append(int(idx))
    st['t_bands'].append(lay['t_bands'])
    st['t_add'].append(0.0)
    st['viol'].append(lay['band_viol'])
    st['n_fb'] += lay['band_fallback']
    st['omegas'].append(lay['omega'])
    if st.get('sat') is not None and lay.get('bh') is not None:
        if st['bh'] is None:
            st['bh'] = lay['bh']
        st['bh_n_diff'] += lay['bh_n_diff'] + int(not (np.array_equal(lay['bh'][0], st['bh'][0])
                                                       and np.array_equal(lay['bh'][1], st['bh'][1])))
    for name, bp in lay['bands'].items():
        st['n_pts'][name] += int(bp['keep'].sum())
    return len(st['subs']) - 1


def regions_map_fig(lay, row, idx, d_list, ang_canto, fig_dir):
    """Figura de verificação das regiões numa amostra (zoom no canto mais perto de r = SOD/2,
    θ ≈ 60°)."""
    apply_style()
    xy, tri = lay['xy'], lay['tri']
    bp = lay['bands']['mesh']
    C = xy[lay['corners']]
    th = np.arctan2(C[:, 1], C[:, 0])
    sod = float(row['stator_outer_diameter [mm]']) / 2
    kc = int(np.argmin(np.abs(np.hypot(*C.T) - sod) + 0.05 * np.abs(th - np.pi / 3)))
    for di, d in enumerate(d_list):
        code = np.full(len(bp['keep']), -1, np.int8)
        code[bp['keep']] = bp['reg'][di]
        title = (f'Amostra {idx}: regiões por distância nos pontos de quadratura da '
                 f'malha, d = {d:.4f} mm (ângulo de canto > {ang_canto:g}°)')
        save_fig(fig_regions_map(xy, tri, code, lay['iface'], lay['corners'], d, title,
                                 zoom=(C[kc, 0], C[kc, 1], max(0.8, 4 * d))),
                 fig_dir / f"regions_map_sample_{idx:06d}_d{f'{d:.4f}'.replace('.', 'p')}",
                 pdf=False, dpi=200)


def regions_add_floors(st, lay, pos):
    """Pisos REGIONS_FLOORS da amostra nos acumuladores (pos = posição de regions_track)."""
    for kd in REGIONS_FLOORS:
        if kd == 'P1_mat':
            mm, mf, w, ps = _split_points(lay, lay['split']['elem_y'])
        else:
            mm, mf, w, ps = _floor_points(kd, lay, with_set=True)
        t = time.perf_counter()
        st['acc_floor'][kd].add(mm, mf, w, lay['bands'][ps])
        if st.get('sat') is not None:
            st['sat']['piso_' + kd].add(mm, mf, w, _sat_masks(lay, ps))
        st['t_add'][pos] += time.perf_counter() - t
        st['pset']['piso_' + kd] = ps


def regions_add_run(st, k, r, lay, pos):
    """Predição de 1 run (r de _regions_runs) em 1 amostra, acumulada; nada fica guardado."""
    if r['arch'] == 'FNO_SplitGNN':
        _, y_base = predict_split(r['model'], r['normalizer'], lay['split'])
        mm, mf, w, ps = _split_points(lay, y_base)
    else:
        out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], to_torch_sample(lay))
        mm, mf, w, _, ps = _surface_points(r['arch'], lay, out_hw, yn, with_set=True)
    t = time.perf_counter()
    st['acc'][k].add(mm, mf, w, lay['bands'][ps])
    if st.get('sat') is not None:
        st['sat'][k].add(mm, mf, w, _sat_masks(lay, ps))
    st['t_add'][pos] += time.perf_counter() - t
    st['pset'][k] = ps


def regions_finalize(st, runs, params, d_list, subset, out_dir: Path, fig_dir: Path, timing,
                     extra=None):
    """JSON (sempre das somas acumuladas), curvas, figura das curvas e tabela no terminal.
    timing: total_s, wall_s_per_sample, n_workers (+ campos do chamador); extra: campos
    acrescentados no topo do JSON (ex.: placeholder, índices, execução)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    acc, acc_floor, omegas = st['acc'], st['acc_floor'], st['omegas']
    subs_arr = np.array(st['subs'])
    ai = np.array([o['area_inner_mm2'] for o in omegas])
    ao = np.array([o['area_outer_mm2'] for o in omegas])
    res = dict(**(extra or {}),
               n_samples=len(st['subs']), n_by_subset={s: int((subs_arr == s).sum()) for s in SUBSETS},
               params=params, regions=list(BANDS) + list(BAND_UNIONS),
               note='by_d[d][região]: L1_A/L2_A normalizados pela área da região, em % do B_ref '
                    'GLOBAL em Ω; area_frac = área da região / área de Ω; L1_contrib_frac = fração '
                    'de ∫e dA total em Ω vinda da região. quinas, fronteira_sem_quinas e interior '
                    'são disjuntas (somam a área e o ∫e dA de Ω); fronteira = Ω_f inteira. Curvas '
                    'em d: regions_curves.npz.',
               omega=dict(
                   labels_inner=sorted({tuple(o['labels_inner']) for o in omegas}),
                   labels_outer=sorted({tuple(o['labels_outer']) for o in omegas}),
                   area_inner_mm2=[float(ai.min()), float(ai.max())],
                   area_outer_mm2=[float(ao.min()), float(ao.max())],
                   area_inner_exact_mm2=[float(min(o['area_inner_exact_mm2'] for o in omegas)),
                                         float(max(o['area_inner_exact_mm2'] for o in omegas))],
                   area_outer_exact_mm2=[float(min(o['area_outer_exact_mm2'] for o in omegas)),
                                         float(max(o['area_outer_exact_mm2'] for o in omegas))],
                   centroid_r_inner=[min(o['centroid_r_inner'][0] for o in omegas if o['centroid_r_inner']),
                                     max(o['centroid_r_inner'][1] for o in omegas if o['centroid_r_inner'])],
                   centroid_r_outer=[min(o['centroid_r_outer'][0] for o in omegas if o['centroid_r_outer']),
                                     max(o['centroid_r_outer'][1] for o in omegas if o['centroid_r_outer'])],
                   applied=params['omega'] != 'malha inteira (validação)'),
               timing=dict(**timing, bands_s_per_sample=float(np.mean(st['t_bands'])),
                           accum_s_per_sample=float(np.mean(st['t_add'])),
                           note='bands = Ω, Γ, 𝒱_c, distâncias e bins no worker (paralelo); accum = '
                                'BandAccum.add de todas as runs e pisos (processo principal); '
                                'wall = tempo total / nº de amostras'),
               checks=dict(max_dist_f_minus_dist_q_mm=float(max(st['viol'])),
                           n_points_shapely_fallback=int(st['n_fb']),
                           mean_points_in_omega_per_sample={k: v / len(st['subs'])
                                                            for k, v in st['n_pts'].items()}),
               runs={k: dict(arch=r['arch'], loss=r['loss'], run=r['run'], run_dir=r['run_dir'],
                             best_epoch=r['epoch'], surface_kind=_regions_kind(r['arch']),
                             point_set=st['pset'][k], **acc[k].report())
                     for k, r in runs.items()},
               floor_surface={kd: dict(point_set=st['pset']['piso_' + kd], **acc_floor[kd].report())
                              for kd in REGIONS_FLOORS})
    (out_dir / 'regions.json').write_text(json.dumps(res, indent=2))
    curves = {k: a.curves() for k, a in acc.items()}
    curves.update({'piso_' + kd: a.curves() for kd, a in acc_floor.items()})
    np.savez_compressed(out_dir / 'regions_curves.npz', keys=np.array(list(curves)),
                        d_marks=np.array(d_list), **_curves_npz(curves))
    apply_style()
    keys = {}
    for k, r in runs.items():
        keys.setdefault(r['loss'], []).append((r['arch'], r['arch'], k))
    save_fig(fig_region_curves(curves, keys, d_list,
                               f"{len(st['subs'])} amostras ({subset}) — fração acumulada da área de "
                               f"Ω e do erro total η = ∫_Ω(d) e dA / ∫_Ω e dA"),
             fig_dir / 'regions_curvas', pdf=True, dpi=200)

    print(f"\n{len(st['subs'])} amostras — erro de superfície por faixa, % do B_ref global em Ω "
          f'(L1_A / L2_A) [fração da área | fração do L1_A total]')
    cols = list(BANDS) + list(BAND_UNIONS)
    items_ = [(k, v) for k, v in res['runs'].items()] + \
             [('piso ' + kd, v) for kd, v in res['floor_surface'].items()]
    for d in d_list:
        print(f'  d = {d:g} mm')
        print(f"  {'':26s}" + ''.join(f'{n:>34s}' for n in cols) + '      total  B_ref')
        for k, v in items_:
            b = v['by_d'][f'{d:g}']
            print(f'  {k:26s}' + ''.join(
                f"{b[n]['L1_area_pct']:6.2f} /{b[n]['L2_area_pct']:6.2f} "
                f"[{b[n]['area_frac']:.3f}|{b[n]['L1_contrib_frac']:.3f}]".rjust(34) for n in cols)
                  + f"  {v['total']['L1_area_pct']:5.2f}/{v['total']['L2_area_pct']:5.2f}"
                    f"  {v['B_ref_global']:.4f}")
    tm = res['timing']
    print(f"tempo total {tm['total_s'] / 60:.1f} min; por amostra: faixas {tm['bands_s_per_sample']:.2f} s "
          f"(worker), acumulação {tm['accum_s_per_sample']:.2f} s, parede {tm['wall_s_per_sample']:.2f} s; "
          f"máx(dist_f - dist_q) = {res['checks']['max_dist_f_minus_dist_q_mm']:.2e} mm; "
          f"fallback {st['n_fb']} pontos")
    print(f"salvo em {out_dir / 'regions.json'}, {out_dir / 'regions_curves.npz'}; figuras em {fig_dir}")
    if st.get('sat') is not None:
        res['saturation'] = regions_saturation(st, runs, res, d_list, subset, out_dir, fig_dir, extra)
    return res


def regions_saturation(st, runs, res, d_list, subset, out_dir: Path, fig_dir: Path, extra=None):
    """saturation_error.json + figura: erro no aço × |B| da referência (SatAccum), 50
    intervalos de mesma área (sat_report), L1 em % do B_ref em Ω da mesma run; μ_eff(|B|)
    da curva B-H do .ans (FemmBH) e joelho (μ_eff = metade do máximo). Retorna o resumo das
    conferências."""
    bh = FemmBH(*st['bh'])
    knee = bh.knee(0.5)
    bref = {k: v['B_ref_global'] for k, v in res['runs'].items()}
    bref.update({'piso_' + kd: v['B_ref_global'] for kd, v in res['floor_surface'].items()})
    rep_ = {k: sat_report(a, bref[k]) for k, a in st['sat'].items()}
    b = np.round(np.arange(0.01, 2.6 + 1e-9, 0.01), 4)
    checks = dict(max_rel_diff_sum_w=max(r[s]['check']['rel_diff_sum_w'] for r in rep_.values() for s in SAT_SETS),
                  max_rel_diff_sum_ew=max(r[s]['check']['rel_diff_sum_ew'] for r in rep_.values() for s in SAT_SETS),
                  max_overflow_area_frac=max((r[s]['check']['overflow_area_frac'] or 0)
                                             for r in rep_.values() for s in SAT_SETS),
                  bh_curves_different_from_first=int(st['bh_n_diff']))
    out = dict(**{k: v for k, v in (extra or {}).items() if k in ('placeholder', 'sample_idx')},
               n_samples=len(st['subs']), subset=subset, d_interior_mm=float(d_list[0]),
               definicao=dict(conjuntos={'aco_int': 'pontos de quadratura no aço (iron) em Ω_int(d) '
                                                    '= dist_f > d, Ω = só o motor',
                                         'aco': 'todos os pontos de quadratura no aço (em Ω)'},
                              variavel='|B| da referência suavizada no ponto (|B|_FEMM da métrica de '
                                       'superfície)',
                              histograma=f'{len(SAT_EDGES) - 1} bins de 0,005 T em [0, 2,6] T + overflow',
                              intervalos='50 de mesma área de aço (na resolução dos bins); L1/L2 em % '
                                         'do B_ref global em Ω da mesma run (regions.json)'),
               bh=dict(B_T=bh.B_in.tolist(), H_A_m=bh.H_in.tolist(), n_repair=bh.n_repair,
                       mu_eff_curve=dict(B_T=b.tolist(), mu_eff=bh.mu_eff(b).tolist()), knee=knee),
               checks=checks,
               runs={k: dict(arch=r['arch'], loss=r['loss'], run_dir=r['run_dir'], best_epoch=r['epoch'],
                             B_ref_omega=bref[k], **rep_[k]) for k, r in runs.items()},
               floors={kd: dict(B_ref_omega=bref['piso_' + kd], **rep_['piso_' + kd]) for kd in REGIONS_FLOORS},
               hist=dict(edges_T=SAT_EDGES.tolist(),
                         sums={k: {s_: a.h[s_].tolist() for s_ in SAT_SETS} for k, a in st['sat'].items()}))
    (out_dir / 'saturation_error.json').write_text(json.dumps(out, indent=1))
    apply_style()
    rep_fig = {k: v for k, v in rep_.items()}
    save_fig(fig_sat_error(rep_fig, bh, knee,
                           f"{len(st['subs'])} amostras ({subset}) — erro de superfície no aço × |B| da "
                           f"referência (50 intervalos de mesma área; d = {d_list[0]:.4f} mm)",
                           REGIONS_ARCHS, MODEL_COLOR,
                           {'FNO_BipartiteGNN_Elem': 'Elem', 'FNO_SplitGNN': 'Split'}),
             fig_dir / 'saturacao_erro_L1_vs_B', pdf=True, dpi=200)
    print(f"saturação: {out_dir / 'saturation_error.json'}; conferência hist × pontos: "
          f"Σw {checks['max_rel_diff_sum_w']:.1e}, Σe·w {checks['max_rel_diff_sum_ew']:.1e} (rel.); "
          f"overflow ≤ {checks['max_overflow_area_frac']:.1e}; joelho {knee.get('B_knee_T')}")
    return checks


def run_regions(out_dir: Path, fig_dir: Path, max_samples=None, d_list=None, d_max=None,
                n_bins=REGIONS_N_BINS, bin_scale='lin', ang_canto=30.0, subset='test',
                mask_omega=True):
    """Erro de superfície por faixas de distância (docs/claude/avaliacao.md §2), elementos
    cortados: cada ponto de quadratura (malha ou overlay grade ∩ malha, os mesmos do eval) é
    atribuído pela SUA distância a Γ (arestas de interface) e a 𝒱_c (vértices de canto),
    com a costura periódica; só os pontos em Ω (motor sem os anéis de ar) entram.
    Runs do artigo (_regions_runs) e os pisos REGIONS_FLOORS: BandAccum (histogramas em d +
    regiões disjuntas por d da lista). Subset como no eval. mask_omega=False só para a
    validação (Ω = malha inteira). Saída: regions.json (tabela por d), regions_curves.npz,
    plots/regions_curvas.{png,pdf} e plots/regions_map_sample_*_d*.png.
    Versão com memória controlada (lotes, checkpoint, retomada): scripts/run_postprocess.py,
    com as mesmas funções regions_*."""
    d_list, edges, params = regions_setup(d_list, d_max, n_bins, bin_scale, ang_canto, subset,
                                          mask_omega)
    print('parâmetros:', params)
    rows = design_rows()
    samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], subset, max_samples, N_SAMPLES)
    runs = _regions_runs()
    for k, r in runs.items():
        print(f"  {k:28s} {r['run_dir']} best ép.{r['epoch']}  superfície {_regions_kind(r['arch'])}")
    st = regions_state(runs, edges, d_list)
    groups = {}
    for name, sub, idx, p in samples:
        groups.setdefault(name, []).append((sub, idx, p))
    t0 = time.time()
    for gi, (name, items) in enumerate(groups.items()):
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items))) as ex:
            futs = [ex.submit(_regions_sample, p, rows[idx], ang_canto, edges, d_list, mask_omega)
                    for _, idx, p in items]
            for i, (sub, idx, p) in enumerate(items):
                lay = futs[i].result()
                futs[i] = None                # solta a amostra já consumida
                pos = regions_track(st, sub, idx, lay)
                if not st['fig_done']:
                    regions_map_fig(lay, rows[idx], idx, d_list, ang_canto, fig_dir)
                    st['fig_done'] = True
                regions_add_floors(st, lay, pos)
                for k, r in runs.items():
                    regions_add_run(st, k, r, lay, pos)
        print(f"  [{gi + 1}/{len(groups)}] {name}  (amostras {len(st['subs'])}, "
              f'{(time.time() - t0) / 60:.1f} min)', flush=True)
    t_total = time.time() - t0
    return regions_finalize(st, runs, params, d_list, subset, out_dir, fig_dir,
                            dict(total_s=float(t_total),
                                 wall_s_per_sample=float(t_total / len(st['subs'])),
                                 n_workers=N_WORKERS))


TORQUE_TMP = Path('data/temp')     # soluções FEMM das varreduras (torque_theta_e{,_batch}/)
TORQUE_WORKERS, TORQUE_FEMM_WORKERS = 6, 8
THETA_E_N, THETA_E_SEED = 5, 0     # amostras padrão do theta-e (teste, 1 por faixa de 2 A)


def _torque_runs(loss, archs):
    """Runs da loss: bateria bldc (load_set) + FNO_SplitGNN da bateria split (best.pth);
    archs = None -> todas."""
    runs = load_set(ROOT_S, losses=(loss,))
    runs = {k: r for k, r in runs.items() if archs is None or r['arch'] in archs}
    if archs is None or 'FNO_SplitGNN' in archs:
        d = find_split_runs(ROOT_S)[loss]
        model, normalizer, _, epoch = load_model(d)
        runs[f'FNO_SplitGNN_{loss}'] = dict(arch='FNO_SplitGNN', loss=loss, run=d.name,
                                            run_dir=str(d), model=model, normalizer=normalizer,
                                            epoch=int(epoch))
    return runs


def _with_split(runs):
    """parse_torque_sample monta o layout da Split só se ela estiver entre as runs."""
    return any(r['arch'] == 'FNO_SplitGNN' for r in runs.values())


def _torque_sweep_dir(idx, work_dir=TORQUE_TMP):
    """Pasta da varredura θe de 1 amostra."""
    return Path(work_dir) / 'torque_theta_e' / f'sample_{idx:06d}'


def _femm_batch_one(idx, k, th, row, work_dir=TORQUE_TMP, femm_ref=False):
    """Resolve 1 ângulo do batch no FEMM (.ans + json do cache; T22 só com femm_ref)."""
    d = Path(work_dir) / 'torque_theta_e_batch' / f'sample_{idx:06d}'
    gz, js = d / f'theta_{k:02d}.ans.gz', d / f'theta_{k:02d}.json'
    if gz.exists() and js.exists() and femm_ref_ok(json.loads(js.read_text()), row) and \
            (not femm_ref or json.loads(js.read_text()).get('T22_sector') is not None):
        return
    tmp = Path(work_dir) / 'torque_theta_e_batch' / 'tmp' / f'{idx}_{k}'
    try:
        t22 = femm_solve(row, th, tmp, gz, femm_ref=femm_ref)
        js.write_text(json.dumps(femm_ref_rec(row, th, t22)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def femm_ref_rec(row, th, t22):
    """Registro do cache de 1 solução (femm_ref de torque.py; T22 None sem --femm-ref)."""
    return femm_ref(row, th, t22) if t22 is not None else \
        dict(femm_ref(row, th, 0.0), T22_sector=None)


def run_torque(mode, out_dir: Path, loss='mae', archs=None, femm=False,
               idxs=None, n_theta=None, n_samples=80, n_per_chunk=8, limit=None,
               raw_dir=None, work_dir=None, subset='test', femm_ref=False, first=None,
               cache_only=False, workers=None):
    """Torque pelo tensor de Maxwell na faixa do entreferro, REFERÊNCIA = smooth (B suavizado
    do .ans no centróide, mesma fórmula das predições; docs/claude/avaliacao.md §3), para as
    runs da loss (bateria bldc + FNO_SplitGNN; archs filtra), com o piso de cada arch
    (target_elem_xy) e o gt (efeito da suavização).
      theta-e : varredura θe das amostras idxs (None = THETA_E_N amostras do TESTE do
                split.json da run de referência, 1 por faixa de 2 A, seed THETA_E_SEED);
      batch   : n_samples amostras do subset estratificadas por corrente × n_theta ângulos
                (first = só as `first` primeiras da seleção, placeholder); resumo
                summarize_batch (preds, pisos, suavização);
      scatter : torque das n_per_chunk primeiras amostras de cada chunk do subset.
    O FEMM só gera malha e .ans (femm = resolve as varreduras; caches com outra profundidade
    são ressolvidos). femm_ref: calcula também o T22 (mo_blockintegral) e grava a curva
    'femm' / T_femm -- fora de todos os resumos.
    limit: só os primeiros `limit` ângulos (theta-e/batch) ou amostras (scatter) -- smoke.
    raw_dir: raw de onde vêm valid_designs.csv e os .ans (None: samples.RAW_DIR); work_dir:
    soluções FEMM e temporários (None: TORQUE_TMP).
    cache_only (batch): nunca chama o FEMM -- ângulos sem cache ou com cache inválido
    (profundidade) são pulados e registrados em meta['cache_skipped']; workers: processos de
    parse (None = TORQUE_WORKERS)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    work_dir = Path(work_dir) if work_dir is not None else TORQUE_TMP
    rows = read_rows(raw_dir)
    runs = _torque_runs(loss, archs)
    split_run = find_runs(ROOT_S)[('FNO_BipartiteGNN_Elem', loss)]   # mesmo split em todas
    meta = dict(loss=loss, split_run=str(split_run), subset=subset,
                reference='smooth (B suavizado do .ans no centróide, mesma fórmula da faixa)',
                femm_T22_recorded=bool(femm_ref),
                runs={r['arch']: dict(run_dir=r['run_dir'], best_epoch=r['epoch']) for r in runs.values()})
    pk = [f"pred_{r['arch']}" for r in runs.values()] + [f"floor_{r['arch']}" for r in runs.values()]
    n_work = workers or TORQUE_WORKERS
    mNm = lambda v: np.asarray(v) * 1e3
    if mode == 'theta-e':
        n_theta = n_theta or 24
        if idxs is None:
            idxs = select_batch_samples(split_run, rows, THETA_E_N, seed=THETA_E_SEED, subset='test')
            print(f'  theta-e: amostras do teste {idxs}', flush=True)
        for idx in idxs:
            row, sw = rows[idx], _torque_sweep_dir(idx, work_dir)
            th_all = torque_thetas(n_theta)
            cache = sw / 'femm_ref.json'
            refs = json.loads(cache.read_text()) if cache.exists() else []
            stale = cache.exists() and not all(femm_ref_ok(r, row) for r in refs)
            no_t22 = femm_ref and any(r.get('T22_sector') is None for r in refs)
            if stale:
                print(f'  s{idx}: femm_ref.json com outra profundidade (ou sem stack_length_mm) '
                      f'-> descartado, ressolvendo', flush=True)
            if femm or stale or no_t22 or not cache.exists():
                ref = [femm_ref_rec(row, th, femm_solve(row, th, sw / 'tmp', sw / 'raw' / f'theta_{k:03d}.ans.gz',
                                                        femm_ref=femm_ref))
                       for k, th in enumerate(th_all)]
                cache.write_text(json.dumps(ref, indent=1))
            ref = json.loads(cache.read_text())[:limit]
            T = {'smooth': [], 'gt': [], **{k: [] for k in pk}}
            if femm_ref:
                T['femm_T22'] = [3 * r['T22_sector'] for r in ref]
            with ProcessPoolExecutor(n_work) as ex:
                futs = [ex.submit(parse_torque_sample, sw / 'raw' / f'theta_{k:03d}.ans.gz', row,
                                  _with_split(runs))
                        for k in range(len(ref))]
                for k, f in enumerate(futs):
                    t = sample_torques(f.result(), row, runs)
                    for kk in ['smooth', 'gt'] + pk:
                        T[kk].append(t[kk])
                    print(f"  s{idx} θe={ref[k]['theta_e_deg']:5.1f}°  ref {t['smooth'] * 1e3:8.1f}  "
                          f"gt {t['gt'] * 1e3:8.1f}  " + '  '.join(f'{kk} {t[kk] * 1e3:7.1f}' for kk in pk)
                          + ' mNm', flush=True)
            T = {k: np.array(v) for k, v in T.items()}
            rms = lambda x: float(np.sqrt(np.mean(np.square(x))) * 1e3)
            res = dict(sample_idx=idx, **meta, current_amplitude_A=float(row['current_amplitude [A]']),
                       rotor_phase_deg=float(row['rotor_phase [deg]']), gap_mm=list(gap(row)),
                       theta_e_deg=[r['theta_e_deg'] for r in ref], T_Nm={k: v.tolist() for k, v in T.items()},
                       rms_mNm={k: rms(v) for k, v in T.items()},
                       max_mNm={k: float(v.max() * 1e3) for k, v in T.items()},
                       min_mNm={k: float(v.min() * 1e3) for k, v in T.items()},
                       rms_err_vs_ref_mNm={k: rms(T[k] - T['smooth']) for k in ['gt'] + pk})
            (out_dir / f'theta_e_s{idx}_{loss}.json').write_text(json.dumps(res, indent=2))
            print(json.dumps(res['rms_err_vs_ref_mNm'], indent=1))
    elif mode == 'batch':
        n_theta = n_theta or 16
        ids = select_batch_samples(split_run, rows, n_samples, subset=subset)[:first]
        root = work_dir / 'torque_theta_e_batch'
        # com --femm, todos os ângulos (_femm_batch_one pula só os caches válidos); sem --femm,
        # só os caches existentes com outra profundidade (ou sem T22 com --femm-ref) são ressolvidos
        tasks = [(i, k, th) for i in ids for k, th in enumerate(torque_thetas(n_theta))]
        if not femm:
            jsons = {(i, k): root / f'sample_{i:06d}' / f'theta_{k:02d}.json' for i, k, _ in tasks}
            bad = lambda j, r: (not femm_ref_ok(j, r)) or (femm_ref and j.get('T22_sector') is None)
            tasks = [(i, k, th) for i, k, th in tasks if jsons[i, k].exists()
                     and bad(json.loads(jsons[i, k].read_text()), rows[i])]
            if tasks:
                print(f'  {len(tasks)} cache(s) a ressolver (profundidade / T22)', flush=True)
        skipped = []
        if cache_only:
            skipped = [dict(idx=int(i), k=int(k), motivo='cache inválido (profundidade/T22)') for i, k, _ in tasks]
            skipped += [dict(idx=int(i), k=int(k), motivo='sem cache')
                        for i in ids for k in range(n_theta)
                        if not (root / f'sample_{i:06d}' / f'theta_{k:02d}.json').exists()]
            bad_set = {(d_['idx'], d_['k']) for d_ in skipped}
            tasks = []
            meta['cache_skipped'] = skipped
            if skipped:
                print(f'  --cache-only: {len(skipped)} ângulo(s) pulados: {skipped[:10]}', flush=True)
        if tasks:
            with ProcessPoolExecutor(TORQUE_FEMM_WORKERS) as ex:
                for f in [ex.submit(_femm_batch_one, i, k, th, rows[i], work_dir, femm_ref)
                          for i, k, th in tasks]:
                    f.result()
        res = {}
        with ProcessPoolExecutor(n_work) as ex:
            for n, i in enumerate(ids, 1):
                d = root / f'sample_{i:06d}'
                ks = [k for k in range(n_theta) if (d / f'theta_{k:02d}.json').exists()
                      and (not cache_only or (int(i), k) not in bad_set)][:limit]
                futs = [ex.submit(parse_torque_sample, d / f'theta_{k:02d}.ans.gz', rows[i],
                                  _with_split(runs))
                        for k in ks]
                cur = dict(theta_e_deg=[], smooth=[], gt=[], **{kk: [] for kk in pk})
                if femm_ref:
                    cur['femm'] = []
                for k, f in zip(ks, futs):
                    ref = json.loads((d / f'theta_{k:02d}.json').read_text())
                    t = sample_torques(f.result(), rows[i], runs)
                    cur['theta_e_deg'].append(ref['theta_e_deg'])
                    if femm_ref:
                        cur['femm'].append(3 * ref['T22_sector'])
                    for kk in ['smooth', 'gt'] + pk:
                        cur[kk].append(t[kk])
                row = rows[i]
                geom = {k: float(row[k]) for k in ('stator_outer_diameter [mm]', 'pole_thickness [mm]',
                                                   'pole_embrance', 'slot_Bs0 [mm]', 'slot_Bs1 [mm]',
                                                   'slot_Bs2 [mm]', 'rotor_inner_diameter [mm]')}
                res[i] = dict(curves=cur, I=float(row['current_amplitude [A]']),
                              rotor_phase=float(row['rotor_phase [deg]']), geom=geom)
                print(f'  [{n}/{len(ids)}] amostra {i}  {len(ks)} ângulos', flush=True)
                if limit and n >= limit:
                    break
        (out_dir / f'theta_e_batch_{loss}.json').write_text(json.dumps(dict(**meta, samples=res)))
        summ = summarize_batch(res)
        (out_dir / f'theta_e_batch_{loss}_summary.json').write_text(json.dumps({**meta, **summ}, indent=1))
        print(json.dumps({k: summ[k] for k in ('summary', 'floor', 'smoothing_effect')}, indent=1))
    elif mode == 'scatter':
        ids = scatter_ids(split_run, n_per_chunk, subset=subset)[:limit]
        out = dict(idx=[], T_smooth=[], T_gt=[], I=[], theta_e=[], rotor_phase=[], **{f'T_{kk}': [] for kk in pk})
        with ProcessPoolExecutor(n_work) as ex:
            for b in range(0, len(ids), 32):
                batch = ids[b:b + 32]
                futs = [ex.submit(parse_torque_sample, raw_dir / f'sample_{i:06d}.ans.gz', rows[i],
                                  _with_split(runs))
                        for i in batch]
                for i, f in zip(batch, futs):
                    t = sample_torques(f.result(), rows[i], runs)
                    r = rows[i]
                    out['idx'].append(i); out['T_gt'].append(t['gt']); out['T_smooth'].append(t['smooth'])
                    for kk in pk:
                        out[f'T_{kk}'].append(t[kk])
                    out['I'].append(float(r['current_amplitude [A]']))
                    out['theta_e'].append(float(r['current_angle [deg]']))
                    out['rotor_phase'].append(float(r['rotor_phase [deg]']))
                print(f"  {len(out['idx'])}/{len(ids)}", flush=True)
        S, G = mNm(out['T_smooth']), mNm(out['T_gt'])
        stats = {kk: scatter_stats(S, mNm(out[f'T_{kk}']), G) for kk in pk}
        stats['gt'] = scatter_stats(S, G, G)          # efeito da suavização
        (out_dir / f'scatter_{loss}.json').write_text(json.dumps(dict(**meta, stats=stats, **out), indent=1))
        print(json.dumps(stats, indent=1))
    print(f'salvo em {out_dir}')


def run_saturation(out_dir: Path, fig_dir: Path, max_samples=None, knee_fracs=(0.5,),
                   check_femm=False, subset='test'):
    """Saturação do aço, só sobre o gabarito: curva B-H do .ans (conferida igual em todas
    as amostras), pontos de operação dos elementos de aço (|B| suavizado no centróide) e
    μ_eff; figura A (curva + distribuição) e B (|B| no aço nas amostras P50 e de maior |B|
    médio no aço, com contornos no joelho). Subset como no eval."""
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


SAT_BH_LABELS = {'FNO_BipartiteGNN_Elem': 'FNO_Bipartite'}


def run_sat_bh(data_dir: Path, fig_dir: Path, loss='mae', cset='aco_int', n_int=50):
    """Erro × ponto de operação sobre a curva B-H, SEM inferência: lê o saturation_error.json
    do regions (histograma Σw/Σe·w em |B| da referência, B_ref de Ω de cada run e a curva B-H
    do .ans). Por arch (REGIONS_ARCHS, loss dada): os n_int intervalos de mesma área de aço do
    sat_report (conjunto cset; padrão aço ∩ Ω_int), L1 e |B| mediano, em H = H(|B| mediano)
    pelo FemmBH. Faixas abaixo do pico de μ_eff / pico–joelho / acima do joelho (sat_bh_bands).
    Saída: <fig_dir>/saturacao_erro_sobre_bh[_n<N> se placeholder].{png,pdf} e
    <data_dir>/saturation_error_bh.json."""
    d = json.loads((data_dir / 'saturation_error.json').read_text(encoding='utf-8'))
    bh = FemmBH(d['bh']['B_T'], d['bh']['H_A_m'])
    knee = bh.knee(0.5)
    kj = d['bh']['knee']
    assert abs(knee['B_knee_T'] - kj['B_knee_T']) < 1e-9 and abs(knee['B_mu_max_T'] - kj['B_mu_max_T']) < 1e-9, \
        'joelho/pico recalculados diferem do saturation_error.json'
    H_pk = float(bh.h(np.array([knee['B_mu_max_T']]))[0])
    edges = np.asarray(d['hist']['edges_T'])
    cuts = [knee['B_mu_max_T'], knee['B_knee_T']]
    series, rep, area = {}, {}, None
    for a in REGIONS_ARCHS:
        k = f'{a}_{loss}'
        r = d['runs'][k]
        iv = r[cset]['intervals']
        assert len(iv) == n_int, f'{k}: {len(iv)} intervalos (esperado {n_int})'
        bmed = np.array([i['B_median_T'] for i in iv])
        l1 = np.array([i['L1_pct'] for i in iv])
        series[a] = (bh.h(bmed), l1)
        bands = sat_bh_bands(d['hist']['sums'][k][cset], edges, r['B_ref_omega'], cuts)
        # conferências: Σ frações = 1 e L1 ponderado pelas áreas = L1 total do sat_report
        tot = r[cset]['total']['L1_pct']
        l1w = sum(b['area_frac'] * b['L1_pct'] for b in bands)
        assert abs(sum(b['area_frac'] for b in bands) - 1) < 1e-9 and abs(l1w - tot) < 1e-9 * max(tot, 1), k
        # a área de aço por faixa depende do conjunto de pontos de quadratura da run
        # (FNO2d: overlay grade × malha; demais: pontos da malha) -> reportada por arch;
        # runs do mesmo conjunto têm de coincidir
        fr = [b['area_frac'] for b in bands]
        ps = 'overlay' if _regions_kind(a) == 'grade' else 'mesh'
        area = area or {}
        assert ps not in area or np.allclose(fr, area[ps], atol=1e-12), f'{k}: área difere no conjunto {ps}'
        area.setdefault(ps, fr)
        rep[a] = dict(run=k, run_dir=r['run_dir'], best_epoch=r['best_epoch'], B_ref_omega=r['B_ref_omega'],
                      point_set=ps, area_frac=fr, L1_total_pct=tot, bands=bands,
                      intervals=[dict(B_median_T=float(b_), H_A_m=float(h_), L1_pct=float(l_))
                                 for b_, h_, l_ in zip(bmed, series[a][0], l1)])
    names = ('abaixo_do_pico', 'pico_joelho', 'acima_do_joelho')
    out = dict(placeholder=d.get('placeholder'), n_samples=d['n_samples'], subset=d['subset'],
               loss=loss, conjunto=cset, d_interior_mm=d['d_interior_mm'],
               pico_mu_eff=dict(B_T=knee['B_mu_max_T'], H_A_m=H_pk, mu_eff=knee['mu_max']),
               joelho=dict(B_T=knee['B_knee_T'], H_A_m=knee['H_knee_A_m'], frac=knee['frac'],
                           mu_eff=knee['frac'] * knee['mu_max']),
               area_frac={ps: dict(zip(names, fr)) for ps, fr in area.items()},
               L1_pct={SAT_BH_LABELS.get(a, a): dict(zip(names, [b['L1_pct'] for b in v['bands']]),
                                                     total=v['L1_total_pct']) for a, v in rep.items()},
               definicao='faixas em |B| da referência: [0, B_pico), [B_pico, B_joelho), [B_joelho, ∞); '
                         'bin de 0,005 T cortado pela fronteira dividido na proporção da largura; L1 em % '
                         'do B_ref global em Ω da mesma run',
               runs=rep)
    (data_dir / 'saturation_error_bh.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                      encoding='utf-8')
    apply_style()
    name = 'saturacao_erro_sobre_bh' + (f"_n{d['n_samples']}" if d.get('placeholder') else '')
    save_fig(fig_sat_error_bh(bh, knee, series, MODEL_COLOR, SAT_BH_LABELS), fig_dir / name, pdf=True, dpi=200)
    print(f"{d['n_samples']} amostras ({d['subset']}), {cset}, {loss.upper()}")
    print(f"pico de μ_eff: B = {knee['B_mu_max_T']:.4f} T, H = {H_pk:.1f} A/m (μ_eff = {knee['mu_max']:.0f})")
    print(f"joelho (μ_eff = máx/2): B = {knee['B_knee_T']:.4f} T, H = {knee['H_knee_A_m']:.1f} A/m")
    for ps, fr in area.items():
        print(f'área de aço (pontos {ps}): ' + ', '.join(f'{n} {100 * f:.3f}%' for n, f in zip(names, fr)))
    for a, v in out['L1_pct'].items():
        print(f'  L1 {a:14s} ' + ', '.join(f'{n} {v[n]:.3f}%' for n in names) + f" | total {v['total']:.3f}%")
    print(f"salvo em {data_dir / 'saturation_error_bh.json'}; figura {fig_dir / name}.png/.pdf")


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
    n_nodes = np.array([t['vertex']['node_x'].shape[0] for t in ts])
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
    """Custo computacional: FEMM (só mi_analyze) | pré-processamento | inferência."""
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
    """Tabela de treinamento: só leitura dos logs das runs de find_runs(ROOT_S)."""
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
              f"{r['hardware']['gpu_name'] or r['hardware']['device']} @ {r['hardware']['hostname']} "
              f"(torch {r['hardware']['torch_version']}, CUDA {r['hardware']['cuda_version']}, "
              f"cuDNN {r['hardware']['cudnn_version']})")
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
    p_rg = sp.add_parser('regions', help='erro de superfície por faixas de distância (quinas/fronteira/interior)')
    p_rg.add_argument('n', nargs='?', type=int, default=None, help='nº de amostras (default: todas)')
    p_rg.add_argument('--d', type=float, nargs='+', default=list(REGIONS_D_MM),
                      help=f'larguras de faixa da tabela, mm (default: {REGIONS_D_MM[0]:.4f} = '
                           f'passo angular da grade no menor raio)')
    p_rg.add_argument('--d-max', type=float, default=REGIONS_D_MAX_MM,
                      help=f'maior d das curvas, mm (default: {REGIONS_D_MAX_MM})')
    p_rg.add_argument('--n-bins', type=int, default=REGIONS_N_BINS,
                      help=f'bins de distância até d-max (default: {REGIONS_N_BINS})')
    p_rg.add_argument('--bin-scale', default='lin', choices=('lin', 'log'),
                      help="bins uniformes ('lin', padrão) ou log (d_max/1e3 a d_max)")
    p_rg.add_argument('--ang-canto', type=float, default=30.0, help='graus (default: 30)')
    p_rg.add_argument('--subset', **SUBSET_ARG)
    p_rg.add_argument('--out', type=Path, default=OUT_ROOT / 'dados', help='diretório de saída')
    p_rg.add_argument('--fig', type=Path, default=OUT_ROOT / 'plots', help='diretório da figura')
    p_tq = sp.add_parser('torque', help='torque FEMM × gabarito × modelos (theta-e/batch/scatter)')
    p_tq.add_argument('mode', choices=('theta-e', 'batch', 'scatter'))
    p_tq.add_argument('--idx', type=int, nargs='+', default=None,
                      help='theta-e: índices das amostras no raw (default: 5 amostras do conjunto de '
                           'TESTE do split.json da run de referência, 1 por faixa de 2 A de corrente, '
                           'seed 0 — select_batch_samples)')
    p_tq.add_argument('--loss', default='mae', choices=('mse', 'mae'))
    p_tq.add_argument('--archs', nargs='+', default=None, help='default: todas (bateria bldc + FNO_SplitGNN)')
    p_tq.add_argument('--femm', action='store_true', help='resolve as varreduras no FEMM antes')
    p_tq.add_argument('--femm-ref', action='store_true',
                      help='calcula e grava também o T22 do FEMM (mo_blockintegral); fora dos resumos '
                           '(a referência é sempre o smooth)')
    p_tq.add_argument('--cache-only', action='store_true',
                      help='batch: não chama o FEMM; ângulos sem cache válido são pulados e registrados')
    p_tq.add_argument('--workers', type=int, default=None, help='processos de parse (default: TORQUE_WORKERS)')
    p_tq.add_argument('--first', type=int, default=None,
                      help='batch: só as K primeiras amostras da seleção (placeholder)')
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
    p_sb = sp.add_parser('sat-bh', help='erro × ponto de operação sobre a curva B-H (lê saturation_error.json)')
    p_sb.add_argument('--data', type=Path, default=OUT_ROOT / 'dados',
                      help='diretório com saturation_error.json (do regions)')
    p_sb.add_argument('--fig', type=Path, default=OUT_ROOT / 'plots')
    p_sb.add_argument('--loss', default='mae', choices=('mse', 'mae'))
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
        run_regions(a.out, a.fig, a.n, a.d, a.d_max, a.n_bins, a.bin_scale, a.ang_canto, a.subset)
    elif a.cmd == 'torque':
        run_torque(a.mode, a.out, a.loss, a.archs, a.femm, a.idx, a.n_theta,
                   a.n_samples, a.n_per_chunk, a.limit, a.raw_dir, a.work_dir, a.subset,
                   a.femm_ref, a.first, a.cache_only, a.workers)
    elif a.cmd == 'saturation':
        run_saturation(a.out, a.fig, a.n, tuple(a.knee_frac), a.check_femm, a.subset)
    elif a.cmd == 'sat-bh':
        run_sat_bh(a.data, a.fig, a.loss)
    elif a.cmd == 'timing':
        run_timing(a.what, a.out, a.n_femm, a.warmup, a.reps, a.max_batch, a.n_infer_samples,
                   a.subset)
    elif a.cmd == 'train-table':
        run_train_table(a.out)
