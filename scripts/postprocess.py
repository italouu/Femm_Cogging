"""
postprocess.py
--------------
Executável do pós-processamento (código em src/postprocess/) — só gabarito SMOOTH.
Nada aqui treina; lê as runs em data/logs/ e grava em resultados/<dataset>/:
  dados/  — números (json/npz), versionados;
  plots/  — figuras, fora do git.

Subcomandos:
  eval [N_AMOSTRAS] — os 10 runs da bateria (5 archs × mse/mae, best.pth) nos chunks fora
      do treino ('teste'/'fora' — 'fora' vazio no split 70/30; N -> dividido entre os subsets
      não vazios). ERRO DE SUPERFÍCIE OFICIAL (2026-10-20): superfície discreta do modelo ×
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

Execução (raiz do projeto):
  python -m scripts.postprocess eval [N_AMOSTRAS] [--out DIR]
  python -m scripts.postprocess figures [--data DIR] [--out DIR]
  python -m scripts.postprocess surface IDX [LOSS]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scripts.build_smooth_ans_chunks_direct import MAX_SAMPLES
from src.configs.datagen import DatagenConfig
# [REMOVIDO 2026-10-20] sem as projeções elemento <-> nó
# from src.postprocess.fields import mag
from src.data_gen.parsers.femm_mesh_unified import _read_ans_gz
# [REMOVIDO 2026-10-20] nodes_to_elem (nível elem removido)
# from src.postprocess.fields import elem_to_nodes, gap_radius_mm, mag, nodes_to_elem
from src.postprocess.fields import elem_to_nodes, gap_radius_mm, mag
# [REMOVIDO 2026-10-20] sem ARCHS/RUN_ORDER (usados pelas figuras)
# from src.postprocess.inference import (INTERP_MODE, find_runs, floor_nodes, load_set, predict,
#                                        to_torch_sample)
# [REMOVIDO 2026-10-20] INTERP_MODE/floor_nodes (piso da grade nos nós, removido)
# from src.postprocess.inference import (ARCHS, INTERP_MODE, RUN_ORDER, find_runs, floor_nodes,
#                                        load_set, predict, to_torch_sample)
# [REMOVIDO 2026-10-20] sem INTERP_MODE/floor_nodes -- voltam com o erro nodal secundário (P1)
# from src.postprocess.inference import (ARCHS, RUN_ORDER, find_runs, load_set, predict,
#                                        to_torch_sample)
from src.postprocess.inference import (ARCHS, INTERP_MODE, RUN_ORDER, find_runs, floor_nodes,
                                       load_set, predict, to_torch_sample)
# [REMOVIDO 2026-10-20] sem surf_err
# from src.postprocess.metrics import Accum, grid_area_weights, merge, node_err, pstats
# [REMOVIDO 2026-10-20] sem a superfície malha × malha
# from src.postprocess.metrics import Accum, grid_area_weights, merge, node_err, pstats, surf_err
# [REMOVIDO 2026-10-20] níveis antigos (grid_area_weights, node_err, surf_err, add_mesh_surface)
# from src.postprocess.metrics import (Accum, add_mesh_surface, grid_area_weights, merge, node_err,
#                                      pstats, surf_err, surface_values_p1)
from src.postprocess.metrics import (Accum, merge, pstats, surface_error_per_triangle,
                                     surface_values_p1)
from src.postprocess.plots.errors import fig_bars, fig_surface_error
from src.postprocess.plots.fields import fig_field
from src.postprocess.plots.primitives import apply_style, save_fig
from src.postprocess.plots.training import fig_training
# [REMOVIDO 2026-10-20] sem sample_path/TMP_ROOT (usados pelas figuras)
# from src.postprocess.samples import SUBSETS, design_rows, parse_smooth, samples_with_subset
from src.postprocess.samples import (SUBSETS, TMP_ROOT, design_rows, parse_smooth,
                                     sample_path, samples_with_subset)

# [REMOVIDO 2026-10-13] nomes com sufixo _smooth e run de referência do split fixa
# (bateria de 4000 amostras). Agora o dataset smooth é o oficial e o split vem da run
# FNO2d mse da própria bateria (find_runs).
# DATASET   = 'mesh_ans_138x276_smooth'
# ROOT_S    = Path('data/logs/mesh_ans_138x276_smooth_battery')        # bateria smooth
# SPLIT_RUN = ROOT_S / 'FNO2d' / 'run_0001'                            # referência do split
# [REMOVIDO 2026-10-14] DATASET   = 'mesh_ans_138x276' — agora DatagenConfig.dataset (= raw)
DATASET   = DatagenConfig().dataset
ROOT_S    = Path('data/logs') / DATASET                              # logs da bateria smooth
N_SAMPLES = MAX_SAMPLES   # amostras do raw no dataset (as mesmas do build dos chunks)
OUT_ROOT  = Path('resultados') / DATASET
N_WORKERS = 12
# [REMOVIDO 2026-10-20] sem grid_inter
# LEVELS    = ('mesh', 'grid', 'elem')                        # 'elem' só em ELEM_ARCHS
# [REMOVIDO 2026-10-20] sem 'surface'
# LEVELS    = ('mesh', 'grid', 'grid_inter', 'elem')          # 'elem' só em ELEM_ARCHS
# [REMOVIDO 2026-10-20] níveis antigos removidos -- só a superfície oficial
# LEVELS    = ('surface', 'mesh', 'grid', 'grid_inter', 'elem')  # 'elem' só em ELEM_ARCHS
GRID_ARCHS = ('FNO2d',)                # saída = grade H×W -> superfície por interseção
SURF_KINDS = ('grade', 'P1_nos', 'P0_elem')   # pisos: o gabarito em cada tipo de superfície
# [REMOVIDO 2026-10-20] ELEM_ARCHS = ('FNO_BipartiteGNN', 'FNO_BipartiteGNN_Elem')  (nível elem)
REF_RUN   = 'FNO_BipartiteGNN_mae'     # run que escolhe as amostras P50/P95 das figuras


def _mag_np(b):
    """|B| de [N,2] numpy."""
    return np.hypot(b[:, 0], b[:, 1])


# [REMOVIDO 2026-10-20] versão com os níveis antigos (mesh nodal, grid × y_hw, grid_inter de
# todas as archs, elem e pisos elem<->nó) -- removidos por decisão do usuário; a avaliação é
# só o erro de superfície oficial (nível surface + pisos de superfície).
# def run_eval(out_dir: Path, max_samples=None):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     rows = design_rows()
#     # [REMOVIDO 2026-10-13] run de referência fixa
#     # samples = samples_with_subset(SPLIT_RUN, max_samples)
#     samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], max_samples, N_SAMPLES)
#     runs = load_set(ROOT_S)
#     for k, r in runs.items():
#         print(f"  {k:24s} {r['run']} best ép.{r['epoch']} interp={r['interp_mode']}")
#
#     # [REMOVIDO 2026-10-20] sem o nível 'elem' (Elem integrada ao eval)
#     # acc = {(k, sub, kind): Accum() for k in runs for sub in SUBSETS for kind in ('mesh', 'grid')}
#     acc = {(k, sub, kind): Accum() for k in runs for sub in SUBSETS for kind in LEVELS}
#     acc_floor = {sub: Accum() for sub in SUBSETS}
#     acc_floor_elem = {(lv, sub): Accum() for lv in ('mesh', 'elem') for sub in SUBSETS}
#     acc_floor_inter = {sub: Accum() for sub in SUBSETS}
#     # superfície oficial: piso de cada tipo de superfície discreta, com o próprio gabarito
#     acc_floor_surf = {(kind, sub): Accum() for kind in SURF_KINDS for sub in SUBSETS}
#     per_surf = {k: [] for k in runs}     # run_key -> list[(L1_A, L2_A)] da superfície
#     per = {k: [] for k in runs}          # run_key -> list[(L1, L2, pt)]
#     per_floor = []
#     ids, subs, n_nodes = [], [], []
#
#     groups = {}
#     for name, sub, idx, p in samples:
#         groups.setdefault(name, []).append((sub, idx, p))
#
#     t0 = time.time()
#     done = 0
#     for gi, (name, items) in enumerate(groups.items()):
#         t = time.time()
#         with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items))) as ex:
#             # [REMOVIDO 2026-10-20] sem o layout da Elem
#             # futs = [ex.submit(parse_smooth, p, rows[idx]) for _, idx, p in items]
#             # [REMOVIDO 2026-10-20] sem a sobreposição grade × malha
#             # futs = [ex.submit(parse_smooth, p, rows[idx], with_elem=True) for _, idx, p in items]
#             futs = [ex.submit(parse_smooth, p, rows[idx], with_elem=True, with_overlay=True)
#                     for _, idx, p in items]
#             for (sub, idx, p), f in zip(items, futs):
#                 lay = f.result()
#                 ts = to_torch_sample(lay)
#                 area = lay['v1']['node_x'][:, 2]
#                 gt = lay['v1']['node_y']
#                 assert np.array_equal(gt, lay['bip']['node_y'])
#                 mag_t = np.hypot(gt[:, 0], gt[:, 1])
#                 y_hw = ts['v1']['y_hw']
#                 agrid = grid_area_weights(*y_hw.shape[-2:])
#                 ids.append(idx)
#                 subs.append(sub)
#                 n_nodes.append(len(area))
#
#                 assert np.array_equal(lay['elem']['y_hw'], lay['v1']['y_hw'])
#                 gt_e = lay['elem']['node_y'].astype(np.float64)      # [M,2] centróides
#                 a_e = lay['elem']['node_x'][:, 4]                    # área do elemento
#                 mag_te = np.hypot(gt_e[:, 0], gt_e[:, 1])
#
#                 mf = mag(floor_nodes(ts))
#                 acc_floor[sub].add(mf, mag_t, area)
#                 per_floor.append(node_err(mf, mag_t, area))
#                 ov = lay['overlay']
#                 acc_floor_inter[sub].add_surface(mag(y_hw).ravel()[ov['cell']], ov['mag'], ov['w'])
#                 mq = lay['mesh_quad']
#                 acc_floor_surf[('grade', sub)].add_surface(mag(y_hw).ravel()[ov['cell']],
#                                                            ov['mag'], ov['w'])
#                 add_mesh_surface(acc_floor_surf[('P1_nos', sub)], mq,
#                                  surface_values_p1(lay['tri'], gt.astype(np.float64), mq['bary']))
#                 add_mesh_surface(acc_floor_surf[('P0_elem', sub)], mq, mag_te)
#                 # piso da troca de suporte: gabarito elem -> nós e nós -> elem
#                 gt_n = gt.astype(np.float64)
#                 acc_floor_elem[('mesh', sub)].add(_mag_np(elem_to_nodes(lay, gt_e)), mag_t, area)
#                 acc_floor_elem[('elem', sub)].add(_mag_np(nodes_to_elem(lay, gt_n)), mag_te, a_e)
#
#                 for k, r in runs.items():
#                     out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
#                     # [REMOVIDO 2026-10-20] saída sempre por nó -- a Elem sai por elemento
#                     # mag_p = mag(yn.float().cpu())
#                     yp = yn.double().cpu().numpy()
#                     if r['arch'] == 'FNO_BipartiteGNN_Elem':
#                         p_e, yp = yp, elem_to_nodes(lay, yp)
#                     elif r['arch'] == 'FNO_BipartiteGNN':
#                         p_e = nodes_to_elem(lay, yp)
#                     else:
#                         p_e = None
#                     mag_p = _mag_np(yp)
#                     if p_e is not None:
#                         acc[(k, sub, 'elem')].add(_mag_np(p_e), mag_te, a_e)
#                     acc[(k, sub, 'mesh')].add(mag_p, mag_t, area)
#                     mag_hw = mag(out_hw.float().cpu())
#                     acc[(k, sub, 'grid')].add(mag_hw, mag(y_hw), agrid)
#                     acc[(k, sub, 'grid_inter')].add_surface(mag_hw.ravel()[ov['cell']],
#                                                             ov['mag'], ov['w'])
#                     per[k].append(node_err(mag_p, mag_t, area))
#                     # superfície discreta do modelo × superfície do FEMM (oficial, 2026-10-20)
#                     a_s = Accum()
#                     if r['arch'] in GRID_ARCHS:
#                         a_s.add_surface(mag_hw.ravel()[ov['cell']], ov['mag'], ov['w'])
#                     elif r['arch'] == 'FNO_BipartiteGNN_Elem':
#                         add_mesh_surface(a_s, mq, _mag_np(p_e))
#                     else:
#                         add_mesh_surface(a_s, mq, surface_values_p1(lay['tri'], yp, mq['bary']))
#                     acc[(k, sub, 'surface')] = merge([acc[(k, sub, 'surface')], a_s])
#                     rs = a_s.report()
#                     per_surf[k].append((rs['L1_area'], rs['L2_area']))
#                 done += 1
#         print(f'  [{gi + 1}/{len(groups)}] {name} ({items[0][0]})  {time.time() - t:.0f}s  '
#               f'(amostras {done}, {(time.time() - t0) / 60:.1f} min)', flush=True)
#
#     # ------------------------------------------------------------------ #
#     subs_arr = np.array(subs)
#     present = [s for s in SUBSETS if (subs_arr == s).any()]
#
#     def reports(getter):
#         r = {sub: getter([sub]).report() for sub in present}
#         r['total'] = getter(list(SUBSETS)).report()
#         return r
#
#     res = dict(n_samples=done, n_by_subset={s: int((subs_arr == s).sum()) for s in SUBSETS},
#                interp_mode=INTERP_MODE, runs={},
#                floor=reports(lambda ss: merge([acc_floor[s] for s in ss])),
#                floor_grid_inter=reports(lambda ss: merge([acc_floor_inter[s] for s in ss])),
#                floor_surface={kd: reports(lambda ss, kd=kd: merge([acc_floor_surf[(kd, s)] for s in ss]))
#                               for kd in SURF_KINDS},
#                floor_elem={lv: reports(lambda ss, lv=lv: merge([acc_floor_elem[(lv, s)] for s in ss]))
#                            for lv in ('mesh', 'elem')})
#     for k, r in runs.items():
#         arr = np.array(per[k])
#         res['runs'][k] = dict(
#             arch=r['arch'], loss=r['loss'], run=r['run'], best_epoch=r['epoch'],
#             surface_kind=('grade' if r['arch'] in GRID_ARCHS else
#                           'P0_elem' if r['arch'] == 'FNO_BipartiteGNN_Elem' else 'P1_nos'),
#             surface=reports(lambda ss, k=k: merge([acc[(k, s, 'surface')] for s in ss])),
#             mesh=reports(lambda ss, k=k: merge([acc[(k, s, 'mesh')] for s in ss])),
#             grid=reports(lambda ss, k=k: merge([acc[(k, s, 'grid')] for s in ss])),
#             grid_inter=reports(lambda ss, k=k: merge([acc[(k, s, 'grid_inter')] for s in ss])),
#             per_sample_L1_A={sub: pstats(arr[subs_arr == sub, 0]) for sub in present})
#         if r['arch'] in ELEM_ARCHS:
#             res['runs'][k]['elem'] = reports(lambda ss, k=k: merge([acc[(k, s, 'elem')] for s in ss]))
#
#     np.savez_compressed(
#         out_dir / 'eval_smooth_per_sample.npz',
#         sample_idx=np.array(ids), subset=subs_arr, n_nodes=np.array(n_nodes),
#         floor=np.array(per_floor), **{k: np.array(v) for k, v in per.items()},
#         **{f'surface_{k}': np.array(v) for k, v in per_surf.items()},
#     )
#     (out_dir / 'eval_smooth.json').write_text(json.dumps(res, indent=2))
#     print(f'\nsalvo em {out_dir} ({(time.time() - t0) / 60:.1f} min)')
#     print(f'{done} amostras — ERRO DE SUPERFÍCIE (oficial): superfície discreta do modelo × '
#           f'FEMM (B suavizado P1), % de B_ref (L1_A / L2_A)')
#     for k, v in res['runs'].items():
#         g = v['surface']['total']
#         print(f"  {k:26s} [{v['surface_kind']:7s}] L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%"
#               f"  (B_ref {g['B_ref']:.4f} T)")
#     for kd in SURF_KINDS:
#         g = res['floor_surface'][kd]['total']
#         print(f"  {'piso ' + kd:26s} [{kd:7s}] L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%")
#     print(f'{done} amostras — erro de |B| nos nós, % de B_ref (pontual / L1_A / L2_A)')
#     for k, v in res['runs'].items():
#         g = v['mesh']['total']
#         print(f"  {k:24s} pt {g['MAE_pt_pct']:5.2f}%  L1 {g['L1_area_pct']:5.2f}%"
#               f"  L2 {g['L2_area_pct']:5.2f}%")
#     g = res['floor']['total']
#     print(f"  {'piso da grade':24s} pt {g['MAE_pt_pct']:5.2f}%  L1 {g['L1_area_pct']:5.2f}%"
#           f"  L2 {g['L2_area_pct']:5.2f}%")
#     g = res['floor_elem']['mesh']['total']
#     print(f"  {'piso elem -> nós':24s} pt {g['MAE_pt_pct']:5.2f}%  L1 {g['L1_area_pct']:5.2f}%"
#           f"  L2 {g['L2_area_pct']:5.2f}%")
#     print('erro de |B| da saída em grade × FEMM (interseção célula x triângulo): L1_A / L2_A '
#           '[grid: × y_hw no centro do pixel]')
#     for k, v in res['runs'].items():
#         g, h = v['grid_inter']['total'], v['grid']['total']
#         print(f"  {k:24s} L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%"
#               f"   [grid L1 {h['L1_area_pct']:5.2f}%  L2 {h['L2_area_pct']:5.2f}%]")
#     g = res['floor_grid_inter']['total']
#     print(f"  {'piso y_hw':24s} L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%")
#     print('erro de |B| por elemento (área do elemento)')
#     for k, v in res['runs'].items():
#         if 'elem' in v:
#             g = v['elem']['total']
#             print(f"  {k:24s} pt {g['MAE_pt_pct']:5.2f}%  L1 {g['L1_area_pct']:5.2f}%"
#                   f"  L2 {g['L2_area_pct']:5.2f}%")
#     g = res['floor_elem']['elem']['total']
#     print(f"  {'piso nós -> elem':24s} pt {g['MAE_pt_pct']:5.2f}%  L1 {g['L1_area_pct']:5.2f}%"
#           f"  L2 {g['L2_area_pct']:5.2f}%")


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
    _, nodes, elems = _read_ans_gz(sample_path(idx), TMP_ROOT / 'parse')
    return nodes, elems


# [REMOVIDO 2026-10-20] figures sobre os níveis antigos (mesh nodal, grid × y_hw, grid_inter de
# todas as archs, elem e pisos elem<->nó) -- removidos por decisão do usuário; a avaliação é
# só o erro de superfície oficial (nível surface + pisos de superfície).
# def run_figures(data_dir: Path, fig_dir: Path):
#     """Origem: tests/resultados/figures_smooth.main (2026-10-20). P50/P95 do L1_A da
#     REF_RUN no subset 'fora' -- ou 'teste', se 'fora' estiver vazio (split 70/30)."""
#     import matplotlib.tri as mtri
#     apply_style()
#     res = json.loads((data_dir / 'eval_smooth.json').read_text())
#     per = np.load(data_dir / 'eval_smooth_per_sample.npz')
#     save_fig(fig_training(find_runs(ROOT_S), ARCHS), fig_dir / '1_curvas_treino')
#     save_fig(fig_bars(res, RUN_ORDER), fig_dir / '2_barras_erro')
#
#     sub_name = 'fora' if (per['subset'] == 'fora').any() else 'teste'
#     sub = per['subset'] == sub_name
#     l1 = per[REF_RUN][:, 0]
#     rows = design_rows()
#     runs = load_set(ROOT_S, losses=('mae',))
#     chosen = {}
#     for tag, q in (('P50', 50), ('P95', 95)):
#         cand = np.flatnonzero(sub)
#         # method='lower': o alvo é sempre o valor de uma amostra (com nº par de amostras a
#         # mediana cai entre duas equidistantes e o empate era decidido por ruído de float)
#         target = np.percentile(l1[cand], q, method='lower')
#         pos = cand[np.argmin(np.abs(l1[cand] - target))]
#         idx = int(per['sample_idx'][pos])
#         chosen[tag] = dict(sample_idx=idx, subset=sub_name, ref_run=REF_RUN,
#                            L1_A_ref=float(l1[pos]))
#
#         lay = parse_smooth(sample_path(idx), rows[idx], with_elem=True)
#         nodes, elems = _mesh(idx)
#         tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=elems[:, :3])
#         gt = _mag_np(lay['v1']['node_y'].astype(np.float64))
#         preds = _node_mags(runs, lay, to_torch_sample(lay))
#         preds = {a: preds[a] for a in ARCHS}
#         r_m, _ = gap_radius_mm(rows[idx])
#         th0, half = np.deg2rad(60.0), 4.5
#         for zoom in (None, (r_m * np.cos(th0), r_m * np.sin(th0), half)):
#             where = (f'zoom no entreferro (r≈{r_m:.1f} mm, θ≈60°, janela {2 * half:.0f} mm)'
#                      if zoom else 'setor de 120°')
#             title = (f'Amostra {idx} ({tag} do L1_A de {REF_RUN} em {sub_name}, '
#                      f'{l1[pos] * 1e3:.1f} mT) — modelos com loss mae — {where}')
#             save_fig(fig_field(tri, gt, preds, title, zoom=zoom),
#                      fig_dir / f"3_campo_{tag}_amostra{idx}{'_zoom' if zoom else ''}")
#         print(f'  {tag}: amostra {idx}')
#     (data_dir / 'figures_smooth.json').write_text(json.dumps(chosen, indent=2))
#     print(f'figuras em {fig_dir}')


# [REMOVIDO 2026-10-20] surface com erro nodal lumped (surf_err nos nós) -- agora a
# mesma superfície oficial do eval, com mapa por triângulo.
# def run_surface(sample_idx, loss, data_dir: Path, fig_dir: Path):
#     """Origem: tests/resultados/surface_error_smooth.main (2026-10-20); 5 archs, sem
#     amostra default (a 3735 era do dataset antigo)."""
#     apply_style()
#     data_dir.mkdir(parents=True, exist_ok=True)
#     row = design_rows()[sample_idx]
#     print(f'amostra {sample_idx}: {sample_path(sample_idx)}')
#     lay = parse_smooth(sample_path(sample_idx), row, with_elem=True)
#     area = lay['v1']['node_x'][:, 2].astype(np.float64)
#     mag_t = _mag_np(lay['bip']['node_y'].astype(np.float64))
#
#     runs = load_set(ROOT_S, losses=(loss,))
#     preds = _node_mags(runs, lay, to_torch_sample(lay))
#     res = dict(sample_idx=sample_idx, loss=loss, n_nodes=int(len(area)), runs={}, erros={})
#     E = {}
#     for r in runs.values():
#         a = r['arch']
#         E[a], res['erros'][a] = surf_err(preds[a], mag_t, area)
#         res['runs'][a] = dict(run_dir=r['run_dir'], epoch=r['epoch'])
#         x = res['erros'][a]
#         print(f"  {a:22s} L1_A={x['L1_A_T']:.4f} T ({x['L1_A_pct']:.2f}%)  "
#               f"L2_A={x['L2_A_T']:.4f} T ({x['L2_A_pct']:.2f}%)  L1_pt={x['L1_pt_T']:.4f} T")
#
#     nodes, elems = _mesh(sample_idx)
#     stem = f'sample_{sample_idx:06d}_{loss}'
#     (data_dir / f'{stem}.json').write_text(json.dumps(res, indent=1))
#     np.savez_compressed(data_dir / f'{stem}.npz', xy=nodes[:, :2], elems=elems[:, :3],
#                         mag_gt=mag_t, **{f'e_{a}': E[a] for a in E})
#     save_fig(fig_surface_error(res, nodes[:, :2], elems[:, :3], E, [a for a in ARCHS if a in E]),
#              fig_dir / stem, pdf=False, dpi=150)
#     print(f'figura: {fig_dir / stem}.png')


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


# [REMOVIDO 2026-10-20] versão sem o erro nodal secundário (P1) -- substituída abaixo
# def run_eval(out_dir: Path, max_samples=None):
#     """Erro de superfície oficial (CLAUDE.md, "Convenção de erro"): superfície discreta de
#     cada modelo × superfície do FEMM (B suavizado P1 descontínuo); L1_A, L2_A, B_ref.
#     Saída: eval_smooth.json (runs + floor_surface) e eval_smooth_per_sample.npz
#     (por run e por piso: [n_amostras, 2] = L1_A, L2_A em T)."""
#     out_dir.mkdir(parents=True, exist_ok=True)
#     rows = design_rows()
#     samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], max_samples, N_SAMPLES)
#     runs = load_set(ROOT_S)
#     for k, r in runs.items():
#         print(f"  {k:26s} {r['run']} best ép.{r['epoch']}  superfície {_surface_kind(r['arch'])}")
#
#     acc = {(k, sub): Accum() for k in runs for sub in SUBSETS}
#     acc_floor = {(kd, sub): Accum() for kd in SURF_KINDS for sub in SUBSETS}
#     per = {k: [] for k in runs}
#     per_floor = {kd: [] for kd in SURF_KINDS}
#     ids, subs = [], []
#
#     groups = {}
#     for name, sub, idx, p in samples:
#         groups.setdefault(name, []).append((sub, idx, p))
#
#     t0 = time.time()
#     done = 0
#     for gi, (name, items) in enumerate(groups.items()):
#         t = time.time()
#         with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items))) as ex:
#             futs = [ex.submit(parse_smooth, p, rows[idx], with_elem=True, with_overlay=True)
#                     for _, idx, p in items]
#             for (sub, idx, p), f in zip(items, futs):
#                 lay = f.result()
#                 ts = to_torch_sample(lay)
#                 ids.append(idx)
#                 subs.append(sub)
#                 for kd in SURF_KINDS:
#                     a = Accum()
#                     a.add_surface(*_floor_points(kd, lay))
#                     acc_floor[(kd, sub)] = merge([acc_floor[(kd, sub)], a])
#                     r_ = a.report()
#                     per_floor[kd].append((r_['L1_area'], r_['L2_area']))
#                 for k, r in runs.items():
#                     out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
#                     mm, mf, w, _ = _surface_points(r['arch'], lay, out_hw, yn)
#                     a = Accum()
#                     a.add_surface(mm, mf, w)
#                     acc[(k, sub)] = merge([acc[(k, sub)], a])
#                     r_ = a.report()
#                     per[k].append((r_['L1_area'], r_['L2_area']))
#                 done += 1
#         print(f'  [{gi + 1}/{len(groups)}] {name} ({items[0][0]})  {time.time() - t:.0f}s  '
#               f'(amostras {done}, {(time.time() - t0) / 60:.1f} min)', flush=True)
#
#     subs_arr = np.array(subs)
#     present = [s for s in SUBSETS if (subs_arr == s).any()]
#
#     def reports(getter):
#         r = {sub: getter([sub]).report() for sub in present}
#         r['total'] = getter(list(SUBSETS)).report()
#         return r
#
#     res = dict(n_samples=done, n_by_subset={s: int((subs_arr == s).sum()) for s in SUBSETS},
#                metric='erro de superfície de |B| (superfície do modelo × FEMM P1 descontínuo)',
#                runs={},
#                floor_surface={kd: reports(lambda ss, kd=kd: merge([acc_floor[(kd, s)] for s in ss]))
#                               for kd in SURF_KINDS})
#     for k, r in runs.items():
#         arr = np.array(per[k])
#         res['runs'][k] = dict(
#             arch=r['arch'], loss=r['loss'], run=r['run'], best_epoch=r['epoch'],
#             surface_kind=_surface_kind(r['arch']),
#             surface=reports(lambda ss, k=k: merge([acc[(k, s)] for s in ss])),
#             per_sample_L1_A={sub: pstats(arr[subs_arr == sub, 0]) for sub in present})
#
#     np.savez_compressed(
#         out_dir / 'eval_smooth_per_sample.npz', sample_idx=np.array(ids), subset=subs_arr,
#         **{k: np.array(v) for k, v in per.items()},
#         **{f'floor_{kd}': np.array(v) for kd, v in per_floor.items()})
#     (out_dir / 'eval_smooth.json').write_text(json.dumps(res, indent=2))
#     print(f'\nsalvo em {out_dir} ({(time.time() - t0) / 60:.1f} min)')
#     print(f'{done} amostras — erro de superfície, % de B_ref (L1_A / L2_A)')
#     for k, v in res['runs'].items():
#         g = v['surface']['total']
#         print(f"  {k:26s} [{v['surface_kind']:7s}] L1 {g['L1_area_pct']:5.2f}%  "
#               f"L2 {g['L2_area_pct']:5.2f}%  (B_ref {g['B_ref']:.4f} T)")
#     for kd in SURF_KINDS:
#         g = res['floor_surface'][kd]['total']
#         print(f"  {'piso ' + kd:26s} [{kd:7s}] L1 {g['L1_area_pct']:5.2f}%  L2 {g['L2_area_pct']:5.2f}%")


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


def run_eval(out_dir: Path, max_samples=None):
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
    samples = samples_with_subset(find_runs(ROOT_S)[('FNO2d', 'mse')], max_samples, N_SAMPLES)
    runs = load_set(ROOT_S)
    for k, r in runs.items():
        print(f"  {k:26s} {r['run']} best ép.{r['epoch']}  superfície {_surface_kind(r['arch'])}"
              f"  interp {r['interp_mode']}")

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
    campo P50/P95 do L1_A de superfície da REF_RUN no subset 'fora' (ou 'teste', se vazio).
    O mapa de campo (3_campo) é só visualização de |B| nos nós."""
    import matplotlib.tri as mtri
    apply_style()
    res = json.loads((data_dir / 'eval_smooth.json').read_text())
    per = np.load(data_dir / 'eval_smooth_per_sample.npz')
    save_fig(fig_training(find_runs(ROOT_S), ARCHS), fig_dir / '1_curvas_treino')
    save_fig(fig_bars(res, RUN_ORDER), fig_dir / '2_barras_erro')

    sub_name = 'fora' if (per['subset'] == 'fora').any() else 'teste'
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


def run_surface(sample_idx, loss, data_dir: Path, fig_dir: Path):
    """Erro de superfície oficial em 1 amostra: L1_A/L2_A de cada modelo e mapa da
    densidade do erro por triângulo do FEMM (surface_error_per_triangle)."""
    apply_style()
    data_dir.mkdir(parents=True, exist_ok=True)
    row = design_rows()[sample_idx]
    print(f'amostra {sample_idx}: {sample_path(sample_idx)}')
    lay = parse_smooth(sample_path(sample_idx), row, with_elem=True, with_overlay=True)
    ts = to_torch_sample(lay)
    n_tri = len(lay['tri'])
    runs = load_set(ROOT_S, losses=(loss,))
    res = dict(sample_idx=sample_idx, loss=loss, n_tri=int(n_tri), runs={}, erros={})
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
    nodes, elems = _mesh(sample_idx)
    stem = f'sample_{sample_idx:06d}_{loss}'
    (data_dir / f'{stem}.json').write_text(json.dumps(res, indent=1))
    np.savez_compressed(data_dir / f'{stem}.npz', xy=nodes[:, :2], elems=elems[:, :3],
                        **{f'e_tri_{a}': E[a] for a in E})
    save_fig(fig_surface_error(res, nodes[:, :2], elems[:, :3], E, [a for a in ARCHS if a in E]),
             fig_dir / stem, pdf=False, dpi=150)
    print(f'figura: {fig_dir / stem}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    sp = ap.add_subparsers(dest='cmd', required=True)
    p_eval = sp.add_parser('eval', help='bateria smooth × gabarito smooth (+ piso da grade)')
    p_eval.add_argument('n', nargs='?', type=int, default=None, help='nº de amostras (default: todas)')
    p_eval.add_argument('--out', type=Path, default=OUT_ROOT / 'dados', help='diretório de saída')
    p_fig = sp.add_parser('figures', help='curvas de treino, barras de erro, campo P50/P95 (após eval)')
    p_fig.add_argument('--data', type=Path, default=OUT_ROOT / 'dados', help='saída do eval')
    p_fig.add_argument('--out', type=Path, default=OUT_ROOT / 'plots', help='diretório das figuras')
    p_sur = sp.add_parser('surface', help='mapa de erro de superfície dos 5 modelos em 1 amostra')
    p_sur.add_argument('idx', type=int, help='índice da amostra no raw')
    p_sur.add_argument('loss', nargs='?', default='mae', choices=('mse', 'mae'))
    p_sur.add_argument('--data', type=Path, default=OUT_ROOT / 'dados' / 'surface_error')
    p_sur.add_argument('--out', type=Path, default=OUT_ROOT / 'plots' / 'surface_error')
    a = ap.parse_args()
    if a.cmd == 'eval':
        run_eval(a.out, a.n)
    elif a.cmd == 'figures':
        run_figures(a.data, a.out)
    elif a.cmd == 'surface':
        run_surface(a.idx, a.loss, a.data, a.out)
