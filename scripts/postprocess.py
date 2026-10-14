"""
postprocess.py
--------------
Executável do pós-processamento (código em src/postprocess/) — só gabarito SMOOTH.
Nada aqui treina; lê as runs em data/logs/ e grava em resultados/<dataset>/:
  dados/  — números (json/npz), versionados;
  plots/  — figuras, fora do git.

Subcomandos:
  eval [N_AMOSTRAS] — os 8 runs da bateria smooth (4 archs × mse/mae, best.pth)
      contra o gabarito smooth, nos chunks fora do treino ('teste37'/'fora51',
      N -> metade de cada). Por amostra:
        mesh : |B| predito nos nós vs node_y (área dual do nó);
        grid : saída do estágio FNO (H×W) vs y_hw (r·dr·dθ);
        piso : y_hw do gabarito interpolado nos nós (INTERP_MODE) vs node_y.
      Saída: dados/eval_smooth.json + eval_smooth_per_sample.npz
      Origem: tests/resultados/eval_smooth.py (2026-10-13), sem mudança de lógica.

Execução (raiz do projeto):
  python -m scripts.postprocess eval [N_AMOSTRAS] [--out DIR]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.postprocess.fields import mag
from src.postprocess.inference import INTERP_MODE, floor_nodes, load_set, predict, to_torch_sample
from src.postprocess.metrics import Accum, grid_area_weights, merge, node_err, pstats
from src.postprocess.samples import SUBSETS, design_rows, parse_smooth, samples_with_subset

DATASET   = 'mesh_ans_138x276_smooth'
ROOT_S    = Path('data/logs/mesh_ans_138x276_smooth_battery')        # bateria smooth
SPLIT_RUN = ROOT_S / 'FNO2d' / 'run_0001'                            # referência do split
OUT_ROOT  = Path('resultados') / DATASET
N_WORKERS = 12


def run_eval(out_dir: Path, max_samples=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = design_rows()
    samples = samples_with_subset(SPLIT_RUN, max_samples)
    runs = load_set(ROOT_S)
    for k, r in runs.items():
        print(f"  {k:24s} {r['run']} best ép.{r['epoch']} interp={r['interp_mode']}")

    acc = {(k, sub, kind): Accum() for k in runs for sub in SUBSETS for kind in ('mesh', 'grid')}
    acc_floor = {sub: Accum() for sub in SUBSETS}
    per = {k: [] for k in runs}          # run_key -> list[(L1, L2, pt)]
    per_floor = []
    ids, subs, n_nodes = [], [], []

    groups = {}
    for name, sub, idx, p in samples:
        groups.setdefault(name, []).append((sub, idx, p))

    t0 = time.time()
    done = 0
    for gi, (name, items) in enumerate(groups.items()):
        t = time.time()
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items))) as ex:
            futs = [ex.submit(parse_smooth, p, rows[idx]) for _, idx, p in items]
            for (sub, idx, p), f in zip(items, futs):
                lay = f.result()
                ts = to_torch_sample(lay)
                area = lay['v1']['node_x'][:, 2]
                gt = lay['v1']['node_y']
                assert np.array_equal(gt, lay['bip']['node_y'])
                mag_t = np.hypot(gt[:, 0], gt[:, 1])
                y_hw = ts['v1']['y_hw']
                agrid = grid_area_weights(*y_hw.shape[-2:])
                ids.append(idx)
                subs.append(sub)
                n_nodes.append(len(area))

                mf = mag(floor_nodes(ts))
                acc_floor[sub].add(mf, mag_t, area)
                per_floor.append(node_err(mf, mag_t, area))

                for k, r in runs.items():
                    out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], ts)
                    mag_p = mag(yn.float().cpu())
                    acc[(k, sub, 'mesh')].add(mag_p, mag_t, area)
                    acc[(k, sub, 'grid')].add(mag(out_hw.float().cpu()), mag(y_hw), agrid)
                    per[k].append(node_err(mag_p, mag_t, area))
                done += 1
        print(f'  [{gi + 1}/{len(groups)}] {name} ({items[0][0]})  {time.time() - t:.0f}s  '
              f'(amostras {done}, {(time.time() - t0) / 60:.1f} min)', flush=True)

    # ------------------------------------------------------------------ #
    subs_arr = np.array(subs)
    present = [s for s in SUBSETS if (subs_arr == s).any()]

    def reports(getter):
        r = {sub: getter([sub]).report() for sub in present}
        r['total'] = getter(list(SUBSETS)).report()
        return r

    res = dict(n_samples=done, n_by_subset={s: int((subs_arr == s).sum()) for s in SUBSETS},
               interp_mode=INTERP_MODE, runs={},
               floor=reports(lambda ss: merge([acc_floor[s] for s in ss])))
    for k, r in runs.items():
        arr = np.array(per[k])
        res['runs'][k] = dict(
            arch=r['arch'], loss=r['loss'], run=r['run'], best_epoch=r['epoch'],
            mesh=reports(lambda ss, k=k: merge([acc[(k, s, 'mesh')] for s in ss])),
            grid=reports(lambda ss, k=k: merge([acc[(k, s, 'grid')] for s in ss])),
            per_sample_L1_A={sub: pstats(arr[subs_arr == sub, 0]) for sub in present})

    np.savez_compressed(
        out_dir / 'eval_smooth_per_sample.npz',
        sample_idx=np.array(ids), subset=subs_arr, n_nodes=np.array(n_nodes),
        floor=np.array(per_floor), **{k: np.array(v) for k, v in per.items()},
    )
    (out_dir / 'eval_smooth.json').write_text(json.dumps(res, indent=2))
    print(f'\nsalvo em {out_dir} ({(time.time() - t0) / 60:.1f} min)')
    print(f'{done} amostras — erro de |B| nos nós, % de B_ref (pontual / L1_A / L2_A)')
    for k, v in res['runs'].items():
        g = v['mesh']['total']
        print(f"  {k:24s} pt {g['MAE_pt_pct']:5.2f}%  L1 {g['L1_area_pct']:5.2f}%"
              f"  L2 {g['L2_area_pct']:5.2f}%")
    g = res['floor']['total']
    print(f"  {'piso da grade':24s} pt {g['MAE_pt_pct']:5.2f}%  L1 {g['L1_area_pct']:5.2f}%"
          f"  L2 {g['L2_area_pct']:5.2f}%")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    sp = ap.add_subparsers(dest='cmd', required=True)
    p_eval = sp.add_parser('eval', help='bateria smooth × gabarito smooth (+ piso da grade)')
    p_eval.add_argument('n', nargs='?', type=int, default=None, help='nº de amostras (default: todas)')
    p_eval.add_argument('--out', type=Path, default=OUT_ROOT / 'dados', help='diretório de saída')
    a = ap.parse_args()
    if a.cmd == 'eval':
        run_eval(a.out, a.n)
