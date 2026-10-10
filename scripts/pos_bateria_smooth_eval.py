"""
pos_bateria_smooth_eval.py
--------------------------
Avaliação da bateria smooth (mesh_ans_138x276_smooth_best_mse_mae) no mesmo
padrão de scripts/eval_surface_integral_table.py / pos_bateria_eval.py da
bateria oficial (mesh_ans_138x276_unified_best_mse_mae). Não treina nada.

Amostras: os 88 chunks de teste da bateria oficial (2816 amostras), que se
dividem em
  - 'teste37' : os 37 chunks de teste da bateria smooth (TEST_SPLIT=0,30 --
                usados na parada/escolha do best.pth das runs smooth);
  - 'fora51'  : os 51 chunks fora do treino E do teste da smooth (avaliação
                independente).
Treino das duas baterias é o mesmo (37 chunks, conferido nos split.json).

Por amostra, parse do raw com os DOIS parsers (unificado e smooth) e:
  S  : modelos smooth   (entrada smooth)   vs gabarito smooth
  U  : modelos oficiais (entrada unified)  vs gabarito unified (média nodal)
  UxS: modelos oficiais (entrada unified)  vs gabarito smooth   (efeito do gabarito)
  piso: y_hw do gabarito interpolado nos nós (INTERP_MODE) vs node_y, para os 2 gabaritos.

Métricas (iguais às anteriores): erro de |B| por nó, e = | |B|_pred - |B|_true |;
  pontual = média simples por nó; L1_A / L2_A = ponderados pela área dual do
  nó (integral de superfície); % relativo a B_ref (RMS de |B|_true ponderado
  por área). "grade": saída do estágio FNO (H×W) vs y_hw, peso r·dr·dθ.

Saída: data/logs/mesh_ans_138x276_smooth_best_mse_mae/pos_bateria/
       eval_smooth.json + eval_smooth_per_sample.npz

Execução (raiz do projeto):  python -m scripts.pos_bateria_smooth_eval [N_AMOSTRAS]
"""
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

import scripts.build_unified_ans_chunks_direct as bd
from scripts.eval_surface_integral_table import load_model, Accum, _mag, grid_area_weights
from scripts.pos_bateria_common import (
    DEVICE, RUN_ORDER, run_key, design_rows, to_torch_sample, predict, INTERP_MODE,
)
from src.data_gen.parsers.femm_mesh_unified import parse_ans_gzip_sample_unified
from src.data_gen.parsers.femm_mesh_smooth import parse_ans_gzip_sample_smooth
from src.neural_op.archs.interp import interpolate_grid_to_nodes

ROOT_S = Path('data/logs/mesh_ans_138x276_smooth_best_mse_mae')
ROOT_U = Path('data/logs/mesh_ans_138x276_unified_best_mse_mae')
OUT_DIR = ROOT_S / 'pos_bateria'
TMP_DIR = Path('data/temp') / 'pos_bateria_smooth_parse'
N_WORKERS = 12
MAX_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else None
SUBSETS = ('teste37', 'fora51')


def find_runs(root):
    """{(arch, loss): run_dir} -- 1 run por par (falha se houver repetição)."""
    out = {}
    for arch, loss in RUN_ORDER:
        hits = [d for d in sorted((root / arch).glob('run_*'))
                if (d / 'checkpoints' / 'best.pth').exists()
                and json.loads((d / 'config.json').read_text())['loss'] == loss]
        assert len(hits) == 1, f'{root}/{arch} loss={loss}: {hits}'
        out[(arch, loss)] = hits[0]
    return out


def load_set(root):
    out = {}
    for (arch, loss), d in find_runs(root).items():
        model, normalizer, cfg, epoch = load_model(d)
        out[run_key(arch, loss)] = dict(arch=arch, loss=loss, run=d.name, model=model,
                                         normalizer=normalizer, epoch=epoch,
                                         interp_mode=getattr(model, 'interp_mode', None))
    return out


def samples_with_subset():
    split_s = json.loads((ROOT_S / 'FNO2d' / 'run_0001' / 'split.json').read_text())
    split_u = json.loads((ROOT_U / 'FNO2d' / 'run_0001' / 'split.json').read_text())
    assert set(split_s['train']) == set(split_u['train']), 'treino difere entre baterias'
    test_s = set(split_s['test'])
    assert test_s <= set(split_u['test'])
    ans = sorted(bd.RAW_DIR.glob('sample_*.ans.gz'), key=bd._sample_idx)
    out = []
    for name in split_u['test']:
        ci = int(name.split('_')[-1].split('.')[0])
        sub = 'teste37' if name in test_s else 'fora51'
        for p in ans[ci * bd.CHUNK_SIZE:(ci + 1) * bd.CHUNK_SIZE]:
            out.append((name, sub, bd._sample_idx(p), p))
    return out


def parse_both(path, row):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    kw = dict(ang_1=bd.ANG_1, ang_2=bd.ANG_2, n_r=bd.N_R, n_a=bd.N_A, tmp_dir=TMP_DIR)
    u = parse_ans_gzip_sample_unified(path, r_in, r_ext, **kw)
    s = parse_ans_gzip_sample_smooth(path, r_in, r_ext, **kw)
    return dict(U=dict(v1=u['FNO_GNN'], bip=u['FNO_BipartiteGNN']),
                S=dict(v1=s['FNO_GNN'], bip=s['FNO_BipartiteGNN']))


def node_err(mag_p, mag_t, area):
    e = np.abs(mag_p - mag_t).astype(np.float64)
    a = area.astype(np.float64)
    return (float((e * a).sum() / a.sum()), float(math.sqrt((e ** 2 * a).sum() / a.sum())),
            float(e.mean()))


def merge(accs):
    m = Accum()
    for a in accs:
        for f in ('num_l1', 'num_l2', 'den_area', 'num_ref', 'sum_abs', 'n_pts'):
            setattr(m, f, getattr(m, f) + getattr(a, f))
    return m


def floor_nodes(ts):
    yh = ts['v1']['y_hw'][None].to(DEVICE)
    nx = ts['v1']['node_x'].to(DEVICE)
    return interpolate_grid_to_nodes(yh, nx[:, 3], nx[:, 4], ts['v1']['L'].to(DEVICE),
                                     mode=INTERP_MODE).cpu()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = design_rows()
    samples = samples_with_subset()
    if MAX_SAMPLES is not None:   # teste rápido: metade de cada subset
        samples = ([x for x in samples if x[1] == 'teste37'][:MAX_SAMPLES // 2]
                   + [x for x in samples if x[1] == 'fora51'][:MAX_SAMPLES - MAX_SAMPLES // 2])
    sets = {'S': load_set(ROOT_S), 'U': load_set(ROOT_U)}
    for tag, runs in sets.items():
        for k, r in runs.items():
            print(f"  {tag} {k:24s} {r['run']} best ép.{r['epoch']} interp={r['interp_mode']}")

    # chaves de avaliação: (conjunto de modelos, gabarito)
    evals = [('S', 'S'), ('U', 'U'), ('U', 'S')]
    ekey = lambda m, g: m if m == g else f'{m}x{g}'
    acc = {(ekey(m, g), k, sub, kind): Accum() for m, g in evals for k in sets[m]
           for sub in SUBSETS for kind in ('mesh', 'grid')}
    acc_floor = {(g, sub): Accum() for g in ('S', 'U') for sub in SUBSETS}
    per = {}          # (ekey, run_key) -> list[(L1, L2, pt)]
    per_floor = {g: [] for g in ('S', 'U')}
    ids, subs, n_nodes, gt_diff = [], [], [], []

    groups = {}
    for name, sub, idx, p in samples:
        groups.setdefault(name, []).append((sub, idx, p))

    t0 = time.time()
    done = 0
    for gi, (name, items) in enumerate(groups.items()):
        t = time.time()
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(items))) as ex:
            futs = [ex.submit(parse_both, p, rows[idx]) for _, idx, p in items]
            for (sub, idx, p), f in zip(items, futs):
                lay = f.result()
                ts = {g: to_torch_sample(lay[g]) for g in ('S', 'U')}
                area = lay['S']['v1']['node_x'][:, 2]
                assert np.array_equal(area, lay['U']['v1']['node_x'][:, 2])
                gt = {g: lay[g]['v1']['node_y'] for g in ('S', 'U')}
                assert np.array_equal(gt['S'], lay['S']['bip']['node_y'])
                mag_t = {g: np.hypot(gt[g][:, 0], gt[g][:, 1]) for g in gt}
                y_hw = {g: ts[g]['v1']['y_hw'] for g in gt}
                H, W = y_hw['S'].shape[-2:]
                agrid = grid_area_weights(H, W)
                ids.append(idx)
                subs.append(sub)
                n_nodes.append(len(area))
                gt_diff.append(node_err(mag_t['U'], mag_t['S'], area))

                for g in ('S', 'U'):
                    mf = _mag(floor_nodes(ts[g]))
                    acc_floor[(g, sub)].add(mf, mag_t[g], area)
                    per_floor[g].append(node_err(mf, mag_t[g], area))

                for mset in ('S', 'U'):
                    for k, r in sets[mset].items():
                        out_hw, yn = predict(r['arch'], r['model'], r['normalizer'], ts[mset])
                        mag_p = _mag(yn.float().cpu())
                        mag_hw = _mag(out_hw.float().cpu())
                        for g in (('S',) if mset == 'S' else ('U', 'S')):
                            ek = ekey(mset, g)
                            acc[(ek, k, sub, 'mesh')].add(mag_p, mag_t[g], area)
                            acc[(ek, k, sub, 'grid')].add(mag_hw, _mag(y_hw[g]), agrid)
                            per.setdefault((ek, k), []).append(node_err(mag_p, mag_t[g], area))
                done += 1
        print(f'  [{gi + 1}/{len(groups)}] {name} ({items[0][0]})  {time.time() - t:.0f}s  '
              f'(amostras {done}, {(time.time() - t0) / 60:.1f} min)', flush=True)

    # ------------------------------------------------------------------ #
    subs_arr = np.array(subs)

    def pstats(v):
        v = np.asarray(v, dtype=np.float64)
        return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)))

    def reports(getter):
        r = {sub: getter([sub]).report() for sub in SUBSETS if (subs_arr == sub).any()}
        r['total88'] = getter(list(SUBSETS)).report()
        return r

    res = dict(n_samples=done, n_by_subset={s: int((subs_arr == s).sum()) for s in SUBSETS},
               interp_mode=INTERP_MODE, evals={}, floor={}, gt_unified_vs_smooth={})
    for m, g in evals:
        ek = ekey(m, g)
        res['evals'][ek] = {}
        for k, r in sets[m].items():
            arr = np.array(per[(ek, k)])
            res['evals'][ek][k] = dict(
                arch=r['arch'], loss=r['loss'], run=r['run'], best_epoch=r['epoch'],
                mesh=reports(lambda ss: merge([acc[(ek, k, s, 'mesh')] for s in ss])),
                grid=reports(lambda ss: merge([acc[(ek, k, s, 'grid')] for s in ss])),
                per_sample_L1_A={sub: pstats(arr[subs_arr == sub, 0]) for sub in SUBSETS
                                 if (subs_arr == sub).any()})
    for g in ('S', 'U'):
        res['floor'][g] = reports(lambda ss: merge([acc_floor[(g, s)] for s in ss]))
    gd = np.array(gt_diff)
    res['gt_unified_vs_smooth'] = dict(per_sample_L1_A_T=pstats(gd[:, 0]),
                                       per_sample_L2_A_T=pstats(gd[:, 1]),
                                       per_sample_pt_T=pstats(gd[:, 2]))

    np.savez_compressed(
        OUT_DIR / 'eval_smooth_per_sample.npz',
        sample_idx=np.array(ids), subset=subs_arr, n_nodes=np.array(n_nodes), gt_diff=gd,
        **{f'floor_{g}': np.array(v) for g, v in per_floor.items()},
        **{f'{ek}__{k}': np.array(v) for (ek, k), v in per.items()},
    )
    (OUT_DIR / 'eval_smooth.json').write_text(json.dumps(res, indent=2))
    print(f'\nsalvo em {OUT_DIR} ({(time.time() - t0) / 60:.1f} min)')
    for ek, d in res['evals'].items():
        for k, v in d.items():
            g = v['mesh']['total88']
            print(f"  {ek:4s} {k:24s} total: pt {g['MAE_pt_pct']:5.2f}%  L1 {g['L1_area_pct']:5.2f}%"
                  f"  L2 {g['L2_area_pct']:5.2f}%")


if __name__ == '__main__':
    main()
