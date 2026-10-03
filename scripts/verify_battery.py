"""
verify_battery.py — Fase 2 (verificação) da bateria definitiva, 2026-10-03.

    python -m scripts.verify_battery v1            # teste unitário da interpolação (B1)
    python -m scripts.verify_battery v2            # escala do FNO@nós após B2
    python -m scripts.verify_battery v3            # smoke test 4 archs × losses da bateria
    python -m scripts.verify_battery all

Resultados acumulados em docs/bateria_definitiva/verify_results.json.

V2/V3 usam os chunks reais de data/torch/data_chunks/mesh_ans_138x276_unified/
se existirem; senão remontam o necessário a partir do raw
(data/raw/mesh_ans_138x276/) — mesmo código de build_unified_ans_chunks_direct.

V3 (opções): --n-chunks K (chunks no dataset temporário, default 4 → com
train_split=0.30, 1 chunk de treino e 3 de teste), --epochs N (default 3),
--keep (não apaga as pastas temporárias), --only ARCH[,ARCH] (subconjunto).
Hiperparâmetros = os da bateria (scripts/run_best_configs.py); só mudam
n_epochs, checkpoint_every=1 (pra existir best.pth em poucas épocas — o
GNN_PostBase precisa dele pra achar a base) e a pasta de logs/dataset.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

OUT_JSON   = Path('docs/bateria_definitiva/verify_results.json')
UNIFIED    = 'mesh_ans_138x276_unified'
CHUNKS_DIR = Path('data/torch/data_chunks') / UNIFIED
ARCHS      = ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN')


def _save(key, value):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(OUT_JSON.read_text(encoding='utf-8')) if OUT_JSON.exists() else {}
    d[key] = value
    OUT_JSON.write_text(json.dumps(d, indent=2, default=str), encoding='utf-8')


# =========================================================================== #
# V1 — interpolação
# =========================================================================== #
def v1():
    from src.neural_op.archs.interp import interpolate_grid_to_nodes
    H, W, ANG = 138, 276, 120.0
    g = torch.Generator().manual_seed(0)

    # 1) campo linear em (r, θ), região entre centros de pixel
    rc = (torch.arange(H, dtype=torch.float64) + 0.5) / H
    cc = (torch.arange(W, dtype=torch.float64) + 0.5) / W
    R, C = torch.meshgrid(rc, cc, indexing='ij')
    a, b, k = 0.7, -1.3, 0.25
    field = (a * R + b * C + k)[None, None]
    n = 20000
    r = 0.5 / H + torch.rand(n, dtype=torch.float64, generator=g) * (1 - 1.0 / H)
    c = 0.5 / W + torch.rand(n, dtype=torch.float64, generator=g) * (1 - 1.0 / W)
    r = torch.cat([r, torch.tensor([0.5 / H, 1 - 0.5 / H, 0.5 / H, 1 - 0.5 / H], dtype=torch.float64)])
    c = torch.cat([c, torch.tensor([0.5 / W, 0.5 / W, 1 - 0.5 / W, 1 - 0.5 / W], dtype=torch.float64)])
    exact = a * r + b * c + k
    L = torch.tensor([r.numel()])
    err = {m: (interpolate_grid_to_nodes(field, r, c, L, mode=m)[:, 0] - exact).abs().max().item()
           for m in ('cell_centered', 'legacy')}
    ok_lin = err['cell_centered'] < 1e-12

    # 2) campo periódico em θ — continuidade através de 0°/120°
    theta = cc * 2 * math.pi
    row = torch.cos(theta) + 0.3 * torch.sin(3 * theta)
    pfield = row.view(1, 1, 1, W).expand(1, 1, H, W).contiguous()
    per = {}
    for th in (0.1, 119.9, 0.0, 120.0):
        u = ((th / ANG) * W - 0.5) % W                # índice contínuo, com wrap
        t = u - (W - 1)                                # entre coluna W−1 e coluna 0
        expected = (1 - t) * row[-1].item() + t * row[0].item()
        got = interpolate_grid_to_nodes(pfield, torch.tensor([0.5], dtype=torch.float64),
                                        torch.tensor([th / ANG], dtype=torch.float64),
                                        torch.tensor([1]), mode='cell_centered')[0, 0].item()
        per[th] = dict(got=got, expected=expected, abs_err=abs(got - expected))
    ok_per = all(v['abs_err'] < 1e-12 for v in per.values()) and \
        abs(per[0.0]['got'] - per[120.0]['got']) < 1e-12

    print(f"[V1.1] campo linear — erro máx: cell_centered={err['cell_centered']:.3e}  "
          f"legacy={err['legacy']:.3e}  -> {'OK' if ok_lin else 'FALHOU'}")
    print("[V1.2] campo periódico (cell_centered) — mistura das colunas 0 e W−1:")
    for th, v in per.items():
        print(f"        θ={th:6.1f}°  obtido={v['got']:+.12f}  esperado={v['expected']:+.12f}")
    print(f"        -> {'OK' if ok_per else 'FALHOU'}")
    res = dict(ok=ok_lin and ok_per, linear_max_err=err, periodic=per)
    _save('V1', res)
    return res['ok']


# =========================================================================== #
# V2 — escala do FNO@nós (B2)
# =========================================================================== #
def _normalizer_for(arch):
    """Stats reais do dataset unificado: Normalizer.fit se houver chunks; senão
    o norm_stats gravado no config.json de uma run (antiga ou nova) desse arch."""
    from src.neural_op.normalization import Normalizer
    if (CHUNKS_DIR / arch).exists() and any((CHUNKS_DIR / arch).glob('data_chunk_*.pt')):
        return Normalizer.fit(f'{UNIFIED}/{arch}', arch), f'Normalizer.fit({UNIFIED}/{arch})'
    for cfg_path in sorted(Path('data/logs').glob(f'{UNIFIED}_best_mse_mae*/{arch}/run_*/config.json')):
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        if cfg.get('dataset') == f'{UNIFIED}/{arch}' and cfg.get('norm_stats'):
            return Normalizer.from_dict(cfg['norm_stats']), str(cfg_path)
    raise FileNotFoundError(f'sem chunks nem config.json com norm_stats para {arch}')


def _test_chunk_name():
    for split in sorted(Path('data/logs').glob(f'{UNIFIED}_best_mse_mae*/*/run_*/split.json')):
        return json.loads(split.read_text())['test'][0]
    return 'data_chunk_0000.pt'


class _FixedFNO(torch.nn.Module):
    """Substitui model.fno: devolve sempre o tensor dado (o próprio y_hw codificado)."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x):
        return self.out


def _chan_stats(t):
    return dict(mean=t.mean(0).tolist(), std=t.std(0).tolist())


def v2():
    from scripts.eval_surface_integral_table import ChunkSource
    from src.neural_op.archs.fno_gnn import FNO_GNN
    from src.neural_op.archs.femm_mesh_v2_gnn import FNO_BipartiteGNN

    name = _test_chunk_name()
    ch = ChunkSource().get(name)
    common = dict(fno_modes1=4, fno_modes2=4, fno_conv_width=4, fno_conv_layers=1,
                  fno_lift_width=8, fno_lift_layers=2, fno_proj_width=8, fno_proj_layers=2,
                  data_res=(138, 276), gnn_node_width=8, gnn_n_layers=1,
                  grid_in_ch=2, grid_out_ch=2, interp_mode='cell_centered')
    from src.neural_op.archs.interp import interpolate_grid_to_nodes
    results, ok = {}, True
    for arch in ('FNO_GNN', 'FNO_BipartiteGNN'):
        nz, src = _normalizer_for(arch)
        d = ch[arch]
        # chunk inteiro (todas as amostras) — o forward do modelo itera por amostra via L
        y_hw_enc = nz.encode(d['y_hw'], 'y_hw')
        node_y_enc = nz.encode(d['node_y'], 'node_y')
        node_x = nz.encode(d['node_x'], 'node_x')
        L = d['L']
        rb, cb = ((d['node_x'][:, 3], d['node_x'][:, 4]) if arch == 'FNO_GNN'
                  else (d['node_x'][:, 0], d['node_x'][:, 1]))
        # referência física: y_hw (gabarito, Tesla) interpolado nos nós
        ref_phys = interpolate_grid_to_nodes(d['y_hw'], rb, cb, L, mode='cell_centered')
        out, phys = {}, {}
        for rescale in (False, True):
            if arch == 'FNO_GNN':
                m = FNO_GNN(**common, edge_dim=4, node_in_ch=5, fno_node_rescale=rescale)
                args = (d['x_hw'], node_x, d['edge_index'], d['edge_attr'], L)
            else:
                m = FNO_BipartiteGNN(**common, edge_dim=3, node_in_ch=2, elem_in_ch=5,
                                     cross_edge_dim=1, fno_node_rescale=rescale)
                args = (d['x_hw'], node_x, nz.encode(d['elem_x'], 'elem_x'),
                        d['edge_index'], d['edge_attr'],
                        d['cross_edge_index'], d['cross_edge_attr'], L)
            m.fno = _FixedFNO(y_hw_enc)
            m.normalizer = nz
            with torch.no_grad():
                _, fno_at_nodes, _ = m(*args, return_components=True)
            key = 'rescale_on' if rescale else 'rescale_off'
            out[key] = _chan_stats(fno_at_nodes)
            # como metric_fn/eval interpretam FNO@nós: decode com stats de node_y
            dec = nz.decode(fno_at_nodes, 'node_y')
            phys[key] = dict(**_chan_stats(dec),
                             max_abs_err_vs_ref_T=(dec - ref_phys).abs().max().item())
        tgt = _chan_stats(node_y_enc)
        phys['node_y_true'] = _chan_stats(d['node_y'])
        phys['y_hw_interp_ref'] = _chan_stats(ref_phys)
        on = out['rescale_on']
        d_mean = max(abs(a - b) for a, b in zip(on['mean'], tgt['mean']))
        r_std = max(abs(a / b - 1) for a, b in zip(on['std'], tgt['std']))
        # critério: com B2, FNO@nós decodificado como node_y == y_hw físico interpolado
        # (exato a menos de float32); sem B2, diverge (escala errada)
        err_on = phys['rescale_on']['max_abs_err_vs_ref_T']
        err_off = phys['rescale_off']['max_abs_err_vs_ref_T']
        arch_ok = err_on < 1e-4 and err_off > 1e-2
        ok &= arch_ok
        results[arch] = dict(chunk=name, n_samples=int(L.numel()), n_nodes=int(L.sum()),
                             stats_source=src, encoded=dict(node_y=tgt, **out), physical_T=phys,
                             encoded_max_abs_mean_diff=d_mean, encoded_max_rel_std_diff=r_std,
                             ok=arch_ok)
        print(f'[V2] {arch}  (chunk {name}, {int(L.numel())} amostras, {int(L.sum())} nós; stats: {src})')
        print('     espaço codificado (z-score):')
        for k, s in (('node_y', tgt), ('FNO@nós B2 off', out['rescale_off']),
                     ('FNO@nós B2 on', out['rescale_on'])):
            print(f"       {k:16s} média={[round(x, 4) for x in s['mean']]}  "
                  f"desvio={[round(x, 4) for x in s['std']]}")
        print('     unidade física (T), FNO@nós decodificado com stats de node_y:')
        for k in ('node_y_true', 'y_hw_interp_ref', 'rescale_off', 'rescale_on'):
            s = phys[k]
            e = f"  |Δ vs y_hw interp|máx={s['max_abs_err_vs_ref_T']:.2e} T" if 'max_abs_err_vs_ref_T' in s else ''
            print(f"       {k:16s} média={[round(x, 4) for x in s['mean']]}  "
                  f"desvio={[round(x, 4) for x in s['std']]}{e}")
        print(f"     -> {'OK' if arch_ok else 'FALHOU'}  (B2 on reproduz y_hw físico; "
              f"diferença de desvio restante vs node_y = suavização da grade, piso do T0a)")
    sig = {k: nz.stats[k]['std'] for k in ('y_hw', 'node_y')}
    print(f'     σ (stats FNO_BipartiteGNN): y_hw={sig["y_hw"]}  node_y={sig["node_y"]}')
    _save('V2', dict(ok=ok, **results))
    return ok


# =========================================================================== #
# V3 — smoke test
# =========================================================================== #
SMOKE_DS      = '_smoke_bateria_definitiva'                 # data/torch/data_chunks/<SMOKE_DS>/<arch>/
SMOKE_PROBLEM = '_smoke_bateria_definitiva'                 # data/logs/<SMOKE_PROBLEM>/


def _prepare_smoke_chunks(k):
    root = Path('data/torch/data_chunks') / SMOKE_DS
    names = [f'data_chunk_{i:04d}.pt' for i in range(k)]
    if all((CHUNKS_DIR / a / nm).exists() for a in ARCHS for nm in names):
        for a in ARCHS:
            (root / a).mkdir(parents=True, exist_ok=True)
            for nm in names:
                dst = root / a / nm
                if not dst.exists():
                    try:
                        os.link(CHUNKS_DIR / a / nm, dst)       # hardlink: sem cópia
                    except OSError:
                        shutil.copy2(CHUNKS_DIR / a / nm, dst)
        return root, 'hardlink/cópia dos chunks reais'
    import scripts.build_unified_ans_chunks_direct as bd
    orig = bd.CHUNKS_ROOT
    bd.CHUNKS_ROOT = root
    try:
        bd.run(max_samples=k * bd.CHUNK_SIZE)
    finally:
        bd.CHUNKS_ROOT = orig
    return root, 'remontado do raw'


def _check_run_files(run_dir: Path, arch: str, n_epochs: int):
    probs = []
    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    for k in ('n_params_real', 'n_params_trainable_real', 'git_commit'):
        if cfg.get(k) in (None, ''):
            probs.append(f'config.json sem {k}')
    if arch == 'GNN_PostBase' and 'postbase_base' not in cfg:
        probs.append('config.json sem postbase_base')
    lines = (run_dir / 'epochs.csv').read_text().strip().splitlines()
    if len(lines) - 1 != n_epochs:
        probs.append(f'epochs.csv com {len(lines) - 1} linhas (esperado {n_epochs})')
    else:
        hdr = lines[0].split(',')
        for ln in lines[1:]:
            row = dict(zip(hdr, ln.split(',')))
            if row['mae_hw'] == '' or (arch != 'FNO2d' and row['mae_graph'] == ''):
                probs.append('epochs.csv com mae vazio'); break
    summ = json.loads((run_dir / 'run_summary.json').read_text())
    for k in ('stop_reason', 'best_epoch', 'wall_time_s', 'n_params_real', 'git_commit'):
        if summ.get(k) is None:
            probs.append(f'run_summary.json sem {k}')
    if torch.cuda.is_available() and summ.get('gpu_peak_mem_bytes') is None:
        probs.append('run_summary.json sem pico de memória')
    for f in ('checkpoints/best.pth', 'model_final.pth', 'metrics.jsonl', 'split.json'):
        if not (run_dir / f).exists():
            probs.append(f'falta {f}')
    return probs, cfg, summ


def v3(n_chunks=4, n_epochs=3, keep=False, only=None):
    import scripts.run_best_configs as rbc
    from scripts.train import run
    from src.configs.monitor import MonitorCfg

    t0 = time.time()
    root, how = _prepare_smoke_chunks(n_chunks)
    print(f'[V3] dataset temporário {root} ({how}, {n_chunks} chunks/arch)', flush=True)

    log_root = Path('data/logs') / SMOKE_PROBLEM
    orig_problem = rbc.PROBLEM
    rbc.PROBLEM = SMOKE_PROBLEM            # _find_base_run_dir procura a base no smoke
    results, ok = [], True
    try:
        for spec in rbc.build_run_list(1):
            if only and spec['arch'] not in only:
                continue
            spec = dict(spec, dataset=f"{SMOKE_DS}/{spec['arch']}")
            label = f"{spec['arch']} / {spec['loss']}"
            print(f"\n{'=' * 70}\n[V3] {label}\n{'=' * 70}", flush=True)
            r = dict(arch=spec['arch'], loss=spec['loss'])
            try:
                nn_cfg = rbc.make_nn_cfg(
                    spec, problem=SMOKE_PROBLEM, n_epochs=n_epochs,
                    monitor_cfg=MonitorCfg(checkpoint_every=1, metrics_every_epoch=True))
                r['status'] = run(nn_cfg)
                run_dir = sorted((log_root / spec['arch']).glob('run_*'))[-1]
                probs, cfg, summ = _check_run_files(run_dir, spec['arch'], n_epochs)
                r.update(run_dir=run_dir.as_posix(), problems=probs,
                         stop_reason=summ.get('stop_reason'),
                         gpu_peak_mem_gib=summ.get('gpu_peak_mem_gib'),
                         n_params_real=summ.get('n_params_real'),
                         wall_time_s=summ.get('wall_time_s'))
                if spec['arch'] == 'GNN_PostBase':
                    pb = cfg.get('postbase_base', {})
                    good = (pb.get('base_arch') == 'FNO2d' and pb.get('base_loss') == spec['loss']
                            and pb.get('base_repeat', 0) == spec['repeat']
                            and SMOKE_PROBLEM in pb.get('base_run_dir', ''))
                    r['postbase_base'] = pb
                    if not good:
                        probs.append(f'base errada: {pb}')
                epochs = (run_dir / 'epochs.csv').read_text().strip().splitlines()
                r['epochs_csv_tail'] = epochs[-1]
                r['ok'] = r['status'] in ('done', 'stopped') and not probs
            except Exception as e:
                traceback.print_exc()
                r.update(status='exception', error=f'{type(e).__name__}: {e}', ok=False)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            ok &= r['ok']
            results.append(r)
    finally:
        rbc.PROBLEM = orig_problem
        if not keep:
            shutil.rmtree(log_root, ignore_errors=True)
            shutil.rmtree(root, ignore_errors=True)
            print(f'\n[V3] pastas temporárias apagadas ({log_root}, {root})')

    print(f"\n[V3] resumo ({(time.time() - t0) / 60:.1f} min)")
    for r in results:
        extra = r.get('error') or '; '.join(r.get('problems', [])) or ''
        print(f"  [{'OK ' if r['ok'] else 'ERR'}] {r['arch']:17s} {r['loss']:12s} "
              f"status={r.get('status')}  parada={r.get('stop_reason')}  "
              f"pico GPU={r.get('gpu_peak_mem_gib')} GiB  {extra}")
    _save('V3', dict(ok=ok, n_chunks=n_chunks, n_epochs=n_epochs, chunks_source=how,
                     device='cuda' if torch.cuda.is_available() else 'cpu',
                     gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                     runs=results))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', choices=('v1', 'v2', 'v3', 'all'))
    ap.add_argument('--n-chunks', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--only', default=None, help='archs separados por vírgula (V3)')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    oks = {}
    if a.which in ('v1', 'all'):
        oks['V1'] = v1()
    if a.which in ('v2', 'all'):
        oks['V2'] = v2()
    if a.which in ('v3', 'all'):
        oks['V3'] = v3(a.n_chunks, a.epochs, a.keep, only)
    print('\n' + '  '.join(f"{k}: {'OK' if v else 'FALHOU'}" for k, v in oks.items()))
    sys.exit(0 if all(oks.values()) else 1)


if __name__ == '__main__':
    main()
