"""
run_postprocess.py
------------------
Script de partida do pós-processamento com uso de memória controlado. Só orquestra: chama
as funções que já existem em scripts/postprocess.py e src/postprocess/ (nenhum cálculo,
métrica, máscara de Ω, região ou modelo é definido aqui).

Etapas (--steps): por ora só 'regions' (erro de superfície por faixas de distância,
docs/claude/avaliacao.md §2). As demais (eval, torque, saturation, timing) entram em STEPS
chamando os run_* existentes do postprocess.py.

Execução do regions em lotes (--chunk amostras):
  1. workers (--workers, max_tasks_per_child = --tasks-per-child) calculam o layout, Ω, Γ,
     𝒱_c e as distâncias de cada amostra (postprocess._regions_sample); o lote seguinte já é
     submetido enquanto o processo principal cuida do lote atual (no máximo 2 lotes na RAM);
  2. processo principal: pisos, depois cada run do artigo, UM MODELO DE CADA VEZ na GPU sobre
     as amostras do lote (só a predição de 1 run × 1 amostra existe a cada momento),
     acumulada em BandAccum (somas; a ordem das amostras em cada acumulador é a do subset,
     a mesma do `postprocess regions`);
  3. fim do lote: del + gc.collect + torch.cuda.empty_cache, checkpoint dos acumuladores
     (<out>/dados/regions_checkpoint.pkl, gravação atômica), log do lote (tempo, estimativa
     do restante, pico de RAM do principal + workers, pico de VRAM).
  --resume continua do último lote salvo (mesma configuração e mesmas amostras, conferidas);
  o JSON final vem sempre das somas acumuladas. --max-ram-gb: se a RAM (principal + workers)
  passar do limite, o lote atual termina, o checkpoint é gravado e o script para.

Amostras: --n N (padrão 10) = as N primeiras do subset na ordem do split (determinístico),
  --all = todo o subset. Os índices vão no JSON.
Saída: --out; padrão resultados/BLDC_138x276/placeholder/ quando N < subset (JSON com
  placeholder: true), resultados/BLDC_138x276/ com --all. dados/regions.json,
  dados/regions_curves.npz, dados/run_postprocess.log, plots/regions_curvas.*,
  plots/regions_map_sample_*.png.

Execução (raiz do projeto):
  python -m scripts.run_postprocess --steps regions --n 10 --workers 2
  python -m scripts.run_postprocess --steps regions --all --workers 3 [--max-ram-gb 26]
  python -m scripts.run_postprocess --steps regions --all --resume
"""
import argparse
import gc
import json
import os
import pickle
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch

import scripts.postprocess as pp
from src.postprocess.inference import find_runs
from src.postprocess.samples import SUBSETS, design_rows, samples_with_subset

CKPT_VERSION = 1


# ---------------------------------------------------------------------------
# memória
# ---------------------------------------------------------------------------
class MemMonitor:
    """Amostra a RAM (RSS do processo principal e soma dos filhos = workers) a cada `dt` s
    numa thread; pico desde o último reset."""

    def __init__(self, dt=0.25):
        self.p = psutil.Process()
        self.dt = dt
        self.lock = threading.Lock()
        self.reset()
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def now(self):
        main = self.p.memory_info().rss
        work = 0
        for c in self.p.children(recursive=True):
            try:
                work += c.memory_info().rss
            except psutil.Error:
                pass
        return main / 2 ** 30, work / 2 ** 30

    def _loop(self):
        while not self._stop.is_set():
            try:
                m, w = self.now()
                with self.lock:
                    self.peak_main = max(self.peak_main, m)
                    self.peak_work = max(self.peak_work, w)
                    self.peak_total = max(self.peak_total, m + w)
            except psutil.Error:
                pass
            self._stop.wait(self.dt)

    def reset(self):
        with getattr(self, 'lock', threading.Lock()):
            self.peak_main = self.peak_work = self.peak_total = 0.0

    def peaks(self):
        with self.lock:
            return dict(ram_main_gib=self.peak_main, ram_workers_gib=self.peak_work,
                        ram_total_gib=self.peak_total)

    def stop(self):
        self._stop.set()
        self.th.join()


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram_reset():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _vram_peak():
    if not torch.cuda.is_available():
        return dict(vram_alloc_gib=None, vram_reserved_gib=None)
    return dict(vram_alloc_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                vram_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def _save_ckpt(path: Path, ck):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(ck, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_ckpt(path: Path, signature):
    with open(path, 'rb') as f:
        ck = pickle.load(f)
    if ck.get('version') != CKPT_VERSION:
        raise SystemExit(f'checkpoint {path}: versão {ck.get("version")} != {CKPT_VERSION}')
    if ck['signature'] != signature:
        diff = [k for k in signature if ck['signature'].get(k) != signature[k]]
        raise SystemExit(f'checkpoint {path} é de outra configuração (difere em: {diff}); '
                         f'rode sem --resume (recomeça) ou com os mesmos argumentos')
    return ck


# ---------------------------------------------------------------------------
# etapa regions
# ---------------------------------------------------------------------------
def step_regions(a, out_root: Path, placeholder: bool, log):
    dados, plots = out_root / 'dados', out_root / 'plots'
    dados.mkdir(parents=True, exist_ok=True)
    d_list, edges, params = pp.regions_setup(a.d, a.d_max, a.n_bins, a.bin_scale, a.ang_canto,
                                             a.subset)
    rows = design_rows()
    samples = samples_with_subset(find_runs(pp.ROOT_S)[('FNO2d', 'mse')], a.subset,
                                  None if a.all else a.n, pp.N_SAMPLES)
    idx_all = [int(s[2]) for s in samples]
    batches = [samples[i:i + a.chunk] for i in range(0, len(samples), a.chunk)]
    runs = pp._regions_runs()
    for k, r in runs.items():
        log(f"  {k:28s} {r['run_dir']} best ép.{r['epoch']}  superfície {pp._regions_kind(r['arch'])}")
    signature = dict(params=params, sample_idx=idx_all, chunk=a.chunk,
                     runs={k: (r['run_dir'], r['epoch']) for k, r in runs.items()})
    ck_path = dados / 'regions_checkpoint.pkl'
    if a.resume and ck_path.exists():
        ck = _load_ckpt(ck_path, signature)
        st, b0, blog, t_prev = ck['state'], ck['next_batch'], ck['batch_log'], ck['elapsed_s']
        log(f'retomando do checkpoint: lote {b0 + 1}/{len(batches)} '
            f'({len(st["subs"])} amostras já acumuladas, {t_prev / 60:.1f} min antes)')
    else:
        if a.resume:
            log(f'--resume: {ck_path} não existe; começando do início')
        st, b0, blog, t_prev = pp.regions_state(runs, edges, d_list), 0, [], 0.0

    mon = MemMonitor()
    stop_reason = None
    t_sess = time.time()
    ex = ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=a.tasks_per_child)

    def submit(b):
        if b >= len(batches):
            return None
        return [ex.submit(pp._regions_sample, p, rows[idx], a.ang_canto, edges, d_list)
                for _, _, idx, p in batches[b]]

    try:
        futs = submit(b0)
        for b in range(b0, len(batches)):
            tb = time.time()
            mon.reset()
            _vram_reset()
            lays = [f.result() for f in futs]
            futs = submit(b + 1)                       # workers seguem no lote seguinte
            t_work = time.time() - tb
            pos = []
            for (_, sub, idx, _), lay in zip(batches[b], lays):
                pos.append(pp.regions_track(st, sub, idx, lay))
                if not st['fig_done']:
                    pp.regions_map_fig(lay, rows[idx], idx, d_list, a.ang_canto, plots)
                    st['fig_done'] = True
                pp.regions_add_floors(st, lay, pos[-1])
            for k, r in runs.items():                 # um modelo de cada vez sobre o lote
                for lay, i in zip(lays, pos):
                    pp.regions_add_run(st, k, r, lay, i)
                _free_memory()
            del lays
            _free_memory()
            mem = mon.peaks()
            vram = _vram_peak()
            elapsed = t_prev + (time.time() - t_sess)
            rec = dict(batch=b, n=len(batches[b]), idx=[int(s[2]) for s in batches[b]],
                       t_s=time.time() - tb, t_wait_workers_s=t_work, **mem, **vram)
            blog.append(rec)
            _save_ckpt(ck_path, dict(version=CKPT_VERSION, signature=signature, state=st,
                                     next_batch=b + 1, batch_log=blog, elapsed_s=elapsed))
            done = len(st['subs'])
            rate = elapsed / done
            eta = rate * (len(samples) - done)
            log(f"  lote {b + 1}/{len(batches)}: {rec['n']} amostras em {rec['t_s']:.0f} s "
                f"(espera dos workers {t_work:.0f} s) | {done}/{len(samples)} amostras, "
                f"{elapsed / 60:.1f} min, {rate:.1f} s/amostra, restante ~{eta / 60:.1f} min | "
                f"RAM pico {mem['ram_total_gib']:.1f} GiB (principal {mem['ram_main_gib']:.1f} + "
                f"workers {mem['ram_workers_gib']:.1f}) | VRAM pico "
                f"{(vram['vram_alloc_gib'] or 0):.2f} GiB alocada / "
                f"{(vram['vram_reserved_gib'] or 0):.2f} reservada")
            if b + 1 < len(batches):
                if a.max_ram_gb is not None and mem['ram_total_gib'] > a.max_ram_gb:
                    stop_reason = (f'RAM pico {mem["ram_total_gib"]:.1f} GiB > --max-ram-gb '
                                   f'{a.max_ram_gb:g} no lote {b + 1}')
                elif a.stop_after is not None and b + 1 - b0 >= a.stop_after:
                    stop_reason = f'--stop-after {a.stop_after} lote(s)'
                if stop_reason:
                    break
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        mon.stop()
    if stop_reason:
        log(f'PARADO: {stop_reason}. Checkpoint em {ck_path}; continue com --resume.')
        return None

    elapsed = t_prev + (time.time() - t_sess)
    n = len(st['subs'])
    ram = max(r['ram_total_gib'] for r in blog)
    vram = max((r['vram_alloc_gib'] or 0) for r in blog)
    timing = dict(total_s=float(elapsed), wall_s_per_sample=float(elapsed / n),
                  n_workers=a.workers, tasks_per_child=a.tasks_per_child, chunk=a.chunk)
    extra = dict(placeholder=placeholder, n_samples_requested=len(samples), sample_idx=idx_all,
                 execution=dict(script='scripts/run_postprocess.py', peak_ram_total_gib=ram,
                                peak_vram_alloc_gib=vram, batches=blog,
                                omega_area_per_sample=_omega_areas(st)))
    res = pp.regions_finalize(st, runs, params, d_list, a.subset, dados, plots, timing, extra)
    _check_regions(res, log)
    if not a.keep_checkpoint:
        ck_path.unlink(missing_ok=True)
    return res


def _omega_areas(st):
    """Área de Ω por amostra (malha − anéis excluídos) × exata ((r_ro² − r_si²)·θ/2, pois os
    anéis exatos vão de r_in a r_si e de r_ro a r_ext)."""
    out = []
    for idx, o in zip(st['idx'], st['omegas']):
        a = o['area_total_mm2'] - o['area_inner_mm2'] - o['area_outer_mm2']
        ex = (o['r_rotor_outer'] ** 2 - o['r_stator_inner'] ** 2) * pp.SEAM_RAD / 2
        out.append(dict(idx=int(idx), area_omega_mm2=float(a), area_omega_exact_mm2=float(ex),
                        rel_diff=float(a / ex - 1),
                        area_inner_rel_diff=float(o['area_inner_mm2'] / o['area_inner_exact_mm2'] - 1),
                        area_outer_rel_diff=float(o['area_outer_mm2'] / o['area_outer_exact_mm2'] - 1)))
    return out


def _check_regions(res, log):
    """Validações rápidas: soma das 3 disjuntas = total em Ω (pp), Ω_q ⊆ Ω_f, área de Ω."""
    worst = 0.0
    for v in list(res['runs'].values()) + list(res['floor_surface'].values()):
        for b in v['by_d'].values():
            parts = [b[r] for r in ('quinas', 'fronteira_sem_quinas', 'interior')]
            l1 = sum(p['L1_area_pct'] * p['area_frac'] for p in parts)
            l2 = np.sqrt(sum(p['L2_area_pct'] ** 2 * p['area_frac'] for p in parts))
            worst = max(worst, abs(l1 - v['total']['L1_area_pct']),
                        abs(l2 - v['total']['L2_area_pct']),
                        abs(sum(p['area_frac'] for p in parts) - 1) * 100,
                        abs(sum(p['L1_contrib_frac'] for p in parts) - 1) * 100)
    oa = res['execution']['omega_area_per_sample']
    rd = [o['rel_diff'] for o in oa]
    log(f"validação: |Σ disjuntas − total| máx = {worst:.1e} pp ({'ok' if worst <= 1e-6 else 'FALHOU'}); "
        f"máx(dist_f − dist_q) = {res['checks']['max_dist_f_minus_dist_q_mm']:.1e} mm "
        f"({'ok' if res['checks']['max_dist_f_minus_dist_q_mm'] <= 1e-9 else 'FALHOU'}); "
        f"fallback {res['checks']['n_points_shapely_fallback']} pontos; área de Ω × exata: "
        f"{100 * min(rd):+.3f}% a {100 * max(rd):+.3f}% (anel interno "
        f"{100 * min(o['area_inner_rel_diff'] for o in oa):+.3f}%..{100 * max(o['area_inner_rel_diff'] for o in oa):+.3f}%, "
        f"externo {100 * min(o['area_outer_rel_diff'] for o in oa):+.3f}%..{100 * max(o['area_outer_rel_diff'] for o in oa):+.3f}%)")
    res['execution']['checks'] = dict(max_abs_sum_disjoint_minus_total_pp=worst,
                                      omega_area_rel_diff=[min(rd), max(rd)])


# Etapas disponíveis: nome -> função(args, out_root, placeholder, log). As demais etapas
# (eval, torque, saturation, timing) entram aqui chamando os run_* do postprocess.py.
STEPS = {'regions': step_regions}


def main():
    ap = argparse.ArgumentParser(description='pós-processamento com memória controlada '
                                             '(orquestra scripts/postprocess.py)')
    ap.add_argument('--steps', nargs='+', required=True, choices=list(STEPS))
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--n', type=int, default=10, help='as N primeiras amostras do subset (padrão 10)')
    g.add_argument('--all', action='store_true', help='todo o subset')
    ap.add_argument('--subset', default='test', choices=SUBSETS)
    ap.add_argument('--workers', type=int, default=2, help='processos do parse/geometria (padrão 2)')
    ap.add_argument('--tasks-per-child', type=int, default=1,
                    help='amostras por processo antes de reciclá-lo (padrão 1: devolve a memória)')
    ap.add_argument('--chunk', type=int, default=8, help='amostras por lote (padrão 8)')
    ap.add_argument('--max-ram-gb', type=float, default=None,
                    help='para (com checkpoint) se o pico de RAM do lote passar disso, GiB')
    ap.add_argument('--resume', action='store_true', help='continua do último lote salvo')
    ap.add_argument('--keep-checkpoint', action='store_true',
                    help='mantém o checkpoint depois do JSON final')
    ap.add_argument('--stop-after', type=int, default=None,
                    help='(teste) para após K lotes nesta sessão, com checkpoint')
    ap.add_argument('--out', type=Path, default=None,
                    help='raiz da saída (dados/, plots/); padrão: placeholder/ se N < subset')
    ap.add_argument('--d', type=float, nargs='+', default=list(pp.REGIONS_D_MM))
    ap.add_argument('--d-max', type=float, default=pp.REGIONS_D_MAX_MM)
    ap.add_argument('--n-bins', type=int, default=pp.REGIONS_N_BINS)
    ap.add_argument('--bin-scale', default='lin', choices=('lin', 'log'))
    ap.add_argument('--ang-canto', type=float, default=30.0)
    a = ap.parse_args()

    n_subset = len(samples_with_subset(find_runs(pp.ROOT_S)[('FNO2d', 'mse')], a.subset, None,
                                       pp.N_SAMPLES))
    if not a.all and a.n >= n_subset:
        a.all = True
    placeholder = not a.all
    out_root = a.out or (pp.OUT_ROOT / 'placeholder' if placeholder else pp.OUT_ROOT)
    (out_root / 'dados').mkdir(parents=True, exist_ok=True)
    log_path = out_root / 'dados' / 'run_postprocess.log'

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {vars(a)} -> {out_root} "
        f"({'PLACEHOLDER' if placeholder else 'subset inteiro'}, {n_subset} amostras no subset)")
    t0 = time.time()
    for s in a.steps:
        STEPS[s](a, out_root, placeholder, log)
    log(f'=== fim ({(time.time() - t0) / 60:.1f} min nesta sessão)')


if __name__ == '__main__':
    main()
