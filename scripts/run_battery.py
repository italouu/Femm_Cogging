"""
run_battery.py — executa uma bateria de treinos (2026-10-13).

    python -m scripts.run_battery smooth              # src/configs/batteries/smooth.py
    python -m scripts.run_battery smooth --dry-run    # só valida e mostra a sequência
    python -m scripts.run_battery tests.minha_bateria # módulo qualquer com BATTERY

A bateria (src/configs/battery.py::BatteryCfg) é uma sequência de funções que devolvem
um NnCfg escrito por inteiro; cada uma é treinada com scripts/train.py::run, sem alteração.

Fluxo:
  1. Pré-checagem: constrói o NnCfg de toda run SEM dependência (valida tipos de
     arch_cfg/loss_cfg, dataset, dimensões, normalizador) e mostra a tabela. Qualquer erro
     aborta ANTES de treinar. Runs com `needs` são validadas na hora (a base ainda não existe).
  2. Para cada repetição, para cada run: constrói o NnCfg, grava NnCfg.repeat = repetição,
     treina, registra status e run_dir em data/logs/_batteries/<nome>/manifest.json.
  3. Retomada: rodar de novo pula as runs já concluídas (done/stopped) cuja config é
     IDÊNTICA à registrada (hash de asdict(NnCfg)). Se a definição mudou, a bateria ABORTA
     mostrando os campos diferentes — nunca mistura runs de definições diferentes.
     Runs failed/interrupted são refeitas. Ctrl+C interrompe a bateria inteira.
"""
import argparse
import dataclasses
import gc
import glob
import hashlib
import importlib
import json
import time
import traceback
from pathlib import Path

from scripts.train import run

MANIFEST_ROOT = Path('data/logs/_batteries')
OK_STATUS = ('done', 'stopped')


def load_battery(name: str):
    module = name if '.' in name else f'src.configs.batteries.{name}'
    return importlib.import_module(module).BATTERY


def _cfg_dict(nn) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(nn), default=str))


def _cfg_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _diff(old: dict, new: dict, prefix='') -> list:
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            out += _diff(a, b, f'{prefix}{k}.')
        elif a != b:
            out.append(f'{prefix}{k}: {a!r} -> {b!r}')
    return out


def _run_dirs(nn) -> set:
    return set(glob.glob(str(Path('data/logs') / nn.problem / nn.arch / 'run_*')))


def _build(r, done: dict):
    nn = r.build({n: done[n] for n in r.needs})
    if nn.repeat != 0:
        raise ValueError(f"run '{r.name}': NnCfg.repeat={nn.repeat} na definição — a repetição "
                         f"é gravada pelo runner (BatteryCfg.n_repeats), deixe no default")
    return nn


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def precheck(battery) -> bool:
    print(f"\nBateria '{battery.name}' — {len(battery.runs)} run(s) × {battery.n_repeats} repetição(ões)")
    print(f"{'run':16s} {'arch':22s} {'loss':5s} {'chunks':>6s}  dataset -> problem")
    ok = True
    for r in battery.runs:
        if r.needs:
            print(f"{r.name:16s} (depende de {', '.join(r.needs)} — validada na hora do treino)")
            continue
        try:
            nn = _build(r, {})
            n = len(glob.glob(f'data/torch/data_chunks/{nn.dataset}/data_chunk_*.pt'))
            print(f"{r.name:16s} {nn.arch:22s} {nn.loss:5s} {n:6d}  {nn.dataset} -> {nn.problem}")
        except Exception as e:
            ok = False
            print(f"{r.name:16s} ERRO: {type(e).__name__}: {e}")
    return ok


def run_battery(battery, dry_run: bool = False) -> int:
    if not precheck(battery):
        print('\nPré-checagem falhou — nenhum treino iniciado.')
        return 2
    if dry_run:
        print('\n--dry-run: nada foi treinado.')
        return 0

    mdir = MANIFEST_ROOT / battery.name
    mdir.mkdir(parents=True, exist_ok=True)
    mpath = mdir / 'manifest.json'
    manifest = (json.loads(mpath.read_text(encoding='utf-8')) if mpath.exists()
                else {'battery': battery.name, 'runs': {}})

    def save():
        mpath.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'\nManifesto: {mpath}')
    for rep in range(battery.n_repeats):
        done = {}   # nome -> run_dir concluído nesta repetição
        for r in battery.runs:
            key = f'{r.name}#rep{rep}'
            print(f"\n{'=' * 80}\n{key}\n{'=' * 80}", flush=True)
            missing = [n for n in r.needs if n not in done]
            if missing:
                print(f'  PULADA — dependência(s) não concluída(s): {missing}')
                manifest['runs'][key] = dict(status='skipped', reason=f'needs {missing}')
                save()
                continue
            try:
                nn = _build(r, done)
            except Exception as e:
                traceback.print_exc()
                manifest['runs'][key] = dict(status='failed', reason=f'build: {e}')
                save()
                continue
            nn.repeat = rep
            cfg = _cfg_dict(nn)
            h = _cfg_hash(cfg)

            prev = manifest['runs'].get(key, {})
            if prev.get('status') in OK_STATUS:
                if prev.get('hash') != h:
                    print(f"\nABORTADO: a definição de '{key}' mudou desde a run "
                          f"{prev.get('run_dir')} ({prev.get('status')}). Diferenças:")
                    for line in _diff(prev.get('config', {}), cfg):
                        print(f'  {line}')
                    print('Use outro nome de bateria/run ou remova a entrada do manifesto '
                          '(conscientemente) para treinar a definição nova.')
                    return 3
                print(f"  já concluída ({prev['status']}) em {prev['run_dir']} — pulada")
                done[r.name] = prev['run_dir']
                continue

            before = _run_dirs(nn)
            entry = dict(status='running', hash=h, arch=nn.arch, loss=nn.loss,
                         dataset=nn.dataset, problem=nn.problem, repeat=rep,
                         start=time.strftime('%Y-%m-%dT%H:%M:%S'), config=cfg)
            manifest['runs'][key] = entry
            save()
            try:
                status = run(nn)
            except Exception as e:
                traceback.print_exc()
                status, entry['reason'] = 'failed', str(e)

            new = sorted(_run_dirs(nn) - before)
            run_dir = Path(new[-1]).as_posix() if len(new) == 1 else None
            if len(new) > 1:
                entry['reason'] = f'mais de um run_dir novo: {new}'
                status = 'failed'
            interrupted = False
            if run_dir:
                summ = Path(run_dir) / 'run_summary.json'
                if summ.exists():
                    sr = json.loads(summ.read_text()).get('stop_reason')
                    interrupted = sr == 'keyboard_interrupt'
            if interrupted:
                status = 'interrupted'
            entry.update(status=status, run_dir=run_dir, end=time.strftime('%Y-%m-%dT%H:%M:%S'))
            save()
            _free_gpu()
            if interrupted:
                print('\nBateria interrompida (Ctrl+C) — rode de novo para retomar.')
                return 130
            if status in OK_STATUS:
                done[r.name] = run_dir

    print(f"\n{'=' * 80}\nResumo — {mpath}\n{'=' * 80}")
    all_ok = True
    for key, e in manifest['runs'].items():
        all_ok &= e['status'] in OK_STATUS
        print(f"  [{e['status']:11s}] {key:22s} {e.get('run_dir') or e.get('reason', '')}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('battery', help="nome em src/configs/batteries/ ou módulo com BATTERY")
    ap.add_argument('--dry-run', action='store_true', help='só valida e mostra a sequência')
    a = ap.parse_args()
    raise SystemExit(run_battery(load_battery(a.battery), dry_run=a.dry_run))
