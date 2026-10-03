"""
run_bateria_definitiva.py — bateria definitiva mesh_ans_138x276_unified, ponta a ponta
(2026-10-03). Um único comando, da raiz do projeto:

    python -m scripts.run_bateria_definitiva

Etapas, em ordem — cada uma é o script já existente, chamado como subprocesso com o
MESMO interpretador (sys.executable); para na primeira etapa que falhar:

    chunks   scripts.build_unified_ans_chunks_direct  (pula chunks já existentes)
    phase0   scripts.phase0_measurements              (só se os 8 best.pth antigos existirem)
    v1, v2   scripts.verify_battery v1 / v2
    params   scripts.count_battery_params
    v3       scripts.verify_battery v3                (smoke 4 archs × losses, 3 épocas)
    archive  scripts.archive_old_battery_logs --execute  (B0 — só se houver runs antigas)
    battery  scripts.run_best_configs                 (só se RUN_BATTERY=True)

Correções B1–B4 já ligadas em scripts/run_best_configs.py; B5/B6 pelas chaves
N_REPEATS/INCLUDE_REL_L2 no topo daquele arquivo. A bateria só começa se o smoke (V3)
passar. A saída de cada etapa também vai para
data/logs/_bateria_definitiva/<data>_<etapa>.log. Rodar de novo é seguro: chunks são pulados e o B0 não roda se a pasta
_oldcriterion já existir (as runs em data/logs/<PROBLEM>/ passam a ser da bateria nova).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Configuração ──────────────────────────────────────────────────────────────
RUN_BATTERY = True          # False = para depois do B0 (só verificações)
SKIP_STEPS  = set()         # ex: {'chunks', 'v3'} para pular etapas já feitas

PROBLEM  = 'mesh_ans_138x276_unified_best_mse_mae'
OLD_LOGS = Path('data/logs') / PROBLEM
ARCHIVED = Path('data/logs') / f'{PROBLEM}_oldcriterion'
RUNNER_LOGS = Path('data/logs') / '_bateria_definitiva'   # saída de cada etapa (<data>_<etapa>.log)


def _step(name, *args):
    if name in SKIP_STEPS:
        print(f'[skip] {name} (SKIP_STEPS)', flush=True)
        return
    cmd = [sys.executable, '-m', *args]
    print(f"\n{'=' * 80}\n[{name}] {' '.join(cmd)}\n{'=' * 80}", flush=True)
    t0 = time.time()
    # [REMOVIDO] saída só no terminal — agora também em RUNNER_LOGS/<etapa>.log
    # ret = subprocess.run(cmd).returncode
    RUNNER_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = RUNNER_LOGS / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.log"
    print(f'[{name}] log: {log_path}', flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write(' '.join(cmd) + '\n')
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        ret = proc.wait()
    if ret != 0:
        sys.exit(f'\n[{name}] FALHOU (código {ret}) — bateria interrompida')
    print(f'[{name}] ok ({(time.time() - t0) / 60:.1f} min)', flush=True)


def _n_old_best():
    return len(list(OLD_LOGS.glob('*/run_*/checkpoints/best.pth')))


def main():
    if not Path('scripts/run_best_configs.py').exists():
        sys.exit('ERRO: rode da raiz do projeto')

    _step('chunks', 'scripts.build_unified_ans_chunks_direct')

    # Fase 0 usa os checkpoints da bateria ANTIGA — precisa rodar antes do B0
    if not ARCHIVED.exists() and _n_old_best() >= 8:
        _step('phase0', 'scripts.phase0_measurements')
        phase0_done = True
    else:
        print(f'\n[phase0] PULADA — {OLD_LOGS} sem os 8 best.pth da bateria antiga '
              f'(ou já arquivada)', flush=True)
        phase0_done = False

    _step('v1', 'scripts.verify_battery', 'v1')
    _step('v2', 'scripts.verify_battery', 'v2')
    _step('params', 'scripts.count_battery_params')
    _step('v3', 'scripts.verify_battery', 'v3')

    # B0 — só na primeira vez (destino ainda não existe) e se houver runs antigas
    if not ARCHIVED.exists() and any(OLD_LOGS.glob('*/run_*')):
        args = ['scripts.archive_old_battery_logs', '--execute']
        if not phase0_done:
            args.append('--skip-phase0-check')
        _step('archive', *args)
    else:
        print('\n[archive] nada a arquivar', flush=True)

    if RUN_BATTERY:
        _step('battery', 'scripts.run_best_configs')
        print(f'\nBateria concluída — logs em {OLD_LOGS}/<arch>/run_XXXX/')
    else:
        print('\nRUN_BATTERY=False — parou antes da bateria')


if __name__ == '__main__':
    main()
