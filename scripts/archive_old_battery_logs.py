"""
archive_old_battery_logs.py — B0 da bateria definitiva (2026-10-03).

Move data/logs/mesh_ans_138x276_unified_best_mse_mae/ (bateria anterior,
critério de parada antigo) para data/logs/mesh_ans_138x276_unified_best_mse_mae_oldcriterion/,
pra que a bateria nova comece com a pasta vazia e o _find_base_run_dir de
scripts/run_best_configs.py nunca encontre um FNO2d antigo como base do
GNN_PostBase (o _find_base_run_dir também já rejeita bases com correções
diferentes — este arquivamento é a segunda trava).

SÓ DEPOIS DA FASE 0 (que usa esses checkpoints): por padrão exige
docs/bateria_definitiva/phase0_results.json gerado sobre o test set INTEIRO.

    python -m scripts.archive_old_battery_logs              # simulação (não move nada)
    python -m scripts.archive_old_battery_logs --execute    # move de fato

Nada é apagado — só renomeado (mesmo disco, operação atômica). Data/logs/* é
gitignored, então não há efeito no git. Observação: os config.json dos
GNN_PostBase antigos apontam base_run_dir para o caminho antigo do FNO2d; eles
continuam avaliáveis (fallback de _load_frozen_base: snapshot base_arch_cfg +
pesos da base embutidos no checkpoint do próprio GNN_PostBase).
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path('data/logs/mesh_ans_138x276_unified_best_mse_mae')
DST = Path('data/logs/mesh_ans_138x276_unified_best_mse_mae_oldcriterion')
PHASE0 = Path('docs/bateria_definitiva/phase0_results.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true', help='move de fato (sem isso: simulação)')
    ap.add_argument('--skip-phase0-check', action='store_true')
    args = ap.parse_args()

    if not SRC.exists():
        sys.exit(f'nada a fazer: {SRC} não existe')
    if DST.exists():
        sys.exit(f'ERRO: destino {DST} já existe — não sobrescrevo')

    if not args.skip_phase0_check:
        if not PHASE0.exists():
            sys.exit(f'ERRO: {PHASE0} não existe — rode a Fase 0 antes '
                     f'(python -m scripts.phase0_measurements)')
        res = json.loads(PHASE0.read_text(encoding='utf-8'))
        test = None
        for split in SRC.glob('*/run_*/split.json'):
            test = json.loads(split.read_text())['test']
            break
        if test is not None and res.get('n_chunks') != len(test):
            sys.exit(f"ERRO: Fase 0 rodou em {res.get('n_chunks')} chunk(s), test set tem "
                     f"{len(test)} — rode a Fase 0 completa antes de arquivar")

    runs = sorted(p.relative_to(SRC).as_posix() for p in SRC.glob('*/run_*'))
    print(f'{SRC}  ->  {DST}')
    for r in runs:
        print(f'  {r}')
    if not args.execute:
        print('\n(simulação — nada foi movido; use --execute)')
        return
    SRC.rename(DST)
    print(f'\nmovido: {len(runs)} runs em {DST}')


if __name__ == '__main__':
    main()
