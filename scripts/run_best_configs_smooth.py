"""
run_best_configs_smooth.py
--------------------------
Bateria com gabarito B SUAVIZADO do FEMM (2026-10-09, ver CLAUDE.md
"Suavização de B do FEMM a partir do `.ans`"): as mesmas 8 runs da bateria
oficial (4 archs × mse/mae), com EXATAMENTE as mesmas configurações --
importa scripts/run_best_configs.py (BEST_CONFIGS, LOSSES, N_REPEATS, B1–B6,
N_EPOCHS, critério de parada, pareamento GNN_PostBase ↔ FNO2d por loss e
repetição) em vez de copiar, então qualquer ajuste lá vale aqui também.

Troca só:
  dataset -> mesh_ans_138x276_smooth/<arch>
             (gerado por scripts/build_smooth_ans_chunks_direct.py a partir do
             MESMO raw da bateria sem smooth, data/raw/mesh_ans_138x276/ --
             mesmo chunking e split_seed, mesmo split treino/teste)
  problem -> mesh_ans_138x276_smooth_best_mse_mae
             (data/logs/<PROBLEM>/<arch>/ -- e é aqui que o GNN_PostBase procura
             o FNO2d base, nunca no da bateria sem smooth)

Execução (a partir da raiz do projeto):
    python -m scripts.run_best_configs_smooth
"""
import traceback
from pathlib import Path

import scripts.run_best_configs as rbc
from scripts.run_best_configs import build_run_list, make_nn_cfg, BEST_CONFIGS
from scripts.build_smooth_ans_chunks_direct import OUT_DATASET as SMOOTH_ROOT, RAW_DIR
from scripts.train import run

PROBLEM = f'{SMOOTH_ROOT}_best_mse_mae'          # data/logs/<PROBLEM>/<arch>/
EXPECTED_CHUNKS = 125                            # 4000 amostras / chunk_size 32


def _smooth_dataset(arch: str) -> str:
    return f'{SMOOTH_ROOT}/{arch}'


def check_chunks(expected: int = EXPECTED_CHUNKS):
    """Falha cedo se algum dataset da bateria estiver incompleto."""
    archs = sorted({c['arch'] for c in BEST_CONFIGS})
    faltando = []
    for arch in archs:
        d = Path('data/torch/data_chunks') / _smooth_dataset(arch)
        n = len(list(d.glob('data_chunk_*.pt')))
        print(f"  {d}: {n}/{expected} chunks")
        if n != expected:
            faltando.append(arch)
    if faltando:
        raise FileNotFoundError(
            f"chunks incompletos para {faltando} -- gere com "
            f"`python -m scripts.build_smooth_ans_chunks_direct` (raw: {RAW_DIR})")


def build_smooth_run_list():
    return [dict(spec, dataset=_smooth_dataset(spec['arch'])) for spec in build_run_list()]


if __name__ == '__main__':
    # _find_base_run_dir (GNN_PostBase) lê rbc.PROBLEM em tempo de execução
    rbc.PROBLEM = PROBLEM
    print(f"Bateria smooth -> data/logs/{PROBLEM}/")
    check_chunks()

    summary = []
    for spec in build_smooth_run_list():
        label = f"{spec['arch']} / {spec['dataset']} / loss={spec['loss']} / rep={spec['repeat']}"
        print(f"\n{'='*80}\n{label}\n{'='*80}", flush=True)
        try:
            status = run(make_nn_cfg(spec, problem=PROBLEM))
            summary.append((label, status, None))
        except Exception as e:
            traceback.print_exc()
            summary.append((label, 'failed', str(e)))

    print(f"\n{'='*80}\nResumo\n{'='*80}")
    for label, status, err in summary:
        line = f"  [{status:8s}] {label}"
        if err:
            line += f"  -- {err}"
        print(line)
