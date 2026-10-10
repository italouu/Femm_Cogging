"""
run_elem_proto_smooth.py
------------------------
FNO_BipartiteGNN_Elem com gabarito B SUAVIZADO do FEMM (2026-10-10) -- versão
smooth de scripts/run_elem_proto.py: importa make_nn_cfg/precheck/LOSSES de
lá (mesmos hiperparâmetros = FNO_BipartiteGNN da bateria) e troca só:

  dataset    -> mesh_ans_138x276_smooth/FNO_BipartiteGNN_Elem
                (python -m scripts.build_elem_smooth_chunks)
  problem    -> mesh_ans_138x276_smooth_elem_proto
  test_split -> TEST_SPLIT da bateria smooth (0,30 -- mesmo treino/teste das
                runs de scripts/run_best_configs_smooth.py)

Atenção: mae_graph aqui é MAE por ELEMENTO (B suavizado no centróide), não
comparável diretamente com o mae_graph por nó da bateria smooth.

Execução (raiz do projeto; Windows ou VM Linux):
    python -m scripts.run_elem_proto_smooth
    python -m scripts.run_elem_proto_smooth --losses mae
"""
import argparse
import traceback

from scripts.run_elem_proto import ARCH, LOSSES, make_nn_cfg, precheck
from scripts.run_best_configs_smooth import TEST_SPLIT
from scripts.build_smooth_ans_chunks_direct import OUT_DATASET as SMOOTH_ROOT
from scripts.train import run

DATASET = f'{SMOOTH_ROOT}/{ARCH}'
PROBLEM = f'{SMOOTH_ROOT}_elem_proto'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--losses', nargs='+', default=LOSSES)
    ap.add_argument('--no-precheck', action='store_true')
    a = ap.parse_args()
    if not a.no_precheck and not precheck(dataset=DATASET):
        print("\nPrecheck falhou — nenhum treino iniciado "
              "(chunks: python -m scripts.build_elem_smooth_chunks).")
        raise SystemExit(2)

    summary = []
    for loss in a.losses:
        label = f"{ARCH} / {DATASET} / loss={loss}"
        print(f"\n{'=' * 80}\n{label}\n{'=' * 80}", flush=True)
        try:
            nn = make_nn_cfg(loss, dataset=DATASET, problem=PROBLEM, test_split=TEST_SPLIT)
            summary.append((label, run(nn), None))
        except Exception as e:
            traceback.print_exc()
            summary.append((label, 'failed', str(e)))

    print(f"\n{'=' * 80}\nResumo\n{'=' * 80}")
    for label, status, err in summary:
        print(f"  [{status:8s}] {label}" + (f"  -- {err}" if err else ""))
    raise SystemExit(0 if all(s in ('done', 'stopped') for _, s, _ in summary) else 1)
