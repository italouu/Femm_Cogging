"""
Inferência de todas as runs nas amostras de teste (ou validação) direto dos chunks.

    python -m scripts.infer_chunks [--subset test|val] [--archs A ...] [--losses mse mae]

Saída: data/temp/inference/<arch>_<loss>_<run>/<tag do best.pth>/<subset>/sample_XXXXXX.npz
(ver src/postprocess/chunk_inference.py). Retomável.
"""
import argparse
import time
from pathlib import Path

from src.postprocess.chunk_inference import infer_run, pred_dir
from src.postprocess.inference import ARCHS, find_runs, find_split_runs

LOG_ROOT = Path('data/logs/BLDC_138x276')
ALL_ARCHS = ARCHS + ('FNO_SplitGNN',)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='test', choices=('test', 'val'))
    ap.add_argument('--archs', nargs='+', default=list(ALL_ARCHS), choices=ALL_ARCHS)
    ap.add_argument('--losses', nargs='+', default=['mse', 'mae'], choices=('mse', 'mae'))
    a = ap.parse_args()
    runs = {k: v for k, v in find_runs(LOG_ROOT).items()}
    runs.update({('FNO_SplitGNN', l): d for l, d in find_split_runs(LOG_ROOT).items()})
    for arch in a.archs:
        for loss in a.losses:
            d = runs[(arch, loss)]
            t = time.perf_counter()
            n_new, n_tot = infer_run(d, a.subset)
            print(f'{arch}_{loss} ({d.name}): {n_new} novas / {n_tot} -> {pred_dir(d, a.subset)} '
                  f'[{time.perf_counter() - t:.0f} s]', flush=True)


if __name__ == '__main__':
    main()
