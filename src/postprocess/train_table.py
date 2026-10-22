"""
Tabela de treinamento a partir dos logs das runs -- só leitura de
config.json, run_summary.json e epochs.csv; nada é recalculado (as medianas por época são
das colunas já gravadas pelo treino). Campo ausente numa run -> None, listado em 'missing'.
"""
import csv
import json
import statistics
from pathlib import Path

# run_summary.json (fallback: config.json) -- nomes como gravados pelo treino
SUMMARY_FIELDS = ('n_params_real', 'n_params_trainable_real', 'gpu_peak_mem_gib', 'wall_time_s',
                  'best_epoch', 'last_epoch', 'n_epochs_cfg', 'stop_reason', 'status')
# colunas de epochs.csv (s por época): total, treino, avaliação (val_loss), métricas B
EPOCH_COLS = ('epoch_time_s', 'train_time_s', 'eval_time_s', 'metric_time_s')
# hardware registrado na run pelo treino
HW_FIELDS = ('device',)


def _read_epochs(path: Path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _median_col(rows, col):
    v = [float(r[col]) for r in rows if r.get(col) not in (None, '')]
    return statistics.median(v) if v else None


def _sum_col(rows, col):
    v = [float(r[col]) for r in rows if r.get(col) not in (None, '')]
    return sum(v) if v else None


def run_row(run_dir: Path):
    """Linha da tabela de 1 run + lista dos campos ausentes."""
    cfg = json.loads((run_dir / 'config.json').read_text())
    sp = run_dir / 'run_summary.json'
    summ = json.loads(sp.read_text()) if sp.exists() else {}
    ep = run_dir / 'epochs.csv'
    rows = _read_epochs(ep) if ep.exists() else []
    out, missing = {}, []
    for f in SUMMARY_FIELDS:
        out[f] = summ.get(f, cfg.get(f))
    for c in EPOCH_COLS:
        out[f'median_{c}'] = _median_col(rows, c)
    out['sum_epoch_time_s'] = _sum_col(rows, 'epoch_time_s')
    out['n_epochs_csv'] = len(rows)
    out['hardware'] = {h: summ.get(h, cfg.get(h)) for h in HW_FIELDS}
    for k, v in out.items():
        if k == 'hardware':
            missing += [f'hardware.{h}' for h, x in v.items() if x is None]
        elif v is None:
            missing.append(k)
    return out, missing


def train_table(runs):
    """runs: {(arch, loss): run_dir} (inference.find_runs) -> dict para o JSON."""
    table, missing = {}, {}
    for (arch, loss), d in runs.items():
        k = f'{arch}_{loss}'
        row, miss = run_row(Path(d))
        table[k] = dict(arch=arch, loss=loss, run=Path(d).name, run_dir=str(d), **row)
        if miss:
            missing[k] = miss
    return dict(source='config.json + run_summary.json + epochs.csv (nada recalculado)',
                epoch_cols=dict(epoch_time_s='época inteira', train_time_s='treino',
                                eval_time_s='avaliação (val_loss)',
                                metric_time_s='métricas de monitoramento (2º forward)'),
                runs=table, missing=missing)
