import csv
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from src.neural_op.training_utils import count_params, count_params_real


# B4 (2026-10-03) — colunas de epochs.csv (uma linha por época)
EPOCH_CSV_FIELDS = ('epoch', 'train_loss', 'test_loss', 'mae_hw', 'mae_graph', 'lr',
                    'epoch_time_s', 'train_time_s', 'eval_time_s', 'metric_time_s')


def _git_info():
    """Hash do commit + flag de árvore suja (arquivos rastreados modificados)."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {'git_commit': commit, 'git_dirty': dirty}
    except Exception as e:   # git ausente / fora de repositório
        return {'git_commit': None, 'git_dirty': None, 'git_error': str(e)}


def _postbase_base_info(arch_cfg):
    """GNN_PostBase: caminho exato (absoluto) da run base, checkpoint usado e
    sua época — além do snapshot base_run_dir/base_checkpoint já gravado em
    arch_cfg. None para outros archs."""
    base_run_dir = getattr(arch_cfg, 'base_run_dir', None)
    if base_run_dir is None:
        return None
    run_dir = Path(base_run_dir)
    ck = getattr(arch_cfg, 'base_checkpoint', 'best')
    ckpt_path = (run_dir / 'model_final.pth') if ck == 'final' \
        else (run_dir / 'checkpoints' / f'{ck}.pth')
    info = {
        'base_run_dir':      run_dir.as_posix(),
        'base_run_dir_abs':  run_dir.resolve().as_posix(),
        'base_checkpoint':   ck,
        'base_checkpoint_path': ckpt_path.resolve().as_posix(),
        'base_checkpoint_exists': ckpt_path.exists(),
        'base_checkpoint_epoch': getattr(arch_cfg, 'base_epoch', None),
    }
    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        base_cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        info.update(base_arch=base_cfg.get('arch'), base_loss=base_cfg.get('loss'),
                    base_repeat=base_cfg.get('repeat', 0),
                    base_git_commit=base_cfg.get('git_commit'))
    return info


class ModelManager:
    """
    Gerencia o ciclo de vida de uma run de treino.

    Cria data/logs/{problem}/{arch}/run_XXXX/ com:
      config.json   — hiperparâmetros + metadados (autossuficiente para reconstrução)
      split.json    — nomes dos chunks de treino/teste desta run (fonte de verdade;
                      evita reconstruir o split via seed sobre um dataset que pode
                      ter mudado desde o treino)
      metrics.jsonl — uma linha por heartbeat
      status.txt    — running → done / stopped / failed
      notes.txt     — vazio; edição manual pós-treino
      checkpoints/
          best.pth      — melhor test_loss
          latest.pth    — último heartbeat (para resume)
      model_final.pth   — pesos ao fim do treino
    """

    def __init__(self, cfg):
        self.cfg  = cfg
        base      = Path('data/logs') / cfg.problem / cfg.arch
        existing  = sorted(base.glob('run_????'))
        run_num   = (int(existing[-1].name[4:]) + 1) if existing else 1

        self.run_dir        = base / f'run_{run_num:04d}'
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.best_path      = self.checkpoint_dir / 'best.pth'
        self.latest_path    = self.checkpoint_dir / 'latest.pth'
        self.final_path     = self.run_dir / 'model_final.pth'
        self._config_path   = self.run_dir / 'config.json'
        self._split_path    = self.run_dir / 'split.json'
        self._metrics_path  = self.run_dir / 'metrics.jsonl'
        self._status_path   = self.run_dir / 'status.txt'
        self._notes_path    = self.run_dir / 'notes.txt'
        self._epochs_path   = self.run_dir / 'epochs.csv'        # B4
        self._summary_path  = self.run_dir / 'run_summary.json'  # B4
        self._t_open        = None
        self._device        = None
        self._n_params      = {}

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._notes_path.write_text('')

    def open(self, model, device: str, resumed_from=None, split=None):
        """
        Escreve config.json e status=running. Deve ser chamado antes de fit().

        split : dict|None
            {'train': [...paths...], 'test': [...paths...]} usados nesta run.
            Gravado em split.json (só os nomes dos arquivos, não o caminho completo,
            para não depender do dataset continuar no mesmo local) — permite que
            scripts/eval.py reproduza o split exato sem recalcular via seed.
        """
        cfg_dict               = asdict(self.cfg)
        cfg_dict['n_params']   = count_params(model)
        cfg_dict['device']     = str(device)
        cfg_dict['start_time'] = datetime.now().isoformat(timespec='seconds')
        if resumed_from is not None:
            cfg_dict['resumed_from'] = str(resumed_from)
        # B4 (2026-10-03) — parâmetros reais (complexo ×2) e treináveis, git,
        # base exata do GNN_PostBase. n_params acima mantido (convenção antiga, numel).
        self._n_params = {
            'n_params_real':           count_params_real(model),
            'n_params_trainable_real': count_params_real(model, trainable_only=True),
        }
        cfg_dict.update(self._n_params)
        cfg_dict.update(_git_info())
        postbase = _postbase_base_info(getattr(self.cfg, 'arch_cfg', None))
        if postbase is not None:
            cfg_dict['postbase_base'] = postbase
        self._config_path.write_text(json.dumps(cfg_dict, indent=2))
        self._git = {k: cfg_dict.get(k) for k in ('git_commit', 'git_dirty')}

        # B4 — epochs.csv (cabeçalho) + relógio de parede + pico de memória da GPU
        with self._epochs_path.open('w', newline='') as f:
            csv.writer(f).writerow(EPOCH_CSV_FIELDS)
        self._t_open = time.perf_counter()
        self._device = str(device)
        if self._device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if split is not None:
            split_dict = {
                'train': [Path(p).name for p in split['train']],
                'test':  [Path(p).name for p in split['test']],
            }
            self._split_path.write_text(json.dumps(split_dict, indent=2))
        self._status_path.write_text('running')
        suffix = f"  (retomado de {Path(resumed_from).name})" if resumed_from else ""
        print(f"  run -> {self.run_dir}{suffix}")

    def log(self, epoch, train_loss, test_loss, lr,
            train_time_s, eval_time_s, samples_per_s,
            mae_hw=None, mae_graph=None):
        """
        Append de uma linha em metrics.jsonl. Chamado pelo TrainingMonitor no heartbeat.

        mae_hw/mae_graph : MAE bruto (Tesla) sobre o test set, calculado só no
        heartbeat via ArchEntry.metric_fn — mae_hw compara sempre a saída em grade
        H×W do FNO; mae_graph compara a saída final em grafo/nós (None se o arch
        não produz saída em grafo, ex: FNO2d/MaskedFNO2d, ou se a run não passou
        metric_fn a fit()).
        """
        entry = {
            'epoch':         epoch,
            'train_loss':    train_loss,
            'test_loss':     test_loss,
            'lr':            lr,
            'train_time_s':  round(train_time_s, 3),
            'eval_time_s':   round(eval_time_s, 3),
            'samples_per_s': round(samples_per_s, 1),
            'mae_hw':        mae_hw,
            'mae_graph':     mae_graph,
        }
        with self._metrics_path.open('a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_epoch(self, **row):
        """B4 — uma linha por época em epochs.csv (ver fit(..., epoch_log_fn=))."""
        with self._epochs_path.open('a', newline='') as f:
            csv.writer(f).writerow(['' if row.get(k) is None else row.get(k)
                                    for k in EPOCH_CSV_FIELDS])

    @staticmethod
    def load_run(run_dir, checkpoint='latest'):
        """
        Carrega checkpoint e reconstrói prev_losses de metrics.jsonl.

        Retorna
        -------
        ckpt        : dict com model_state_dict, optimizer_state_dict,
                      scheduler_state_dict, epoch
        prev_losses : {'train': list[float], 'test': list[float]}
                      uma entrada por heartbeat registrado em metrics.jsonl
        """
        run_path  = Path(run_dir)
        ckpt_path = run_path / 'checkpoints' / f'{checkpoint}.pth'
        ckpt      = torch.load(ckpt_path, map_location='cpu')

        prev_losses  = {'train': [], 'test': []}
        metrics_path = run_path / 'metrics.jsonl'
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    prev_losses['train'].append(entry['train_loss'])
                    prev_losses['test'].append(entry['test_loss'])

        print(f"  checkpoint carregado: {ckpt_path}  (epoch {ckpt['epoch']})"
              f"  |  {len(prev_losses['train'])} heartbeats anteriores")
        return ckpt, prev_losses

    def close(self, status: str, epoch, model, optimizer, scheduler,
              stop_reason=None, best_epoch=None, best_test_loss=None, n_epochs_cfg=None):
        """Salva model_final.pth e atualiza status.txt. Deve ser chamado no finally do script.

        B4 (2026-10-03): grava também run_summary.json — pico de memória da GPU
        (max_memory_allocated desde open()), parâmetros reais/treináveis, tempo
        de parede total, época/test_loss do best, motivo da parada e hash git."""
        sd = {k: v for k, v in model.state_dict().items() if k != '_metadata'}
        torch.save({
            'model_state_dict':     sd,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'epoch':                epoch,
        }, self.final_path)
        self._status_path.write_text(status)

        peak = None
        if self._device and self._device.startswith('cuda') and torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
        summary = {
            'status':                 status,
            'stop_reason':            stop_reason,
            'last_epoch':             epoch,
            'n_epochs_cfg':           n_epochs_cfg,
            'best_epoch':             best_epoch,
            'best_test_loss':         best_test_loss,
            'wall_time_s':            (round(time.perf_counter() - self._t_open, 1)
                                       if self._t_open is not None else None),
            'gpu_peak_mem_bytes':     peak,
            'gpu_peak_mem_gib':       round(peak / 2**30, 3) if peak is not None else None,
            'device':                 self._device,
            'checkpoints': {
                'best':  self.best_path.as_posix() if self.best_path.exists() else None,
                'final': self.final_path.as_posix(),
            },
            'end_time':               datetime.now().isoformat(timespec='seconds'),
            **self._n_params,
            **getattr(self, '_git', {}),
        }
        self._summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  run {self.run_dir.name} -> {status}  ({self.final_path.name}, "
              f"parada: {stop_reason}, best epoch {best_epoch})")
