import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from src.neural_op.training_utils import count_params


class ModelManager:
    """
    Gerencia o ciclo de vida de uma run de treino.

    Cria data/logs/{problem}/{arch}/run_XXXX/ com:
      config.json   — hiperparâmetros + metadados (autossuficiente para reconstrução)
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
        self._metrics_path  = self.run_dir / 'metrics.jsonl'
        self._status_path   = self.run_dir / 'status.txt'
        self._notes_path    = self.run_dir / 'notes.txt'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._notes_path.write_text('')

    def open(self, model, device: str, resumed_from=None):
        """Escreve config.json e status=running. Deve ser chamado antes de fit()."""
        cfg_dict               = asdict(self.cfg)
        cfg_dict['n_params']   = count_params(model)
        cfg_dict['device']     = str(device)
        cfg_dict['start_time'] = datetime.now().isoformat(timespec='seconds')
        if resumed_from is not None:
            cfg_dict['resumed_from'] = str(resumed_from)
        self._config_path.write_text(json.dumps(cfg_dict, indent=2))
        self._status_path.write_text('running')
        suffix = f"  (retomado de {Path(resumed_from).name})" if resumed_from else ""
        print(f"  run -> {self.run_dir}{suffix}")

    def log(self, epoch, train_loss, test_loss, lr,
            train_time_s, eval_time_s, samples_per_s):
        """Append de uma linha em metrics.jsonl. Chamado pelo TrainingMonitor no heartbeat."""
        entry = {
            'epoch':         epoch,
            'train_loss':    train_loss,
            'test_loss':     test_loss,
            'lr':            lr,
            'train_time_s':  round(train_time_s, 3),
            'eval_time_s':   round(eval_time_s, 3),
            'samples_per_s': round(samples_per_s, 1),
        }
        with self._metrics_path.open('a') as f:
            f.write(json.dumps(entry) + '\n')

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

    def close(self, status: str, epoch, model, optimizer, scheduler):
        """Salva model_final.pth e atualiza status.txt. Deve ser chamado no finally do script."""
        sd = {k: v for k, v in model.state_dict().items() if k != '_metadata'}
        torch.save({
            'model_state_dict':     sd,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'epoch':                epoch,
        }, self.final_path)
        self._status_path.write_text(status)
        print(f"  run {self.run_dir.name} -> {status}  ({self.final_path.name})")
