from pathlib import Path
from src.neural_op.training_utils import save_checkpoint


class TrainingMonitor:
    """
    Gerencia checkpointing e critério de parada no heartbeat (a cada checkpoint_every épocas).

    patience conta em heartbeats, não em épocas individuais:
        patience=3, checkpoint_every=50 → para após 150 épocas sem melhora.

    Quando mgr (ModelManager) é fornecido, os caminhos de checkpoint são derivados dele
    e as métricas de cada heartbeat são registradas via mgr.log().
    """

    def __init__(self, cfg, checkpoint_path=None, mgr=None):
        self.cfg          = cfg
        self.mgr          = mgr
        self.last_epoch   = 0    # atualizado por fit() a cada época
        self.stopped_early = False

        if mgr is not None:
            self.checkpoint_path = mgr.latest_path
            self.best_path       = mgr.best_path
        else:
            self.checkpoint_path = Path(checkpoint_path)
            self.best_path       = (
                self.checkpoint_path.parent / (self.checkpoint_path.stem + '_best.pth')
            )

        # [REMOVIDO] ckpt_extra migrado para config.json via ModelManager
        # self.ckpt_extra = ckpt_extra or {}

        self._best_loss      = float('inf')
        self._patience_count = 0

    def step(self, epoch, train_losses, test_losses, model, optimizer, scheduler,
             *, lr=0.0, train_time_s=0.0, eval_time_s=0.0, samples_per_s=0.0) -> bool:
        """
        Chamado a cada heartbeat. Salva checkpoint, atualiza best, loga métricas,
        verifica early stop. Retorna True se o treino deve parar.
        """
        test_loss = test_losses[-1]
        improved  = test_loss < self._best_loss - self.cfg.early_stop_min_delta

        if improved:
            self._best_loss      = test_loss
            self._patience_count = 0
            if self.cfg.save_best:
                save_checkpoint(str(self.best_path), epoch, model, optimizer, scheduler)
                print(f"  best -> {self.best_path.name}  (test {test_loss:.4e})")
        else:
            self._patience_count += 1

        save_checkpoint(str(self.checkpoint_path), epoch, model, optimizer, scheduler)
        print(f"  ckpt -> {self.checkpoint_path.name}")

        if self.mgr is not None:
            self.mgr.log(epoch, train_losses[-1], test_losses[-1], lr,
                         train_time_s, eval_time_s, samples_per_s)

        if self.cfg.early_stop_patience is not None:
            if self._patience_count >= self.cfg.early_stop_patience:
                print(f"  early stop: {self._patience_count} heartbeats sem melhora "
                      f"(patience={self.cfg.early_stop_patience})")
                self.stopped_early = True
                return True
        return False
