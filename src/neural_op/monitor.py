from pathlib import Path
from src.neural_op.training_utils import save_checkpoint


class TrainingMonitor:
    """
    Gerencia checkpointing e critério de parada no heartbeat (a cada checkpoint_every épocas).

    patience conta em heartbeats, não em épocas individuais:
        patience=3, checkpoint_every=50 → para após 150 épocas sem melhora.

    gl_threshold (critério ativo por padrão, Prechelt 1998): para quando o test_loss do
    heartbeat atual fica gl_threshold% (ou mais) pior que o melhor test_loss já visto —
    GL(t) = 100·(test_loss(t)/melhor_test_loss - 1). Dimensionless (independe da escala
    da loss), ao contrário de early_stop_min_delta. Os dois critérios (patience e GL)
    podem coexistir; cada um só age se seu campo não for None.

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
        self._gl_count       = 0    # heartbeats consecutivos com GL > gl_threshold

    def step(self, epoch, train_losses, test_losses, model, optimizer, scheduler,
             *, lr=0.0, train_time_s=0.0, eval_time_s=0.0, samples_per_s=0.0,
             mae_hw=None, mae_graph=None) -> bool:
        """
        Chamado a cada heartbeat. Salva checkpoint, atualiza best, loga métricas,
        verifica early stop. Retorna True se o treino deve parar.

        mae_hw/mae_graph : MAE bruto opcional (ver fit()/ArchEntry.metric_fn);
                            None quando a run não passou metric_fn a fit().
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
                         train_time_s, eval_time_s, samples_per_s,
                         mae_hw=mae_hw, mae_graph=mae_graph)

        # GL calculado sempre (mesmo antes de min_epochs), pra contagem consecutiva
        # refletir o estado real ao sair do warm-up
        if self.cfg.gl_threshold is not None:
            gl = 100.0 * (test_loss / self._best_loss - 1.0)
            self._gl_count = self._gl_count + 1 if gl > self.cfg.gl_threshold else 0

        # warm-up: nenhum critério de parada age antes de min_epochs (ruído do início)
        if epoch + 1 < getattr(self.cfg, 'min_epochs', 0):
            return False

        if self.cfg.early_stop_patience is not None:
            if self._patience_count >= self.cfg.early_stop_patience:
                print(f"  early stop: {self._patience_count} heartbeats sem melhora "
                      f"(patience={self.cfg.early_stop_patience})")
                self.stopped_early = True
                return True

        # [REMOVIDO 2026-10-02] GL parava no PRIMEIRO heartbeat acima do limiar — um único
        # heartbeat ruidoso (ex: FNO2d mse parou na época 39, +6,8%) ou um pico de
        # instabilidade da otimização (train_loss também explode e depois recupera) bastava
        # pra encerrar o treino. Substituído por GL sustentado (gl_patience heartbeats
        # consecutivos) + warm-up min_epochs, logo acima/abaixo.
        # if self.cfg.gl_threshold is not None:
        #     gl = 100.0 * (test_loss / self._best_loss - 1.0)
        #     if gl > self.cfg.gl_threshold:
        #         print(f"  early stop: generalization loss {gl:.2f}% > {self.cfg.gl_threshold}% "
        #               f"(best test={self._best_loss:.4e}, atual={test_loss:.4e})")
        #         self.stopped_early = True
        #         return True
        if self.cfg.gl_threshold is not None:
            gl_patience = getattr(self.cfg, 'gl_patience', 1)
            if self._gl_count >= gl_patience:
                print(f"  early stop: generalization loss {gl:.2f}% > {self.cfg.gl_threshold}% "
                      f"por {self._gl_count} heartbeats consecutivos "
                      f"(best test={self._best_loss:.4e}, atual={test_loss:.4e})")
                self.stopped_early = True
                return True

        return False
