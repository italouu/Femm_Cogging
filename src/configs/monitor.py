from dataclasses import dataclass
from typing import Optional


@dataclass
class MonitorCfg:
    checkpoint_every:     int           = 10
    # [REMOVIDO 2026-10-03] default None deixava runs estagnadas rodando até n_epochs
    # (ex: FNO2d mae unified, best na época 209, rodou até 499 sem ganho)
    # early_stop_patience:  Optional[int] = None  # None = desativado; conta heartbeats
    early_stop_patience:  Optional[int] = 10    # heartbeats sem novo best (estagnação);
                                                 # None = desativado; respeita min_epochs
    early_stop_min_delta: float         = 1e-6
    gl_threshold:         Optional[float] = 5.0  # % — generalization loss (Prechelt 1998);
                                                  # None = desativado; ativo por padrão (GL_5)
    gl_patience:          int           = 3      # heartbeats CONSECUTIVOS com GL > gl_threshold
                                                  # antes de parar (piora sustentada, não um
                                                  # heartbeat ruidoso/pico isolado); 1 = antigo
    min_epochs:           int           = 100    # nenhum critério de parada age antes disso
                                                  # (ruído do início do treino); 0 = desativado
    log_grad_norm:        bool          = False  # TODO: não implementado
    save_best:            bool          = True
    # B4 (2026-10-03) — calcula mae_hw/mae_graph (test set, unidade física) em
    # TODA época, gravados em epochs.csv; False = só no heartbeat (antigo).
    # Custo: uma passada forward extra sobre o test set por época.
    metrics_every_epoch:  bool          = False
