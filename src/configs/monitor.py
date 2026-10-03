from dataclasses import dataclass
from typing import Optional


@dataclass
class MonitorCfg:
    checkpoint_every:     int           = 10
    early_stop_patience:  Optional[int] = None  # None = desativado; conta heartbeats
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
