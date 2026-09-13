from dataclasses import dataclass
from typing import Optional


@dataclass
class MonitorCfg:
    checkpoint_every:     int           = 10
    early_stop_patience:  Optional[int] = None  # None = desativado; conta heartbeats
    early_stop_min_delta: float         = 1e-6
    gl_threshold:         Optional[float] = 5.0  # % — generalization loss (Prechelt 1998);
                                                  # None = desativado; ativo por padrão (GL_5)
    log_grad_norm:        bool          = False  # TODO: não implementado
    save_best:            bool          = True
