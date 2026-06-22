from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalCfg:
    run_dir              : str   = 'data/logs/motor_default_v2_135x270/GNN_PostBase/run_0005' # caminho para run_XXXX/ (data/logs/{problem}/{arch}/run_XXXX)
    checkpoint           : str   = 'best'              # 'best', 'latest' ou 'final'
    split                : str   = 'test'               # 'train' ou 'test' — conjunto de onde o chunk vem
    chunk_index          : Optional[int] = 1         # N-ésimo chunk (0-indexed) dentro de `split`, reproduzindo
                                                         # o mesmo shuffle/seed do treino da run; se None, usa chunk_name
    chunk_name           : str   = 'data_chunk_0000'   # sem extensão .pt; usado só se chunk_index=None; dataset lido de run_dir/config.json
    sample_idx           : int   = 0
    irrelevance_threshold: float = 0.0001
    show_qtree_overlay   : bool  = False               # overlay de refinamento (FNO_GNN)
    error_cap_enabled    : bool  = False                # capa vmax do colormap de erro em error_cap
    error_cap            : float = 20.0                # vmax do erro normalizado [%] quando error_cap_enabled=True