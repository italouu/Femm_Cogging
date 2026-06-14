from dataclasses import dataclass


@dataclass
class EvalCfg:
    run_dir              : str   = 'data/logs/motor_fixed_geometry_135x270/FNO2d/run_0001' # caminho para run_XXXX/ (data/logs/{problem}/{arch}/run_XXXX)
    checkpoint           : str   = 'best'              # 'best', 'latest' ou 'final'
    chunk_name           : str   = 'data_chunk_0000'   # sem extensão .pt; dataset lido de run_dir/config.json
    sample_idx           : int   = 0
    irrelevance_threshold: float = 0.0001
    show_qtree_overlay   : bool  = False               # overlay de refinamento (FNO_GNN)
    error_cap_enabled    : bool  = False                # capa vmax do colormap de erro em error_cap
    error_cap            : float = 20.0                # vmax do erro normalizado [%] quando error_cap_enabled=True
