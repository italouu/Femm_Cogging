from dataclasses import dataclass


@dataclass
class BenchCfg:
    run_dir              : str   = 'data/logs/motor_default_v2_135x270/GNN_PostBase/run_0005' # caminho para run_XXXX/ (data/logs/{problem}/{arch}/run_XXXX)
    checkpoint           : str   = 'best'              # 'best', 'latest' ou 'final'
    irrelevance_threshold: float = 0.0001               # |y|>=thr define a região relevante usada nas métricas
                                                         # avalia sempre TODAS as chunks (treino+teste) do dataset da run
