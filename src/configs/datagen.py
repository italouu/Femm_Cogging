from dataclasses import dataclass
from typing import Optional


@dataclass
class DatagenConfig:
    dataset: str = 'test_motor_v5_135x270'

    # Grade / Geometria
    n_r: int = 135
    n_a: int = 270
    ang_1: int = 0
    ang_2: int = 120

    # Geração de dados
    mode: str = 'qtree'
    distribution: str = 'uniform'
    sample_method: str = 'legacy'  # 'fixed_geometry' |'constrained' | 'legacy' | 'constrained_lhs'
    n_samples: int = 4000
    max_depth: int = 1
    datagen_seed: int = 12
    cascade_buffer: Optional[int] = 1
    homogeneity_threshold: float = 0.90

    # Prepare / chunks
    chunk_size: int = 32

    # gen_npz_structures
    npz_parser:             str           = 'FNO_GNN'
    npz_samples_per_worker: int           = 2
    npz_max_workers:        int           = 12
    npz_max_samples:        Optional[int] = None
