from dataclasses import dataclass
from typing import Optional


@dataclass
class DatagenConfig:
    dataset: str = 'fixed_geometry_v3_135x270'

    # Grade / Geometria
    n_r: int = 135
    n_a: int = 270
    ang_1: int = 0
    ang_2: int = 120

    # Geração de dados
    mode: str = 'qtree'
    distribution: str = 'uniform'
    sample_method: str = 'fixed_geometry'  # 'fixed_geometry' |'constrained' | 'legacy'
    n_samples: int = 1
    max_depth: int = 1
    datagen_seed: int = 12
    cascade_buffer: Optional[int] = None
    homogeneity_threshold: float = 0.90

    # Prepare / chunks
    chunk_size: int = 32

    # gen_npz_structures
    npz_parser:             str           = 'FNO_GNN'
    npz_samples_per_worker: int           = 2
    npz_max_workers:        int           = 12
    npz_max_samples:        Optional[int] = None
