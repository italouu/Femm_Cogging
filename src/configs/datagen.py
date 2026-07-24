from dataclasses import dataclass
from typing import Optional


@dataclass
class DatagenConfig:
    dataset: str = 'mesh_138x276'

    # Grade / Geometria
    n_r: int = 138
    n_a: int = 276
    ang_1: int = 0
    ang_2: int = 120

    # Geração de dados
    # 'grid'/'qtree' -> pipeline CSV (generate_data.py + gen_npz_structures.py);
    # 'femm_mesh'    -> pipeline via malha real do FEMM (generate_data_femm_mesh.py),
    #                   grafo extraído do .ans do solver em vez de quadtree Shapely
    #                   (ver src/data_gen/femm_mesh.py). Não usa check_data/generate_one_batch
    #                   (esses só aceitam 'grid'/'qtree').
    mode: str = 'femm_mesh'
    distribution: str = 'uniform'
    sample_method: str = 'legacy'  # 'fixed_geometry' |'constrained' | 'legacy' | 'constrained_lhs'
    n_samples: int = 4000
    max_depth: int = 1
    datagen_seed: int = 12
    cascade_buffer: Optional[int] = 1
    homogeneity_threshold: float = 0.90

    # Prepare / chunks
    chunk_size: int = 32

    # gen_npz_structures (só mode='grid'/'qtree')
    npz_parser:             str           = 'FNO_GNN'
    npz_samples_per_worker: int           = 2
    npz_max_workers:        int           = 12
    npz_max_samples:        Optional[int] = None

    # generate_data_femm_mesh (só mode='femm_mesh')
    femm_mesh_max_workers: int = 8
