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

    # npz_parser: chave em PARSER_REGISTRY. Em ambos os modos, consumido por
    # gen_npz_structures.py (mesmo comando `python -m scripts.gen_npz_structures`
    # pros três modos):
    #   mode='grid'/'qtree'  -> filtra colunas na criação do .npz a partir das
    #                           CSVs de data/raw/<dataset>/
    #   mode='femm_mesh'     -> branch próprio em gen_npz_structures.py, filtra
    #                           colunas do staging bruto já gerado em
    #                           data/raw/<dataset>/ (sample_*.npz) — usar
    #                           'FEMM_MESH' ou variante futura
    npz_parser:             str           = 'FEMM_MESH'
    npz_samples_per_worker: int           = 2
    npz_max_workers:        int           = 12
    npz_max_samples:        Optional[int] = None

    # generate_data_femm_mesh (só mode='femm_mesh')
    femm_mesh_max_workers: int = 8

    @property
    def parsed_dataset_name(self) -> str:
        """Nome combinado dataset+parser (mode='femm_mesh') — usado como
        subdiretório em data/temp/samples_mesh_parsed/ e
        data/torch/data_chunks/, pra não misturar chunks de parsers
        diferentes (o layout de node_x/edge_attr muda conforme o parser)."""
        return f"{self.dataset}_{self.npz_parser}"
