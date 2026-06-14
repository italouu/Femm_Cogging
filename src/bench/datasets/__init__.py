from dataclasses import dataclass


@dataclass
class DatasetEntry:
    in_channels  : int
    out_channels : int
    resolution   : tuple
    has_interface: bool
    chunk_dir    : str
    description  : str


DATASET_REGISTRY: dict[str, DatasetEntry] = {}


# Registros importados aqui para que `from src.bench.datasets import DATASET_REGISTRY`
# já inclua todas as entradas conhecidas.
from src.bench.datasets import darcy as _darcy  # noqa: E402, F401
