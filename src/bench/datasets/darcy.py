from src.bench.datasets import DATASET_REGISTRY, DatasetEntry

DATASET_REGISTRY['darcy_beta1'] = DatasetEntry(
    in_channels=1,
    out_channels=1,
    resolution=(128, 128),
    has_interface=True,
    chunk_dir='darcy_beta1',
    description='PDEBench 2D DarcyFlow beta=1.0, 128x128',
)
