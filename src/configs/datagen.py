from dataclasses import dataclass


@dataclass
class DatagenConfig:
    # raw com correntes de fase (current_peak abaixo), 6000 amostras — 2026-10-14
    dataset: str = 'BLDC_138x276'

    # Grade / Geometria
    n_r: int = 138
    n_a: int = 276
    ang_1: int = 0
    ang_2: int = 120

    # Geração do raw (scripts/generate_data_femm_mesh_v2.py): desenha + malha + mi_analyze()
    # e salva o .ans inteiro em data/raw/<dataset>/sample_XXXXXX.ans.gz. Geometrias sorteadas
    # por BLDC_Process.generate_samples.
    distribution: str = 'uniform'
    n_samples: int = 6000
    datagen_seed: int = 12
    # Correntes de fase (2026-10-14): pico [A] por fase do sorteio equilibrado
    # (I ~ U[0, current_peak], θe ~ U[0, 360°) — BLDC_Process.add_phase_currents).
    # 0 = sem correntes (raw sem as colunas current_* -> 0 A, como antes).
    current_peak: float = 10.0

    # Prepare / chunks
    chunk_size: int = 32
    # workers do parse raw -> chunks (scripts/build_smooth_ans_chunks_direct.py,
    # scripts/build_elem_smooth_chunks.py, scripts/build_smooth_datasets.py; até 2026-10-22
    # npz_max_workers)
    chunk_max_workers: int = 6

    # workers da geração do raw (scripts/generate_data_femm_mesh_v2.py)
    femm_mesh_max_workers: int = 8
