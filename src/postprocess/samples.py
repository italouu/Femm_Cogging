"""
Amostras do raw (data/raw/mesh_ans_138x276/): designs válidos, caminhos,
subsets teste/fora (split_seed=12), parse do gabarito smooth.

Origem: tests/resultados/common.py (2026-10-13), sem mudança de lógica; a run
de referência do split passa a ser argumento (antes fixa em ROOT_S/FNO2d/run_0001).

Subsets: os chunks fora do treino, reconstruídos com o mesmo split do treino
(split_chunk_paths, train_split=0,30):
  - 'teste37' : os 37 chunks de teste (parada / escolha do best.pth);
  - 'fora51'  : os 51 chunks nem de treino nem de teste (avaliação independente).
"""
import csv
import json
from pathlib import Path

from src.configs.datagen import DatagenConfig
from src.data_gen.parsers.femm_mesh_smooth import parse_ans_gzip_sample_smooth
from src.neural_op.dataloaders.grid_loader import split_chunk_paths

RAW_DIR  = Path('data/raw/mesh_ans_138x276')
TMP_ROOT = Path('data/temp') / 'postprocess'

_dg = DatagenConfig()
N_R, N_A     = _dg.n_r, _dg.n_a
ANG_1, ANG_2 = _dg.ang_1, _dg.ang_2
CHUNK_SIZE   = _dg.chunk_size

N_CHUNKS = 125            # 4000 amostras / 32
SUBSETS  = ('teste37', 'fora51')


def _sample_idx(path: Path) -> int:
    return int(path.name.split('_')[1].split('.')[0])


def design_rows():
    with open(RAW_DIR / 'valid_designs.csv', newline='') as f:
        return list(csv.DictReader(f))


def sample_path(idx):
    return RAW_DIR / f'sample_{idx:06d}.ans.gz'


def samples_with_subset(split_run_dir: Path, max_samples=None):
    """[(chunk_name, subset, sample_idx, ans_path)] dos chunks fora do treino, na
    ordem do split de `split_run_dir` (config.json + split.json de uma run da
    bateria); max_samples -> metade de cada subset."""
    cfg = json.loads((split_run_dir / 'config.json').read_text())
    split = json.loads((split_run_dir / 'split.json').read_text())
    names = [f'data_chunk_{i:04d}.pt' for i in range(N_CHUNKS)]
    train, rest = split_chunk_paths(names, cfg['train_split'], seed=cfg['split_seed'])
    assert train == split['train'] and rest[:len(split['test'])] == split['test'], \
        f'split reconstruído difere do split.json de {split_run_dir}'
    test = set(split['test'])
    ans = sorted(RAW_DIR.glob('sample_*.ans.gz'), key=_sample_idx)
    out = []
    for name in rest:
        ci = int(name.split('_')[-1].split('.')[0])
        sub = 'teste37' if name in test else 'fora51'
        for p in ans[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]:
            out.append((name, sub, _sample_idx(p), p))
    if max_samples is not None:
        out = ([x for x in out if x[1] == 'teste37'][:max_samples // 2]
               + [x for x in out if x[1] == 'fora51'][:max_samples - max_samples // 2])
    return out


def parse_smooth(path, row, tmp_dir=TMP_ROOT / 'parse'):
    """1 .ans.gz -> layouts do gabarito smooth {'v1': FNO_GNN, 'bip': FNO_BipartiteGNN}."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    s = parse_ans_gzip_sample_smooth(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
                                     n_r=N_R, n_a=N_A, tmp_dir=tmp_dir)
    return dict(v1=s['FNO_GNN'], bip=s['FNO_BipartiteGNN'])
