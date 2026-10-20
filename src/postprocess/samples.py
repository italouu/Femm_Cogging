"""
Amostras do raw (data/raw/mesh_ans_138x276/): designs válidos, caminhos,
subsets teste/fora (split_seed=12), parse do gabarito smooth.

Origem: tests/resultados/common.py (2026-10-13), sem mudança de lógica; a run
de referência do split passa a ser argumento (antes fixa em ROOT_S/FNO2d/run_0001).

Subsets: os chunks fora do treino, reconstruídos com o mesmo split do treino
(split_chunk_paths com train_split/test_split/split_seed do config.json da run):
  - 'teste' : os chunks de teste (parada / escolha do best.pth);
  - 'fora'  : os chunks nem de treino nem de teste (avaliação independente) — vazio quando
              test_split=None (teste = complemento do treino, ex.: 70/30 da bateria de 2000).
O nº de chunks do dataset não é fixo: é o que reproduz o split.json da run (2026-10-13;
antes N_CHUNKS=125 e subsets 'teste37'/'fora51', bateria de 4000 amostras com 30/30/40).

Referencial (2026-10-17): o parse smooth entrega node_y/y_hw POLARES (Br, Bθ). Runs
treinadas antes disso (alvo cartesiano Bx, By) só são comparáveis por |B|; por
componente, girar a saída delas pelo ângulo do ponto (ans_parsing._to_polar).
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

from src.configs.datagen import DatagenConfig
from src.data_gen.parsers.ans_b_smoothing import element_b, load_ans, nodal_b
from src.data_gen.parsers.ans_parsing import _node_material_priority, _parse_label_materials
from src.data_gen.parsers.femm_mesh_elem_smooth import parse_ans_gzip_sample_elem_smooth
from src.data_gen.parsers.femm_mesh_smooth import parse_ans_gzip_sample_smooth
from src.data_gen.parsers.femm_mesh_unified import _read_ans_gz
from src.neural_op.dataloaders.grid_loader import split_chunk_paths
from src.postprocess.metrics import grid_mesh_overlay, mesh_surface_quadrature

_dg = DatagenConfig()

# [REMOVIDO 2026-10-14] RAW_DIR  = Path('data/raw/mesh_ans_138x276') — agora DatagenConfig.dataset
RAW_DIR  = Path('data/raw') / _dg.dataset
TMP_ROOT = Path('data/temp') / 'postprocess'
N_R, N_A     = _dg.n_r, _dg.n_a
ANG_1, ANG_2 = _dg.ang_1, _dg.ang_2
CHUNK_SIZE   = _dg.chunk_size

# [REMOVIDO 2026-10-13] nº de chunks fixo e rótulos da bateria de 4000 amostras (30/30/40);
# agora o nº de chunks é deduzido do split.json (_n_chunks) e os rótulos são genéricos.
# N_CHUNKS = 125            # 4000 amostras / 32
# SUBSETS  = ('teste37', 'fora51')
SUBSETS  = ('teste', 'fora')


def _sample_idx(path: Path) -> int:
    return int(path.name.split('_')[1].split('.')[0])


def design_rows():
    with open(RAW_DIR / 'valid_designs.csv', newline='') as f:
        return list(csv.DictReader(f))


def sample_path(idx):
    return RAW_DIR / f'sample_{idx:06d}.ans.gz'


# [REMOVIDO 2026-10-13] versão com N_CHUNKS fixo, rótulos 'teste37'/'fora51' e
# max_samples sempre meio a meio (quebra com o subset 'fora' vazio do split 70/30):
# def samples_with_subset(split_run_dir: Path, max_samples=None):
#     """[(chunk_name, subset, sample_idx, ans_path)] dos chunks fora do treino, na
#     ordem do split de `split_run_dir` (config.json + split.json de uma run da
#     bateria); max_samples -> metade de cada subset."""
#     cfg = json.loads((split_run_dir / 'config.json').read_text())
#     split = json.loads((split_run_dir / 'split.json').read_text())
#     names = [f'data_chunk_{i:04d}.pt' for i in range(N_CHUNKS)]
#     train, rest = split_chunk_paths(names, cfg['train_split'], seed=cfg['split_seed'])
#     assert train == split['train'] and rest[:len(split['test'])] == split['test'], \
#         f'split reconstruído difere do split.json de {split_run_dir}'
#     test = set(split['test'])
#     ans = sorted(RAW_DIR.glob('sample_*.ans.gz'), key=_sample_idx)
#     out = []
#     for name in rest:
#         ci = int(name.split('_')[-1].split('.')[0])
#         sub = 'teste37' if name in test else 'fora51'
#         for p in ans[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]:
#             out.append((name, sub, _sample_idx(p), p))
#     if max_samples is not None:
#         out = ([x for x in out if x[1] == 'teste37'][:max_samples // 2]
#                + [x for x in out if x[1] == 'fora51'][:max_samples - max_samples // 2])
#     return out


def _n_chunks(cfg, split, n_max):
    """Nº de chunks do dataset da run: o n (entre treino+teste e n_max) cujo shuffle
    reproduz o split.json — o shuffle depende de n, então só o n correto bate."""
    n_min = len(split['train']) + len(split['test'])
    for n in range(n_min, n_max + 1):
        names = [f'data_chunk_{i:04d}.pt' for i in range(n)]
        train, test = split_chunk_paths(names, cfg['train_split'], seed=cfg['split_seed'],
                                        test_split=cfg.get('test_split'))
        if train == split['train'] and test == split['test']:
            return n
    raise AssertionError(f'nenhum nº de chunks em [{n_min}, {n_max}] reproduz o split.json')


def samples_with_subset(split_run_dir: Path, max_samples=None, n_samples=None):
    """[(chunk_name, subset, sample_idx, ans_path)] dos chunks fora do treino, na
    ordem do split de `split_run_dir` (config.json + split.json de uma run da
    bateria); max_samples -> dividido igualmente entre os subsets não vazios.
    n_samples: nº de amostras do raw usadas no dataset da run (as n_samples primeiras,
    como MAX_SAMPLES em scripts/build_smooth_ans_chunks_direct.py); None = todas."""
    cfg = json.loads((split_run_dir / 'config.json').read_text())
    split = json.loads((split_run_dir / 'split.json').read_text())
    ans = sorted(RAW_DIR.glob('sample_*.ans.gz'), key=_sample_idx)[:n_samples]
    n = _n_chunks(cfg, split, math.ceil(len(ans) / CHUNK_SIZE))   # já confere o split.json
    names = [f'data_chunk_{i:04d}.pt' for i in range(n)]
    _, rest = split_chunk_paths(names, cfg['train_split'], seed=cfg['split_seed'])
    test = set(split['test'])
    out = []
    for name in rest:
        ci = int(name.split('_')[-1].split('.')[0])
        sub = 'teste' if name in test else 'fora'
        for p in ans[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]:
            out.append((name, sub, _sample_idx(p), p))
    if max_samples is not None:
        present = [sub for sub in SUBSETS if any(x[1] == sub for x in out)]
        quota = [max_samples // len(present) + (i < max_samples % len(present))
                 for i in range(len(present))]
        out = [x for sub, q in zip(present, quota) for x in [y for y in out if y[1] == sub][:q]]
    return out


# [REMOVIDO 2026-10-20] sem o layout da Elem -- substituída pela versão com with_elem
# (elem_smooth_eval integrado ao eval).
# def parse_smooth(path, row, tmp_dir=TMP_ROOT / 'parse'):
#     """1 .ans.gz -> layouts do gabarito smooth {'v1': FNO_GNN, 'bip': FNO_BipartiteGNN}."""
#     tmp_dir.mkdir(parents=True, exist_ok=True)
#     r_in = float(row['inner_diameter [mm]']) / 2
#     r_ext = float(row['outer_diameter [mm]']) / 2
#     s = parse_ans_gzip_sample_smooth(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2,
#                                      n_r=N_R, n_a=N_A, tmp_dir=tmp_dir)
#     return dict(v1=s['FNO_GNN'], bip=s['FNO_BipartiteGNN'])
# [REMOVIDO 2026-10-20] assinatura sem with_overlay
# def parse_smooth(path, row, tmp_dir=TMP_ROOT / 'parse', with_elem=False):
def parse_smooth(path, row, tmp_dir=TMP_ROOT / 'parse', with_elem=False, with_overlay=False):
    """1 .ans.gz -> layouts do gabarito smooth {'v1': FNO_GNN, 'bip': FNO_BipartiteGNN}.
    with_elem: acrescenta 'elem' (layout FNO_BipartiteGNN_Elem, gabarito por elemento) e a
    malha para as projeções elemento <-> nó: 'tri' [M,3], 'elem_mat' [M], 'node_mat' [S]
    (material do nó por prioridade ferro>ímã>cobre>ar -- mesma regra do gabarito).
    Origem do with_elem: tests/resultados/elem_smooth_eval.parse_worker (2026-10-20).
    with_overlay: acrescenta 'overlay' = metrics.grid_mesh_overlay (erro grade × malha por
    interseção célula ∩ triângulo, FEMM = B suavizado P1) -- ~8 s/amostra -- e 'mesh_quad' =
    metrics.mesh_surface_quadrature (erro de superfície malha × malha) -- ~0,03 s/amostra."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    kw = dict(ang_1=ANG_1, ang_2=ANG_2, n_r=N_R, n_a=N_A, tmp_dir=tmp_dir)
    s = parse_ans_gzip_sample_smooth(path, r_in, r_ext, **kw)
    out = dict(v1=s['FNO_GNN'], bip=s['FNO_BipartiteGNN'])
    if with_elem:
        out['elem'] = parse_ans_gzip_sample_elem_smooth(path, r_in, r_ext, **kw)
        lines, nodes, elems = _read_ans_gz(path, tmp_dir)
        label_material_id, _, _ = _parse_label_materials(lines)
        out['elem_mat'] = label_material_id[elems[:, 3]]
        out['node_mat'] = _node_material_priority(elems, out['elem_mat'], nodes.shape[0])
        out['tri'] = elems[:, :3].astype(np.int64)
    if with_overlay:
        mesh = load_ans(path)
        B1, B2 = element_b(mesh)
        b1, b2 = nodal_b(mesh, B1, B2)
        out['overlay'] = grid_mesh_overlay(np.stack([mesh.x, mesh.y], axis=1), mesh.p, b1, b2,
                                           r_in, r_ext, N_R, N_A, ang_1=ANG_1, ang_2=ANG_2)
        out['mesh_quad'] = mesh_surface_quadrature(np.stack([mesh.x, mesh.y], axis=1),
                                                   mesh.p, b1, b2)
        assert np.array_equal(mesh.p, out.get('tri', mesh.p)), 'malha do grafo != malha do .ans'
    return out
