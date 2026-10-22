"""
Amostras do raw (data/raw/<DatagenConfig.dataset>/): designs válidos, caminhos,
subsets do split de uma run, parse do gabarito smooth (alvo polar Br, Bθ).

Subsets: os chunks fora do treino, reconstruídos com o split da run
(split_chunk_paths com train_split/val_split/split_seed do config.json, conferido
contra o split.json):
  - 'test' : só avaliação final, nunca visto no treino (padrão do pós-processamento);
  - 'val'  : parada antecipada e escolha do best.pth (só por opção explícita).
O nº de chunks do dataset é o que reproduz o split.json da run.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

from src.configs.datagen import DatagenConfig
from src.data_gen.parsers.ans_b_smoothing import element_b, nodal_b
from src.data_gen.parsers.ans_parsing import (_node_material_priority, _parse_label_materials,
                                              read_ans)
from src.data_gen.parsers.elem_smooth import parse_elem_smooth
from src.data_gen.parsers.smooth_layouts import parse_smooth_layouts
from src.neural_op.dataloaders.loaders import split_chunk_paths
from src.postprocess.metrics import grid_mesh_overlay, mesh_surface_quadrature

_dg = DatagenConfig()

RAW_DIR  = Path('data/raw') / _dg.dataset
N_R, N_A     = _dg.n_r, _dg.n_a
ANG_1, ANG_2 = _dg.ang_1, _dg.ang_2
CHUNK_SIZE   = _dg.chunk_size

SUBSETS  = ('val', 'test')


def _sample_idx(path: Path) -> int:
    return int(path.name.split('_')[1].split('.')[0])


def design_rows():
    with open(RAW_DIR / 'valid_designs.csv', newline='') as f:
        return list(csv.DictReader(f))


def sample_path(idx):
    return RAW_DIR / f'sample_{idx:06d}.ans.gz'


def _n_chunks(cfg, split, n_max):
    """Nº de chunks do dataset da run: o n (entre treino+validação+teste e n_max) cujo
    shuffle reproduz o split.json — o shuffle depende de n, então só o n correto bate."""
    n_min = len(split['train']) + len(split['val']) + len(split['test'])
    for n in range(n_min, n_max + 1):
        names = [f'data_chunk_{i:04d}.pt' for i in range(n)]
        try:
            parts = split_chunk_paths(names, cfg['train_split'], cfg['val_split'], cfg['split_seed'])
        except ValueError:
            continue
        if all(p == split[k] for p, k in zip(parts, ('train', 'val', 'test'))):
            return n
    raise AssertionError(f'nenhum nº de chunks em [{n_min}, {n_max}] reproduz o split.json')


def samples_with_subset(split_run_dir: Path, subset='test', max_samples=None, n_samples=None):
    """[(chunk_name, subset, sample_idx, ans_path)] dos chunks do subset ('test' ou 'val'),
    na ordem do split de `split_run_dir` (config.json + split.json de uma run da bateria);
    max_samples -> as max_samples primeiras. n_samples: nº de amostras do raw usadas no
    dataset da run (as n_samples primeiras, como MAX_SAMPLES em
    scripts/build_chunks_main.py); None = todas."""
    if subset not in SUBSETS:
        raise ValueError(f'subset {subset!r} (esperado um de {SUBSETS})')
    cfg = json.loads((split_run_dir / 'config.json').read_text())
    split = json.loads((split_run_dir / 'split.json').read_text())
    ans = sorted(RAW_DIR.glob('sample_*.ans.gz'), key=_sample_idx)[:n_samples]
    _n_chunks(cfg, split, math.ceil(len(ans) / CHUNK_SIZE))   # confere o split.json
    out = []
    for name in split[subset]:
        ci = int(name.split('_')[-1].split('.')[0])
        for p in ans[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]:
            out.append((name, subset, _sample_idx(p), p))
    return out[:max_samples]


def parse_smooth(path, row, with_elem=False, with_overlay=False, ans=None):
    """1 .ans.gz -> layouts do gabarito smooth {'vertex': FNO_GNN, 'bip': FNO_BipartiteGNN}.
    with_elem: acrescenta 'elem' (layout FNO_BipartiteGNN_Elem, gabarito por elemento) e a
    malha para as projeções elemento <-> nó: 'tri' [M,3], 'elem_mat' [M], 'node_mat' [S]
    (material do nó por prioridade ferro>ímã>cobre>ar -- mesma regra do gabarito).
    with_overlay: acrescenta 'overlay' = metrics.grid_mesh_overlay (erro grade × malha por
    interseção célula ∩ triângulo, FEMM = B suavizado P1) -- ~8 s/amostra -- e 'mesh_quad' =
    metrics.mesh_surface_quadrature (erro de superfície malha × malha) -- ~0,03 s/amostra.
    ans: `.ans` já lido (ans_parsing.read_ans); None = lê de `path` -- uma leitura só por
    amostra."""
    if ans is None:
        ans = read_ans(path)
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    kw = dict(ang_1=ANG_1, ang_2=ANG_2, n_r=N_R, n_a=N_A, ans=ans)
    s = parse_smooth_layouts(path, r_in, r_ext, **kw)
    out = dict(vertex=s['FNO_GNN'], bip=s['FNO_BipartiteGNN'])
    if with_elem:
        out['elem'] = parse_elem_smooth(path, r_in, r_ext, **kw)
        lines, nodes, elems = ans.lines, ans.nodes, ans.elems
        label_material_id, _, _ = _parse_label_materials(lines)
        out['elem_mat'] = label_material_id[elems[:, 3]]
        out['node_mat'] = _node_material_priority(elems, out['elem_mat'], nodes.shape[0])
        out['tri'] = elems[:, :3].astype(np.int64)
    if with_overlay:
        mesh = ans.mesh
        B1, B2 = element_b(mesh)
        b1, b2 = nodal_b(mesh, B1, B2)
        out['overlay'] = grid_mesh_overlay(np.stack([mesh.x, mesh.y], axis=1), mesh.p, b1, b2,
                                           r_in, r_ext, N_R, N_A, ang_1=ANG_1, ang_2=ANG_2)
        out['mesh_quad'] = mesh_surface_quadrature(np.stack([mesh.x, mesh.y], axis=1),
                                                   mesh.p, b1, b2)
        assert np.array_equal(mesh.p, out.get('tri', mesh.p)), 'malha do grafo != malha do .ans'
    return out
