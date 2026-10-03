"""
eval_surface_integral_table.py
------------------------------
Tabela de erro (loss mse x mae) dos 8 runs de scripts/run_best_configs.py sobre
os datasets unificados mesh_ans_138x276_unified/<arch> -- reconstrução (versionada)
de .claude_scratch/eval_surface_integral*.py, que só sobraram em .pyc.

Mesma convenção das análises de 2026-09-15/16/21:
  - test set inteiro de cada run (split.json salvo no treino), checkpoint best.pth;
  - err = |mag_pred - mag_true|, mag = hypot(Bx,By) (diferença de MAGNITUDE);
  - "malha" (comparação principal): superfície GT na MALHA REAL do FEMM -- todos
    os archs avaliados nos vértices; FNO2d interpolado nos nós via
    _interpolate_fno_to_nodes (r_base/c_base do layout v1, pasta FNO_GNN/ -- mesma
    ordem de amostras). Peso de área: node_dual_area (v1, node_x[:,2]) ou área do
    elemento/3 somada por vértice via cross_edge_index (bipartite) -- os dois são
    a mesma área lumped;
  - "grade": saída do estágio FNO (grade H×W) contra y_hw, peso dA = r·dr·dθ;
  - L1_area = ∫|e|dA / A_total ; L2_area = sqrt(∫e² dA / A_total) ; MAE ponto a
    ponto = média simples por nó/pixel. % = relativo a B_ref = RMS de |B_true|
    ponderado por área (sqrt(∫|B|² dA / A_total)).

Fonte dos dados: chunks em data/torch/data_chunks/mesh_ans_138x276_unified/<arch>/
se existirem; senão cada chunk de teste é remontado do raw (data/raw/mesh_ans_138x276/)
com as funções de build_unified_ans_chunks_direct.py (mesmo agrupamento e mesmo
_flush), gravado em data/temp/ e apagado após a avaliação.

Execução (a partir da raiz do projeto):
    python -m scripts.eval_surface_integral_table
"""
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from src.neural_op.archs import ARCH_REGISTRY
from src.neural_op.archs.fno_gnn import _interpolate_fno_to_nodes
from src.neural_op.normalization import Normalizer

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
PROBLEM  = 'mesh_ans_138x276_unified_best_mse_mae'
DATASET  = 'mesh_ans_138x276_unified'
LOG_ROOT = Path('data/logs') / PROBLEM
CHUNKS_ROOT = Path('data/torch/data_chunks') / DATASET
TMP_CHUNKS  = Path('data/temp') / f'{DATASET}_eval_chunks'
OUT_JSON    = Path('data/logs') / PROBLEM / 'surface_integral_table.json'

# Domínio de amostragem fixo (generate_samples_constrained)
R_IN_MM, R_EXT_MM   = 28.5, 46.5
ANG1_DEG, ANG2_DEG  = 0.0, 120.0

ARCHS = ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN')
# None = test set inteiro
N_CHUNKS = None


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def load_model(run_dir: Path):
    cfg   = json.loads((run_dir / 'config.json').read_text())
    entry = ARCH_REGISTRY[cfg['arch']]
    if hasattr(entry.cfg_cls, 'from_dict'):
        arch_cfg = entry.cfg_cls.from_dict(cfg['arch_cfg'])
    else:
        arch_cfg = entry.cfg_cls(**cfg['arch_cfg'])
    model = entry.make_model(arch_cfg)
    ckpt  = torch.load(run_dir / 'checkpoints' / 'best.pth', map_location='cpu', weights_only=False)
    sd    = {k: v for k, v in ckpt['model_state_dict'].items() if k != '_metadata'}
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    normalizer = (Normalizer.from_dict(cfg['norm_stats'])
                  if cfg.get('normalize') and cfg.get('norm_stats') else None)
    model.normalizer = normalizer
    return model, normalizer, cfg, ckpt['epoch']


def discover_runs():
    runs = []
    for arch in ARCHS:
        for run_dir in sorted((LOG_ROOT / arch).glob('run_*')):
            if not (run_dir / 'checkpoints' / 'best.pth').exists():
                print(f'  [skip] {run_dir} -- sem best.pth')
                continue
            cfg = json.loads((run_dir / 'config.json').read_text())
            runs.append(dict(arch=arch, run=run_dir.name, run_dir=run_dir, loss=cfg['loss']))
    return runs


# --------------------------------------------------------------------------- #
# Chunks (existentes ou remontados do raw)
# --------------------------------------------------------------------------- #
class ChunkSource:
    """Entrega {arch_layout: chunk_dict} para 'FNO_GNN' (v1) e 'FNO_BipartiteGNN'."""
    LAYOUTS = ('FNO_GNN', 'FNO_BipartiteGNN')

    def __init__(self):
        self._pool = None

    def get(self, name: str) -> dict:
        if all((CHUNKS_ROOT / a / name).exists() for a in self.LAYOUTS):
            return {a: torch.load(CHUNKS_ROOT / a / name, map_location='cpu', weights_only=False)
                    for a in self.LAYOUTS}
        return self._rebuild(name)

    def _rebuild(self, name: str) -> dict:
        import csv
        import scripts.build_unified_ans_chunks_direct as bd
        chunk_idx = int(name.split('_')[-1].split('.')[0])
        ans_paths = sorted(bd.RAW_DIR.glob('sample_*.ans.gz'), key=bd._sample_idx)
        group = ans_paths[chunk_idx * bd.CHUNK_SIZE:(chunk_idx + 1) * bd.CHUNK_SIZE]
        with open(bd.RAW_DIR / 'valid_designs.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        bd.TMP_PARSE.mkdir(parents=True, exist_ok=True)
        # pool novo por chunk (reciclagem de worker -- mesmo motivo do builder)
        with ProcessPoolExecutor(max_workers=min(bd.MAX_WORKERS, len(group))) as ex:
            futs = [ex.submit(bd._parse_one, p,
                              float(rows[bd._sample_idx(p)]['inner_diameter [mm]']) / 2,
                              float(rows[bd._sample_idx(p)]['outer_diameter [mm]']) / 2,
                              bd.TMP_PARSE) for p in group]
            layouts = [f.result() for f in futs]

        builders = bd._builders()
        out = {}
        orig_root = bd.CHUNKS_ROOT
        bd.CHUNKS_ROOT = TMP_CHUNKS
        try:
            for a in self.LAYOUTS:
                bd._write_chunk(chunk_idx, a, [l[a] for l in layouts], builders)
                out[a] = torch.load(TMP_CHUNKS / a / name, map_location='cpu', weights_only=False)
                (TMP_CHUNKS / a / name).unlink()
        finally:
            bd.CHUNKS_ROOT = orig_root
        return out


# --------------------------------------------------------------------------- #
# Acumulador de erro
# --------------------------------------------------------------------------- #
class Accum:
    def __init__(self):
        self.num_l1 = self.num_l2 = self.den_area = self.num_ref = 0.0
        self.sum_abs = 0.0
        self.n_pts = 0

    def add(self, mag_pred, mag_true, area):
        e = np.abs(mag_pred - mag_true).astype(np.float64)
        a = np.broadcast_to(area, e.shape).astype(np.float64)
        t = mag_true.astype(np.float64)
        self.num_l1   += float((e * a).sum())
        self.num_l2   += float((e ** 2 * a).sum())
        self.num_ref  += float((t ** 2 * a).sum())
        self.den_area += float(a.sum())
        self.sum_abs  += float(e.sum())
        self.n_pts    += e.size

    def report(self):
        b_ref = math.sqrt(self.num_ref / self.den_area)
        l1 = self.num_l1 / self.den_area
        l2 = math.sqrt(self.num_l2 / self.den_area)
        mae = self.sum_abs / self.n_pts
        return dict(B_ref=b_ref, MAE_pt=mae, MAE_pt_pct=100 * mae / b_ref,
                    L1_area=l1, L1_area_pct=100 * l1 / b_ref,
                    L2_area=l2, L2_area_pct=100 * l2 / b_ref, n_pts=self.n_pts)


def grid_area_weights(H, W):
    """dA = r·dr·dθ por linha (radial); angular uniforme."""
    dr     = (R_EXT_MM - R_IN_MM) / H
    dtheta = math.radians(ANG2_DEG - ANG1_DEG) / W
    r      = R_IN_MM + (np.arange(H) + 0.5) * dr
    return np.broadcast_to((r * dr * dtheta)[:, None], (H, W))


def _mag(t):
    """|B|: nós [S,2] (Bx,By nas colunas) ou grade [2,H,W] (canais)."""
    if t.dim() == 2:
        return torch.hypot(t[:, 0], t[:, 1]).numpy()
    return torch.hypot(t[0], t[1]).numpy()


def _enc(normalizer, t, key):
    return normalizer.encode(t, key) if normalizer is not None else t


def _dec(normalizer, t, key):
    return normalizer.decode(t, key) if normalizer is not None else t


# --------------------------------------------------------------------------- #
# Avaliação por amostra
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_chunk(arch, model, normalizer, chunks, acc_mesh, acc_grid):
    v1 = chunks['FNO_GNN']
    v2 = chunks['FNO_BipartiteGNN']
    d  = v2 if arch == 'FNO_BipartiteGNN' else v1
    B  = d['x_hw'].shape[0]
    H, W = d['x_hw'].shape[-2:]
    area_grid = grid_area_weights(H, W)

    n_off = torch.cat([torch.zeros(1, dtype=torch.long), d['L'].cumsum(0)])
    e_off = torch.cat([torch.zeros(1, dtype=torch.long), d['E_L'].cumsum(0)])
    if arch == 'FNO_BipartiteGNN':
        m_off = torch.cat([torch.zeros(1, dtype=torch.long), d['elem_L'].cumsum(0)])
        c_off = torch.cat([torch.zeros(1, dtype=torch.long), d['C_L'].cumsum(0)])

    for b in range(B):
        ns, ne = int(n_off[b]), int(n_off[b + 1])
        es, ee = int(e_off[b]), int(e_off[b + 1])
        x_hw   = d['x_hw'][b:b + 1]
        y_hw   = d['y_hw'][b]
        node_x = d['node_x'][ns:ne]
        node_y = d['node_y'][ns:ne]
        Li     = d['L'][b:b + 1]
        x_in   = _enc(normalizer, x_hw, 'x_hw').to(DEVICE)

        if arch == 'FNO2d':
            out_hw = _dec(normalizer, model(x_in), 'y_hw')
            # B1 — interpolação registrada no config da run ('legacy' em runs antigas)
            y_nodes = _interpolate_fno_to_nodes(out_hw, node_x.to(DEVICE), Li.to(DEVICE),
                                                mode=getattr(model, 'interp_mode', 'legacy'))
            node_area = node_x[:, 2].numpy()
        elif arch in ('FNO_GNN', 'GNN_PostBase'):
            ei = d['edge_index'][:, es:ee] - ns
            out_hw, y_nodes = model(x_in, _enc(normalizer, node_x, 'node_x').to(DEVICE),
                                    ei.to(DEVICE), d['edge_attr'][es:ee].to(DEVICE),
                                    Li.to(DEVICE))
            out_hw  = _dec(normalizer, out_hw, 'y_hw')
            y_nodes = _dec(normalizer, y_nodes, 'node_y')
            node_area = node_x[:, 2].numpy()
        else:  # FNO_BipartiteGNN
            ms, me = int(m_off[b]), int(m_off[b + 1])
            cs, ce = int(c_off[b]), int(c_off[b + 1])
            elem_x = d['elem_x'][ms:me]
            ei  = d['edge_index'][:, es:ee] - ns
            cei = d['cross_edge_index'][:, cs:ce].clone()
            cei[0] -= ms
            cei[1] -= ns
            out_hw, y_nodes = model(
                x_in, _enc(normalizer, node_x, 'node_x').to(DEVICE),
                _enc(normalizer, elem_x, 'elem_x').to(DEVICE),
                ei.to(DEVICE), d['edge_attr'][es:ee].to(DEVICE),
                cei.to(DEVICE), d['cross_edge_attr'][cs:ce].to(DEVICE), Li.to(DEVICE))
            out_hw  = _dec(normalizer, out_hw, 'y_hw')
            y_nodes = _dec(normalizer, y_nodes, 'node_y')
            node_area = np.zeros(ne - ns, dtype=np.float64)
            np.add.at(node_area, cei[1].numpy(), elem_x[cei[0].numpy(), 2].numpy() / 3.0)

        acc_mesh.add(_mag(y_nodes.cpu()), _mag(node_y), node_area)
        acc_grid.add(_mag(out_hw[0].cpu()), _mag(y_hw), area_grid)


# --------------------------------------------------------------------------- #
def main():
    runs = discover_runs()
    print(f'{len(runs)} runs em {LOG_ROOT}  |  device={DEVICE}')

    # mesmo split em todas as archs (split_seed=12, mesma ordem de amostras) --
    # confere e usa o do primeiro run
    splits = {json.dumps(json.loads((r['run_dir'] / 'split.json').read_text())['test'])
              for r in runs}
    if len(splits) != 1:
        sys.exit('ERRO: split de teste difere entre runs -- comparação não seria justa')
    test_files = json.loads(next(iter(splits)))
    if N_CHUNKS is not None:
        test_files = test_files[:N_CHUNKS]

    models = {}
    for r in runs:
        model, normalizer, cfg, epoch = load_model(r['run_dir'])
        models[(r['arch'], r['run'])] = (model, normalizer)
        r['epoch'] = epoch
        r['acc_mesh'], r['acc_grid'] = Accum(), Accum()
        print(f"  carregado {r['arch']:17s} {r['run']} loss={r['loss']:4s} epoch={epoch}")

    src = ChunkSource()
    t0 = time.time()
    for i, name in enumerate(test_files):
        t = time.time()
        chunks = src.get(name)
        # sanidade: GT dos dois layouts tem que ser idêntico
        assert torch.equal(chunks['FNO_GNN']['y_hw'], chunks['FNO_BipartiteGNN']['y_hw'])
        assert torch.equal(chunks['FNO_GNN']['L'], chunks['FNO_BipartiteGNN']['L'])
        t_load = time.time() - t
        for r in runs:
            model, normalizer = models[(r['arch'], r['run'])]
            eval_chunk(r['arch'], model, normalizer, chunks, r['acc_mesh'], r['acc_grid'])
        del chunks
        print(f'  [{i + 1}/{len(test_files)}] {name}  (dados {t_load:.0f}s, total '
              f'{time.time() - t:.0f}s, acumulado {(time.time() - t0) / 60:.1f} min)', flush=True)
    shutil.rmtree(TMP_CHUNKS, ignore_errors=True)

    results = []
    for r in runs:
        results.append(dict(arch=r['arch'], run=r['run'], loss=r['loss'], epoch=r['epoch'],
                            n_samples=32 * len(test_files),
                            mesh=r['acc_mesh'].report(), grid=r['acc_grid'].report()))
    OUT_JSON.write_text(json.dumps(results, indent=2))

    print('\n=== Superfície GT em MALHA (nós FEMM) -- |B| em T (% de B_ref) ===')
    print(f"{'arch':17s} {'loss':4s} {'ep':>4s} {'MAE pt':>16s} {'L1 área':>16s} {'L2 área':>16s}")
    for x in results:
        m = x['mesh']
        print(f"{x['arch']:17s} {x['loss']:4s} {x['epoch']:4d} "
              f"{m['MAE_pt']:.4f} ({m['MAE_pt_pct']:5.2f}%) "
              f"{m['L1_area']:.4f} ({m['L1_area_pct']:5.2f}%) "
              f"{m['L2_area']:.4f} ({m['L2_area_pct']:5.2f}%)")
    print(f"B_ref malha = {results[0]['mesh']['B_ref']:.4f} T | "
          f"B_ref grade = {results[0]['grid']['B_ref']:.4f} T")
    print(f'\nresultados salvos em {OUT_JSON}')


if __name__ == '__main__':
    main()
