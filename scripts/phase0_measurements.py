"""
phase0_measurements.py — Fase 0 da preparação da bateria definitiva
(mesh_ans_138x276_unified). Só leitura: usa os checkpoints 'best' das runs já
existentes em data/logs/<PROBLEM>/ e NÃO depende de nenhuma mudança no código de
treino (as runs antigas são reconstruídas com interp_mode='legacy' e sem
reescala B2 — defaults dos configs antigos).

Rodar da raiz do projeto, ANTES do B0 (arquivamento dos logs):
    python -m scripts.phase0_measurements            # test set inteiro
    python -m scripts.phase0_measurements --n-chunks 2   # smoke rápido

Saída: docs/bateria_definitiva/phase0_results.json  +  phase0_results.md

Métrica ε_mesh (todas as tabelas de malha): erro do MÓDULO
    e = | hypot(Bx,By)_pred − hypot(Bx,By)_true |      contra node_y,
agregado GLOBALMENTE sobre todo o teste (somas de numerador/denominador sobre
todas as amostras, não média de % por amostra):
    L1    = ∫|e| dA / A_tot
    L2    = sqrt(∫e² dA / A_tot)
    B_ref = sqrt(∫|B_true|² dA / A_tot)
reportados em % de B_ref. Dois estimadores de área:
  (a) 'dual'  — node_dual_area por nó (layout v1, pasta FNO_GNN/, node_x[:,2]);
  (b) 'elem'  — integral por elemento P1, área do triângulo (elem_x[:,2] do layout
                bipartite) com os 3 vértices dados por cross_edge_index:
                ∫|e|  ≈ A·mean(e_1,e_2,e_3)            (pedido no enunciado)
                ∫e²   = A/6·(Σe_i² + Σ_{i<j} e_i e_j)  (quadratura exata de P1)
                ∫|B|² idem com |B_true|.
                OBS: para L1, (b) é algebricamente idêntico a (a) se
                node_dual_area == Σ_{e∋n} A_e/3 (definição de
                _node_material_stats) — a diferença numérica entre (a) e (b) em
                L1 serve como checagem dessa identidade; L2/B_ref diferem de fato.

ε_grid (T0c): erro do módulo na grade H×W contra y_hw, peso dA = r·dr·dθ,
mesmas definições L1/L2/B_ref.

Tabelas
  T0a — piso de representação: y_hw (gabarito, sem modelo) interpolado nos nós
        vs node_y, interpolação 'legacy' e 'cell_centered'.
  T0b — FNO2d (mse/mae) interpolado nos nós, 'legacy' e 'cell_centered'.
  T0c — estágio FNO (y_hw_fno) de FNO_GNN e FNO_BipartiteGNN (mse/mae): ε_grid
        decodificando com stats de y_hw (como hoje) e com stats de node_y.
        Extra: ε_mesh do mesmo estágio interpolado nos nós (legacy, como no
        forward dos modelos antigos), nas duas decodificações.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.eval_surface_integral_table import (
    ChunkSource, load_model, grid_area_weights,
)
from src.neural_op.archs.interp import interpolate_grid_to_nodes

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
PROBLEM = 'mesh_ans_138x276_unified_best_mse_mae'
OUT_DIR = Path('docs/bateria_definitiva')
MODES   = ('legacy', 'cell_centered')


# --------------------------------------------------------------------------- #
# Acumuladores
# --------------------------------------------------------------------------- #
class MeshAccum:
    """ε_mesh com os dois estimadores de área (dual / elem)."""

    def __init__(self):
        self.s = {k: 0.0 for k in ('d_l1', 'd_l2', 'd_ref', 'd_A',
                                   'e_l1', 'e_l2', 'e_ref', 'e_A')}
        self.n_nodes = 0
        self.n_elems = 0

    def add(self, mag_pred, mag_true, dual_area, tri, tri_area):
        e = np.abs(mag_pred - mag_true).astype(np.float64)
        t = mag_true.astype(np.float64)
        a = dual_area.astype(np.float64)
        s = self.s
        s['d_l1'] += float((a * e).sum()); s['d_l2'] += float((a * e * e).sum())
        s['d_ref'] += float((a * t * t).sum()); s['d_A'] += float(a.sum())

        A = tri_area.astype(np.float64)
        e3, t3 = e[tri], t[tri]                                  # [M,3]
        s['e_l1'] += float((A * e3.mean(axis=1)).sum())
        s['e_l2'] += float((A / 6.0 * _p1_sq(e3)).sum())
        s['e_ref'] += float((A / 6.0 * _p1_sq(t3)).sum())
        s['e_A'] += float(A.sum())
        self.n_nodes += e.size
        self.n_elems += A.size

    def report(self):
        s, out = self.s, {}
        for k, name in (('d', 'dual'), ('e', 'elem')):
            A = s[f'{k}_A']
            b_ref = math.sqrt(s[f'{k}_ref'] / A)
            l1 = s[f'{k}_l1'] / A
            l2 = math.sqrt(s[f'{k}_l2'] / A)
            out[name] = dict(B_ref=b_ref, L1=l1, L2=l2,
                             L1_pct=100 * l1 / b_ref, L2_pct=100 * l2 / b_ref, A_tot=A)
        out['n_nodes'] = self.n_nodes
        out['n_elems'] = self.n_elems
        return out


class GridAccum:
    def __init__(self):
        self.l1 = self.l2 = self.ref = self.A = 0.0

    def add(self, mag_pred, mag_true, area):
        e = np.abs(mag_pred - mag_true).astype(np.float64)
        t = mag_true.astype(np.float64)
        a = np.broadcast_to(area, e.shape).astype(np.float64)
        self.l1 += float((a * e).sum()); self.l2 += float((a * e * e).sum())
        self.ref += float((a * t * t).sum()); self.A += float(a.sum())

    def report(self):
        b_ref = math.sqrt(self.ref / self.A)
        l1, l2 = self.l1 / self.A, math.sqrt(self.l2 / self.A)
        return dict(B_ref=b_ref, L1=l1, L2=l2, L1_pct=100 * l1 / b_ref, L2_pct=100 * l2 / b_ref)


def _p1_sq(v3):
    """Σv_i² + Σ_{i<j} v_i v_j por linha (∫v² = A/6 · isso, v linear no triângulo)."""
    a, b, c = v3[:, 0], v3[:, 1], v3[:, 2]
    return a * a + b * b + c * c + a * b + b * c + a * c


def _mag_nodes(t):
    return torch.hypot(t[:, 0], t[:, 1]).cpu().numpy()


def _mag_grid(t):
    return torch.hypot(t[0], t[1]).cpu().numpy()


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def find_runs(log_root: Path, arch: str):
    """{loss: run_dir} — run mais recente com best.pth por loss."""
    out = {}
    for run_dir in sorted((log_root / arch).glob('run_*')):
        if not (run_dir / 'checkpoints' / 'best.pth').exists():
            continue
        cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        out[cfg['loss']] = run_dir
    return out


def _decode_with(normalizer, t, key):
    return normalizer.decode(t, key) if normalizer is not None else t


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-root', default=f'data/logs/{PROBLEM}')
    ap.add_argument('--n-chunks', type=int, default=None, help='None = test set inteiro')
    args = ap.parse_args()
    log_root = Path(args.log_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = {a: find_runs(log_root, a) for a in ('FNO2d', 'FNO_GNN', 'FNO_BipartiteGNN')}
    print(f'device={DEVICE}  runs: ' + ', '.join(
        f'{a}:{sorted(r)}' for a, r in runs.items()), flush=True)

    # split de teste — único e idêntico entre as runs
    splits = {json.dumps(json.loads((rd / 'split.json').read_text())['test'])
              for r in runs.values() for rd in r.values()}
    if len(splits) != 1:
        sys.exit('ERRO: split de teste difere entre runs')
    test_files = json.loads(next(iter(splits)))
    if args.n_chunks is not None:
        test_files = test_files[:args.n_chunks]

    models = {}
    for arch, by_loss in runs.items():
        for loss, rd in by_loss.items():
            model, normalizer, cfg, epoch = load_model(rd)
            models[(arch, loss)] = (model, normalizer, rd, epoch)
            print(f'  {arch:17s} {loss:4s} {rd.name} best-epoch={epoch}', flush=True)

    # acumuladores
    t0a = {m: MeshAccum() for m in MODES}
    t0b = {(loss, m): MeshAccum() for loss in runs['FNO2d'] for m in MODES}
    t0c_grid = {(a, loss, dk): GridAccum()
                for a in ('FNO_GNN', 'FNO_BipartiteGNN') for loss in runs[a]
                for dk in ('y_hw', 'node_y')}
    t0c_mesh = {(a, loss, dk): MeshAccum()
                for a in ('FNO_GNN', 'FNO_BipartiteGNN') for loss in runs[a]
                for dk in ('y_hw', 'node_y')}
    checks = dict(max_dual_vs_elem_lumped=0.0, max_rbase_v1_v2=0.0, max_nodey_v1_v2=0.0)

    src = ChunkSource()
    tstart = time.time()
    with torch.no_grad():
        for ci, name in enumerate(test_files):
            tc = time.time()
            ch = src.get(name)
            v1, v2 = ch['FNO_GNN'], ch['FNO_BipartiteGNN']
            assert torch.equal(v1['L'], v2['L']), 'ordem/contagem de nós difere entre layouts'
            H, W = v2['x_hw'].shape[-2:]
            area_grid = grid_area_weights(H, W)
            n_off = torch.cat([torch.zeros(1, dtype=torch.long), v2['L'].cumsum(0)])
            m_off = torch.cat([torch.zeros(1, dtype=torch.long), v2['elem_L'].cumsum(0)])
            c_off = torch.cat([torch.zeros(1, dtype=torch.long), v2['C_L'].cumsum(0)])

            for b in range(v2['x_hw'].shape[0]):
                ns, ne = int(n_off[b]), int(n_off[b + 1])
                ms, me = int(m_off[b]), int(m_off[b + 1])
                cs, ce = int(c_off[b]), int(c_off[b + 1])
                Li = v2['L'][b:b + 1]
                r_base = v2['node_x'][ns:ne, 0].double()
                c_base = v2['node_x'][ns:ne, 1].double()
                node_y = v2['node_y'][ns:ne]
                mag_true = _mag_nodes(node_y)
                dual = v1['node_x'][ns:ne, 2].numpy()

                # elementos: cross_edge_index (elem, vtx), 3 por elemento
                cei = v2['cross_edge_index'][:, cs:ce]
                order = torch.argsort(cei[0], stable=True)
                tri = (cei[1][order] - ns).view(-1, 3).numpy()
                tri_area = v2['elem_x'][ms:me, 2].numpy()
                assert tri.shape[0] == me - ms

                # checagens de consistência
                lumped = np.zeros(ne - ns); np.add.at(lumped, tri.ravel(), np.repeat(tri_area / 3.0, 3))
                checks['max_dual_vs_elem_lumped'] = max(
                    checks['max_dual_vs_elem_lumped'], float(np.abs(lumped - dual).max() / dual.max()))
                checks['max_rbase_v1_v2'] = max(checks['max_rbase_v1_v2'], float(
                    (v1['node_x'][ns:ne, 3:5] - v2['node_x'][ns:ne, 0:2]).abs().max()))
                checks['max_nodey_v1_v2'] = max(checks['max_nodey_v1_v2'], float(
                    (v1['node_y'][ns:ne] - node_y).abs().max()))

                # T0a — gabarito da grade interpolado nos nós
                y_hw = v2['y_hw'][b:b + 1].double()
                for m in MODES:
                    yn = interpolate_grid_to_nodes(y_hw, r_base, c_base, Li, mode=m)
                    t0a[m].add(_mag_nodes(yn), mag_true, dual, tri, tri_area)

                x_hw = v2['x_hw'][b:b + 1]

                # T0b — FNO2d nos nós
                for loss in runs['FNO2d']:
                    model, nz, _, _ = models[('FNO2d', loss)]
                    x_in = (nz.encode(x_hw, 'x_hw') if nz is not None else x_hw).to(DEVICE)
                    out = _decode_with(nz, model(x_in), 'y_hw').double()
                    for m in MODES:
                        yn = interpolate_grid_to_nodes(out, r_base.to(DEVICE), c_base.to(DEVICE),
                                                       Li.to(DEVICE), mode=m)
                        t0b[(loss, m)].add(_mag_nodes(yn), mag_true, dual, tri, tri_area)

                # T0c — estágio FNO de FNO_GNN / FNO_BipartiteGNN
                for a in ('FNO_GNN', 'FNO_BipartiteGNN'):
                    for loss in runs[a]:
                        model, nz, _, _ = models[(a, loss)]
                        x_in = (nz.encode(x_hw, 'x_hw') if nz is not None else x_hw).to(DEVICE)
                        y_fno = model.fno(x_in)                     # espaço normalizado
                        for dk in ('y_hw', 'node_y'):
                            dec = _decode_with(nz, y_fno, dk).double()
                            t0c_grid[(a, loss, dk)].add(_mag_grid(dec[0]), _mag_grid(v2['y_hw'][b]),
                                                        area_grid)
                            yn = interpolate_grid_to_nodes(dec, r_base.to(DEVICE), c_base.to(DEVICE),
                                                           Li.to(DEVICE), mode='legacy')
                            t0c_mesh[(a, loss, dk)].add(_mag_nodes(yn), mag_true, dual, tri, tri_area)
            del ch
            print(f'  [{ci + 1}/{len(test_files)}] {name}  {time.time() - tc:.0f}s '
                  f'(acum. {(time.time() - tstart) / 60:.1f} min)', flush=True)

    res = dict(
        n_chunks=len(test_files), test_files=test_files, device=DEVICE,
        runs={f'{a}/{l}': dict(run=str(models[(a, l)][2]), best_epoch=models[(a, l)][3])
              for (a, l) in models},
        checks=checks,
        T0a={m: t0a[m].report() for m in MODES},
        T0b={f'{l}/{m}': t0b[(l, m)].report() for (l, m) in t0b},
        T0c_grid={f'{a}/{l}/{dk}': t0c_grid[(a, l, dk)].report() for (a, l, dk) in t0c_grid},
        T0c_mesh={f'{a}/{l}/{dk}': t0c_mesh[(a, l, dk)].report() for (a, l, dk) in t0c_mesh},
    )
    (OUT_DIR / 'phase0_results.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    md = render_md(res)
    (OUT_DIR / 'phase0_results.md').write_text(md, encoding='utf-8')
    print('\n' + md)
    print(f'salvo em {OUT_DIR}/phase0_results.(json|md)')


def _row_mesh(label, r):
    d, e = r['dual'], r['elem']
    return (f"| {label} | {d['L1_pct']:.3f} | {d['L2_pct']:.3f} | {e['L1_pct']:.3f} | "
            f"{e['L2_pct']:.3f} |")


def render_md(res):
    hdr = ('| caso | L1 dual (%B_ref) | L2 dual (%B_ref) | L1 elem (%B_ref) | L2 elem (%B_ref) |\n'
           '|---|---|---|---|---|')
    a0 = res['T0a']['legacy']
    lines = [
        f"# Fase 0 — resultados ({res['n_chunks']} chunks de teste)",
        '',
        f"B_ref (malha): dual = {a0['dual']['B_ref']:.4f} T, elem = {a0['elem']['B_ref']:.4f} T  "
        f"| nós = {a0['n_nodes']}, elementos = {a0['n_elems']}",
        '',
        '## T0a — piso de representação (y_hw → nós vs node_y)', '', hdr,
    ]
    for m, r in res['T0a'].items():
        lines.append(_row_mesh(m, r))
    lines += ['', '## T0b — FNO2d nos nós', '', hdr]
    for k, r in res['T0b'].items():
        lines.append(_row_mesh(f'FNO2d {k}', r))
    lines += ['', '## T0c — estágio FNO na grade (ε_grid, peso r·dr·dθ)', '',
              '| arch / loss | decod. | B_ref (T) | L1 (%B_ref) | L2 (%B_ref) |', '|---|---|---|---|---|']
    for k, r in res['T0c_grid'].items():
        a, l, dk = k.split('/')
        lines.append(f"| {a} / {l} | {dk} | {r['B_ref']:.4f} | {r['L1_pct']:.3f} | {r['L2_pct']:.3f} |")
    lines += ['', '### T0c (extra) — mesmo estágio FNO interpolado nos nós (legacy)', '', hdr]
    for k, r in res['T0c_mesh'].items():
        lines.append(_row_mesh(k, r))
    c = res['checks']
    lines += ['', '## Checagens', '',
              f"- máx |node_dual_area − Σ A_e/3| / máx(dual): {c['max_dual_vs_elem_lumped']:.2e}",
              f"- máx |r_base,c_base (v1) − (bipartite)|: {c['max_rbase_v1_v2']:.2e}",
              f"- máx |node_y (v1) − node_y (bipartite)|: {c['max_nodey_v1_v2']:.2e}"]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
