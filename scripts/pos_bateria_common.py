"""
pos_bateria_common.py
---------------------
Funções compartilhadas pelas análises pós-bateria de
mesh_ans_138x276_unified_best_mse_mae (scripts/pos_bateria_*.py).

Nada aqui treina ou altera o código de treino -- só reaproveita:
  - parse_ans_gzip_sample_unified (mesmo layout dos chunks unificados);
  - load_model / discover_runs de scripts/eval_surface_integral_table.py
    (mesmo carregamento de best.pth + Normalizer);
  - interpolate_grid_to_nodes (B1, cell_centered) para FNO2d e piso da grade.

Convenções (iguais às do relatório anterior):
  - |B| = hypot(Bx,By); erro de módulo por nó e = | |B|_pred - |B|_true |;
  - A_n = área lumped por nó (node_dual_area do layout v1 = Σ área_elem/3);
  - polares: B_r = Bx cosθ + By sinθ ; B_θ = -Bx sinθ + By cosθ.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.tri as mtri

import scripts.build_unified_ans_chunks_direct as bd
from scripts.eval_surface_integral_table import (
    load_model, discover_runs, _enc, _dec, LOG_ROOT, R_IN_MM, R_EXT_MM, ANG1_DEG, ANG2_DEG,
)
from src.data_gen.parsers.femm_mesh_unified import parse_ans_gzip_sample_unified, _read_ans_gz
from src.data_gen.parsers.ans_parsing import (
    _parse_block_materials, _element_areas, _element_b_from_A,
)
from src.data_gen.motor_constants import MATERIAL_ID
from src.neural_op.archs.interp import interpolate_grid_to_nodes
from scripts.run_best_configs import INTERP_MODE   # B1b: mesmo modo dos modelos da bateria

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = LOG_ROOT / 'pos_bateria'
FIG_DIR = OUT_DIR / 'figuras'
TMP_DIR = Path('data/temp') / 'pos_bateria_parse'

MU0      = 4e-7 * np.pi
IRON_ID  = MATERIAL_ID['iron_1008']
N_ARC    = 1200
FUND     = 7          # 14 polos em 120° -> 7 períodos no setor
N_HARM   = 100        # THD até a ordem 100
SAT_THR  = (1.4, 1.6, 1.8)
MU_BINS  = np.linspace(0.0, 4.0, 401)   # log10(mu_r efetivo), histograma 5b

RUN_ORDER = [('FNO2d', 'mse'), ('FNO2d', 'mae'), ('FNO_GNN', 'mse'), ('FNO_GNN', 'mae'),
             ('GNN_PostBase', 'mse'), ('GNN_PostBase', 'mae'),
             ('FNO_BipartiteGNN', 'mse'), ('FNO_BipartiteGNN', 'mae')]


def run_key(arch, loss):
    return f'{arch}_{loss}'


# --------------------------------------------------------------------------- #
# Amostras de teste
# --------------------------------------------------------------------------- #
def design_rows():
    with open(bd.RAW_DIR / 'valid_designs.csv', newline='') as f:
        return list(csv.DictReader(f))


def test_samples():
    """[(chunk_name, sample_idx, ans_path)] na ordem dos chunks de teste
    (split.json, idêntico nos 8 runs -- conferido em eval_surface_integral_table)."""
    runs = discover_runs()
    splits = {json.dumps(json.loads((r['run_dir'] / 'split.json').read_text())['test'])
              for r in runs}
    assert len(splits) == 1, 'split de teste difere entre runs'
    test_files = json.loads(next(iter(splits)))
    ans_paths = sorted(bd.RAW_DIR.glob('sample_*.ans.gz'), key=bd._sample_idx)
    out = []
    for name in test_files:
        ci = int(name.split('_')[-1].split('.')[0])
        for p in ans_paths[ci * bd.CHUNK_SIZE:(ci + 1) * bd.CHUNK_SIZE]:
            out.append((name, bd._sample_idx(p), p))
    return out


# --------------------------------------------------------------------------- #
# Geometria auxiliar (worker, CPU)
# --------------------------------------------------------------------------- #
def gap_radius_mm(row):
    """r_m = stator_outer_d/2 + gap/2, gap = (rotor_inner_d - stator_outer_d)/2
    (valid_designs.csv não tem coluna 'gap' -- rotor_inner = stator_outer + 2·gap)."""
    so = float(row['stator_outer_diameter [mm]'])
    ri = float(row['rotor_inner_diameter [mm]'])
    gap = (ri - so) / 2.0
    return so / 2.0 + gap / 2.0, gap


def arc_theta_deg():
    return np.arange(N_ARC) * (ANG2_DEG - ANG1_DEG) / N_ARC + ANG1_DEG


def _barycentric(nodes_xy, elems, tri_idx, px, py):
    p0 = nodes_xy[elems[tri_idx, 0]]
    p1 = nodes_xy[elems[tri_idx, 1]]
    p2 = nodes_xy[elems[tri_idx, 2]]
    det = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
    w0 = ((p1[:, 1] - p2[:, 1]) * (px - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (py - p2[:, 1])) / det
    w1 = ((p2[:, 1] - p0[:, 1]) * (px - p2[:, 0]) + (p0[:, 0] - p2[:, 0]) * (py - p2[:, 1])) / det
    return np.stack([w0, w1, 1.0 - w0 - w1], axis=1)


def arc_weights(nodes_xy, elems, r_m):
    """Para cada ponto do arco: 3 nós do triângulo que o contém + pesos
    baricêntricos (interpolação linear P1, coerente com o gabarito)."""
    th = np.deg2rad(arc_theta_deg())
    px, py = r_m * np.cos(th), r_m * np.sin(th)
    tri = mtri.Triangulation(nodes_xy[:, 0], nodes_xy[:, 1], triangles=elems[:, :3])
    tidx = tri.get_trifinder()(px, py)
    n_fallback = int((tidx < 0).sum())
    if n_fallback:
        # pontos exatamente sobre o corte θ=0: desloca 1e-9 rad pra dentro do setor
        bad = tidx < 0
        th2 = th[bad] + 1e-9
        tidx[bad] = tri.get_trifinder()(r_m * np.cos(th2), r_m * np.sin(th2))
        assert (tidx >= 0).all(), 'ponto do arco fora da malha'
    w = _barycentric(nodes_xy, elems, tidx, px, py)
    return elems[tidx, :3].astype(np.int64), w, n_fallback


def parse_bh_curve(lines):
    """Curva BH do primeiro bloco com BHPoints>0 ([BlockProps]); confere que
    todos os blocos com curva têm a mesma."""
    curves, i = [], 0
    while i < len(lines):
        l = lines[i]
        if '<BHPoints>' in l:
            n = int(l.split('=')[1])
            if n > 0:
                pts = np.array([[float(v) for v in lines[i + 1 + k].split()] for k in range(n)])
                curves.append(pts)
                i += n
        i += 1
    assert curves, 'nenhuma curva BH no .ans'
    for c in curves[1:]:
        assert np.array_equal(c, curves[0]), 'curvas BH diferentes entre blocos de ferro'
    return curves[0]        # [N,2] (B [T], H [A/m])


def mu_eff_from_B(Bmag, bh):
    """mu_r efetivo = |B| / (mu0·H(|B|)); H interpolado linearmente na curva BH
    (extrapolação linear além do último ponto; |B|->0 usa a inclinação inicial)."""
    B, H = bh[:, 0], bh[:, 1]
    Hb = np.interp(Bmag, B, H)
    over = Bmag > B[-1]
    if over.any():
        slope = (H[-1] - H[-2]) / (B[-1] - B[-2])
        Hb[over] = H[-1] + (Bmag[over] - B[-1]) * slope
    mu0_init = B[1] / (MU0 * H[1])
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = np.where(Bmag > 1e-12, Bmag / (MU0 * Hb), mu0_init)
    return mu, int(over.sum())


def element_fields(lines, nodes, elems):
    """Por elemento: material_id, área (mm²), |B| (curl(A) P1, T), mu_r efetivo
    (só significativo no ferro)."""
    block_material_id, _ = _parse_block_materials(lines)
    mat = block_material_id[elems[:, 3]]
    area = _element_areas(nodes, elems)
    bx, by = _element_b_from_A(nodes, elems, nodes[:, 2])
    bmag = np.hypot(bx.astype(np.float64), by.astype(np.float64))
    bh = parse_bh_curve(lines)
    mu, n_over = mu_eff_from_B(bmag, bh)
    return dict(mat=mat, area=area, bmag=bmag, mu_eff=mu, bh=bh, n_bh_extrap=n_over)


def saturation_stats(ef):
    iron = ef['mat'] == IRON_ID
    a = ef['area'][iron]
    b = ef['bmag'][iron]
    mu = ef['mu_eff'][iron]
    at = a.sum()
    frac = {f'{t:.1f}': float(a[b > t].sum() / at) for t in SAT_THR}
    hist, _ = np.histogram(np.log10(np.clip(mu, 10 ** MU_BINS[0], 10 ** MU_BINS[-1])),
                           bins=MU_BINS, weights=a)
    iron_extrap = int((b > ef['bh'][-1, 0]).sum())
    return dict(frac=frac, iron_area_mm2=float(at), mu_hist=hist,
                n_iron_elems=int(iron.sum()), n_iron_bh_extrap=iron_extrap,
                bmax_iron=float(b.max()))


def parse_worker(path, row, full=False):
    """Worker (processo separado): 1 .ans.gz -> layouts dos modelos + arco +
    saturação. full=True devolve também a malha completa (figuras)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    r_in = float(row['inner_diameter [mm]']) / 2
    r_ext = float(row['outer_diameter [mm]']) / 2
    lay = parse_ans_gzip_sample_unified(path, r_in, r_ext, ang_1=bd.ANG_1, ang_2=bd.ANG_2,
                                        n_r=bd.N_R, n_a=bd.N_A, tmp_dir=TMP_DIR)
    lines, nodes, elems = _read_ans_gz(Path(path), TMP_DIR)
    r_m, gap = gap_radius_mm(row)
    arc_idx, arc_w, n_fb = arc_weights(nodes[:, :2], elems, r_m)
    ef = element_fields(lines, nodes, elems)
    out = dict(v1=lay['FNO_GNN'], bip=lay['FNO_BipartiteGNN'],
               arc_idx=arc_idx, arc_w=arc_w, arc_fallback=n_fb, r_m=r_m, gap=gap,
               r_in=r_in, r_ext=r_ext, sat=saturation_stats(ef),
               n_bh_extrap_all=ef['n_bh_extrap'])
    if full:
        out.update(nodes=nodes, elems=elems[:, :3], elem=ef)
    return out


# --------------------------------------------------------------------------- #
# Modelos (GPU)
# --------------------------------------------------------------------------- #
def _t(a, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t if dtype is None else t.to(dtype)


def to_torch_sample(s):
    """dicts numpy de parse_worker -> tensores CPU no formato de 1 amostra do chunk."""
    v1, bp = s['v1'], s['bip']
    return dict(
        v1=dict(x_hw=_t(v1['x_hw'])[None], y_hw=_t(v1['y_hw']), node_x=_t(v1['node_x']),
                node_y=_t(v1['node_y']), edge_index=_t(v1['edge_index']),
                edge_attr=_t(v1['edge_attr']), L=_t(v1['L'])),
        bip=dict(x_hw=_t(bp['x_hw'])[None], node_x=_t(bp['node_x']), elem_x=_t(bp['elem_x']),
                 edge_index=_t(bp['edge_index']), edge_attr=_t(bp['edge_attr']),
                 cross_edge_index=_t(bp['cross_edge_index']),
                 cross_edge_attr=_t(bp['cross_edge_attr']), L=_t(bp['L'])),
    )


@torch.inference_mode()
def predict(arch, model, normalizer, ts):
    """Retorna (out_hw [2,H,W] em T no device, y_nodes [S,2] em T no device).
    Mesma sequência de eval_surface_integral_table.eval_chunk (amostra única)."""
    if arch == 'FNO_BipartiteGNN':
        d = ts['bip']
        out_hw, y_nodes = model(
            _enc(normalizer, d['x_hw'], 'x_hw').to(DEVICE),
            _enc(normalizer, d['node_x'], 'node_x').to(DEVICE),
            _enc(normalizer, d['elem_x'], 'elem_x').to(DEVICE),
            d['edge_index'].to(DEVICE), d['edge_attr'].to(DEVICE),
            d['cross_edge_index'].to(DEVICE), d['cross_edge_attr'].to(DEVICE),
            d['L'].to(DEVICE))
        return _dec(normalizer, out_hw, 'y_hw')[0], _dec(normalizer, y_nodes, 'node_y')
    d = ts['v1']
    x_in = _enc(normalizer, d['x_hw'], 'x_hw').to(DEVICE)
    if arch == 'FNO2d':
        out_hw = _dec(normalizer, model(x_in), 'y_hw')
        nx = d['node_x'].to(DEVICE)
        y_nodes = interpolate_grid_to_nodes(out_hw, nx[:, 3], nx[:, 4], d['L'].to(DEVICE),
                                            mode=getattr(model, 'interp_mode', 'legacy'))
        return out_hw[0], y_nodes
    out_hw, y_nodes = model(x_in, _enc(normalizer, d['node_x'], 'node_x').to(DEVICE),
                            d['edge_index'].to(DEVICE), d['edge_attr'].to(DEVICE),
                            d['L'].to(DEVICE))
    return _dec(normalizer, out_hw, 'y_hw')[0], _dec(normalizer, y_nodes, 'node_y')


def load_runs():
    """{run_key: dict(arch, loss, run, model, normalizer, epoch, interp_mode)} na ordem RUN_ORDER."""
    found = {(r['arch'], r['loss']): r for r in discover_runs()}
    out = {}
    for arch, loss in RUN_ORDER:
        r = found[(arch, loss)]
        model, normalizer, cfg, epoch = load_model(r['run_dir'])
        out[run_key(arch, loss)] = dict(arch=arch, loss=loss, run=r['run'], run_dir=str(r['run_dir']),
                                         model=model, normalizer=normalizer, epoch=epoch,
                                         interp_mode=getattr(model, 'interp_mode', None))
    return out


# --------------------------------------------------------------------------- #
# Arco: valores, erros, harmônicos
# --------------------------------------------------------------------------- #
def arc_values_from_nodes(node_B, arc_idx, arc_w):
    """node_B [S,2] numpy -> [N_ARC,2] (Bx,By) por interpolação baricêntrica."""
    return (node_B[arc_idx] * arc_w[:, :, None]).sum(axis=1)


def arc_values_fno(out_hw, r_m, r_in, r_ext):
    """FNO2d: grade -> pontos do arco via interpolate_grid_to_nodes (cell_centered)."""
    th = arc_theta_deg()
    rb = torch.full((N_ARC,), (r_m - r_in) / (r_ext - r_in), device=out_hw.device)
    cb = torch.as_tensor((th - ANG1_DEG) / (ANG2_DEG - ANG1_DEG), device=out_hw.device,
                         dtype=out_hw.dtype)
    L = torch.tensor([N_ARC], device=out_hw.device)
    # [REMOVIDO 2026-10-06] 'cell_centered' fixo (wrap circular — obsoleto, ver interp.py)
    # v = interpolate_grid_to_nodes(out_hw[None], rb.to(out_hw.dtype), cb, L, mode='cell_centered')
    v = interpolate_grid_to_nodes(out_hw[None], rb.to(out_hw.dtype), cb, L, mode=INTERP_MODE)
    return v.double().cpu().numpy()


def to_polar(bxy):
    th = np.deg2rad(arc_theta_deg())
    c, s = np.cos(th), np.sin(th)
    return bxy[:, 0] * c + bxy[:, 1] * s, -bxy[:, 0] * s + bxy[:, 1] * c


def harmonics(br):
    """Amplitudes (pico) e fases (graus) das ordens 0..N_HARM, período base = 120°."""
    X = np.fft.rfft(br) / len(br)
    amp = np.abs(X) * 2.0
    amp[0] /= 2.0
    return amp[:N_HARM + 1], np.rad2deg(np.angle(X[:N_HARM + 1]))


def thd(amp):
    """THD até a ordem N_HARM: todas as ordens 1..100 exceto DC e a fundamental (7)."""
    k = np.arange(1, N_HARM + 1)
    k = k[k != FUND]
    return float(np.sqrt((amp[k] ** 2).sum()) / amp[FUND])


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def percentile_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x, 95)))
