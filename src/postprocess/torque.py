"""
Torque a partir do campo B: FEMM × gabarito × modelos.

  theta-e : femm_solve, band_torque, curve_row (curvas T(θe) por amostra)
  batch   : select_batch_samples, fundamental, curve_row, batch_summary (method_check)
  scatter : scatter_ids, scatter_stats

Método:
  FEMM   : T22 = mo_blockintegral(22) nos blocos do rotor (rótulos com r > r_m), × 3 setores.
  campo  : tensor de Maxwell médio na faixa de ar do entreferro, com B por elemento:
           T = −3·L/(μ0·(r2−r1)) · Σ_e r·Br·Bθ·A_e (elementos de ar com centróide em (r1, r2);
           sinal: torque no rotor externo), L = stack length do design (coluna stack_length do
           valid_designs.csv, fallback motor_constants.STACK_LENGTH_MM) -- a mesma profundidade
           do mi_probdef em femm_solve.
  gt     : B exato curl(A) por elemento (parse_elem_structure) -- o method_check é
           gt × FEMM T22 (mede o erro do próprio método da faixa, sem modelo).
  smooth : B suavizado no centróide (gabarito da Elem).
  pred   : saída do modelo levada aos elementos -- Elem: por elemento; demais: valores
           nodais -> média dos 3 vértices (fields.nodes_to_elem, P1 no centróide).
Generalização: qualquer run/loss das 5 archs. Referencial: band_torque recebe (Bx, By);
o gabarito smooth e as saídas das runs são POLARES (Br, Bθ) -> girados para xy pelo ângulo
do ponto (centróide; nó, antes da média).
"""
import csv
import gzip
import json
import os
import shutil
from pathlib import Path

import numpy as np

MU0 = 4e-7 * np.pi


def stack_length_m(row):
    """Stack length [m] de uma linha do valid_designs.csv; fallback STACK_LENGTH_MM."""
    from src.data_gen.motor_constants import STACK_LENGTH_MM
    return float(row.get('stack_length [mm]', STACK_LENGTH_MM)) * 1e-3


def femm_ref(row, theta_e, t22):
    """Registro de cache de 1 solução FEMM: T22 do setor + a profundidade
    (stack length) com que foi resolvido -- a mesma de femm_solve(row, ...)."""
    return dict(theta_e_deg=float(theta_e), T22_sector=float(t22),
                stack_length_mm=stack_length_m(row) * 1e3)


def femm_ref_ok(ref, row):
    """True se o cache foi resolvido com o stack length de `row` (cache sem o campo
    stack_length_mm -> inválido)."""
    sl = ref.get('stack_length_mm')
    return sl is not None and abs(float(sl) - stack_length_m(row) * 1e3) <= 1e-9 * max(1.0, abs(float(sl)))


# --------------------------------------------------------------------------- #
# designs / geometria
# --------------------------------------------------------------------------- #
def params_of(row):
    """valid_designs.csv -> motor_params {chave: {'unit','value'}}."""
    p = {}
    for col, v in row.items():
        key, unit = (col.split(' [')[0], col.split(' [')[1].rstrip(']')) if ' [' in col else (col, '')
        p[key] = {'unit': unit, 'value': float(v)}
    return p


def gap(row):
    """(r1, r2) mm: raio externo do estator, raio interno do rotor (faixa do entreferro)."""
    return float(row['stator_outer_diameter [mm]']) / 2, float(row['rotor_inner_diameter [mm]']) / 2


def thetas(n_theta):
    return np.arange(n_theta) * 360.0 / n_theta


def block_labels(ans_path):
    lines = Path(ans_path).read_text('utf-8', errors='replace').splitlines()
    j = next(i for i, l in enumerate(lines) if l.startswith('[NumBlockLabels]'))
    n = int(lines[j].split('=')[1])
    return np.array([[float(v) for v in l.split()[:2]] for l in lines[j + 1:j + 1 + n]])


# --------------------------------------------------------------------------- #
# FEMM (Windows: pyfemm)
# --------------------------------------------------------------------------- #
def femm_solve(row, theta_e, work_dir: Path, gz_out: Path):
    """Redesenha o design (fase do rotor e amplitude da corrente do CSV, ângulo θe dado),
    resolve, grava o .ans comprimido em gz_out e devolve T22 do setor (rotor externo)."""
    import femm
    from src.data_gen.motor_model import BLDC_FEMM_Model_Sym120_Annular
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    p = params_of(row)
    p['current_angle']['value'] = float(theta_e)
    r_m = sum(gap(row)) / 2
    cwd0 = os.getcwd()
    os.chdir(work_dir)
    try:
        model = BLDC_FEMM_Model_Sym120_Annular(motor_params=p, phase=p['rotor_phase']['value'])
        femm.openfemm('bHide'); femm.main_resize(1000, 1000); femm.newdocument(0)
        femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, stack_length_m(row) * 1e3, 200)
        model.draw_motor(); femm.mi_zoomnatural(); femm.mi_saveas(str(work_dir / 'model.fem'))
        femm.mi_createmesh(); femm.mi_analyze(); femm.mi_loadsolution()
        lab = block_labels(work_dir / 'model.ans')
        # Rotor = rótulos com r > r_m. O rótulo do ar do entreferro fica exatamente em
        # r = (RID + SOD)/4 = r_m (motor_model.py, r_avg); a seleção depende de o raio dele,
        # relido do .ans, não passar de r_m em ponto flutuante — funciona porque rótulo e
        # r_m saem da mesma conta (conferido igual em 100 amostras), e o > estrito o exclui.
        for x, y in lab[np.hypot(lab[:, 0], lab[:, 1]) > r_m]:
            femm.mo_selectblock(x, y)
        t22 = float(np.atleast_1d(femm.mo_blockintegral(22))[0])
        femm.mo_clearblock(); femm.closefemm()
        Path(gz_out).parent.mkdir(parents=True, exist_ok=True)
        with open(work_dir / 'model.ans', 'rb') as f, gzip.open(str(gz_out) + '.part', 'wb') as g:
            shutil.copyfileobj(f, g)
        os.replace(str(gz_out) + '.part', gz_out)
    finally:
        os.chdir(cwd0)
    return t22


# --------------------------------------------------------------------------- #
# torque pelo campo
# --------------------------------------------------------------------------- #
def band_torque(B, cen, area, mat, r1, r2, depth_m):
    """Torque no rotor (×3, N·m): média do tensor de Maxwell na faixa de ar do entreferro.
    B [M,2] = (Bx, By) por elemento; cen [M,2] mm; area [M] mm²; depth_m = stack length [m]
    (stack_length_m(row), o mesmo da profundidade do FEMM)."""
    from src.data_gen.motor_constants import MATERIAL_ID
    r = np.hypot(cen[:, 0], cen[:, 1])
    sel = (r > r1) & (r < r2) & (mat == MATERIAL_ID['vacuum'])
    th = np.arctan2(cen[sel, 1], cen[sel, 0])
    bx, by = B[sel, 0], B[sel, 1]
    br = bx * np.cos(th) + by * np.sin(th)
    bt = -bx * np.sin(th) + by * np.cos(th)
    s = (r[sel] * 1e-3 * br * bt * area[sel] * 1e-6).sum()
    # torque no conteúdo do círculo (estator) = +L/μ0/(r2−r1)·Σ; no rotor externo é o oposto
    return -3 * depth_m / (MU0 * (r2 - r1) * 1e-3) * s, int(sel.sum())


def polar_to_xy(B, x, y):
    """(Br, Bθ) no ponto (x, y) -> (Bx, By) (inversa de ans_parsing._to_polar)."""
    th = np.arctan2(y, x)
    c, s = np.cos(th), np.sin(th)
    B = np.asarray(B, np.float64)
    return np.stack([B[:, 0] * c - B[:, 1] * s, B[:, 0] * s + B[:, 1] * c], axis=1)


def parse_torque_sample(path, row):
    """1 .ans.gz -> layouts para os modelos (samples.parse_smooth com Elem) + o que o torque
    usa: gt exato por elemento (xy), smooth no centróide (xy), material, centróide, área.
    O `.ans` é lido uma vez (read_ans) e passado aos dois parsers."""
    from src.data_gen.parsers.elem_structure import parse_elem_structure
    from src.data_gen.parsers.ans_parsing import read_ans
    from src.postprocess.samples import ANG_1, ANG_2, N_A, N_R, parse_smooth
    ans = read_ans(path)
    lay = parse_smooth(path, row, with_elem=True, ans=ans)
    r_in, r_ext = float(row['inner_diameter [mm]']) / 2, float(row['outer_diameter [mm]']) / 2
    d_ex = parse_elem_structure(path, r_in, r_ext, ang_1=ANG_1, ang_2=ANG_2, n_r=N_R,
                                      n_a=N_A, ans=ans)
    nodes, elems = ans.nodes, ans.elems
    cen = nodes[elems[:, :3], :2].mean(axis=1).astype(np.float64)
    lay['torque'] = dict(gt=d_ex['node_y'].astype(np.float64),
                         smooth=polar_to_xy(lay['elem']['node_y'], cen[:, 0], cen[:, 1]),
                         mat=lay['elem_mat'].astype(np.int64), cen=cen,
                         area=d_ex['node_x'][:, 4].astype(np.float64),
                         node_xy=nodes[:, :2].astype(np.float64))
    return lay


def pred_elem_xy(arch, run, lay, ts):
    """Saída do modelo -> (Bx, By) por elemento [M,2]."""
    from src.postprocess.fields import nodes_to_elem
    from src.postprocess.inference import predict
    _, yn = predict(arch, run['model'], run['normalizer'], ts)
    yp = yn.float().cpu().numpy().astype(np.float64)
    tq = lay['torque']
    if arch == 'FNO_BipartiteGNN_Elem':
        return polar_to_xy(yp, tq['cen'][:, 0], tq['cen'][:, 1])
    return nodes_to_elem(lay, polar_to_xy(yp, tq['node_xy'][:, 0], tq['node_xy'][:, 1]))


def sample_torques(lay, row, runs):
    """{'gt', 'smooth', 'pred_<arch>'...}: torque (N·m, motor completo) de 1 solução."""
    from src.postprocess.inference import to_torch_sample
    tq = lay['torque']
    r1, r2 = gap(row)
    g = lambda B: band_torque(B, tq['cen'], tq['area'], tq['mat'], r1, r2, stack_length_m(row))[0]
    out = dict(gt=g(tq['gt']), smooth=g(tq['smooth']))
    ts = to_torch_sample(lay)
    for r in runs.values():
        out[f"pred_{r['arch']}"] = g(pred_elem_xy(r['arch'], r, lay, ts))
    return out


# --------------------------------------------------------------------------- #
# amostras e resumos (batch / scatter)
# --------------------------------------------------------------------------- #
def subset_chunks(split_run_dir: Path, subset='test'):
    """Índices dos chunks do subset ('test' ou 'val') no split.json da run."""
    split = json.loads((Path(split_run_dir) / 'split.json').read_text())
    return sorted(int(s.split('_')[-1][:4]) for s in split[subset])


def select_batch_samples(split_run_dir, rows, n_samples=80, seed=0, chunk_size=32, subset='test'):
    """n_samples amostras do subset estratificadas por corrente (n/5 por faixa de 2 A).
    Índices além do raw (último chunk incompleto) ficam fora do sorteio."""
    pool = np.array([c * chunk_size + j for c in subset_chunks(split_run_dir, subset)
                     for j in range(chunk_size) if c * chunk_size + j < len(rows)])
    I = np.array([float(rows[i]['current_amplitude [A]']) for i in pool])
    rng = np.random.default_rng(seed)
    per = n_samples // 5
    out = [rng.choice(pool[(I >= a) & (I < a + 2)], per, replace=False) for a in range(0, 10, 2)]
    return sorted(int(i) for i in np.concatenate(out))


def scatter_ids(split_run_dir, n_per_chunk=8, chunk_size=32, subset='test'):
    """As n_per_chunk primeiras amostras de cada chunk do subset."""
    return [ch * chunk_size + j for ch in subset_chunks(split_run_dir, subset) for j in range(n_per_chunk)]


def fundamental(th_deg, T):
    """Ajuste T ≈ T0 + A·cos(θe − φ) -> (T0, A, φ [°])."""
    th = np.deg2rad(th_deg)
    M = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    c, *_ = np.linalg.lstsq(M, T, rcond=None)
    return c[0], np.hypot(c[1], c[2]), np.rad2deg(np.arctan2(c[2], c[1]))


def curve_row(th, F, P, G):
    """Curva do modelo P × FEMM F (gt G para o method_check), em mNm."""
    T0f, Af, phf = fundamental(th, F)
    T0p, Ap, php = fundamental(th, P)
    return dict(rms_femm=float(np.sqrt(np.mean(F ** 2))), rmse=float(np.sqrt(np.mean((P - F) ** 2))),
                rmse_gt_femm=float(np.sqrt(np.mean((G - F) ** 2))), A_femm=float(Af),
                A_ratio=float(Ap / Af), dphi=float((php - phf + 180) % 360 - 180),
                dT0=float(T0p - T0f), T0_femm=float(T0f), peak_ratio=float(np.ptp(P) / np.ptp(F)))


def batch_summary(rows_):
    S = {k: np.array([r[k] for r in rows_]) for k in rows_[0] if not isinstance(rows_[0][k], dict)}
    S['rel'] = 100 * S['rmse'] / S['rms_femm']
    return dict(n=len(rows_), rmse_median_mNm=float(np.median(S['rmse'])),
                rel_median_pct=float(np.median(S['rel'])), rel_p90_pct=float(np.percentile(S['rel'], 90)),
                A_ratio_mean=float(S['A_ratio'].mean()), A_ratio_std=float(S['A_ratio'].std()),
                dphi_mean_deg=float(S['dphi'].mean()), dphi_std_deg=float(S['dphi'].std()),
                dT0_mean_mNm=float(S['dT0'].mean()), dT0_std_mNm=float(S['dT0'].std()),
                method_check_rmse_gt_vs_femm_median_mNm=float(np.median(S['rmse_gt_femm'])),
                rel_median_por_faixa={f'{a}-{a + 2}A': float(np.median(S['rel'][(S['I'] >= a) & (S['I'] < a + 2)]))
                                      for a in range(0, 10, 2) if ((S['I'] >= a) & (S['I'] < a + 2)).any()})


def scatter_stats(gt, pr, sm):
    """gt, pr, sm em mNm."""
    e = pr - gt
    st = dict(n=len(gt), rms_gt_mNm=float(np.sqrt(np.mean(gt ** 2))),
              rmse_mNm=float(np.sqrt(np.mean(e ** 2))), mae_mNm=float(np.mean(np.abs(e))),
              bias_mNm=float(e.mean()), p95_abs_err_mNm=float(np.percentile(np.abs(e), 95)),
              max_abs_err_mNm=float(np.abs(e).max()),
              r2=float(1 - np.sum(e ** 2) / np.sum((gt - gt.mean()) ** 2)),
              rmse_smooth_vs_gt_mNm=float(np.sqrt(np.mean((sm - gt) ** 2))))
    st['rmse_pct_of_rms_gt'] = 100 * st['rmse_mNm'] / st['rms_gt_mNm']
    return st


def read_rows(raw_dir: Path):
    return list(csv.DictReader(open(Path(raw_dir) / 'valid_designs.csv')))
