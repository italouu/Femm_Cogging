"""
Diagnóstico: torque pelo tensor de Maxwell PONDERADO do FEMM (mo_blockintegral(22)) aplicado
a A reconstruído por mínimos quadrados a partir de B, comparado com o fluxo concatenado e
com a faixa de Arkkio (band_torque). Não altera treino, modelos nem o torque atual.

    python -m scripts.maxwell_rec_torque [IDX ...] [--keep-ans] [--tag _10]
    (sem IDX: as 10 amostras de dados/torque/fluxlink_split_mae.json, 16 θe cada)

Por (amostra, θe) do cache data/temp/torque_theta_e_batch/ (nunca alterado):
  fontes de B por elemento (Bx, By):
    P0      B exato rot(A) do FEMM (element_b)
    smooth  B suavizado no centróide (gabarito; referência de todos os métodos)
    <run>   predição das 8 runs do artigo (regions.json; FNO2d, FNO_GNN, Elem, Split × mse/mae),
            levada aos elementos como em torque.pred_elem_xy
  reconstrução de A (src/postprocess/a_reconstruction.py):
    a   sem restrição (A[0] = 0), deslocada p/ média 0 nos nós Dirichlet (B inalterado)
    b   com restrição: A = 0 nos nós Dirichlet e nós periódicos amarrados (lidos do .ans)
  métodos (torque da máquina inteira, N·m):
    mx22    3 × mo_blockintegral(22) nos rótulos do rotor (r > r_m), no .ans copiado com o A
            trocado (ans_rewrite.write_ans_with_A); 'orig' = .ans original
    flux    fluxo concatenado (Clarke) com o A: orig, a, b
    arkkio  band_torque com B direto ('direct') ou rot(A_b) ('b')
Validações (param no 1º erro): (b) com P0 = A do FEMM (‖ΔA‖/‖A‖ ≤ 1e-8, o [Precision]);
ida e volta (cópia com o A original: A relido idêntico e T22 idêntico ao do original).

Métricas por curva (torque.curve_row): ε_T = 100·RMSE/RMS da referência, A_pred/A_ref, Δφ, ΔT0
do ajuste T0 + A·cos(θe − φ). Referência de cada (método, reconstrução) = smooth pelo mesmo
caminho. Medianas geral e por faixa de corrente (0,7–2,6 / 3,1–5,2 / 6,4–9,8 A).

Saída: resultados/BLDC_138x276/dados/torque/maxwell_rec<tag>.json (+ _tabela.md);
cache por θe em data/temp/maxwell_rec/cache/ (retomável); cópias .ans em data/temp/maxwell_rec/work/
(apagadas após a medição, salvo --keep-ans).
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from src.data_gen.parsers.ans_b_smoothing import element_b
from src.data_gen.parsers.ans_parsing import read_ans
from src.postprocess import a_reconstruction as ar
from src.postprocess.ans_rewrite import ans_constraints, write_ans_with_A
from src.postprocess.samples import design_rows
from src.postprocess.torque import (band_torque, block_labels, curve_row, gap, parse_torque_sample,
                                    pred_elem_xy, stack_length_m, thetas)

ROOT  = Path(__file__).resolve().parents[1]
DADOS = ROOT / 'resultados' / 'BLDC_138x276' / 'dados'
CACHE = ROOT / 'data' / 'temp' / 'torque_theta_e_batch'
TMP   = ROOT / 'data' / 'temp' / 'maxwell_rec'
RUN_KEYS = ('FNO2d_mse', 'FNO2d_mae', 'FNO_GNN_mse', 'FNO_GNN_mae',
            'FNO_BipartiteGNN_Elem_mse', 'FNO_BipartiteGNN_Elem_mae',
            'FNO_SplitGNN_mse', 'FNO_SplitGNN_mae')
N_THETA = 16
TOL_A = 1e-8                     # [Precision] do .ans
BANDS = (('baixa 0,7–2,6 A', 0.0, 3.0), ('média 3,1–5,2 A', 3.0, 6.0), ('alta 6,4–9,8 A', 6.0, 99.0))


# --------------------------------------------------------------------------- FEMM
class Femm:
    def __init__(self):
        import femm
        self.f = femm
        femm.openfemm(1)
        self.times = []          # (abrir, integral) [s]

    def t22(self, path, r_m):
        f = self.f
        t0 = time.perf_counter(); f.opendocument(str(path)); t1 = time.perf_counter()
        lab = block_labels(path)
        for x, y in lab[np.hypot(lab[:, 0], lab[:, 1]) > r_m]:     # mesma regra de femm_solve
            f.mo_selectblock(x, y)
        T = 3 * float(np.atleast_1d(f.mo_blockintegral(22))[0])
        f.mo_clearblock(); t2 = time.perf_counter(); f.mo_close()
        self.times.append((t1 - t0, t2 - t1))
        return T

    def close(self):
        self.f.closefemm()


def load_runs():
    from src.postprocess.inference import load_model
    reg = json.loads((DADOS / 'regions.json').read_text(encoding='utf-8'))['runs']
    runs = {}
    for k in RUN_KEYS:
        d = ROOT / reg[k]['run_dir'].replace('\\', '/')
        m, nz, cfg, ep = load_model(d)
        assert f"{cfg['arch']}_{cfg['loss']}" == k, (k, d)
        runs[k] = dict(arch=cfg['arch'], model=m, normalizer=nz, run_dir=str(d), epoch=int(ep))
    return runs


# --------------------------------------------------------------------------- 1 (amostra, θe)
def theta_record(idx, k, row, runs, fm, keep_ans):
    from src.postprocess.inference import to_torch_sample
    src = CACHE / f'sample_{idx:06d}' / f'theta_{k:02d}.ans.gz'
    raw = gzip.open(src, 'rb').read(); text = raw.decode('utf-8')
    ans = read_ans(src); xy = ans.nodes[:, :2].astype(np.float64); E = ans.elems[:, :3]; A0 = ans.nodes[:, 2]
    lay = parse_torque_sample(src, row, with_split=True); tq = lay['torque']
    cons = ans_constraints(ans.lines, ans.nodes)
    te = float(thetas(N_THETA)[k]); I = float(row['current_amplitude [A]'])
    depth = stack_length_m(row); r1, r2 = gap(row); r_m = (r1 + r2) / 2; i_ph = ar.phase_currents(I, te)
    work = TMP / 'work' / f'{idx:06d}_{k:02d}'; work.mkdir(parents=True, exist_ok=True)

    # fontes de B
    B = dict(P0=np.stack(element_b(ans.mesh), 1), smooth=tq['smooth'])
    ts = to_torch_sample(lay)
    for rk, r in runs.items():
        B[rk] = pred_elem_xy(r['arch'], r, lay, ts)

    flux = lambda A: float(ar.flux_torque(ans.lines, xy, ans.elems, A, depth, i_ph)[0])
    ark = lambda Bx: float(band_torque(Bx, tq['cen'], tq['area'], tq['mat'], r1, r2, depth)[0])
    mx = lambda A, name: fm.t22(write_ans_with_A(text, work / f'{name}.ans', A), r_m)

    # original + validações
    orig = work / 'orig.ans'; orig.write_bytes(raw)
    T_orig = fm.t22(orig, r_m)
    cp = write_ans_with_A(text, work / 'roundtrip.ans', A0)
    A_back = read_ans_text(cp)
    T_rt = fm.t22(cp, r_m)
    T_saved = 3 * json.loads(src.with_suffix('').with_suffix('.json').read_text())['T22_sector']
    val = dict(roundtrip_A_identico=bool(np.array_equal(A_back, A0)), T22_orig=T_orig, T22_roundtrip=T_rt,
               T22_gravado=T_saved, n_dirichlet=int(len(cons['dirichlet'])), n_pbc=int(len(cons['pbc'])),
               dirichlet_eq_A0=bool(set(cons['dirichlet']) == set(np.flatnonzero(A0 == 0))))
    if not (val['roundtrip_A_identico'] and T_rt == T_orig):
        raise RuntimeError(f'ida e volta falhou em #{idx} θ{k}: {val}')

    T = dict(mx22=dict(orig=T_orig), flux=dict(orig=flux(A0)), arkkio=dict(orig=ark(B['P0'])))
    t_rec = []
    for name, Bx in B.items():
        t0 = time.perf_counter()
        Ab = ar.reconstruct_A(xy, E, Bx, cons)
        Aa = ar.shift_to_dirichlet(ar.reconstruct_A(xy, E, Bx), cons)
        t_rec.append(time.perf_counter() - t0)
        if name == 'P0':
            eA = float(np.linalg.norm(Ab - A0) / np.linalg.norm(A0))
            val.update(P0_b_rel_normA=eA, P0_b_max_absA=float(np.abs(Ab - A0).max()))
            if eA > TOL_A:
                raise RuntimeError(f'(b) com P0 não devolve o A do FEMM em #{idx} θ{k}: rel {eA:.2e}')
        for v, A in (('a', Aa), ('b', Ab)):
            T['mx22'][f'{name}_{v}'] = mx(A, f'{name}_{v}')
            T['flux'][f'{name}_{v}'] = flux(A)
        T['arkkio'][f'{name}_direct'] = ark(Bx)
        T['arkkio'][f'{name}_b'] = ark(ar.curl_P0(xy, E, Ab))
    if not keep_ans:
        for p in work.glob('*.ans'):
            p.unlink()
        work.rmdir()
    return dict(idx=idx, k=k, theta_e_deg=te, I=I, T=T, validacao=val, t_reconstrucao_s=t_rec)


def read_ans_text(path):
    """A nodal relido de um .ans não comprimido (para a ida e volta)."""
    from src.postprocess.ans_rewrite import _strip, solution_layout
    L = _strip(Path(path).read_text('utf-8').splitlines())
    i, N, *_ = solution_layout(L)
    return np.array([float(L[i + q].split('\t')[2]) for q in range(N)])


# --------------------------------------------------------------------------- métricas
def curves(recs):
    """[(θe, T)] -> {método: {chave: array[16]}} em mN·m."""
    out = {}
    for m in recs[0]['T']:
        out[m] = {key: np.array([r['T'][m][key] for r in recs]) * 1e3 for key in recs[0]['T'][m]}
    return out


def comparisons():
    """(grupo, método, chave_pred, chave_ref)."""
    C = []
    for rk in RUN_KEYS:
        for v in ('b', 'a'):
            C.append(('pred', 'mx22', f'{rk}_{v}', f'smooth_{v}'))
            C.append(('pred', 'flux', f'{rk}_{v}', f'smooth_{v}'))
        C.append(('pred', 'arkkio', f'{rk}_direct', 'smooth_direct'))
        C.append(('pred', 'arkkio', f'{rk}_b', 'smooth_b'))
    for m in ('mx22', 'flux'):
        for s in ('smooth', 'P0'):
            for v in ('b', 'a'):
                C.append(('piso', m, f'{s}_{v}', 'orig'))
        C.append(('piso', m, 'smooth_b', 'P0_b'))
    C += [('piso', 'arkkio', 'smooth_direct', 'orig'), ('piso', 'arkkio', 'smooth_b', 'orig'),
          ('piso', 'arkkio', 'P0_b', 'orig'),
          ('entre_metodos', 'flux|mx22', 'orig', 'orig'), ('entre_metodos', 'arkkio|mx22', 'orig', 'orig'),
          ('entre_metodos', 'arkkio|mx22', 'smooth_direct', 'orig')]
    return C


def per_sample_rows(samples):
    rows = []
    for s in samples:
        th = np.array([r['theta_e_deg'] for r in s['thetas']]); C = curves(s['thetas'])
        for g, m, kp, kr in comparisons():
            mp, mr = (m.split('|') + [m])[:2] if '|' in m else (m, m)
            row = curve_row(th, C[mr][kr], C[mp][kp])
            row['eps_T'] = 100 * row['rmse'] / row['rms_ref']
            rows.append(dict(grupo=g, metodo=m, pred=kp, ref=kr, idx=s['idx'], I=s['I'], **row))
    return rows


def summarize(rows):
    out = {}
    for r in rows:
        out.setdefault((r['grupo'], r['metodo'], r['pred'], r['ref']), []).append(r)
    S = []
    for (g, m, kp, kr), rr in out.items():
        I = np.array([r['I'] for r in rr])
        f = lambda key, sel=slice(None): float(np.median(np.array([r[key] for r in rr])[sel]))
        d = dict(grupo=g, metodo=m, pred=kp, ref=kr, n=len(rr),
                 eps_T_med=f('eps_T'), A_ratio_med=f('A_ratio'), dphi_med=f('dphi'), dT0_med_mNm=f('dT0'),
                 eps_T_p90=float(np.percentile([r['eps_T'] for r in rr], 90)))
        for name, a, b in BANDS:
            sel = (I >= a) & (I < b)
            if sel.any():
                d[f'eps_T_med[{name}]'] = f('eps_T', sel)
        S.append(d)
    return S


def table_md(S, t_femm, t_rec):
    br = lambda x, nd=1: f'{x:.{nd}f}'.replace('.', ',')
    get = {(d['grupo'], d['metodo'], d['pred'], d['ref']): d for d in S}
    L = ['# Torque: Maxwell ponderado (FEMM 22) após reconstrução × fluxo concatenado × Arkkio', '',
         f'{S[0]["n"]} amostras × {N_THETA} θe. ε_T = 100·RMSE/RMS da referência (mediana; p90 entre parênteses). '
         'Referência = smooth pelo mesmo método e reconstrução. a = sem restrição, b = com restrição '
         '(Dirichlet + periódico); Arkkio "direto" = B da predição sem reconstrução.', '',
         '## ε_T mediano [%] por arquitetura e perda', '',
         '| run | MST22 b | MST22 a | fluxo b | fluxo a | Arkkio direto | Arkkio rot(A_b) |',
         '|---|---|---|---|---|---|---|']
    cols = (('mx22', '_b', 'smooth_b'), ('mx22', '_a', 'smooth_a'), ('flux', '_b', 'smooth_b'),
            ('flux', '_a', 'smooth_a'), ('arkkio', '_direct', 'smooth_direct'), ('arkkio', '_b', 'smooth_b'))
    for rk in RUN_KEYS:
        cells = []
        for m, suf, ref in cols:
            d = get[('pred', m, rk + suf, ref)]
            cells.append(f"{br(d['eps_T_med'])} ({br(d['eps_T_p90'])})")
        L.append(f'| {rk} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Por faixa de corrente (ε_T mediano [%], reconstrução b; Arkkio direto)', '',
          '| run | método | ' + ' | '.join(n for n, *_ in BANDS) + ' |', '|---|---|' + '---|' * len(BANDS)]
    for rk in RUN_KEYS:
        for m, suf, ref in (('mx22', '_b', 'smooth_b'), ('flux', '_b', 'smooth_b'), ('arkkio', '_direct', 'smooth_direct')):
            d = get[('pred', m, rk + suf, ref)]
            L.append(f'| {rk} | {m} | ' + ' | '.join(br(d.get(f"eps_T_med[{n}]", np.nan)) for n, *_ in BANDS) + ' |')
    L += ['', '## Fundamental: A_pred/A_ref, Δφ [°], ΔT0 [mN·m] (medianas; b; Arkkio direto)', '',
          '| run | método | A/A_ref | Δφ | ΔT0 |', '|---|---|---|---|---|']
    for rk in RUN_KEYS:
        for m, suf, ref in (('mx22', '_b', 'smooth_b'), ('flux', '_b', 'smooth_b'), ('arkkio', '_direct', 'smooth_direct')):
            d = get[('pred', m, rk + suf, ref)]
            L.append(f"| {rk} | {m} | {br(d['A_ratio_med'], 3)} | {br(d['dphi_med'], 2)} | {br(d['dT0_med_mNm'], 1)} |")
    L += ['', '## Pisos (GT por reconstrução × A original) e diferença entre métodos no GT', '',
          '| grupo | método | curva | referência | ε_T med [%] | p90 | A/A_ref | Δφ [°] | ΔT0 [mN·m] |',
          '|---|---|---|---|---|---|---|---|---|']
    for d in S:
        if d['grupo'] != 'pred':
            L.append(f"| {d['grupo']} | {d['metodo']} | {d['pred']} | {d['ref']} | {br(d['eps_T_med'], 2)} | "
                     f"{br(d['eps_T_p90'], 2)} | {br(d['A_ratio_med'], 4)} | {br(d['dphi_med'], 3)} | {br(d['dT0_med_mNm'], 2)} |")
    to, ti = np.array(t_femm).T if len(t_femm) else (np.array([0.]), np.array([0.]))
    L += ['', '## Tempo', '',
          f'- Pós-processador do FEMM por solução: abrir o .ans {br(np.median(to) * 1e3, 0)} ms (mediana), '
          f'mo_blockintegral(22) com seleção {br(np.median(ti) * 1e3, 0)} ms; total {br(np.median(to + ti) * 1e3, 0)} ms '
          f'({len(to)} soluções nesta execução).',
          f'- Reconstrução de A (a + b) por fonte: {br(np.median(t_rec) * 1e3, 0)} ms (mediana).']
    return '\n'.join(L) + '\n'


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('idx', nargs='*', type=int)
    ap.add_argument('--keep-ans', action='store_true')
    ap.add_argument('--tag', default='_10')
    a = ap.parse_args()
    ids = a.idx or [int(s['idx']) for s in
                    json.loads((DADOS / 'torque' / 'fluxlink_split_mae.json').read_text())['samples']]
    rows = design_rows(); runs = load_runs()
    ck = TMP / 'cache'; fm = None
    samples, t_femm, t_rec = [], [], []
    try:
        for n, idx in enumerate(ids, 1):
            recs = []
            for k in range(N_THETA):
                fj = ck / f'sample_{idx:06d}' / f'theta_{k:02d}.json'
                if fj.exists():
                    r = json.loads(fj.read_text())
                else:
                    fm = fm or Femm()
                    nt = len(fm.times); t0 = time.perf_counter()
                    r = theta_record(idx, k, rows[idx], runs, fm, a.keep_ans)
                    r['t_femm_s'] = fm.times[nt:]; r['t_total_s'] = time.perf_counter() - t0
                    fj.parent.mkdir(parents=True, exist_ok=True); fj.write_text(json.dumps(r))
                t_femm += r.get('t_femm_s', []); t_rec += r['t_reconstrucao_s']; recs.append(r)
            s = dict(idx=idx, I=recs[0]['I'], thetas=recs); samples.append(s)
            v = [r['validacao'] for r in recs]
            print(f"[{n}/{len(ids)}] #{idx} I={s['I']:.1f} A | (b) P0 max rel ‖ΔA‖ {max(x['P0_b_rel_normA'] for x in v):.1e} | "
                  f"T_orig×T_gravado max {max(abs(x['T22_orig'] - x['T22_gravado']) for x in v):.1e} N·m", flush=True)
    finally:
        if fm:
            fm.close()
    per = per_sample_rows(samples); S = summarize(per)
    out = dict(amostras=ids, runs={k: dict(run_dir=r['run_dir'], epoch=r['epoch']) for k, r in runs.items()},
               faixas=[n for n, *_ in BANDS], tolerancia_A=TOL_A,
               tempo=dict(femm_abrir_med_s=float(np.median([t[0] for t in t_femm])) if t_femm else None,
                          femm_integral_med_s=float(np.median([t[1] for t in t_femm])) if t_femm else None,
                          n_solucoes_femm=len(t_femm), reconstrucao_a_b_med_s=float(np.median(t_rec))),
               resumo=S, por_amostra=per,
               curvas={str(s['idx']): dict(I=s['I'], theta_e_deg=[r['theta_e_deg'] for r in s['thetas']],
                                           T_Nm={m: {kk: [r['T'][m][kk] for r in s['thetas']] for kk in s['thetas'][0]['T'][m]}
                                                 for m in s['thetas'][0]['T']},
                                           validacao=[r['validacao'] for r in s['thetas']]) for s in samples})
    pj = DADOS / 'torque' / f'maxwell_rec{a.tag}.json'
    pj.write_text(json.dumps(out, indent=1), encoding='utf-8')
    pm = DADOS / 'torque' / f'maxwell_rec{a.tag}_tabela.md'
    pm.write_text(table_md(S, t_femm, t_rec), encoding='utf-8')
    print(f'dados em {pj}\ntabela em {pm}')


if __name__ == '__main__':
    main()
