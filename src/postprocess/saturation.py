"""
Saturação do aço ligada à curva B-H -- só o gabarito, nenhum modelo.

Curva B-H: lida do próprio .ans ([BlockProps], blocos com <BHPoints> > 0) e conferida
igual em todas as amostras. Interpolação como o FEMM (porte de xfemm, REOptimize-Systems/
xfemm, cfemm/libfemm/CMaterialProp.cpp, CMMaterialProp::GetSlopes / GetH, magnetostático):
  - inclinações dH/dB nos pontos por spline cúbica com condição natural nas pontas (sistema
    tridiagonal);
  - se algum trecho da cúbica não for monótono (raiz de dH/dB dentro do trecho), a curva é
    reparada por média móvel de 3 pontos nos pontos internos (até 100 vezes);
  - H(B) = Hermite cúbico entre os pontos; acima do último ponto, extrapolação linear com a
    inclinação do último ponto.
Conferido contra o μ do pós-processador do FEMM (mo_getpointvalues) -- ver check_against_femm
(amostra 224, 300 elementos de aço: diferença relativa máx. 3e-15; 0 reparos na curva).

Ponto de operação por elemento de aço: |B| suavizado no centróide (mesmo gabarito da Elem,
elem_smooth) e μ_eff = |B| / (μ0·H(|B|)).
"""
import gzip
from pathlib import Path

import numpy as np

MU0 = 4e-7 * np.pi


def parse_bh_curves(lines):
    """[(nome, B [T], H [A/m])] dos blocos de [BlockProps] com BHPoints > 0."""
    out, name, i = [], None, 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('<BlockName>'):
            name = s.split('=', 1)[1].strip().strip('"')
        elif s.startswith('<BHPoints>'):
            n = int(s.split('=', 1)[1])
            if n > 0:
                pts = np.array([[float(v) for v in l.split()[:2]] for l in lines[i + 1:i + 1 + n]])
                out.append((name, pts[:, 0], pts[:, 1]))
                i += n
        elif s.startswith('[NumPoints]'):
            break
        i += 1
    return out


class FemmBH:
    """Curva B-H com a interpolação do FEMM (CMMaterialProp::GetSlopes/GetH, omega = 0)."""

    def __init__(self, B, H, max_repair=100):
        self.B_in, self.H_in = np.asarray(B, float).copy(), np.asarray(H, float).copy()
        B, H = self.B_in.copy(), self.H_in.copy()
        assert len(B) >= 2 and np.all(np.diff(B) > 0), 'curva B-H inválida'
        self.n_repair = 0
        for _ in range(max_repair + 1):
            s = self._slopes(B, H)
            if self._monotone(B, H, s):
                break
            Bn, Hn = B.copy(), H.copy()
            Bn[1:-1] = (B[:-2] + B[1:-1] + B[2:]) / 3
            Hn[1:-1] = (H[:-2] + H[1:-1] + H[2:]) / 3
            if np.array_equal(Bn, B) and np.array_equal(Hn, H):
                raise ValueError('reparo da curva B-H sem progresso')
            B, H = Bn, Hn
            self.n_repair += 1
        self.B, self.H, self.s = B, H, s

    @staticmethod
    def _slopes(B, H):
        n = len(B)
        M = np.zeros((n, n))
        b = np.zeros(n)
        l1 = B[1] - B[0]
        M[0, 0], M[0, 1], b[0] = 4 / l1, 2 / l1, 6 * (H[1] - H[0]) / l1 ** 2
        l1 = B[-1] - B[-2]
        M[-1, -1], M[-1, -2], b[-1] = 4 / l1, 2 / l1, 6 * (H[-1] - H[-2]) / l1 ** 2
        for i in range(1, n - 1):
            l1, l2 = B[i] - B[i - 1], B[i + 1] - B[i]
            M[i, i - 1], M[i, i], M[i, i + 1] = 2 / l1, 4 * (l1 + l2) / (l1 * l2), 2 / l2
            b[i] = 6 * (H[i] - H[i - 1]) / l1 ** 2 + 6 * (H[i + 1] - H[i]) / l2 ** 2
        return np.linalg.solve(M, b)

    @staticmethod
    def _monotone(B, H, s):
        for i in range(1, len(B)):
            L, d0, d1, u0, u1 = B[i] - B[i - 1], s[i - 1], s[i], H[i - 1], H[i]
            c0 = d0
            c1 = -2 * (2 * d0 * L + d1 * L + 3 * u0 - 3 * u1) / L ** 2
            c2 = 3 * (d0 * L + d1 * L + 2 * u0 - 2 * u1) / L ** 3
            X0 = X1 = -1.0
            disc = c1 ** 2 - 4 * c0 * c2
            if c2 == 0:
                if c1 != 0:
                    X0 = -c0 / c1
            elif disc > 0:
                X0 = -(c1 + np.sqrt(disc)) / (2 * c2)
                X1 = (-c1 + np.sqrt(disc)) / (2 * c2)
            if 0 <= X0 <= L or 0 <= X1 <= L:
                return False
        return True

    def h(self, b):
        """H(|B|) [A/m] (vetorizado)."""
        b = np.abs(np.asarray(b, float))
        B, H, s = self.B, self.H, self.s
        out = np.zeros_like(b)
        hi = b > B[-1]
        out[hi] = H[-1] + s[-1] * (b[hi] - B[-1])
        i = np.clip(np.searchsorted(B, b[~hi], side='right') - 1, 0, len(B) - 2)
        l = B[i + 1] - B[i]
        z = (b[~hi] - B[i]) / l
        z2 = z * z
        out[~hi] = ((1 - 3 * z2 + 2 * z2 * z) * H[i] + z * (1 - 2 * z + z2) * l * s[i]
                    + z2 * (3 - 2 * z) * H[i + 1] + z2 * (z - 1) * l * s[i + 1])
        return out

    def mu_eff(self, b):
        """μ_eff = |B| / (μ0·H(|B|)) (relativo); |B| = 0 -> limite B[1]/(μ0·H[1]) não usado."""
        b = np.abs(np.asarray(b, float))
        h = self.h(b)
        return np.divide(b, MU0 * h, out=np.full_like(b, np.nan), where=h > 0)

    def knee(self, frac=0.5, n=200001):
        """Joelho: menor B acima do pico de μ_eff em que μ_eff cai a frac·máx."""
        b = np.linspace(self.B[1] * 1e-3, self.B[-1], n)
        mu = self.mu_eff(b)
        k = int(np.nanargmax(mu))
        below = np.flatnonzero(mu[k:] <= frac * mu[k])
        if not len(below):
            return dict(frac=frac, B_knee_T=None, mu_max=float(mu[k]), B_mu_max_T=float(b[k]))
        j = k + below[0]
        # interpolação linear entre as amostras vizinhas do cruzamento
        t = (mu[j - 1] - frac * mu[k]) / (mu[j - 1] - mu[j])
        bk = b[j - 1] + t * (b[j] - b[j - 1])
        return dict(frac=frac, B_knee_T=float(bk), H_knee_A_m=float(self.h(np.array([bk]))[0]),
                    mu_max=float(mu[k]), B_mu_max_T=float(b[k]))


def steel_operating_points(path, row, iron_id=0):
    """|B| suavizado no centróide (gabarito da Elem) dos elementos de aço de 1 amostra.
    Retorna dict(bmag [Ms], area [Ms] mm², elem [Ms], curves = parse_bh_curves)."""
    from src.data_gen.parsers.ans_b_smoothing import element_b, load_ans, nodal_b, point_b
    from src.data_gen.parsers.ans_parsing import _parse_label_materials
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    label_mat, _, _ = _parse_label_materials(lines)
    mesh = load_ans(path)
    B1, B2 = element_b(mesh)
    b1, b2 = nodal_b(mesh, B1, B2)
    P = np.stack([mesh.x, mesh.y], 1)[mesh.p]                     # [M,3,2]
    cen = P.mean(axis=1)
    m_idx = np.arange(len(mesh.p))
    bx, by = point_b(mesh, B1, B2, b1, b2, m_idx, cen[:, 0], cen[:, 1], smooth=True)
    steel = np.flatnonzero(label_mat[mesh.lbl] == iron_id)
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    return dict(bmag=np.hypot(bx, by)[steel], area=area[steel], elem=steel,
                b_elem_p0=np.hypot(B1, B2)[steel], cen=cen[steel],
                curves=parse_bh_curves(lines))


def weighted_percentile(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(c, q / 100.0)])


def check_against_femm(ans_gz, bh, cen, b_smooth, n_points=200, seed=0):
    """μ do FEMM (mo_getpointvalues, Mu1, mo_smooth on) nos centróides de n_points elementos
    de aço × μ_eff desta interpolação com o |B| suavizado no ponto (o pós-processador do FEMM
    avalia μ = |B|/(μ0·H(|B|)) com o B suavizado do ponto -- conferido: com o |B| P0 do
    elemento a diferença chega a ~80%). Windows/pyfemm. Retorna as diferenças relativas."""
    import shutil
    import tempfile
    import femm
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(cen), min(n_points, len(cen)), replace=False)
    tmp = Path(tempfile.mkdtemp())
    ans = tmp / 'check.ans'
    with gzip.open(ans_gz, 'rb') as f, open(ans, 'wb') as g:
        shutil.copyfileobj(f, g)
    femm.openfemm('bHide')
    try:
        femm.opendocument(str(ans))
        mu_f = np.array([femm.mo_getpointvalues(float(cen[i, 0]), float(cen[i, 1]))[9] for i in sel])
        femm.mo_close()
    finally:
        femm.closefemm()
        shutil.rmtree(tmp, ignore_errors=True)
    mu_l = bh.mu_eff(b_smooth[sel])
    rel = np.abs(mu_l - mu_f) / mu_f
    return dict(n_points=int(len(sel)), max_rel_diff=float(rel.max()),
                median_rel_diff=float(np.median(rel)),
                B_range_T=[float(b_smooth[sel].min()), float(b_smooth[sel].max())])


# ---------------------------------------------------------------------------
# erro de superfície × ponto de operação (acumulado dentro do regions)
# ---------------------------------------------------------------------------
SAT_EDGES = np.round(np.arange(0.0, 2.6 + 1e-9, 0.005), 6)   # 520 bins de 0,005 T + overflow
SAT_SETS = ('aco_int', 'aco')     # aço ∩ Ω_int(d) | aço inteiro (em Ω)


class SatAccum:
    """Erro de superfície no aço em função de |B| da referência suavizada no ponto de
    quadratura (|B|_FEMM da métrica de superfície): histograma fino em |B| (SAT_EDGES; o
    último índice = |B| ≥ 2,6 T, overflow) com Σw, Σe·w, Σe²·w, para cada conjunto de
    SAT_SETS. `direct` = as mesmas somas feitas direto nos pontos (validação do histograma)."""

    def __init__(self, edges=SAT_EDGES):
        self.edges = np.asarray(edges, float)
        nb = len(self.edges)                       # nb-1 bins regulares + 1 overflow
        self.h = {k: np.zeros((3, nb)) for k in SAT_SETS}
        self.direct = {k: np.zeros(3) for k in SAT_SETS}

    def add(self, mag_model, mag_femm, w, masks):
        """masks {conjunto: bool [Q]} na ordem dos pontos de mag_*/w."""
        mm, mf, w = (np.asarray(v, np.float64) for v in (mag_model, mag_femm, w))
        e = np.abs(mm - mf)
        nb = len(self.edges)
        for k, m in masks.items():
            b = np.minimum(np.searchsorted(self.edges, mf[m], side='right') - 1, nb - 1)
            wm, em = w[m], e[m]
            vals = (wm, em * wm, em ** 2 * wm)
            for j, v in enumerate(vals):
                self.h[k][j] += np.bincount(b, v, minlength=nb)
            self.direct[k] += [v.sum() for v in vals]


def sat_report(acc, b_ref, n_int=50):
    """Intervalos de MESMA ÁREA de aço (n_int, na resolução dos bins) por conjunto:
    |B| mediano (interpolado no bin), faixa de |B|, área (mm² e fração), L1/L2 (% do b_ref,
    o B_ref global em Ω da mesma run) + total e a conferência histograma × pontos."""
    out = {}
    E = acc.edges
    width = np.append(np.diff(E), np.nan)          # overflow sem largura
    for k in SAT_SETS:
        W, EW, E2W = acc.h[k]
        A = W.sum()
        dsum = acc.direct[k]
        chk = dict(rel_diff_sum_w=float(abs(A - dsum[0]) / max(dsum[0], 1e-300)),
                   rel_diff_sum_ew=float(abs(EW.sum() - dsum[1]) / max(dsum[1], 1e-300)),
                   overflow_area_frac=float(W[-1] / A) if A else None)
        if A <= 0:
            out[k] = dict(intervals=[], total=None, check=chk)
            continue
        cum = np.cumsum(W)
        grp = np.minimum((n_int * (cum - W / 2) / A).astype(int), n_int - 1)
        iv = []
        for g in range(n_int):
            sel = np.flatnonzero((grp == g) & (W > 0))
            if not len(sel):
                continue
            w, ew, e2w = W[sel].sum(), EW[sel].sum(), E2W[sel].sum()
            c = np.cumsum(W[sel])
            j = int(np.searchsorted(c, w / 2))
            prev = c[j - 1] if j else 0.0
            b0 = E[sel[j]]
            bmed = b0 + (width[sel[j]] if np.isfinite(width[sel[j]]) else 0.0) * (w / 2 - prev) / W[sel[j]]
            hi = E[sel[-1] + 1] if sel[-1] + 1 < len(E) else float('inf')
            iv.append(dict(B_median_T=float(bmed), B_lo_T=float(E[sel[0]]), B_hi_T=float(hi),
                           area_mm2=float(w), area_frac=float(w / A),
                           L1_pct=float(100 * ew / w / b_ref), L2_pct=float(100 * np.sqrt(e2w / w) / b_ref)))
        out[k] = dict(intervals=iv, check=chk,
                      total=dict(area_mm2=float(A), L1_pct=float(100 * EW.sum() / A / b_ref),
                                 L2_pct=float(100 * np.sqrt(E2W.sum() / A) / b_ref)))
    return out


def sat_bh_bands(sums, edges, b_ref, cuts):
    """Faixas de |B| delimitadas por `cuts` (crescentes, ex.: [B do pico de μ_eff, B do
    joelho]) a partir do histograma do SatAccum (sums = [Σw, Σe·w, Σe²·w] por bin; o último
    bin = overflow ≥ edges[-1]): fração da área de aço e L1/L2 (% do b_ref) em cada faixa
    [0, c1), [c1, c2), ..., [c_k, ∞). Um bin cortado pela fronteira é dividido na proporção
    da largura (|B| uniforme dentro do bin de 0,005 T). Nenhum cálculo novo de erro: só
    reagrupa as somas já acumuladas."""
    W, EW, E2W = (np.asarray(s, float) for s in sums)
    E = np.asarray(edges, float)
    A = W.sum()

    def below(c):
        f = np.clip((c - E[:-1]) / np.diff(E), 0.0, 1.0)        # bins regulares
        f = np.append(f, 1.0 if c > E[-1] else 0.0)             # overflow (|B| ≥ E[-1])
        return np.array([(f * W).sum(), (f * EW).sum(), (f * E2W).sum()])

    tot = np.array([A, EW.sum(), E2W.sum()])
    acc = [np.zeros(3)] + [below(c) for c in cuts] + [tot]
    out = []
    for k in range(len(acc) - 1):
        w, ew, e2w = acc[k + 1] - acc[k]
        lo = 0.0 if k == 0 else float(cuts[k - 1])
        hi = float(cuts[k]) if k < len(cuts) else None
        out.append(dict(B_lo_T=lo, B_hi_T=hi, area_frac=float(w / A),
                        L1_pct=float(100 * ew / w / b_ref) if w > 0 else None,
                        L2_pct=float(100 * np.sqrt(e2w / w) / b_ref) if w > 0 else None))
    return out
