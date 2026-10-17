"""
Métricas de erro de |B| (convenção do CLAUDE.md, "Convenção de erro dos resultados"):
  e = | |B|_pred - |B|_true | por ponto; L1_A = Σe·a/Σa, L2_A = √(Σe²·a/Σa);
  MAE_pt = média simples de e; % relativo a B_ref = √(Σ|B|_true²·a/Σa);
  a = área dual do nó (malha) / área do elemento / r·dr·dθ (grade).

Origem: tests/resultados/common.py (2026-10-13), sem mudança de lógica.
Pendente: L2 ponto a ponto na Accum; erro grade × malha por interseção
célula ∩ triângulo (célula tratada como quadrilátero de lados retos).
"""
import math

import numpy as np

# Domínio de amostragem fixo (generate_samples_constrained) -- pesos da grade
R_IN_MM, R_EXT_MM  = 28.5, 46.5
ANG1_DEG, ANG2_DEG = 0.0, 120.0


class Accum:
    """Acumula e = | |B|_pred - |B|_true | ponderado por área sobre amostras."""
    # [REMOVIDO 2026-10-15] sem sum_sq (L2 ponto a ponto) -- acrescentado
    # FIELDS = ('num_l1', 'num_l2', 'den_area', 'num_ref', 'sum_abs', 'n_pts')
    FIELDS = ('num_l1', 'num_l2', 'den_area', 'num_ref', 'sum_abs', 'sum_sq', 'n_pts')

    def __init__(self):
        self.num_l1 = self.num_l2 = self.den_area = self.num_ref = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0          # Σe² sem peso -> L2 ponto a ponto (2026-10-15)
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
        self.sum_sq   += float((e ** 2).sum())
        self.n_pts    += e.size

    def report(self):
        b_ref = math.sqrt(self.num_ref / self.den_area)
        l1 = self.num_l1 / self.den_area
        l2 = math.sqrt(self.num_l2 / self.den_area)
        mae = self.sum_abs / self.n_pts
        l2pt = math.sqrt(self.sum_sq / self.n_pts)
        return dict(B_ref=b_ref, MAE_pt=mae, MAE_pt_pct=100 * mae / b_ref,
                    L1_area=l1, L1_area_pct=100 * l1 / b_ref,
                    L2_area=l2, L2_area_pct=100 * l2 / b_ref,
                    L2_pt=l2pt, L2_pt_pct=100 * l2pt / b_ref, n_pts=self.n_pts)


def merge(accs):
    m = Accum()
    for a in accs:
        for f in Accum.FIELDS:
            setattr(m, f, getattr(m, f) + getattr(a, f))
    return m


def node_err(mag_p, mag_t, area):
    """(L1_A, L2_A, pontual) em T de 1 amostra."""
    e = np.abs(mag_p - mag_t).astype(np.float64)
    a = area.astype(np.float64)
    return (float((e * a).sum() / a.sum()), float(math.sqrt((e ** 2 * a).sum() / a.sum())),
            float(e.mean()))


def pstats(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)))


def grid_area_weights(H, W):
    """dA = r·dr·dθ por linha (radial); angular uniforme."""
    dr     = (R_EXT_MM - R_IN_MM) / H
    dtheta = math.radians(ANG2_DEG - ANG1_DEG) / W
    r      = R_IN_MM + (np.arange(H) + 0.5) * dr
    return np.broadcast_to((r * dr * dtheta)[:, None], (H, W))
