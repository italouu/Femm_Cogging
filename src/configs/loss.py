from dataclasses import dataclass


@dataclass
class LossCfg:
    """
    Hiperparâmetros do termo de cauda (top-k), combinado de forma convexa com
    uma loss base (ex: mse).

    Cálculo (ver topk_tail_term em src/neural_op/losses.py):
        mag_true = magnitude vetorial de y      (sqrt(Bx²+By²), ou |y| se 1 canal)
        mag_pred = magnitude vetorial de out
        b_ref²   = mean(mag_true²)                       — RMS² global de |y| no batch
        err_rel  = (mag_pred - mag_true)² / b_ref²        — erro quadrático relativo por elemento
        k        = max(1, tail_k_frac * n_elementos)
        tail     = mean(top-k maiores valores de err_rel) — foco só na cauda (piores elementos)

        loss_final = (1 - tail_alpha) * loss_base(out, y) + tail_alpha * tail

    tail_alpha=0 → tail não é computado, loss_final == loss_base (comportamento idêntico ao anterior).
    Manter tail_alpha baixo: perto de 1 a parcela de loss_base praticamente desaparece.
    """
    tail_alpha:  float = 0.2    # peso do termo de cauda; 0 = desligado; manter baixo (ex: 0.1-0.3)
    tail_k_frac: float = 0.05   # fração dos piores elementos penalizados (top 5%, alinhado ao p95 já reportado em src/bench/metrics.py)
